import io
import os
import random
import warnings

import numpy as np
from AaronTools.atoms import Atom
from AaronTools.comp_output import CompOutput
from AaronTools.config import Config
from AaronTools.const import AARONLIB
from AaronTools.geometry import Geometry
from AaronTools.theory import GAUSSIAN_ROUTE
from fabric.connection import Connection
from fireworks import Firework, LaunchPad, ScriptTask, Workflow
from fireworks.user_objects.queue_adapters.common_adapter import (
    CommonAdapter as QueueAdapter,
)
from jinja2 import Environment, FileSystemLoader, StrictUndefined

warnings.filterwarnings("ignore", message=".*EllipticCurve.*")
warnings.filterwarnings("ignore", module=".*queue.*")
LAUNCHPAD = LaunchPad.auto_load()


class Job:
    def __init__(self, structure, config=None, quiet=False, step=0.0):
        self.quiet = quiet
        self.fw_id = None
        self.parent_fw_id = None
        if isinstance(step, int) and int(step) == step:
            self.step = int(step)
        else:
            self.step = step
        # job from fw spec
        if isinstance(structure, Firework):
            self.fw_id = structure.fw_id
            spec = structure.as_dict()["spec"]
            self._load_metadata(spec)
        else:
            if isinstance(structure, Geometry):
                self.structure = structure
            else:
                self.structure = Geometry(structure)
            if isinstance(config, Config):
                self.config = config
            else:
                self.config = Config(config, quiet=quiet)
            self._make_changes()
        self.step_list = self.get_steps()

    def find_fw(self, spec=None, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        if spec is None:
            spec = self._get_metadata(config=config)
        query_spec = {}
        for key, val in spec.items():
            if key in [
                "starting_structure",
                "_args",
                "_kwargs",
            ]:
                continue
            if "#" in key:
                continue
            query_spec["spec." + key] = val
        fw_id = LAUNCHPAD.get_fw_ids(query=query_spec)
        if len(fw_id):
            fw = LAUNCHPAD.get_fw_by_id(fw_id[0])
            return fw

    def add_fw(self, parent_fw_id=None, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        if parent_fw_id is None:
            parent_fw_id = self.parent_fw_id
        # make firework
        fw = Firework(
            [self._get_exec_task(config=config)],
            spec=self._get_metadata(config=config),
            name=config["HPC"]["job_name"],
        )
        if isinstance(parent_fw_id, list):
            LAUNCHPAD.append_wf(Workflow([fw]), parent_fw_id)
        elif parent_fw_id:
            LAUNCHPAD.append_wf(Workflow([fw]), [parent_fw_id])
        else:
            LAUNCHPAD.add_wf(Workflow([fw], name=config["Job"]["name"]))
        self.fw_id = fw.fw_id
        return fw

    def add_workflow(self, parent_fw_id=None):
        # create workflow
        self.parent_fw_id = parent_fw_id
        fws = []
        for step in self.step_list:
            self.step = step
            if step < 2 and not self.config._changed_list:
                continue
            fw = self.find_fw()
            if fw is not None:
                self.fw_id = fw.fw_id
            else:
                fw = self.add_fw()
            fws += [fw.fw_id]
            self.parent_fw_id = self.fw_id
        return fws

    def _make_changes(self):
        if not self.config._changes:
            return
        changed = []
        for name, (changes, kind) in self.config._changes.items():
            for key, val in changes.items():
                if kind == "Substitution":
                    for k in key.split(","):
                        k = k.strip()
                        sub = self.structure.substitute(val, k)
                    for atom in sub:
                        changed += [atom.name]
                elif kind == "Mapping":
                    key = [k.strip() for k in key.split(",")]
                    new_ligands = self.structure.map_ligand(val, key)
                    for atom in [ligand for ligand in new_ligands]:
                        changed += [atom.name]
        for con in list(
            eval(self.config["Geometry"].get("constraints", "[]"))
        ):
            for c in con:
                try:
                    changed.remove(str(c))
                except ValueError:
                    pass
        self.config._changed_list = changed

    def write(self, override_style=None, send=True, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        theory = config.get_theory(self.structure)
        exec_type = config["HPC"]["exec_type"]
        if exec_type == "gaussian":
            style = "com"
        if override_style is not None:
            style = override_style
        theory.geometry.write(
            name=os.path.join(
                config["Job"]["top_dir"], config["HPC"]["job_name"]
            ),
            style=style,
            theory=theory,
            **config._kwargs
        )
        if "host" in config["HPC"] and send:
            self.transfer_input(config=config)

    def transfer_input(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        name = self._get_input_name(config=config)
        source = os.path.join(config["Job"].get("top_dir"), name)
        target = os.path.join(config["Job"].get("remote_dir"), name)
        if not self.quiet:
            print(
                "Sending {} to {}...".format(name, config["HPC"].get("host"))
            )
        try:
            self.remote_put(source, target, config=config)
        except FileNotFoundError:
            self.remote_run(
                "mkdir -p {}".format(os.path.dirname(target)), config=config
            )
            self.remote_put(source, target, config=config)

    def launch_job(self, fw_id=None, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        if fw_id is None:
            fw_id = self.fw_id
        fw = LAUNCHPAD.get_fw_by_id(fw_id)
        if fw.state != "READY":
            return
        # transfer qadapter
        qadapter = self._make_qadapter(fw.fw_id, config=config)
        qlaunch_cmd = 'qlaunch -r -l "$AARONLIB/my_launchpad.yaml" -q "{}" --launch_dir "{}" singleshot --fw_id {}'.format(
            qadapter,
            config["HPC"]["work_dir"],
            fw.fw_id,
        )
        if config["HPC"].get("host"):
            out, err = self.remote_run(qlaunch_cmd, hide=True, config=config)
            if out:
                print("out:", out)
            for e in err.split("\n"):
                if (
                    "specified in qadapter but it is not present in template"
                    in e
                    or ".format(subs_key, self.template_file)" in e
                    or e.strip() == ""
                ):
                    continue
                print("err", e.strip())
        else:
            os.system(qlaunch_cmd)

    def transfer_output(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        name = self._get_ouptut_name(config=config)
        source = os.path.join(config["Job"]["remote_dir"], name)
        target = os.path.join(config["Job"]["top_dir"], name)
        new = self.remote_get(source, target, config=config)
        if not self.quiet and new:
            print("Downloaded {} from {}".format(name, config["HPC"]["host"]))

    def validate(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        name = self._get_ouptut_name(config=config)
        output = CompOutput(os.path.join(config["Job"]["top_dir"], name))
        if output.error:
            LAUNCHPAD.defuse_fw(self.fw_id)
        return output

    def update_structure(self, structure, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        for i, a in enumerate(structure):
            self.structure.atoms[i].coords = a.coords
        LAUNCHPAD.update_spec([self.fw_id], self._get_metadata(config=config))

    def resolve_error(self):
        config = self.config_for_step()
        resolved = False
        name = self._get_ouptut_name(config=config)
        output = CompOutput(os.path.join(config["Job"]["top_dir"], name))
        print("Trying to resolve {} error".format(output.error))
        if output.error in ["REDUND", "UNKNOWN"]:
            resolved = True
            self.update_structure(output.geometry)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["CONV_LINK"]:
            resolved = True
            old_fw = self.fw_id
            if self.step > 2:
                self.step = 2
            self.find_fw()
            LAUNCHPAD.defuse_fw(self.fw_id)
            self.update_structure(output.geometry)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.reignite_fw(old_fw)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["CONSTR"]:
            resolved = True
            self.update_structure(output.geometry)
            for i, j in list(eval(config["Geometry"]["constraints"])):
                a = self.structure.find(str(i))[0]
                b = self.structure.find(str(j))[0]
                a.coords = np.round(a.coords, 4)
                b.coords = np.round(b.coords, 4)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["EIGEN"]:
            resolved = True
            self.update_structure(output.geometry)
            if self.config["HPC"]["exec_type"] == "gaussian":
                self.config._kwargs[GAUSSIAN_ROUTE] = {"opt": ["noeigen"]}
            else:
                raise NotImplementedError
            LAUNCHPAD.update_spec([self.fw_id], self._get_metadata())
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["SCF_CONV"]:
            resolved = True
            if self.config["HPC"]["exec_type"] == "gaussian":
                self.config._kwargs[GAUSSIAN_ROUTE] = {
                    "maxstep": [15],
                    "scf": ["xqc"],
                }
            else:
                raise NotImplementedError
            old_fw = self.fw_id
            if self.step > 2:
                self.step = 2
            self.find_fw()
            LAUNCHPAD.defuse_fw(self.fw_id)
            self.update_structure(output.geometry)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.reignite_fw(old_fw)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error == "CHARGEMULT":
            raise Exception("Fatal error: Invalid charge/multiplicity.")
        return resolved

    def get_steps(self):
        step_list = []
        add_low = False
        add_high = False
        for option in self.config["Job"]:
            if "type" not in option:
                continue
            step = option.split()[0]
            try:
                if int(step) == float(step):
                    step_list += [int(step)]
                else:
                    step_list += [float(step)]
            except ValueError:
                if step.lower() == "low":
                    add_low = True
                if step.lower() == "high":
                    add_high = True
        if add_low:
            step_list += [1]
        step_list.sort()
        if add_high:
            step_list += [int(step_list[-1]) + 1]
        if self.config._changes:
            return step_list
        else:
            return step_list[1:]

    def config_for_step(self, step=None):
        if step is None:
            step = self.step
        config = self.config.copy()
        # find step-specific options
        for section in config._sections:
            for key, val in config[section].items():
                remove_key = key
                key = key.strip().split()
                if len(key) == 1:
                    continue
                key_step = key[0]
                key = " ".join(key[1:])
                try:
                    key_step = float(key_step)
                except ValueError:
                    key_step = key_step.strip()
                # screen based on step
                if key_step == "low" and step < 2 and step != 0:
                    config[section][key] = val
                if key_step == "high" and step >= 5:
                    config[section][key] = val
                if isinstance(key_step, float) and key_step == float(step):
                    config[section][key] = val
                # clean up metadata
                del config[section][remove_key]
        # other job-specific additions
        if "host" in config["HPC"]:
            try:
                config["HPC"]["work_dir"] = config["Job"].get("remote_dir")
            except TypeError:
                raise RuntimeError(
                    "Must specify remote working directory for HPC (remote_dir = /path/to/HPC/work/dir)"
                )
        else:
            config["HPC"]["work_dir"] = config["Job"].get("top_dir")
        if self.step:
            config["HPC"]["job_name"] = "{}.{}".format(
                self.structure.name, self.step
            )
        else:
            config["HPC"]["job_name"] = self.structure.name
        # parse user-supplied functions in config file
        config.parse_functions()
        return config

    def remote_put(self, source, target, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        host = config["HPC"].get("transfer_host")
        with Connection(host) as c:
            c.put(source, remote=target)
        return True

    def remote_get(self, source, target, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        host = config["HPC"].get("transfer_host")
        remote_mtime, _ = self.remote_run(
            "stat --format=%Y {}".format(source), hide=True, config=config
        )
        try:
            local_mtime = os.stat(target).st_mtime
        except FileNotFoundError:
            local_mtime = None
        if local_mtime is not None and int(local_mtime) > int(remote_mtime):
            return False
        with Connection(host) as c:
            c.get(source, target)
        return True

    def remote_run(self, cmd, hide=None, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        host = config["HPC"].get("host")
        with Connection(host) as c:
            result = c.run(cmd, hide=hide)
        return result.stdout, result.stderr

    def _repeat_step(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        self.find_fw(config=config)
        LAUNCHPAD.update_spec([self.fw_id], self._get_metadata(config=config))
        LAUNCHPAD.rerun_fw(self.fw_id)

    def _get_input_name(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        name = config["HPC"]["job_name"]
        exec_type = config["HPC"]["exec_type"]
        if exec_type == "gaussian":
            name += ".com"
        return name

    def _get_ouptut_name(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        name = config["HPC"]["job_name"]
        exec_type = config["HPC"]["exec_type"]
        if exec_type == "gaussian":
            name += ".log"
        return name

    def _get_exec_task(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        environment = Environment(
            loader=FileSystemLoader(AARONLIB), undefined=StrictUndefined
        )
        exec_template = environment.get_template(
            config["HPC"].get("exec_type") + ".template"
        )
        options = dict(config["HPC"].items())
        options["scratch_dir"] = os.path.join(
            options["scratch_dir"], str(random.random())
        )
        script = exec_template.render(**options)
        script = [line.strip() for line in script.split("\n") if line.strip()]
        return ScriptTask.from_str(" ; ".join(script))

    def _make_qadapter(self, fw_id, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        # find template
        if config["HPC"].get("host"):
            AARONLIB_REMOTE, _ = self.remote_run(
                "echo -n $AARONLIB", hide=True, config=config
            )
            qadapter_template = os.path.join(
                AARONLIB_REMOTE,
                "{}_qadapter.template".format(config["HPC"]["queue_type"]),
            )
        else:
            qadapter_template = os.path.join(
                AARONLIB,
                "{}_qadapter.template".format(config["HPC"]["queue_type"]),
            )
        if "rocket_launch" not in config["HPC"]:
            config["HPC"][
                "rocket_launch"
            ] = 'rlaunch -l "$AARONLIB/my_launchpad.yaml" singleshot'
        # create qadapter
        qadapter = QueueAdapter(
            q_type=config["HPC"]["queue_type"],
            q_name=config["HPC"]["queue"],
            template_file=qadapter_template,
            **dict(config["HPC"].items())
        )
        # save qadapter; transfer if remote
        rel_dir = os.path.dirname(config["HPC"]["job_name"])
        if config["HPC"].get("host"):
            filename = os.path.join(
                config["HPC"].get("remote_dir"),
                rel_dir,
                "qadapter_{}.yaml".format(fw_id),
            )
            content = qadapter.to_format(f_format="yaml")
            self.remote_put(
                io.StringIO(content),
                filename,
                config=config,
            )
        else:
            filename = os.path.join(
                config["HPC"].get("top_dir"),
                rel_dir,
                "qadapter_{}.yaml".format(fw_id),
            )
            qadapter.to_file(filename)
        return filename

    def _get_metadata(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        spec = {
            "step": self.step,
            "starting_structure": list(
                zip(
                    self.structure.elements,
                    self.structure.coords.tolist(),
                    [a.name for a in self.structure.atoms],
                )
            ),
        }
        for attr in ["_changes", "_changed_list", "_args", "_kwargs"]:
            spec[attr] = config.__dict__[attr]
        for section in config._sections:
            for key, val in config.items(section=section):
                spec["{}/{}".format(section, key).replace(".", "#")] = val
        return spec

    def _load_metadata(self, spec):
        self.config = Config(quiet=True)
        for attr in spec:
            if attr == "step":
                self.step = spec[attr]
            if attr == "starting_structure":
                atoms = []
                for element, coord, name in spec["starting_structure"]:
                    atoms += [Atom(element=element, coords=coord, name=name)]
                self.structure = Geometry(atoms)
            if attr in [
                "_changes",
                "_changed_list",
                "_args",
                "_kwargs",
            ]:
                self.config.__dict__[attr] = spec[attr]
            if "/" in attr:
                section, key = attr.replace("#", ".").split("/")
                if section not in self.config:
                    self.config.add_section(section)
                self.config[section][key] = spec[attr]
        self.structure.name = ".".join(
            self.config["HPC"]["job_name"].split(".")[0:-1]
        )
