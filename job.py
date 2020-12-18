import datetime
import io
import itertools as it
import os
import shutil
from time import sleep

import numpy as np
from AaronTools.atoms import Atom
from AaronTools.comp_output import CompOutput
from AaronTools.config import Config
from AaronTools.const import AARONLIB
from AaronTools.fileIO import Frequency
from AaronTools.geometry import Geometry
from AaronTools.theory import GAUSSIAN_ROUTE
from fabric.connection import Connection
from fireworks import (
    Firework,
    FWAction,
    FWorker,
    LaunchPad,
    ScriptTask,
    Workflow,
)
from fireworks.user_objects.queue_adapters.common_adapter import (
    CommonAdapter as QueueAdapter,
)
from invoke.exceptions import UnexpectedExit
from jinja2 import Environment, FileSystemLoader, StrictUndefined

LAUNCHPAD = LaunchPad.auto_load()


class Job:
    """
    Attributes:
    :structure: (Geometry) the structure
    :config: (Config) the configuration object
    :metadata: (dict) the workflow metadata (for searching)
    :step: (float or int) the current step
    :step_list: ([float or int]) all steps to perform
    :root_fw_id: (int) FW id for root FW of this job's workflow
    :fw_id: (int) the FW id for the current step
    :parent_fw_id: (int) FW id for direct parent of fw_id
    :quiet: (bool) supress extraneous printing if true
    """

    @classmethod
    def get_query_spec(cls, spec, skip_keys=None):
        """
        Builds a suitable pymongo-style query from metadata spec.
        Returns: dict()

        :spec: the metadata dictionary, like that returned by Job._get_metadata()
        :skip_keys: list of keys in spec to skip when building return dict()
        """
        if skip_keys is None:
            skip_keys = []
        query_spec = {}
        for key, val in spec.items():
            if key in skip_keys:
                continue
            # these values are variable during the course of the job
            if key in [
                "starting_structure",
                "_args",
                "_kwargs",
                "_changed_list",
                "HPC/name",
            ]:
                continue
            # shouldn't create new FWs if we move everything to a new directory
            # local directory stuff is built from other values, so this is fine
            if "_dir" in key:
                continue
            # these have been parsed and saved as "_changes"
            if (
                "reopt" in key
                or key.startswith("Substitution")
                or key.startswith("Mapping")
            ):
                continue
            # these are from unused include sections
            if "#" in key:
                continue
            query_spec["spec." + key] = val
        return query_spec

    def __init__(
        self, structure, config=None, quiet=False, step=0.0, make_changes=True
    ):
        """
        :structure: the geometry structure
        :config: the configuration object to associate with this job
        :quiet: print extra info when loading config if False
        :step: the current step for this job
        :make_changes: do structure modification as specified in config
        """
        self.quiet = quiet
        self.fw_id = None
        self.root_fw_id = None
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
            if config:
                self.config = config
        else:
            if isinstance(structure, Geometry):
                self.structure = structure
            else:
                self.structure = Geometry(structure)
            if isinstance(config, Config):
                self.config = config
            else:
                self.config = Config(config, quiet=quiet)
            if make_changes:
                self._make_changes()
        self.step_list = self.get_steps()

    def set_fw_id(self, spec=None, step=None, config=None, skip_keys=None):
        """
        Sets self.fw_id if able to find the FW specified, if found, or None

        :spec: the metadata spec of the FW to find, defaults to self._get_metadata()
        :step: which step to apply to config, defaults to self.step
        :config: the config associated with spec, defaults to self.config
        :skip_keys: list of keys in spec to skip when searching for FW
        """
        if skip_keys is None:
            skip_keys = []
        fw = self.find_fw(spec, step, config, skip_keys)
        if fw:
            fw_id = fw.fw_id
        else:
            fw_id = None
        self.fw_id = fw_id

    def find_fw(self, spec=None, step=None, config=None, skip_keys=None):
        """
        Searches the launchpad for a particular FW
        Returns the FW, if unique FW found, or None

        :spec: the metadata spec of the FW to find, defaults to self._get_metadata()
        :step: which step to apply to config, defaults to self.step
        :config: the config associated with spec, defaults to self.config
        :skip_keys: list of keys in spec to skip when searching for FW
        """
        if skip_keys is None:
            skip_keys = []
        if config is None:
            config = self.config_for_step(step)
        else:
            config = self.config_for_step(step, config)
        if spec is None:
            spec = self._get_metadata(config=config)
        query_spec = Job.get_query_spec(spec, skip_keys=skip_keys)
        query_spec["state"] = {"$not": {"$eq": "ARCHIVED"}}
        fw_id = LAUNCHPAD.get_fw_ids(query=query_spec)
        fws = []
        for fw in fw_id:
            # archived FWs are soft-deleted, so we don't want those
            fws += [LAUNCHPAD.get_fw_by_id(fw)]
        if fws and len(fws) == 1:
            return fws.pop()
        elif fws:
            raise RuntimeError(
                "fw spec returns multiple fireworks (ids: {})".format(
                    [fw.fw_id for fw in fws]
                )
            )
        return None

    def set_root(self, fw_id=None):
        """
        Sets self.root_fw_id to the root FW for the workflow containing `fw_id`

        :fw_id: defaults to self.fw_id
        """
        if fw_id is None:
            fw_id = self.fw_id
        links = LAUNCHPAD.get_wf_summary_dict(fw_id, mode="all")["links"]
        children = set(it.chain.from_iterable(links.values()))
        for key in links:
            if key not in children:
                self.root_fw_id = int(key.split("--")[1])
                break

    def add_fw(self, parent_fw_id=None, step=None, config=None):
        """
        Adds FW to the appropriate workflow (creating if necessary) and updates self.fw_id accordingly
        Returns: the created FW

        :parent_fw_id: for linking within workflow (default to self.parent_fw_id)
        :step: defaults to self.step
        :config: defaults to self.config
        """
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
            LAUNCHPAD.add_wf(
                Workflow(
                    [fw],
                    name=config["Job"]["name"],
                    metadata=self.config.metadata,
                )
            )
        self.fw_id = fw.fw_id
        return fw

    def _root_fw(self):
        """
        Makes the root fw, if we need one, and marks as completed.
        The root fw is for organizational purposes for multi-step jobs, nothing is run.
        """
        # root fw only needed for multi-step jobs
        if not self.step_list:
            return None
        # if we can find a FW for the original structure, use that FW's root
        # instead of making a new one (changes are children of original)
        spec = self._get_metadata()
        spec["_changes"] = {"": [{}, None]}
        spec = self.get_query_spec(
            spec, skip_keys=["step", "Job/name", "HPC/job_name"]
        )
        wfs = set({})
        for fw_id in LAUNCHPAD.get_fw_ids(spec):
            wf = LAUNCHPAD.get_wf_by_fw_id(fw_id)
            for i, links in wf.links.items():
                if not links:
                    wfs.add(i)
        if len(wfs) == 1:
            return wfs.pop()

        # build spec for new root fw
        config = self.config_for_step(0)
        spec = {}
        for key, val in self._get_metadata(config=config).items():
            if "#" in key:
                continue
            if key.startswith("Substitution") or key.startswith("Mapping"):
                continue
            if "name" in key:
                continue
            if "dir" in key:
                continue
            if key in ["step", "starting_structure", "_args", "_kwargs"]:
                continue
            spec[key] = val
        name = ""
        if "Reaction/reaction" in spec:
            name = spec["Reaction/reaction"]
        if "Reaction/template" in spec:
            if name:
                name = os.path.join(name, spec["Reaction/template"])
            else:
                name = spec["Reaction/template"]
        if "_changes" in spec:
            tmp = "_".join(spec["_changes"].keys())
            if name and tmp:
                name = os.path.join(name, tmp)
            elif not name:
                name = tmp
        if name:
            name = os.path.join(name, self.structure.name)
        else:
            name = self.structure.name
        spec["Reaction/name"] = os.path.basename(name)
        fw = self.find_fw(spec=spec)
        if fw is None:
            fw = Firework([], spec=spec, name=name)
            LAUNCHPAD.add_wf(
                Workflow([fw], name=name, metadata=self.config.metadata)
            )
        if fw.state == "READY":
            fw, launch_id = LAUNCHPAD.checkout_fw(
                FWorker(), "", fw_id=fw.fw_id
            )
            LAUNCHPAD.complete_launch(launch_id, action=FWAction())
        return fw.fw_id

    def add_workflow(self, parent_fw_id=None):
        if not self.step_list and not "".join(self.config._changes.keys()):
            if parent_fw_id is not None:
                self.parent_fw_id = parent_fw_id
            fw = self.find_fw()
            if fw is not None:
                self.fw_id = fw.fw_id
                LAUNCHPAD.update_spec([self.fw_id], self._get_metadata())
            else:
                fw = self.add_fw()
            return [fw.fw_id]

        if parent_fw_id is None:
            self.parent_fw_id = self._root_fw()
        else:
            self.parent_fw_id = parent_fw_id
        # create workflow
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
                        if val.lower() == "none":
                            self.structure -= self.structure.get_fragment(k)
                        else:
                            sub = self.structure.substitute(val, k)
                            for atom in sub:
                                changed += [atom.name]
                elif kind == "Mapping":
                    key = [k.strip() for k in key.split(",")]
                    new_ligands = self.structure.map_ligand(val, key)
                    for ligand in new_ligands:
                        for atom in ligand:
                            changed += [atom.name]
        try:
            con_list = list(
                eval(self.config["Geometry"].get("constraints", "[]"), {})
            )
        except KeyError:
            self.structure.parse_comment()
            try:
                con_list = self.structure.other["constraint"]
            except KeyError:
                con_list = []
        for con in con_list:
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
        if "transfer_host" in config["HPC"] and send:
            self.transfer_input(config=config)

    def transfer_input(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        name = self._get_input_name(config=config)
        source = os.path.join(config["Job"].get("top_dir"), name)
        target = os.path.join(config["HPC"].get("remote_dir"), name)
        if not self.quiet:
            print(
                "Sending {} to {}...".format(name, config["HPC"].get("host"))
            )
        try:
            self.remote_put(source, target, config=config)
        except FileNotFoundError:
            sleep(5)
            self.remote_run(
                "mkdir -p {}".format(os.path.dirname(target)),
                hide=True,
                config=config,
            )
            sleep(5)
            self.remote_run(
                "touch {}".format(target), hide=True, config=config
            )
            for _ in range(5):
                # for some reason this doesn't go through every time even
                # though it absolutely should... Some bug in fabric module?
                # regardless, trying a few more time tends to resolve the issue
                sleep(5)
                try:
                    self.remote_put(source, target, config=config)
                    break
                except FileNotFoundError:
                    pass

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
            os.path.join(
                config["HPC"]["work_dir"],
                os.path.dirname(config["HPC"]["job_name"]),
            ),
            fw.fw_id,
        )
        if config["HPC"].get("host"):
            out, err = self.remote_run(qlaunch_cmd, hide=True, config=config)
            if out:
                print("out:", out)
            for e in err.split("\n"):
                if "has been specified in qadapter" in e:
                    continue
                if e.strip():
                    print("err", e.strip())
        else:
            os.system(qlaunch_cmd)

    def transfer_output(self, step=None, config=None):
        if config is None:
            config = self.config_for_step(step)
        name = self._get_output_name(config=config)
        source = os.path.join(config["HPC"]["remote_dir"], name)
        target = os.path.join(config["Job"]["top_dir"], name)
        new = self.remote_get(source, target, config=config)
        if not self.quiet and new:
            print("Downloaded {} from {}".format(name, config["HPC"]["host"]))

    def get_output(self, load_geom=False):
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        if fw.state != "COMPLETED":
            return None
        output = CompOutput()
        for key, val in fw.launches[-1].action.stored_data.items():
            if key == "geometry" and load_geom:
                atoms = []
                for element, coords in val:
                    atoms.append(Atom(element, coords))
                val = Geometry(atoms)
            if key == "frequency" and val:
                val = Frequency(
                    [Frequency.Data(**item) for item in val["data"]]
                )
            setattr(output, key, val)
        return output

    def validate(self, step=None, config=None, get_all=False):
        if config is None:
            config = self.config_for_step(step)
        name = self._get_output_name(config=config)
        try:
            output = CompOutput(
                os.path.join(config["Job"]["top_dir"], name), get_all=get_all
            )
        except Exception:
            LAUNCHPAD.defuse_fw(self.fw_id)
            LAUNCHPAD.update_spec([self.fw_id], self._get_metadata())
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
            return None
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        if fw.state == "RUNNING":
            launch = fw.launches[-1]
            state_history = launch.state_history
            for history in state_history:
                if history["state"] == "RUNNING":
                    runtime = (
                        datetime.datetime.utcnow() - history["created_on"]
                    )
                    walltime = datetime.timedelta(
                        hours=int(config["HPC"]["wall"])
                    )
                    if runtime > walltime:
                        output.error = "WALLTIME"
                        data = output.to_dict(skip_attrs=["opts"])
                        data["error"] = "WALLTIME"
                        LAUNCHPAD.complete_launch(
                            launch.launch_id,
                            action=FWAction(stored_data=data),
                        )
                        LAUNCHPAD.defuse_fw(fw.fw_id)
                        return output
        if fw.state in ["COMPLETED", "DEFUSED", "FIZZLED"]:
            try:
                launch = fw.launches[-1]
            except IndexError:
                launch = fw.archived_launches[-1]
            LAUNCHPAD.complete_launch(
                launch.launch_id,
                action=FWAction(
                    stored_data=output.to_dict(skip_attrs=["opts"])
                ),
            )
            if output.error:
                LAUNCHPAD.defuse_fw(fw.fw_id)
        return output

    def update_structure(
        self, structure, step=None, config=None, kind="output"
    ):
        if step is None:
            step = self.step
        if config is None:
            config = self.config_for_step(step)
        if structure is None:
            name = None
            if kind == "original":
                template = config.get_template()
                if isinstance(template, Geometry):
                    structure = template
                else:
                    for struct, kind in template:
                        if self.structure.name.endswith(struct.name):
                            structure = struct
                            break
                self.structure = structure
                self._make_changes()
                LAUNCHPAD.update_spec(
                    [self.fw_id], self._get_metadata(config=config)
                )
                return
            if kind == "input":
                name = self._get_input_name(config=config)
            else:
                name = self._get_output_name(config=config)
            if name is not None:
                structure = Geometry(
                    os.path.join(config["Job"]["top_dir"], name)
                )
        for i, a in enumerate(structure):
            self.structure.atoms[i].coords = a.coords
        LAUNCHPAD.update_spec([self.fw_id], self._get_metadata(config=config))

    def resolve_error(self, force_rerun=False):
        """
        Makes changes to automatically resolve computational errors
        Returns: True if error was resolved, else False

        :force_rerun: re-runs (after updating) a job that had failed due to error
            that the user must resolve by hand (eg: typos that cause job failure)
        """
        config = self.config_for_step()
        name = self._get_output_name(config=config)
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        output = CompOutput(os.path.join(config["Job"]["top_dir"], name))
        if output.error is None:
            # just in case something got marked that shouldn't have
            # (b/c of race condition???) eg: output file update needed
            LAUNCHPAD.reignite_fw(self.fw_id)
            print("Normal termination detected! Marking COMPLETED")
            return True
        resolved = False
        if fw.archived_launches[-1].action.stored_data["error"] == "WALLTIME":
            output.error = "WALLTIME"
        print("Trying to resolve {} error".format(output.error))
        # archive log files of failed jobs
        new_name = os.path.join(config["Job"]["top_dir"], "archived")
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        try:
            launch_id = fw.launches[-1].launch_id
        except IndexError:
            launch_id = fw.archived_launches[-1].launch_id
        new_name = os.path.join(
            new_name,
            "{}_{}.{}.{}".format(
                name,
                output.error,
                fw.fw_id,
                launch_id,
            ),
        )
        os.makedirs(os.path.dirname(new_name), exist_ok=True)
        shutil.copyfile(os.path.join(config["Job"]["top_dir"], name), new_name)
        # attempt to fix error
        if output.error in ["UNKNOWN", "WALLTIME"]:
            resolved = True
            self.update_structure(output.geometry)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["REDUND", "FBX"]:
            resolved = True
            for a in output.geometry:
                a.coords = np.array(
                    [round(c, 4) for c in a.coords], dtype=float
                )
            self.update_structure(output.geometry)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["CONV_LINK", "CONV_CDS", "LINK"]:
            resolved = True
            old_fw = self.fw_id
            if self.step > 2:
                self.step = 2
            config = self.config_for_step()
            self.find_fw(config=config)
            LAUNCHPAD.defuse_fw(self.fw_id)
            self.update_structure(output.geometry)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.reignite_fw(old_fw)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["CONSTR"]:
            resolved = True
            self.update_structure(output.geometry)
            for i, j in list(eval(config["Geometry"]["constraints"], {})):
                a = self.structure.find(str(i))[0]
                b = self.structure.find(str(j))[0]
                a.coords = np.round(a.coords, 4)
                b.coords = np.round(b.coords, 4)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["COORD"]:
            resolved = True
            self.update_structure(output.geometry)
            self.config._kwargs[GAUSSIAN_ROUTE] = {"opt": ["cartesian"]}
            LAUNCHPAD.update_spec([self.fw_id], self._get_metadata())
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["EIGEN"]:
            resolved = True
            self.update_structure(output.geometry)
            fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
            for launch in fw.archived_launches:
                if "EIGEN" == launch.action.stored_data["error"]:
                    if self.config["HPC"]["exec_type"] == "gaussian":
                        self.config._kwargs[GAUSSIAN_ROUTE] = {
                            "opt": ["noeigen"]
                        }
                        LAUNCHPAD.update_spec(
                            [self.fw_id], self._get_metadata()
                        )
                        LAUNCHPAD.reignite_fw(self.fw_id)
                        LAUNCHPAD.rerun_fw(self.fw_id)
                    else:
                        raise NotImplementedError
                    break
            else:
                LAUNCHPAD.update_spec([self.fw_id], self._get_metadata())
                LAUNCHPAD.reignite_fw(self.fw_id)
                self.step = 2
                config = self.config_for_step()
                self.find_fw()
                LAUNCHPAD.defuse_fw(self.fw_id)
                if self.config["HPC"]["exec_type"] == "gaussian":
                    self.config._kwargs[GAUSSIAN_ROUTE] = {
                        "opt": ["recalcfc=10"]
                    }
                    LAUNCHPAD.update_spec([self.fw_id], self._get_metadata())
                    LAUNCHPAD.reignite_fw(self.fw_id)
                    LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error in ["SCF_CONV"]:
            fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
            maxstep = 0
            for launch in fw.archived_launches:
                if launch.action.stored_data["error"] in ["SCF_CONV"]:
                    maxstep += 1
            try:
                maxstep = [15, 12, 10, 8, 5, 2][maxstep]
            except IndexError:
                maxstep = 0
            resolved = True
            if self.config["HPC"]["exec_type"] == "gaussian":
                if maxstep:
                    self.config._kwargs[GAUSSIAN_ROUTE] = {
                        "opt": ["maxstep={}".format(maxstep)],
                    }
                else:
                    self.config._kwargs[GAUSSIAN_ROUTE] = {"scf": ["xqc"]}
            else:
                raise NotImplementedError
            old_fw = self.fw_id
            if self.step > 2:
                self.step = 2
            self.find_fw()
            LAUNCHPAD.defuse_fw(self.fw_id)
            if self.step > 1:
                self.update_structure(None, kind="input")
            else:
                self.update_structure(None, kind="original")
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.reignite_fw(old_fw)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if output.error == "CLASH":
            geom = output.geometry
            bad_subs = geom.remove_clash()
            self.update_structure(geom)
            if not bad_subs:
                resolved = True
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)

        # stuff Aaron can't fix
        if output.error == "GALLOC":
            # this happens sometimes and is generally a node problem.
            # restarting is usually fine, may not be if job is picked up
            # on the same node as before (TODO: node-specific submission)
            resolved = True
            self.update_structure(output.geometry)
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
        if force_rerun:
            LAUNCHPAD.update_spec([self.fw_id], self._get_metadata())
            LAUNCHPAD.reignite_fw(self.fw_id)
            LAUNCHPAD.rerun_fw(self.fw_id)
            resolved = True
        elif output.error == "CHARGEMULT":
            raise RuntimeError(
                "Bad charge/multiplicity provided. Please fix you Aaron input file or template geometries."
            )
        elif output.error == "ATOM":
            raise RuntimeError("Bad atomic symbol, check template XYZ files.")
        elif output.error == "BASIS":
            raise RuntimeError(
                "Error reading basis set. Confirm that gen=/path/to/basis/ is correct in your Aaron config/input files and that the basis set file requested exists, or switch to an internally provided basis set."
            )
        elif output.error == "QUOTA":
            raise RuntimeError(
                "Erroneous write. Check quota or disk space, then restart Aaron."
            )
        elif output.error == "MEM":
            raise RuntimeError(
                "Node(s) out of memory. Increase memory requested in Aaron input file"
            )
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

    def config_for_step(self, step=None, config=None):
        if step is None:
            step = self.step
        if config is None:
            config = self.config.copy()
        return config.for_step(step)

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
        try:
            remote_mtime, _ = self.remote_run(
                "stat --format=%Y {}".format(source), hide=True, config=config
            )
        except UnexpectedExit:
            # catch race condition:
            # log file not yet created on HPC even though job running
            return False
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

    def _get_output_name(self, step=None, config=None):
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
            options["scratch_dir"], str(hash(self))
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
        job_name = config["HPC"]["job_name"]
        config["HPC"]["job_name"] = job_name.replace("/", " ")
        qadapter = QueueAdapter(
            q_type=config["HPC"]["queue_type"],
            q_name=config["HPC"]["queue"],
            template_file=qadapter_template,
            **dict(config["HPC"].items())
        )
        config["HPC"]["job_name"] = job_name
        # save qadapter; transfer if remote
        rel_dir = os.path.dirname(config["HPC"]["job_name"])
        if config["HPC"].get("host"):
            filename = os.path.join(
                config["HPC"].get("remote_dir"),
                rel_dir,
                "qadapter_{}.yaml".format(fw_id),
            )
            content = qadapter.to_format(f_format="yaml")
            self.remote_run(
                "mkdir -p {}".format(os.path.dirname(filename)), hide=True
            )
            self.remote_run("touch {}".format(filename), hide=True)
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
            "starting_structure": [self.structure.comment]
            + list(
                zip(
                    self.structure.elements,
                    self.structure.coords.tolist(),
                    [a.name for a in self.structure.atoms],
                )
            ),
        }
        spec = config.get_spec(spec)
        return spec

    def _load_metadata(self, spec):
        self.config = Config(quiet=True)
        for attr in spec:
            if attr == "step":
                self.step = spec[attr]
            if attr == "starting_structure":
                comment = spec["starting_structure"][0]
                atoms = []
                for element, coord, name in spec["starting_structure"][1:]:
                    atoms += [Atom(element=element, coords=coord, name=name)]
                self.structure = Geometry(atoms)
                self.structure.comment = comment
                self.structure.parse_comment()
            elif not hasattr(self, "structure"):
                self.structure = Geometry()
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
        if not self.structure.name:
            self.structure.name = self.config["Job"]["name"]
        config = self.config_for_step()
        self.structure.name = ".".join(
            config["HPC"]["job_name"].split(".")[0:-1]
        )
