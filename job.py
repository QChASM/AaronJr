import datetime
import hashlib
import io
import itertools as it
import os
import re
import shutil

import numpy as np
import paramiko
from AaronTools import addlogger, getlogger
from AaronTools.atoms import Atom
from AaronTools.comp_output import CompOutput
from AaronTools.config import Config
from AaronTools.const import AARONLIB, RMSD_CUTOFF, UNIT
from AaronTools.fileIO import Frequency
from AaronTools.geometry import Geometry
from AaronTools.theory import GAUSSIAN_ROUTE
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
from jinja2 import Environment, FileSystemLoader

TEMPLATES = [
    os.path.join(AARONLIB, "templates"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
]
LAUNCHPAD = LaunchPad.auto_load()
MAX_ATTEMPTS = 10  # number of attempts, regardless of error
MAX_SUBMIT = (
    3  # number of submission attempts (errors with queue submission itself)
)


def find_qadapter_template(qadapter_template):
    for path in TEMPLATES:
        rv = os.path.join(path, qadapter_template)
        if os.access(rv, os.R_OK):
            return rv
    log = getlogger()
    log.exception("Could not find %s in %s", qadapter_template, str(TEMPLATES))


@addlogger
class Job:
    """
    Attributes:
    :structure: (Geometry) the structure
    :config: (Config) the configuration object
    :step: (float or int) the current step
    :step_list: ([float or int]) all steps to perform
    :root_fw_id: (int) FW id for root FW of this job's workflow
    :fw_id: (int) the FW id for the current step
    :parent_fw_id: (int) FW id for direct parent of fw_id
    :quiet: (bool) supress extraneous printing if true
    """

    # spec items to skip when building query dict
    SKIP_SPEC = [
        ("Geometry", "(?!constrain(ts)?).*"),
        ("Results", ".*"),
        ("Plot", ".*"),
        ("HPC", ".*"),
        # should be able to change conformer restrictions and still find the FW
        ("Job", "(max_conformers|energy_cutoff|rmsd_cutoff)"),
        ("Substitution", ".*"),  # already included in config._changes
        ("Mapping", ".*"),  # already included in config._changes
        # these can change without needing new FW
        (
            ".*",
            "(.*_dir|local_only|log_level|.*_citations)",
        ),
        (".*", "project"),  # this is included in metadata
        (".*", "include"),  # this has been parsed into the main body
    ]
    ENVIRONMENT = Environment(loader=FileSystemLoader(TEMPLATES))
    LOG = None
    LOGLEVEL = "INFO"
    SKIP_CONNECT = False

    def __init__(
        self, structure, config, quiet=True, make_root=True, testing=False
    ):
        """
        :structure: the geometry structure
        :config: the configuration object to associate with this job
        :quiet: print extra info when loading config if False
        :make_changes: do structure modification as specified in config
        :make_root: set to False to prevent addition of root FW (it will still check to see if it's there)
        :testing: this should only be set to True for testing purposes
        """
        self.quiet = quiet
        self.config = None
        self.step_list = []
        self.structure = None
        self.structure_hash = None

        self.step = 0
        self.conformer = 0
        self.fw_id = None
        self.parent_fw_id = None
        self.root_fw_id = None

        self.RUN_CONNECTION = None
        self.XFER_CONNECTION = None

        # loading in config-defined stuff
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config(config, quiet=quiet)
        self.step_list = self.get_steps()
        try:
            self.step = self.step_list[0]
        except IndexError:
            pass

        run_user = self.config.get("HPC", "user", fallback=False)
        run_host = self.config.get("HPC", "host", fallback=False)
        xfer_user = self.config.get("HPC", "transfer_user", fallback=run_user)
        xfer_host = self.config.get("HPC", "transfer_host", fallback=run_host)
        if not self.SKIP_CONNECT and self.RUN_CONNECTION is None and run_host:
            self.RUN_CONNECTION = paramiko.client.SSHClient()
            self.RUN_CONNECTION.load_system_host_keys()
            self.RUN_CONNECTION.connect(run_host, username=run_user)
        if (
            not self.SKIP_CONNECT
            and self.XFER_CONNECTION is None
            and xfer_host
        ):
            self.XFER_CONNECTION = paramiko.client.SSHClient()
            self.XFER_CONNECTION.load_system_host_keys()
            self.XFER_CONNECTION.connect(xfer_host, username=xfer_user)

        make_changes = True
        # load structure from fw spec
        if isinstance(structure, Firework):
            self.load_fw(structure)
            make_changes = False
        # or from passed geometry
        elif isinstance(structure, Geometry):
            self.structure = structure
        else:
            self.structure = Geometry(structure)
        if make_changes:
            self._make_changes()
        if not self.structure_hash:
            old_level = Geometry.LOG.level
            Geometry.LOG.setLevel("ERROR")
            self.structure_hash = hash(self.structure.copy())
            Geometry.LOG.setLevel(old_level)

        # find fw id for root and current step's job
        if not testing:
            self.set_root(make_root=make_root)

    # remote connection commands
    def remote_mkdir(self, dirname):
        if not dirname:
            return
        with self.XFER_CONNECTION.open_sftp() as sftp:
            try:
                sftp.mkdir(dirname)
            except FileNotFoundError:
                self.remote_mkdir(os.path.dirname(dirname))
                sftp.mkdir(dirname)
            except OSError:
                self.LOG.exception(
                    "Couldn't mkdir %s on remote host. This can often be resolved by "
                    "simply restarting AaronJr. If you still have problems, try "
                    "creating the directory manually.",
                    dirname,
                )

    def remote_put(self, source, target):
        if self.XFER_CONNECTION is None:
            return False

        def _put(source, target, put_method):
            try:
                put_method(source, target)
            except FileNotFoundError:
                self.remote_mkdir(os.path.dirname(target))
                put_method(source, target)

        with self.XFER_CONNECTION.open_sftp() as sftp:
            try:
                _put(source, target, sftp.put)
            except TypeError:
                _put(source, target, sftp.putfo)
        return True

    def remote_get(self, source, target):
        if self.XFER_CONNECTION is None:
            return False
        try:
            with self.XFER_CONNECTION.open_sftp() as sftp:
                remote_mtime = sftp.stat(source).st_mtime
        except FileNotFoundError:
            return False
        try:
            local_mtime = os.stat(target).st_mtime
        except FileNotFoundError:
            local_mtime = None
        if local_mtime is not None and int(local_mtime) > int(remote_mtime):
            return False
        try:
            with self.XFER_CONNECTION.open_sftp() as sftp:
                sftp.get(source, target)
        except FileNotFoundError:
            os.mkdir(os.path.dirname(target))
            with self.XFER_CONNECTION.open_sftp() as sftp:
                sftp.get(source, target)
        return True

    def remote_run(self, cmd):
        if self.RUN_CONNECTION is None:
            return None, None
        _, stdout, stderr = self.RUN_CONNECTION.exec_command(cmd)
        stdout = stdout.read().decode()
        stderr = stderr.read().decode()
        # self.LOG.debug("%s %s", stdout, stderr)
        return stdout, stderr

    # utilities
    def _make_changes(self):
        self.structure = self.config.make_changes(self.structure)

    def _root_fw(self, workflow=None, make_root=True):
        """
        Makes the root fw, if we need one, and marks as completed.
        The root fw is for organizational purposes for multi-step jobs, nothing is run.
        """
        # if we can find a FW for the original structure, use that FW's root
        # instead of making a new one (changes are children of original)
        self.set_fw()
        if self.fw_id:
            self.set_root()
            return self.root_fw_id

        # build spec for new root fw
        spec = self.get_spec(step="all")
        for key in self.config.SPEC_ATTRS:
            if key in ["infile", "metadata"]:
                continue
            del spec[key]
        name = self.get_workflow_name()
        fw = self.find_fw(spec=spec)
        if fw is None and make_root:
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

    def _get_exec_task(self, step=None, conformer=None, structure=None):
        config = self.config_for_step(step=step, conformer=conformer)
        if structure is None:
            structure = self.structure
        exec_type = config["Job"]["exec_type"]
        exec_template = self.ENVIRONMENT.get_template(exec_type + ".template")
        options = dict(
            list(config["HPC"].items()) + list(config["Job"].items())
        )
        filename = os.path.join(
            config["HPC"]["work_dir"],
            self.get_basename(step=step, conformer=conformer),
        )
        options["work_dir"], options["job_name"] = os.path.split(filename)
        # build hash for scratch subdirectory using search spec and filename
        h = hashlib.sha1(
            str(self.query_spec(step=step, conformer=conformer)).encode()
        )
        h.update(filename.encode())
        options["scratch_dir"] = os.path.join(
            config["HPC"].get("scratch_dir", fallback="scratch"),
            h.hexdigest(),
        )
        if exec_type in ["xtb", "crest"]:
            theory = config.get_theory(structure)
            cmdline = theory.get_xtb_cmdline(config)
            if "--optts" in cmdline:
                options["optts"] = "true"
                del cmdline["--optts"]
            options["cmdline"] = ""
            for key, val in cmdline.items():
                if val is not None:
                    options["cmdline"] += "{} {} ".format(key, val)
                else:
                    options["cmdline"] += "{} ".format(key)
            options["cmdline"] = options["cmdline"].rstrip()

        script = exec_template.render(**options)
        return ScriptTask.from_str(script)

    def _make_qadapter(self, fw_id):
        config = self.config_for_step()
        qadapter_template = self._find_qadapter_template()
        # transfer template, if needed
        if config["HPC"].get("host"):
            AARONLIB_REMOTE, _ = self.remote_run("echo -n $AARONLIB")
            remote_template = os.path.join(
                AARONLIB_REMOTE,
                "templates",
                "{}_qadapter.template".format(config["HPC"]["queue_type"]),
            )
            self.remote_put(qadapter_template, remote_template)
            qadapter_template = remote_template
        if "rocket_launch" not in config["HPC"]:
            config["HPC"][
                "rocket_launch"
            ] = 'rlaunch -l "$AARONLIB/my_launchpad.yaml" singleshot'
        # create qadapter
        qadapter = QueueAdapter(
            q_type=config["HPC"]["queue_type"],
            q_name=config["HPC"]["queue"],
            template_file=qadapter_template,
            **dict(list(config["Job"].items()) + list(config["HPC"].items())),
        )
        # save qadapter; transfer if remote
        rel_dir = os.path.dirname(config["Job"]["name"])
        if config["HPC"].get("host"):
            filename = os.path.join(
                config["HPC"].get("remote_dir"),
                rel_dir,
                "qadapter_{}.yaml".format(fw_id),
            )
            content = qadapter.to_format(f_format="yaml")
            self.remote_run("mkdir -p {}".format(os.path.dirname(filename)))
            self.remote_run("touch {}".format(filename))
            self.remote_put(io.StringIO(content), filename)
        else:
            filename = os.path.join(
                config["HPC"].get("top_dir"),
                rel_dir,
                "qadapter_{}.yaml".format(fw_id),
            )
            qadapter.to_file(filename)
        return filename

    def _find_qadapter_template(self, step=None):
        if step is None:
            step = self.step
        config = self.config_for_step(step=step)
        qadapter_template = "{}_qadapter.template".format(
            config["HPC"]["queue_type"]
        )
        return find_qadapter_template(qadapter_template)

    def _structure_dict(self, structure=None):
        if structure is None:
            structure = self.structure
        if isinstance(structure, Geometry):
            atoms = [str(a) for a in structure]
            comment = structure.comment
        elif isinstance(structure, tuple):
            comment, atoms = structure
        else:
            self.LOG.debug("%s %s", type(structure), structure)
        return {"comment": comment, "atoms": atoms}

    def _last_launch(self, fw):
        try:
            return fw.launches[-1]
        except IndexError:
            return fw.archived_launches[-1]

    def _get_stored_data(self, output):
        skip_attrs = ["opts", "other"]
        data = {}
        for key, val in output.to_dict().items():
            if key in skip_attrs:
                continue
            if key not in ["error", "error_msg", "finished"]:
                try:
                    if len(val) == 0:
                        continue
                except TypeError:
                    pass
                if val is None:
                    continue
            if key == "conformers" and val is not None:
                tmp = {}
                for i, v in enumerate(val):
                    tmp[str(i)] = self._structure_dict(v)
                data[str(key)] = tmp
                continue
            if key == "geometry" and val is not None:
                data[str(key)] = self._structure_dict(val)
                continue
            if key == "frequency":
                # this dict can easily be recreated by calling sort_frequencies
                # and the float keys mess up bson storage (can only have str keys)
                del val["by_frequency"]
            data[str(key)] = val
        return data

    def get_steps(self):
        step_dict = {}
        for option, val in self.config["Job"].items():
            key = option.split()
            if len(key) == 1 and key == "type":
                step_dict[0] = val
                continue
            step = key[0]
            key = "".join(key[1:])
            if "type" != key:
                continue
            if "changes" in val and not "".join(self.config._changes.keys()):
                continue
            if int(step) == float(step):
                step_dict[int(step)] = val
            else:
                step_dict[float(step)] = val
        self.step_list = sorted(step_dict.keys())
        return self.step_list

    def config_for_step(self, step=None, conformer=None):
        if step is None:
            step = self.step
        config = self.config.copy()
        if conformer is not None:
            config.conformer = conformer
        return config.for_step(step)

    def get_spec(self, step=None, conformer=None, structure=None, skip=None):
        if step is None:
            step = self.step
        if conformer is None:
            conformer = self.conformer
        if skip is None:
            skip = []
        skip += [("DEFAULT", ".*")]
        if step == "all":
            config = self.config
        else:
            config = self.config_for_step(step=step)

        spec = {
            "starting_structure": self.structure_hash,
            "structure": self._structure_dict(structure=structure),
            "step_list": self.step_list,
        }
        if step != "all":
            spec["step"] = step
            spec["conformer"] = conformer
        spec = config.as_dict(spec, skip=skip)
        return spec

    def query_spec(self, step=None, conformer=None, spec=None):
        """
        Builds a suitable pymongo-style query for searching for the job
        Returns: dict()
        """
        if step is None:
            step = self.step
        if conformer is None:
            conformer = self.conformer
        if spec is None:
            spec = self.get_spec(
                step=step,
                conformer=conformer,
                skip=self.SKIP_SPEC,
            )
        query_spec = {}
        # changing resource requested shouldn't be a separate job
        qadapter_template = self._find_qadapter_template(step=step)
        with open(qadapter_template) as f:
            qadapter_template = f.read()
        res = re.findall("\$\${(.*?)}", qadapter_template)
        exclude_pattern = re.compile(".*?/({})".format("|".join(res)))

        for key, val in spec.items():
            # these values are variable during the course of the job
            if key in ["structure", "step_list", "_args", "_kwargs", "infile"]:
                continue
            if exclude_pattern.fullmatch(key):
                continue
            if key.endswith(
                ("basis", "method", "solvent", "solvent_model", "ecp")
            ):
                query_spec["spec." + key] = {
                    "$regex": "^{}$".format(re.escape(val)),
                    "$options": "i",
                }
            else:
                query_spec["spec." + key] = val
        return query_spec

    def find_fw(self, step=None, conformer=None, spec=None):
        """
        Searches the launchpad for a particular FW
        Returns the FW, if unique FW found, or None
        """
        query_spec = self.query_spec(step=step, conformer=conformer, spec=spec)
        # archived FWs are soft-deleted, so we don't want those
        query_spec["state"] = {"$not": {"$eq": "ARCHIVED"}}
        fw_id = LAUNCHPAD.get_fw_ids(query=query_spec)
        fws = []
        for fw in fw_id:
            fw = LAUNCHPAD.get_fw_by_id(fw)
            fws += [fw]
        if fws and len(fws) == 1:
            return fws.pop()
        elif fws:
            raise RuntimeError(
                "fw spec returns multiple fireworks (ids: {})".format(
                    [fw.fw_id for fw in fws]
                )
            )
        return None

    def set_fw(self, step=None, conformer=None, spec=None, fw_id=None):
        if fw_id is None:
            fw = self.find_fw(step=step, conformer=conformer, spec=spec)
        else:
            fw = LAUNCHPAD.get_fw_by_id(fw_id)
        if fw:
            self.load_fw(fw)
        return fw

    def load_fw(self, fw):
        self.fw_id = fw.fw_id
        self.config._args = fw.spec["_args"]
        self.config._kwargs = fw.spec["_kwargs"]
        if "step" in fw.spec:
            self.step = fw.spec["step"]
        if "conformer" in fw.spec:
            self.conformer = fw.spec["conformer"]
        if "structure" in fw.spec:
            comment = fw.spec["structure"]["comment"]
            atoms = []
            for a in fw.spec["structure"]["atoms"]:
                name, element, x, y, z = a.split()
                atoms += [
                    Atom(
                        element=element,
                        coords=[x, y, z],
                        name=name,
                    )
                ]
            self.structure = Geometry(atoms)
            self.structure.comment = comment
            self.structure.parse_comment()
            self.structure_hash = fw.spec["starting_structure"]
        if not hasattr(self, "structure"):
            self.structure = Geometry()
        self.structure.name = self.get_basename().rstrip("." + str(self.step))
        self.set_root()

    def set_root(self, make_root=True):
        """
        Sets self.root_fw_id to the root FW for the workflow containing `fw_id`

        :fw_id: defaults to self.fw_id
        """
        if self.fw_id is None:
            self.set_fw()
        if self.fw_id is None:
            self.root_fw_id = self._root_fw(make_root=make_root)
            if self.parent_fw_id is None:
                self.parent_fw_id = self.root_fw_id
            return
        links = LAUNCHPAD.get_wf_by_fw_id(self.fw_id).links
        self.parent_fw_id = []
        for key, val in links.items():
            if self.fw_id in val:
                self.parent_fw_id += [key]
        children = set(it.chain.from_iterable(links.values()))
        for key in links:
            if key not in children:
                self.root_fw_id = key
                break

    def get_workflow_name(self):
        wf_name = os.path.join(
            self.config.get("DEFAULT", "project", fallback=""),
            self.config["DEFAULT"]["name"],
        )
        if self.config.get("Reaction", "reaction", fallback=""):
            wf_name = os.path.join(
                wf_name, self.config["Reaction"]["reaction"]
            )
        if self.config.get("Reaction", "template", fallback=""):
            wf_name = os.path.join(
                wf_name, self.config["Reaction"]["template"]
            )
        wf_name = "{}/{}".format(wf_name, self.config["Job"]["name"])
        return wf_name

    # file management
    def get_basename(self, step=None, conformer=None):
        if step is None:
            step = self.step
        if conformer is None:
            conformer = self.conformer
        name = self.config["Job"]["name"]
        if conformer:
            name = "{}_{}".format(name, conformer)
        if step:
            name = "{}.{}".format(name, step)
        return name

    def get_input_name(self, skip_ext=False):
        config = self.config_for_step()
        exec_type = config["Job"]["exec_type"]
        name = self.get_basename()
        if exec_type == "gaussian":
            if skip_ext:
                return name
            name += ".com"
        elif exec_type == "crest":
            name = os.path.join(name, "ref")
            if skip_ext:
                return name
            name += ".xyz"
        elif exec_type == "xtb":
            if skip_ext:
                return name
            name += ".xyz"
        else:
            raise NotImplementedError(
                "File type not added to AaronJr yet...", exec_type
            )
        return name

    def get_output_name(self):
        config = self.config_for_step()
        name = self.get_basename()
        exec_type = config["Job"]["exec_type"]
        if exec_type == "gaussian":
            name += ".log"
        elif exec_type == "crest":
            name = os.path.join(name, "out.crest")
        elif exec_type == "xtb":
            if "ts" in config["Job"]["type"]:
                name = (name + ".xtb", name + ".freq")
            else:
                name += ".xtb"
        else:
            raise Exception
        return name

    def write(self, override_style=None, send=True):
        config = self.config_for_step()
        theory = config.get_theory(self.structure)
        name = os.path.join(
            config["Job"]["top_dir"], self.get_input_name(skip_ext=True)
        )
        exec_type = config["Job"]["exec_type"]
        xcontrol = None
        ref_name = None
        auxfiles = []
        if exec_type == "gaussian":
            style = "com"
        elif exec_type == "crest":
            style = "xyz"
            xcontrol = os.path.join(self.get_basename(), "xcontrol")
        elif exec_type == "xtb":
            style = "xyz"
            xcontrol = os.path.join("{}.xcontrol".format(self.get_basename()))
            ref_name = os.path.join("{}.xyz".format(self.get_basename()))
        else:
            raise Exception
        if override_style is not None:
            style = override_style
        theory.geometry.write(
            name=name, style=style, theory=theory, **config._kwargs
        )
        if xcontrol is not None:
            auxfiles.append(
                (
                    os.path.join(config["Job"]["top_dir"], xcontrol),
                    os.path.join(config["HPC"]["remote_dir"], xcontrol),
                )
            )
            with open(auxfiles[-1][0], "w") as f:
                f.write(theory.get_xcontrol(config, ref=ref_name))
        if "transfer_host" in config["HPC"] and send:
            if auxfiles:
                self.transfer_input(aux_files=auxfiles)
            else:
                self.transfer_input()
        return config

    def transfer_input(self, aux_files=None):
        config = self.config_for_step()
        name = self.get_input_name()
        source = os.path.join(config["Job"].get("top_dir"), name)
        target = os.path.join(config["HPC"].get("remote_dir"), name)

        args = [(source, target)]
        if aux_files is not None:
            args.extend(aux_files)
        for source, target in args:
            # self.LOG.debug("{} -> {}".format(source, target))
            if not self.quiet:
                print(
                    "Sending {} to {}...".format(
                        os.path.basename(target), config["HPC"].get("host")
                    )
                )
            self.remote_put(source, target)

    def transfer_output(self, state=None, update=False):
        config = self.config_for_step()
        name = self.get_output_name()
        if isinstance(name, tuple):
            name, freq_name = name
        else:
            freq_name = None
        source = os.path.join(config["HPC"]["remote_dir"], name)
        target = os.path.join(config["Job"]["top_dir"], name)

        new = self.remote_get(source, target)
        if not self.quiet and new:
            self.LOG.info(
                "Downloaded {} from {}".format(name, config["HPC"]["host"])
            )

        exec_type = config["Job"]["exec_type"]
        if exec_type == "crest" and (update or state == "COMPLETED"):
            # this doesn't get written until the very end
            try:
                new = self.remote_get(
                    os.path.join(
                        os.path.dirname(source), "crest_conformers.xyz"
                    ),
                    os.path.join(
                        os.path.dirname(target), "crest_conformers.xyz"
                    ),
                )
                if not self.quiet and new:
                    self.LOG.info(
                        "Downloaded {} from {}".format(
                            "crest_conformers.xyz", config["HPC"]["host"]
                        )
                    )
            except FileNotFoundError:
                if update:
                    pass
        if exec_type == "xtb" and freq_name is not None:
            new = self.remote_get(
                os.path.join(config["HPC"]["remote_dir"], freq_name),
                os.path.join(config["Job"]["top_dir"], freq_name),
            )
            if not self.quiet and new:
                self.LOG.info(
                    "Downloaded {} from {}".format(
                        ".".join(freq_name),
                        config["HPC"]["host"],
                    )
                )
        return new

    def archive_output(self, error_type):
        config = self.config_for_step()
        name = self.get_output_name()
        if isinstance(name, tuple):
            name, _ = name
        local_name = os.path.join(config["Job"]["top_dir"], name)
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        try:
            launch_id = fw.launches[-1].launch_id
        except IndexError:
            launch_id = fw.archived_launches[-1].launch_id
        new_name = os.path.join(
            config["Job"]["top_dir"],
            "archived",
            "{}_{}.{}.{}".format(
                name,
                error_type,
                fw.fw_id,
                launch_id,
            ),
        )
        os.makedirs(os.path.dirname(new_name), exist_ok=True)
        if os.path.isfile(local_name):
            shutil.move(local_name, new_name)

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

    # workflow management
    def add_fw(
        self, parent_fw_id=None, step=None, conformer=None, structure=None
    ):
        """
        Adds FW to the appropriate workflow (creating if necessary) and updates self.fw_id accordingly
        Returns: the created FW

        :parent_fw_id: for linking within workflow (default to self.parent_fw_id)
        :step: the step number
        :conformer: the conformer number
        :structure: the structure to use
        """
        # make firework
        fw = Firework(
            [
                self._get_exec_task(
                    step=step, conformer=conformer, structure=structure
                )
            ],
            spec=self.get_spec(
                step=step, conformer=conformer, structure=structure
            ),
            name=self.get_basename(step=step, conformer=conformer),
        )
        if not self.quiet:
            print(
                "  Adding Firework for {} to workflow".format(
                    self.get_basename(step=step, conformer=conformer)
                )
            )
        if parent_fw_id is None:
            parent_fw_id = self.parent_fw_id
        if isinstance(parent_fw_id, list):
            LAUNCHPAD.append_wf(
                Workflow([fw]), parent_fw_id, pull_spec_mods=False
            )
        elif parent_fw_id:
            LAUNCHPAD.append_wf(
                Workflow([fw]), [parent_fw_id], pull_spec_mods=False
            )
        else:
            wf_name = self.get_workflow_name()
            LAUNCHPAD.add_wf(
                Workflow(
                    [fw],
                    name=wf_name,
                    metadata=self.config.metadata,
                )
            )
        return fw

    def add_workflow(
        self, parent_fw_id=None, step_list=None, conformer=0, structure=None
    ):
        """
        :parent_fw_id: assign the new workflow as children of this fw_id, default is
            either the previous step's fw_id (if applicable) or the root_fw_id
        :step_list: list(float) corresponding to the step numbers, defaults to the
            return value of self.get_steps
        :conformer: the conformer number to associate with the workflow
        :structure: use this Geometry as starting structure

        Returns: list(fw_ids), bool(isNew)
            list(fw_ids) are the fw_ids for the steps in step_list
            bool(isNew) is True if a new FW was added, False if all previously created
        """
        isNew = False
        if step_list is None:
            step_list = self.step_list
        if not step_list and not "".join(self.config._changes.keys()):
            if parent_fw_id is not None:
                self.parent_fw_id = parent_fw_id
            fw = self.set_fw(conformer=conformer)
            if fw is not None:
                LAUNCHPAD.update_spec(
                    [self.fw_id],
                    self.get_spec(conformer=conformer, structure=structure),
                )
            else:
                fw = self.add_fw(conformer=conformer, structure=structure)
            return [fw.fw_id]

        if parent_fw_id is None:
            self.set_root()
        else:
            self.parent_fw_id = parent_fw_id

        # create workflow
        fws = []
        # self.LOG.debug(self.get_workflow_name())
        for step in step_list:
            fw = self.find_fw(step=step, conformer=conformer)
            if fw is None:
                isNew = True
                fw = self.add_fw(
                    step=step, conformer=conformer, structure=structure
                )
            elif structure is not None and fw.state in [
                "READY",
                "WAITING",
                "DEFUSED",
            ]:
                LAUNCHPAD.update_spec(
                    [fw.fw_id],
                    self.get_spec(
                        step=step, conformer=conformer, structure=structure
                    ),
                )
            fws += [fw.fw_id]
            self.parent_fw_id = fw.fw_id
        return fws, isNew

    def launch_job(self, override_style=None, send=True):
        # this should be the only place that self.write() is called
        config = self.write(override_style=override_style, send=send)
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        if fw.state != "READY":
            return
        qadapter = self._make_qadapter(fw.fw_id)
        qlaunch_cmd = 'qlaunch -r -l "$AARONLIB/my_launchpad.yaml" -q "{}" --launch_dir "{}" singleshot --fw_id {}'.format(
            qadapter,
            os.path.join(
                config["HPC"]["work_dir"],
                os.path.dirname(config["Job"]["name"]),
            ),
            fw.fw_id,
        )
        if config["HPC"].get("host"):
            out, err = self.remote_run(qlaunch_cmd)
            if out:
                print(out)
            for e in err.split("\n"):
                if "has been specified in qadapter" in e:
                    continue
                if e.strip():
                    print("err:", e.strip())
        else:
            os.system(qlaunch_cmd)

    def validate(self, get_all=False, update=False, transfer=True):
        config = self.config_for_step()
        name = self.get_output_name()
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)

        if isinstance(name, tuple):
            name, freq_name = (
                os.path.join(config["Job"]["top_dir"], n) for n in name
            )
        else:
            name = os.path.join(config["Job"]["top_dir"], name)
            freq_name = None

        try:
            if transfer:
                self.transfer_output(state=fw.state, update=update)
            output = CompOutput(name, get_all=get_all, freq_name=freq_name)
        except (IndexError, KeyError):
            # this can be caused by partial log files missing all or some of the
            # atomic coordinate listing
            if fw.state != "COMPLETED":
                return None
            # if COMPLETED, there's something wrong with our error detection,
            # so this will let us know there's a problem
            raise
        except FileNotFoundError:
            if fw.state in ["READY", "WAITING", "RESERVED", "PAUSED"]:
                # there's no output file to parse before RUNNING
                return None
            if fw.state == "RUNNING":
                # give it a little while to start writing to output file
                state_history = fw.launches[-1].state_history[-1]
                tdelta = (
                    datetime.datetime.utcnow() - state_history["created_on"]
                )
                if tdelta < datetime.timedelta(minutes=2):
                    return None
            self.LOG.warning(
                ("Cannot find", name, "for validation check"), exc_info=True
            )
            if len(fw.archived_launches) < MAX_SUBMIT:
                self.LOG.warning("Trying to rerun firework")
                self.rerun("input")
            else:
                self.LOG.warning(
                    "Too many submission attempts. Verify that you have access to "
                    "the host running the computational jobs, and ensure AaronJr "
                    "connection configuration is correct. Verify that you can manually "
                    "submit to the queue, and that AaronJr queue configuration is "
                    "correct. If this still hasn't fixed the issue, check your "
                    "executable template files for errors."
                )
                self.LOG.warning(
                    "To force restart after fixing any issues, run `remove_fw %s` "
                    "before starting AaronJr again.",
                    fw.fw_id,
                )
            return None

        if update and fw.state not in ["WAITING", "PAUSED", "RESERVED"]:
            if output.finished:
                self.complete_launch(fw, output, update=True)
            elif output.error is not None:
                self.complete_launch(fw, output, update=True)
                LAUNCHPAD.defuse_fw(self.fw_id)
                return output
            fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        if fw.state == "RUNNING":
            # Catch walltime error manually, FireWorks can't tell if job gets kicked.
            # This will also fix jobs that have been kicked from the queue for any
            # other reason, but only if they have been dead longer than the walltime
            state = fw.launches[-1].state_history[-1]
            runtime = datetime.datetime.utcnow() - state["created_on"]
            walltime = datetime.timedelta(hours=int(config["Job"]["wall"]))
            if runtime > walltime:
                self.LOG.debug("runtime:%s walltime:%s", runtime, walltime)
                output.error = "WALLTIME"
                self.complete_launch(fw, output)
                LAUNCHPAD.defuse_fw(fw.fw_id)
        if fw.state in ["READY", "WAITING", "RESERVED", "RUNNING", "PAUSED"]:
            return output
        # if we get here, fw.state can only be COMPLETED, DEFUSED, or FIZZLED
        # so something is wrong if output.finished != True
        try:
            self.complete_launch(fw, output)
        except ValueError:
            self.LOG.exception("%s %s", fw.fw_id, fw.state)
        if not output.finished:
            LAUNCHPAD.defuse_fw(fw.fw_id)
            return output
        # if we get here, fw.state == "COMPLETED" and output.finished = True

        # need to run this last so that conformer structures get picked up
        # self.complete_launch() will set structure=best_conformer for all direct
        # children, so we need this to update conformers appropriately
        if "conformers" in config["Job"]["type"]:
            self.add_conformers(fw.fw_id, output)
        return output

    def resolve_error(self, force_rerun=False):
        """
        Makes changes to automatically resolve computational errors
        Returns: True if job is fixed and should be re-launched, else False

        :force_rerun: updates and marks resolved a job that had failed due to error
            that the user must resolve by hand (eg: typos that cause job failure)
        """
        fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
        if fw.state == "COMPLETED":
            return False
        # clear old added route options
        self.config._args = []
        self.config._kwargs = {}

        config = self.config_for_step()
        constrain_step = None
        try:
            for step in self.step_list:
                tmp = self.config_for_step(step)
                if (
                    "constrain" in tmp["Job"]["type"]
                    and "opt" in tmp["Job"]["type"]
                ):
                    constrain_step = step
        except IndexError:
            pass

        name = self.get_output_name()
        if isinstance(name, tuple):
            name, freq_name = (
                os.path.join(config["Job"]["top_dir"], n) for n in name
            )
        else:
            name = os.path.join(config["Job"]["top_dir"], name)
            freq_name = None
        try:
            output = CompOutput(name, freq_name=freq_name)
        except IndexError:
            self.rerun("input")
            return True

        if output.finished and output.error is None:
            # just in case something got marked that shouldn't have
            # (b/c of race condition???) eg: output file update needed
            print("  Normal termination detected! Marking COMPLETED")
            self.complete_launch(fw, output)
            self.config._args = []
            self.config._kwargs = {}
            # return False here since we don't need to re-run
            return False

        # set maximum number of retries
        if len(fw.archived_launches) > MAX_ATTEMPTS:
            return False

        # print status
        resolved = False
        try:
            stored_data = self._last_launch(fw).action.stored_data
            if stored_data["error"] == "WALLTIME":
                output.error = "WALLTIME"
        except IndexError:
            pass
        print("  Trying to resolve {} error".format(output.error))
        # archive log files of failed jobs
        self.archive_output(output.error)

        # attempt to fix error
        if output.error in ["UNKNOWN", "WALLTIME", "CHK"]:
            resolved = True
            self.rerun(output.geometry)
        if output.error in ["REDUND", "FBX"]:
            resolved = True
            for a in output.geometry:
                a.coords = np.array(
                    [round(c, 4) for c in a.coords], dtype=float
                )
            self.rerun(output.geometry)
        if output.error in ["CONV_LINK", "CONV_CDS", "LINK"]:
            resolved = True
            self.rerun(output.geometry)
            if constrain_step is not None and self.step > constrain_step:
                self.set_fw(step=constrain_step)
                self.rerun(output.geometry)
        if output.error in ["CONSTR"]:
            resolved = True
            output.geometry.comment = self.structure.comment
            constraints = config.get_constraints(output.geometry)
            for atoms in constraints.values():
                atoms = it.chain.from_iterable(atoms)
                atoms = list(set(atoms))
                for a in output.geometry.find(atoms):
                    a.coords = np.round(a.coords, 4)
            self.rerun(output.geometry)
        if output.error in ["COORD"]:
            resolved = True
            self.config._kwargs[GAUSSIAN_ROUTE] = {"opt": ["cartesian"]}
            self.rerun(output.geometry)
        if output.error in ["EIGEN"]:
            resolved = True
            fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
            count = 0
            for launch in fw.archived_launches:
                # if we've seen this error before for this job
                if (
                    launch.action is not None
                    and "EIGEN" == launch.action.stored_data["error"]
                ):
                    count += 1
            if count > 3:
                if self.config["Job"]["exec_type"] == "gaussian":
                    self.config._kwargs[GAUSSIAN_ROUTE] = {"opt": ["noeigen"]}
                    self.rerun(output.geometry)
                else:
                    raise NotImplementedError
            else:
                self.rerun(output.geometry)
                self.set_fw(step=constrain_step)
                if self.config["Job"]["exec_type"] == "gaussian":
                    self.config._kwargs[GAUSSIAN_ROUTE] = {
                        "opt": [
                            "CalcFC",
                            "maxstep={}".format(int(round(30 / (count + 1)))),
                        ]
                    }
                    self.rerun(output.geometry)
                else:
                    raise NotImplementedError
        if output.error in ["SCF_CONV"]:
            resolved = True
            fw = LAUNCHPAD.get_fw_by_id(self.fw_id)
            maxstep = 0
            for launch in fw.archived_launches:
                if (
                    launch.action is not None
                    and "SCF_CONF" == launch.action.stored_data["error"]
                ):
                    maxstep += 1
            try:
                maxstep = [15, 12, 10, 8, 5, 2][maxstep]
            except IndexError:
                maxstep = 0
            if self.config["Job"]["exec_type"] == "gaussian":
                if maxstep:
                    self.config._kwargs[GAUSSIAN_ROUTE] = {
                        "opt": ["maxstep={}".format(maxstep)],
                    }
                else:
                    self.config._kwargs[GAUSSIAN_ROUTE] = {"scf": ["xqc"]}
            else:
                raise NotImplementedError
            if self.step != self.step_list[0]:
                structure = "input"
            else:
                structure = "original"
            self.rerun(structure)
            if constrain_step is not None and self.step > constrain_step:
                self.set_fw(step=constrain_step)
                self.rerun(structure)
        if output.error == "CLASH":
            bad_subs = output.geometry.remove_clash()
            if not bad_subs:
                resolved = True
            self.rerun(output.geometry)

        # stuff Aaron can't fix
        if output.error == "GALLOC":
            # this happens sometimes and is generally a node problem.
            # restarting is usually fine, may not be if job is picked up
            # on the same node as before (TODO: node-specific submission)
            resolved = True
            self.rerun(output.geometry)
        if force_rerun and not resolved:
            resolved = True
            if self.step != self.step_list[0]:
                structure = "input"
            else:
                structure = "original"
            self.rerun(structure)
        elif output.error == "CHARGEMULT":
            raise RuntimeError(
                "Bad charge/multiplicity provided. Please fix AaronJr configuration."
            )
        elif output.error == "ATOM":
            raise RuntimeError("Bad atomic symbol, check template XYZ files.")
        elif output.error == "BASIS":
            raise RuntimeError(
                "Error applying basis set. Ensure basis set contains definitions for all atoms in structure."
            )
        elif output.error == "BASIS_READ":
            raise RuntimeError(
                "Error reading basis set. Confirm that gen=/path/to/basis/ is correct in your AaronJr configuration and that the basis set file requested exists, or switch to an internally provided basis set."
            )
        elif output.error == "QUOTA":
            raise RuntimeError(
                "Erroneous write. Check quota or disk space, then restart AaronJr."
            )
        elif output.error == "MEM":
            raise RuntimeError(
                "Node(s) out of memory. Increase memory requested in AaronJr configuration"
            )
        if not resolved:
            self.LOG.error(
                "No resolution for error %s implemented!", output.error
            )
        return resolved

    def complete_launch(self, fw, output, update=True):
        """
        :update: True if we want to allow for creation of dummy launches when no
            launches are found in the fw history (eg: if the launchpad has been
            reset and we're reloading in the computational data)
        """
        try:
            launch_id = self._last_launch(fw).launch_id
        except IndexError:
            if update:
                fw, launch_id = LAUNCHPAD.checkout_fw(
                    FWorker(), "", fw_id=fw.fw_id
                )
            else:
                raise
        if output.geometry is None:
            self.LOG.error(
                "No geometry found in output for FW %s. Child-job geometries will not be updated.",
                fw.fw_id,
            )
            fw_action = FWAction(
                stored_data=self._get_stored_data(output),
            )
        else:
            for a, b in zip(self.structure, output.geometry):
                b.name = a.name
            fw_action = FWAction(
                stored_data=self._get_stored_data(output),
                update_spec={
                    "structure": self._structure_dict(output.geometry)
                },
            )
        LAUNCHPAD.complete_launch(launch_id, action=fw_action)

    def update_structure(self, structure, kind="output"):
        config = self.config_for_step()
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
            if kind == "input":
                name = self.get_input_name()
            else:
                name = self.get_output_name()
                if isinstance(name, tuple):
                    name, _ = name
            if name is not None:
                try:
                    structure = Geometry(
                        os.path.join(config["Job"]["top_dir"], name)
                    )
                except Exception:
                    structure = None
        if structure is not None:
            for i, a in enumerate(structure):
                self.structure.atoms[i].coords = a.coords
        spec = self.get_spec()
        # update script task too, in case the user needed to fix a template
        spec["_tasks"] = [self._get_exec_task().to_dict()]
        LAUNCHPAD.update_spec([self.fw_id], spec)

    def rerun(self, structure):
        LAUNCHPAD.defuse_fw(self.fw_id)
        if isinstance(structure, str) and structure in [
            "original",
            "input",
            "output",
        ]:
            self.update_structure(None, kind=structure)
        else:
            self.update_structure(structure)
        LAUNCHPAD.reignite_fw(self.fw_id)
        LAUNCHPAD.rerun_fw(self.fw_id)

    def add_conformers(self, fw_id, output):
        """ """
        # load screening options
        max_conformers = self.config["Job"].getint(
            "max_conformers", fallback=None
        )
        energy_cutoff = self.config["Job"].getfloat(
            "energy_cutoff", fallback=None
        )
        rmsd_cutoff = self.config["Job"].getfloat(
            "rmsd_cutoff", fallback=RMSD_CUTOFF
        )
        # determine what steps we still need to run for conformer child
        wf = LAUNCHPAD.get_wf_by_fw_id(fw_id)
        steps = set([])
        children = wf.links[fw_id].copy()
        while children:
            fw = LAUNCHPAD.get_fw_by_id(children.pop())
            steps.add(fw.spec["step"])
            children += wf.links[fw.fw_id]
        # add conformer children to workflow
        best_energy = float(output.other["best_energy"])
        skipped = 0
        i = 0
        is_new = False
        fws = []
        kept_confs = [output.geometry]
        comment_special = re.compile("((?:F|C|L|K):\S+)")
        for i, conformer in enumerate(output.conformers):
            energy = conformer.comment.split()[0]
            # need to do this to keep constraints, etc. available in later steps
            self_match = comment_special.findall(self.structure.comment)
            conf_match = comment_special.findall(conformer.comment)
            for match in self_match:
                if match not in conf_match:
                    conformer.comment += " %s" % match
            # screening
            if max_conformers is not None and i >= max_conformers:
                break
            energy_diff = UNIT.HART_TO_KCAL * (float(energy) - best_energy)
            if energy_cutoff is not None and energy_diff > energy_cutoff:
                skipped += 1
                continue
            for geom in kept_confs:
                rmsd = conformer.RMSD(geom)
                if rmsd_cutoff is not None and rmsd < rmsd_cutoff:
                    skipped += 1
                    break
            else:
                kept_confs += [conformer]
                for a, b in zip(self.structure, conformer):
                    b.name = a.name
                fws, is_new = self.add_workflow(
                    parent_fw_id=fw_id,
                    step_list=sorted(steps),
                    conformer=i + 1,
                    structure=conformer,
                )
        if is_new:
            print(
                "  Workflow created for {} conformers of {}".format(
                    i + 1 - skipped, self.get_basename()
                )
            )
        return fws, is_new
