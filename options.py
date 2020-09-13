import glob
import hashlib
import json
import os
import pickle
import re
from copy import copy

import Aaron.json_extension as json_ext
from AaronTools.component import Component
from AaronTools.const import AARONLIB, ELEMENTS, QCHASM, TMETAL
from AaronTools.fileIO import FileWriter
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils import utils
from AaronTools.theory import *


class CatalystMetaData:
    """
    :template_file:     file location of template.xyz
    :reaction_type:     name of the reaction type
    :template_name:     name of the template subdirectory
    :selectivity:       selectivity of the template used
    :ligand_change:     changes to the ligand made
    :substrate_change:  changes to the substrate made
    :ts_directory:      where the conformers are stored
    :catalyst:
    :requested_changes:
    :status:
    """

    def __init__(
        self,
        template_file=None,
        reaction_type=None,
        template_name=None,
        selectivity=None,
        ligand_change=None,
        substrate_change=None,
        ts_number=None,
        ts_directory=None,
        catalyst=None,
        catalyst_change=None,
    ):
        self.template_file = template_file
        if isinstance(reaction_type, Reaction):
            self.reaction_type = reaction_type.reaction_type
            if template_name is None:
                self.template_name = reaction_type.template
            else:
                self.template_name = template_name
        else:
            self.reaction_type = reaction_type
            self.template_name = template_name
        self.selectivity = selectivity
        self.ts_number = ts_number
        self.ligand_change = ligand_change
        self.substrate_change = substrate_change
        self.catalyst = catalyst
        self.ts_directory = ts_directory
        if template_file is not None:
            path, basename = os.path.split(template_file)
            if basename.lower().startswith("ts"):
                num = re.search("(\d+)\.xyz", basename).group(1)
            else:
                num = re.search("(\d+)$", path).group(1)
            self.ts_number = int(num)
        self.catalyst_change = catalyst_change
        self.conf_spec = 1

    def __lt__(self, other):
        def get_str(thing):
            if thing is None:
                return ""
            if isinstance(thing, tuple):
                return thing[0]

        self_str = [get_str(self.ligand_change)]
        self_str += [get_str(self.substrate_change)]
        self_str += [get_str(self.template_file)]
        other_str = [get_str(other.ligand_change)]
        other_str += [get_str(other.substrate_change)]
        other_str += [get_str(other.template_file)]
        for s, o in zip(self_str, other_str):
            if s != o:
                return s < o
        else:
            return False

    def get_basename(self):
        basename = ""
        # first part is lig-sub
        if self.ligand_change is not None:
            basename += self.ligand_change[0] + "."
        else:
            basename += "original."
        if self.substrate_change is not None:
            basename += self.substrate_change[0] + "."
        else:
            basename += "original."
        if self.selectivity:
            basename += "{}.".format(self.selectivity)
        basename += "ts{}.".format(self.ts_number)
        basename += "cf{}".format(self._get_cf_num())
        return basename

    def get_cf_dir(self):
        """
        Get SHA-256 hash of conformer number, convert to base-10, and return
        the first three digits as a string. This limits the number of sub-
        directories to 1,000, so we won't accidentally hit the OS limit on
        number of subdirectories or cause unnecessary performance issues with
        shell commands or completion.

        Returns: str(3-digit integer)
        """
        rv = self._get_cf_num()
        rv = hashlib.sha256(bytes(rv, "utf-8")).hexdigest()
        rv = str(int(rv, base=16))[:3]
        return rv

    def get_relative_path(self):
        cat_change = os.path.join(
            self.ligand_change[0], self.substrate_change[0]
        )
        ts = self.ts_directory.split(cat_change)[-1][1:]
        cf = self.get_cf_dir()
        return os.path.join(cat_change, ts, cf)

    def get_catalyst_name(self):
        basename = self.get_basename()
        cf_dir = self.get_cf_dir()
        return os.path.join(self.ts_directory, cf_dir, basename)

    def load_lib_ligand(self, lig_name):
        """
        Loads a Component() from a ligand XYZ file in the library

        :lig_name: the name of the ligand
        :returns: Component()
        """
        if not lig_name.endswith(".xyz"):
            name = lig_name + ".xyz"
        path = [os.path.join(AARONLIB, "Ligands", name)]
        path += [os.path.join(QCHASM, "AaronTools/Ligands", name)]
        for p in path:
            if os.path.isfile(p):
                return Component(p)
        else:
            raise FileNotFoundError("Could not find ligand: " + name)

    def pickle_update(self):
        if not self.ts_directory:
            return False
        os.makedirs(self.ts_directory, exist_ok=True)
        fname = os.path.join(
            self.ts_directory, "{}_pickle".format(self.get_basename)
        )
        if not self.catalyst and os.access(fname, os.W_OK):
            with open(fname, "rb") as f:
                print("Loading {} from pickle".format(self.get_basename()))
                self = pickle.load(f)
            return True
        elif self.catalyst and os.access(fname, os.W_OK):
            with open(fname, "wb") as f:
                pickle.dump(self, f)
            return True
        elif not os.access(fname, os.W_OK):
            return False

    def generate_structure(self, old_aaron=False):
        """
        Generates a mapped/substituted catalyst structure from the information
        stored in a CatalystMetaData object

        Returns: Catalyst()
        """
        if self.pickle_update():
            return self.catalyst
        self.catalyst = Catalyst(self.template_file)
        if self.ligand_change is None and self.substrate_change is None:
            if old_aaron:
                self.conf_spec = {}
            return self.catalyst
        # do mappings/substitution
        requested_changes = self._do_ligand_change()
        requested_changes += self._do_substrate_change()
        self.catalyst_change = requested_changes

        # try to get better starting structure
        self.catalyst.remove_clash()
        self.catalyst.minimize()

        # only do conformers for requested changes if old_aaron
        if old_aaron and requested_changes:
            self.conf_spec = {}
            for sub in requested_changes:
                sub = self.catalyst.find_substituent(sub)
                if sub:
                    self.conf_spec[sub.end] = (1, [])
        self.catalyst.name = self.get_catalyst_name()
        self.pickle_update()
        return self.catalyst

    def _get_cf_num(self):
        """
        :conf_spec: the Catalyst().conf_spec dictionary, uses the one
            associated with self.catalyst if None
        """
        match = re.search(".*/cf(\d+).xyz", self.template_file, flags=re.I)
        if match is None:
            s = ""
        else:
            s = match.group(1) + "-"
        if isinstance(self.conf_spec, dict):
            for start, (conf_num, skip) in sorted(self.conf_spec.items()):
                s += str(conf_num)
        else:
            s += str(self.conf_spec)
        return s

    def _get_nconfs(self):
        nconfs = 1
        for start in self.conf_spec:
            sub = self.catalyst.find_substituent(start)
            nconfs *= sub.conf_num
        return nconfs

    def _do_ligand_change(self):
        requested_changes = []
        if self.ligand_change[1] is not None:
            lig, change = self.ligand_change
            # mappings
            if "map" in change:
                name, old_keys = change["map"]
                if not old_keys:
                    old_keys = []
                    for lig in self.catalyst.components["ligand"]:
                        old_keys += list(lig.key_atoms)
                new_lig = self.load_lib_ligand(name)
                new_lig.tag("changed")
                self.catalyst.map_ligand(new_lig, old_keys)
            # substitutions
            for atom, sub in change.items():
                if atom == "map":
                    continue
                # convert to relative numbering
                if "map" in change:
                    name, old_keys = change["map"]
                    if not old_keys:
                        atom = "*.{}".format(atom)
                    else:
                        atom = "{}.{}".format(old_keys[0], atom)
                else:
                    n_before_lig = len(self.catalyst.atoms) - sum(
                        [
                            len(i.atoms)
                            for i in self.catalyst.components["ligand"]
                        ]
                    )
                    atom = str(int(atom) + n_before_lig)
                sub = self.catalyst.substitute(sub, atom)
                sub.tag("changed")
                requested_changes += [atom]
        return requested_changes

    def _do_substrate_change(self):
        requested_changes = []
        if self.substrate_change[1] is not None:
            sub, change = self.substrate_change
            # substitutions
            for atom, sub in change.items():
                requested_changes += [atom]
                sub = self.catalyst.substitute(sub, atom)
                sub.tag("changed")
        return requested_changes

    def _get_log_names(self, top_dir=None, old_aaron=False):
        """
        Returns: list of log file names
        """
        rv = []

        # build log file name base
        if top_dir is None:
            top_dir = self.ts_directory
        if os.path.isfile(top_dir):
            top_dir = os.path.dirname(top_dir)
        if old_aaron:
            # backwards compatibility for directory structure
            cat_change = "{}".format(self.ligand_change[0])
            if self.substrate_change[1]:
                cat_change += "-{}".format(self.substrate_change[0])
            cat_change = os.path.join(self.ligand_change[0], cat_change)
        else:
            cat_change = os.path.join(
                self.ligand_change[0], self.substrate_change[0]
            )
        log_name_base = os.path.join(top_dir, cat_change)
        if self.selectivity:
            log_name_base = os.path.join(log_name_base, self.selectivity)
        log_name_base = os.path.join(
            log_name_base, "ts{}".format(self.ts_number)
        )

        # get log file names
        nconfs_sub = 1
        if self.substrate_change[1]:
            for sub_name in self.substrate_change[1].values():
                sub = Substituent(sub_name)
                nconfs_sub *= sub.conf_num
        nconfs_lig = 1
        if self.ligand_change[1]:
            for sub_name in self.ligand_change[1].values():
                sub = Substituent(sub_name)
                nconfs_lig *= sub.conf_num
        if old_aaron:
            if self.substrate_change[1]:
                xyz_cf = []
                if self.ligand_change[1]:
                    cf_num = (
                        int(os.path.basename(self.template_file)[2:-4]) - 1
                    )
                    cf_num = cf_num * nconfs_lig + 1
                    log_name = os.path.join(
                        log_name_base.replace(
                            cat_change,
                            os.path.join(
                                self.ligand_change[0], self.ligand_change[0]
                            ),
                        ),
                        "Cf{}/*.4.log",
                    )
                    for i, f in enumerate(
                        sorted(glob.glob(log_name.format("*")))
                    ):
                        num = int(os.path.basename(f).split(".")[-3][2:]) - 1
                        num = num % nconfs_lig
                        if num:
                            continue
                        xyz_cf += [i]

                tmp = log_name_base.replace(cat_change, cat_change + "_XYZ")
                for xyz in glob.glob("{}/Cf*.xyz".format(tmp)):
                    cf_num = int(os.path.basename(xyz)[2:-4]) - 1
                    if xyz_cf and cf_num not in xyz_cf:
                        continue
                    cf_num = cf_num * nconfs_sub + 1
                    log_name = os.path.join(
                        log_name_base, "Cf{}/*.4.log".format(cf_num)
                    )
                    rv += glob.glob(log_name)
            elif self.ligand_change[1]:
                cf_num = int(os.path.basename(self.template_file)[2:-4]) - 1
                cf_num = cf_num * nconfs_lig + 1
                log_name = os.path.join(
                    log_name_base, "Cf{}/*.4.log".format(cf_num)
                )
                rv += glob.glob(log_name)
        return rv

    def _update_json(self):
        name = os.path.join(
            self.ts_directory,
            os.path.basename(self.template_file).rsplit(".", maxsplit=1)[0],
        )
        with open(os.path.join(name, ".json"), "w") as f:
            json.dump(self, f, cls=json_ext.JSONEncoder)

    def _read_json(self):
        name = os.path.join(
            self.ts_directory,
            os.path.basename(self.template_file).rsplit(".", maxsplit=1)[0],
        )
        with open(os.path.join(name, ".json")) as f:
            obj = json.load(f, cls=json_ext.JSONDecoder)
            for key, val in obj.__dict__.items():
                if key == "ts_directory":
                    continue
                self.__dict__[key] = val


class Reaction:
    """
    Attributes:
    :reaction_type: templates stored in directories
    :template:          in Aaron library named as:
    :selectivity:       TS_geoms/rxn_type/template/selectivity
    :ligand:        ligand mapping/substitutions
    :substrate:     substrate substitutions
    :con_thresh:    connectivity threshold
    :template_XYZ:  list(tuple(template_directory, subdirectory, filename.xyz))
                        eg: [($AARONLIB/TS_geoms/rxn/lig/R, ts1, Cf1.xyz)]
                            --> $AARONLIB/TS_geoms/rxn/lig/R/ts1/Cf1.xyz
    :catalyst_data: metadata for generating catalyst structures
    :by_lig_change: for sorting catalyst_data
    :by_sub_change: for sorting catalyst_data
    """

    def __init__(self, config=None):
        self.reaction_type = ""
        self.template = ""
        self.template_XYZ = []
        self.selectivity = []
        self.ligand = {}
        self.substrate = {}
        self.catalyst_data = []
        self.by_lig_change = {}
        self.by_sub_change = {}
        self.con_thresh = None
        if config is None:
            return

        self.reaction_type = config.get("options", "reaction type")
        self.template = config.get("options", "template")

        for option in config.options("options"):
            if option.lower() == "selectivity":
                self.selectivity = config.getlist("options", option, delim=";")
            
            elif option.lower() == "connectivity threshold":
                self.con_thresh = config.getfloat(option)

        if config.has_section("ligands"):
            self.ligand = {}
            for option in config.options("ligands", ignore_defaults=True):
                if option not in self.ligand:
                    self.ligand[option] = {}
                
                info = config.getlist("ligands", option, delim=" ")
                if len(info) == 1:
                    self.ligand[option] = None

                i = 0
                while i < len(info):
                    if info[i].lower().startswith("swap"):
                        i += 1
                        old_keys = info[i].split("=")[0]
                        new_lig = "=".join(info[i].split("=")[1:])
                        self.ligand[option]["map"] = (new_lig, old_keys)

                    else:
                        ndx = info[i].split("=")[0]
                        new_sub = "=".join(info[i].split("=")[1:])
                        self.ligand[ndx] = new_sub

                    i += 1
       
        if config.has_section("substrates"):
            self.substrate = {}
            for option in config.options("substrates", ignore_defaults=True):
                self.substrate[option] = {}
                info = config.getlist("substrates", option, delim=" ")
                for sub in info:
                    ndx = sub.split("=")[0]
                    new_sub = "=".join(sub.split("=")[1:])
                    self.substrate[option][ndx] = new_sub

    def get_templates(self):
        """
        Finds template files in either user's or built-in library
        """
        selectivity = self.selectivity
        if len(selectivity) == 0:
            # ensure we do the loop at least once
            selectivity += ["", ""]

        templates = []
        for sel in selectivity:
            # get relative path
            path = os.path.join("TS_geoms", self.reaction_type, self.template)
            if sel:
                path = os.path.join(path, sel)
            # try to find in user's library
            if os.access(os.path.join(AARONLIB, path), os.R_OK):
                path = os.path.join(AARONLIB, path)
            # or in built-in library
            elif os.access(os.path.join(QCHASM, path), os.R_OK):
                path = os.path.join(QCHASM, "Aaron", path)
            else:
                msg = "Cannot find/access TS structure templates ({})"
                raise OSError(msg.format(path))
            # find template XYZ files in subdirectories
            for top, dirs, files in os.walk(path):
                if not files:
                    continue
                for f in files:
                    if f.endswith(".xyz"):
                        d = os.path.relpath(top, path)
                        # fix for path joining later
                        if d == ".":
                            d = ""
                        # autodetect R/S selectivity
                        elif d == "R" or d == "S":
                            path = os.path.join(top, d)
                            if d not in self.selectivity:
                                self.selectivity += [d]
                        templates += [(path, d, f)]
        self.template_XYZ = templates

    def generate_structure_data(self, top_dir):
        """
        Generates catalyst metadata for requested mapping/substitution
        combinations. See CatalystMetaData for data structure info

        Sets self.catalyst_data = [CatalystMetaData(), ...]
        """
        if not self.template_XYZ:
            self.get_templates()
        for i, template in enumerate(self.template_XYZ):
            template_file = os.path.join(*template)
            template_path = os.path.join(*template[:2]).rstrip("/")
            # get selectivity, if necessary
            split_path = os.path.split(template_path)
            if split_path[1].lower().startswith("ts"):
                split_path = os.path.split(split_path[0])
            if split_path[1] in self.selectivity:
                selectivity = split_path[1]
            elif split_path[1] in ["R", "S"]:
                selectivity = split_path[1]
                self.selectivity += [selectivity]
            else:
                selectivity = None
            # get changes to catalyst
            for lig_change in self.ligand.items():
                if not lig_change[1]:
                    lig_change = (lig_change[0], None)
                else:
                    # want ligand changes without substrate changes
                    self.catalyst_data += [
                        CatalystMetaData(
                            template_file=template_file,
                            reaction_type=self,
                            selectivity=selectivity,
                            ligand_change=lig_change,
                            substrate_change=("orig", None),
                        )
                    ]
                for sub_change in self.substrate.items():
                    done = False
                    if not sub_change[1]:
                        for cat in self.catalyst_data:
                            if cat.template_file != template_file:
                                continue
                            if cat.reaction_type != self:
                                continue
                            if cat.selectivity != selectivity:
                                continue
                            if cat.ligand_change != lig_change:
                                continue
                            else:
                                cat.substrate_change[0] = sub_change[0]
                                done = True
                    if done:
                        continue
                    self.catalyst_data += [
                        CatalystMetaData(
                            template_file=template_file,
                            reaction_type=self,
                            selectivity=selectivity,
                            ligand_change=lig_change,
                            substrate_change=sub_change,
                        )
                    ]
            # even if no changes requested, still want to optimize templates
            if len(self.ligand.keys()) < 1 and len(self.substrate.keys()) < 1:
                self.catalyst_data += [
                    CatalystMetaData(
                        template_file=template_file,
                        reaction_type=self,
                        selectivity=selectivity,
                        ligand_change=("orig", None),
                        substrate_change=("orig", None),
                    )
                ]

        self.by_lig_change = {}
        self.by_sub_change = {}
        for cat in self.catalyst_data:
            if cat.ligand_change[0] in self.by_lig_change:
                self.by_lig_change[cat.ligand_change[0]] += [cat]
            else:
                self.by_lig_change[cat.ligand_change[0]] = [cat]
            if cat.substrate_change[0] in self.by_sub_change:
                self.by_sub_change[cat.substrate_change[0]] += [cat]
            else:
                self.by_sub_change[cat.substrate_change[0]] = [cat]

            # generate child directory name
            lig_str = cat.ligand_change[0]
            sub_str = cat.substrate_change[0]
            # child_dir is top/lig_spec/sub_spec...
            child_dir = os.path.join(top_dir, lig_str, sub_str)
            # ...[/selectivity]...
            if cat.selectivity:
                child_dir = os.path.join(child_dir, cat.selectivity)
            # .../ts#
            dir_up, tname = os.path.split(cat.template_file)
            dir_up = os.path.basename(dir_up)
            tname = os.path.splitext(tname)[0]
            if tname.lower().startswith("ts"):
                # template files are ts structures
                child_dir = os.path.join(child_dir, tname)
            elif dir_up.lower().startswith("ts"):
                # template files are conformer structures within ts directory
                child_dir = os.path.join(child_dir, dir_up)
                cat.ts_number = int(dir_up[2:])
            cat.ts_directory = child_dir

    def load_from_logs(self, top_dir, old_aaron=False):
        new_cat_data = []
        cat_change_done = []
        if not self.template_XYZ:
            self.get_templates()
        if not self.catalyst_data:
            self.generate_structure_data(top_dir)
        while len(self.catalyst_data):
            cat_data = self.catalyst_data.pop()
            if cat_data.substrate_change[1]:
                cat_change = (
                    cat_data.ligand_change[1],
                    cat_data.substrate_change[1],
                    cat_data.selectivity,
                )
                if cat_change in cat_change_done:
                    continue
            cat_change_done += [cat_change]
            rv = cat_data._get_log_names(top_dir=top_dir, old_aaron=old_aaron)
            for log in rv:
                tmp = copy(cat_data)
                tmp.template_file = log
                if old_aaron:
                    if tmp.substrate_change[1]:
                        tmp.ligand_change = tmp.ligand_change[0], None
                    elif tmp.ligand_change[1]:
                        tmp.substrate_change = "orig", None
                new_cat_data += [tmp]
        self.catalyst_data = new_cat_data
        for cd in self.catalyst_data:
            print(cd.__dict__, end="\n\n")


class AaronTheory(Theory):
    """
    Class Attributes:
    :by_step:       dictionary sorted by step
    :nobasis:       list of methods for which no basis set is required
    :_skip_keys:    route_kwargs keys that are handled specially

    Attributes:
    :other_kwargs:  other kwargs (route, etc.)
    :temperature:   temperature for FrequencyJobs
    
    :top_dir:       local directory to store files
    :remote_dir:    directory to store computation files
    :host:          cluster host (where to submit jobs)
    :transfer_host: host to use to transfer files (defaults to :host:)
    :queue_type:    queuing software used by cluster
    :queue:         name of queue to submit to
    :procs:         number of cores per node
    :nodes:         number of nodes
    :wall:          wall time
    :queue_memory:  memory requested (can be function)
    :memory:        memory requested for computation (can be function)
    """

    by_step = {}  # key 0.0 corresponds to default
    kwargs_by_step = {}
    conditional_kwargs_by_step = {}

    def __init__(self, config=None, gaussian_config=None, orca_config=None, psi4_config=None):
        super().__init__()

        self.step = 0.0
        self.method = None
        self.basis = BasisSet()
        self.other_kwargs = {}
        self.temperature = 298.15

        self.top_dir = None
        self.user = None
        self.host = None
        self.transfer_host = None
        self.remote_dir = None
        self.queue_type = None
        self.queue = None
        self.nodes = 1
        self.processors = None
        self.walltime = None
        self.queue_memory = None
        self.memory = None

        self.config = config

        if config is None:
            return
        else:
            self.read_config(config, gaussian_config, orca_config, psi4_config)

    # class methods
    @classmethod
    def read_config(cls, config, gaussian_config, orca_config, psi4_config):
        # if the user didn't specify these sections, add them in
        # ConfigParser will just use the defaults
        if not config.has_section("options"):
            config.add_section("options")
        
        if not gaussian_config.has_section("Gaussian options"):
            gaussian_config.add_section("Gaussian options")
        
        if not orca_config.has_section("ORCA options"):
            orca_config.add_section("ORCA options")
        
        if not psi4_config.has_section("Psi4 options"):
            psi4_config.add_section("Psi4 options")

        if "host" in config["options"] and "transfer_host" not in config["options"]:
            config.set("options", "transfer_host", config.get("options", "host"))
        for name in config.options("options"):
            info = config.get("options", name)
            step = name.split()[0]
            try:
                step = float(step)
                name = " ".join(name.split()[1:])
            except ValueError:
                if step.lower() in ["low", "short"]:
                    step = 1.0
                    name = " ".join(name.split()[1:])
                elif step.lower() == "high":
                    step = 5.0
                    name = " ".join(name.split()[1:])
                else:
                    step = 0.0  # default
            # change to correct Theory instance
            if step not in cls.by_step:
                cls.by_step[step] = cls()
                cls.by_step[step].config = config

            if step not in cls.kwargs_by_step:
                cls.kwargs_by_step[step] = {}

            if step not in cls.conditional_kwargs_by_step:
                cls.conditional_kwargs_by_step[step] = {}

            obj = cls.by_step[step]
            kwargs = cls.kwargs_by_step[step]
            conditional_kwargs = cls.conditional_kwargs_by_step[step]
            obj.step = step
            
            # cluster options
            solvent = None
            solvent_model = None
            if name.lower() == "remote_dir":
                obj.remote_dir = info
            elif name.lower() == "user":
                obj.user = info
            elif name.lower() == "host":
                obj.host = info
            elif name.lower() == "transfer_host":
                obj.transfer_host = info
            elif name.lower() == "top_dir":
                obj.top_dir = info
            elif name.lower() == "queue_type":
                obj.queue_type = info
            elif name.lower() == "queue":
                obj.queue = info
            elif re.search("^(n_?)?procs$", name.strip(), re.I):
                obj.processors = config.getint("options", name)
            elif re.search("^(n_?)?nodes$", name.strip(), re.I):
                obj.nodes = info
            elif re.search("^wall([ _]?time)?", name.strip(), re.I):
                obj.walltime = info
            elif name.lower() == "memory":
                obj.queue_memory = config.getint("options", name)
            elif name.lower() == "exec_memory":
                obj.memory = config.getint("options", name)
            elif name.lower() == "program":
                obj.program = info

            # theory options
            elif name.lower() == "method":
                obj.method = Method(info)
            elif name.lower().startswith("basis"):
                obj.set_basis(name, info)
            elif name.lower().startswith("ecp"):
                obj.set_ecp(name, info)
            elif name.lower() == "temperature":
                obj.temperature = info
            elif name.lower() == "solvent model":
                solvent_model = info
            elif name.lower() == "solvent":
                solvent = info
            elif name.lower() == "charge":
                obj.charge = config.getint("options", name)
            elif re.search("^mult(iplicity)?$", name.strip(), re.I):
                obj.multiplicity = config.getint("options", name)
            elif re.search("^(int(egration)?_?)?grid$", name.strip(), re.I):
                obj.grid = IntegrationGrid(info)
            elif re.search(
                "^emp(erical_?)?disp(ersion)?$", name.strip(), re.I
            ):
                obj.empirical_dispersion = EmpiricalDispersion(info)
            elif re.search("^den(sity_?)?fit(ting)$", name.strip(), re.I):
                denfit = config.getbool("options", name)
                if denfit:
                    if GAUSSIAN_ROUTE not in kwargs:
                        kwargs[GAUSSIAN_ROUTE] = {}

                    kwargs[GAUSSIAN_ROUTE]["DensityFit"] = []

        if solvent is not None and solvent.lower() != "gas" and solvent_model is not None:
            obj.solvent = ImplicitSolvent(solvent_model, solvent)

        # go through each program-specific option and parse keyword arguments
        for prgm_config, prgm_section, one_layer_positions, two_layer_positions in \
            zip(
                [gaussian_config, orca_config, psi4_config], 
                ["Gaussian options", "ORCA options", "Psi4 options"], 
                [[GAUSSIAN_POST], [ORCA_ROUTE], [PSI4_BEFORE_GEOM, PSI4_AFTER_JOB]], 
                [[GAUSSIAN_PRE_ROUTE, GAUSSIAN_ROUTE], [ORCA_BLOCKS], [PSI4_SETTINGS, PSI4_COORDINATES, PSI4_JOB, PSI4_OPTKING]],
            ):

            for name in prgm_config.options(prgm_section):
                info = prgm_config.get(prgm_section, name)
                step = name.split()[0]
                try:
                    step = float(step)
                    name = " ".join(name.split()[1:])
                except ValueError:
                    if step.lower() in ["low", "short"]:
                        step = 1.0
                        name = " ".join(name.split()[1:])
                    elif step.lower() == "high":
                        step = 5.0
                        name = " ".join(name.split()[1:])
                    else:
                        step = 0.0  # default
                # change to correct Theory instance
                if step not in cls.by_step:
                    cls.by_step[step] = cls()
                    cls.by_step[step].config = config
                    cls.kwargs_by_step = {}
                    cls.conditional_kwargs_by_step

                obj = cls.by_step[step]
                kwargs = cls.kwargs_by_step[step]
                conditional_kwargs = cls.conditional_kwargs_by_step[step]

                kw_info = name.split()[0]
                # two-layer options - things like 'route freq = HPModes, NoRaman'
                for position in two_layer_positions:
                    if kw_info[0].lower() == position:
                        if len(kw_info) == 1:
                            if kw_info[0] not in kwargs:
                                kwargs[position] = {}
                            for value in config.getlist(prgm_section, name):
                                kwargs[position][value] = []
                        else:
                            keyword = kw_info[1]
                            if position not in conditional_kwargs:
                                conditional_kwargs[position] = {}
                            if keyword not in conditional_kwargs[position]:
                                conditional_kwargs[position][keyword] = []
                            conditional_kwargs[position][keyword].extend(prgm_config.getlist(prgm_section, name))
                
                # one-layer options - things like 'simple = TightSCF'
                for position in one_layer_positions:
                    if kw_info[0].lower() == position:
                        if position not in kwargs:
                            kwargs[position] = []
                        kwargs[position].extend(prgm_config.get_list(prgm_section, name))


        # cascade defaults to steps without settings
        cls.cascade_defaults()
        
        # check for nobasis
        for obj in cls.by_step.values():
            if obj.method is None:
                raise RuntimeError("method is a required paramter in the [options] section of the input file or the defined default section")

    @classmethod
    def cascade_defaults(cls):
        # get default object
        if 0.0 not in cls.by_step:
            return
        else:
            defaults = cls.by_step[0.0]
            default_kwargs = cls.kwargs_by_step[0.0]
            default_conditional_kwargs = cls.conditional_kwargs_by_step[0.0]
            defaults.processors = int(re.findall("\d+", str(defaults.nodes))[0])
            defaults.processors *= int(re.findall("\d+", str(defaults.processors))[0])
            if not defaults.memory:
                defaults.memory = defaults.queue_memory
        # add keyword defaults

        # update step objects with defaults
        defaults = defaults.__dict__
        for step, obj in cls.by_step.items():
            if step == 0.0:
                continue
            if not obj.memory:
                obj.memory = obj.queue_memory
            for key, val in obj.__dict__.items():
                if val in ["processors"]:
                    # do these after
                    continue
                if val is not None:
                    # don't overwrite
                    continue
                obj.__dict__[key] = defaults[key]

            obj.processors = int(re.findall("\d+", str(obj.nodes))[0])
            obj.processors *= int(re.findall("\d+", str(obj.processors))[0])

        for step, kwargs in cls.kwargs_by_step.items():
            if step == 0.0:
                continue
            cls.kwargs_by_step[step] = utils.combine_dicts(kwargs, default_kwargs)

        for step, conditional_kwargs in cls.conditional_kwargs_by_step.items():
            if step == 0.0:
                continue
            cls.conditional_kwargs_by_step[step] = utils.combine_dicts(conditional_kwargs, default_conditional_kwargs)


    @classmethod
    def get_step(cls, step):
        step = float(step)
        if step not in cls.by_step:
            step = 0.0
        return cls.by_step[step], cls.kwargs_by_step[step], cls.conditional_kwargs_by_step[step]

    @classmethod
    def set_top_dir(cls, top_dir):
        for theory in cls.by_step.values():
            theory.top_dir = top_dir

    @classmethod
    def write_input(cls, geometry, step=0.0, **kwargs):
        obj, step_kwargs, conditional_kwargs = cls.theory_for_step(geometry, step)
        kwargs = utils.combine_dicts(kwargs, step_kwargs)

        if obj.program.lower() == "gaussian":
            style = "com"
        elif obj.program.lower() == "orca":
            style = "inp"
        elif obj.program.lower() == "psi4":
            style = "in"

        FileWriter.write(geometry, theory=obj, step=step, style=style, conditional_kwargs=conditional_kwargs, **kwargs)

    @classmethod
    def theory_for_step(cls, geometry, step):
        """
        gets theory for the step
        Returns: Theory(), dict, dict
        1st dict keys are keywords for FileWriter.write
        2nd dict keys are for FileWriter.write(conditional_kwargs)

        :geometry:  the geometry object or name string
        :step:
        """
        theory, kwargs, conditional_kwargs = cls.get_step(step)

        # calcuation type
        if step < 2:
            constraints = {'atoms': [atom for atom in geometry.atoms if atom.flag != 0], 
                           'bonds': geometry.get_constraints()}
            job = OptimizationJob(constraints=constraints, geometry=geometry)
        elif step < 3:
            constraints = {'bonds': geometry.get_constraints}
            job = OptimizationJob(constraints=constraints, geometry=geometry)
        elif step < 4:
            job = OptimizationJob(transition_state=True)
            if hasattr(geometry, name):
                name = geometry.name
            else:
                # assume it's a string then
                name = geometry
            if program == "gaussian":
                if GAUSSIAN_ROUTE not in kwargs:
                    kwargs[GAUSSIAN_ROUTE] = {}
                for key in kwargs[GAUSSIAN_ROUTE].keys():
                    if key.lower() == "opt":
                        if os.access("{}.chk".format(name), os.R_OK):
                            kwargs[GAUSSIAN_ROUTE][key].append("ReadFC")
                        else:
                            kwargs[GAUSSIAN_ROUTE][key].append("CalcFC")
                        break
                else:
                    if os.access("{}.chk".format(name), os.R_OK):
                        kwargs[GAUSSIAN_ROUTE]["opt"].append("ReadFC")
                    else:
                        kwargs[GAUSSIAN_ROUTE]["opt"].append("CalcFC")
                    
        elif step < 5:
            # TODO: if program is ORCA, we need to check if ORCA has analytical frequencies
            #       because you need to request NumFreq (Freq errors out, which we could catch...but we could also not)
            job = FrequencyJob(temperature=theory.temperature)

        else:
            job = SinglePointJob()

        theory.job_type = [job]

        return theory, kwargs, conditional_kwargs

    # instance methods
    def parse_function(self, func, params=None, as_int=False):
        # TODO: move this to an interpolater for AaronConfigParser
        original = func
        if params is None:
            params = self.params
        groups = re.match("{(.*?)}", func).groups()
        for func in groups:
            all_subst = []
            subst = ""
            new_func = func
            for i, f in enumerate(func):
                if f == "$":
                    subst = f
                    continue
                if subst and re.match("[a-zA-Z_]", f):
                    subst += f
                    continue
                elif subst:
                    all_subst += [subst]
                    subst = ""
            for subst in all_subst:
                val = subst[1:]
                try:
                    val = re.match(r"-?\d+\.?\d*", params[val]).group(0)
                except AttributeError:
                    tmp = self.parse_function(params[val], params, as_int)
                    val = re.match(r"-?\d+\.?\d*", tmp).group(0)
                new_func = new_func.replace(subst, val)
            if as_int:
                new_func = int(eval(new_func))
            else:
                new_func = eval(new_func)
            original = original.replace("{" + func + "}", str(new_func))
        return original

    def set_basis(self, spec, basis_info):
        """
        Stores basis set information as a dictionary keyed by atoms

        :spec: element specification string from Aaron input file
               may include element symbols or tm, possibly prefixed by !
               to indicate that that element or all transition metals are
               to be excluded
        
        :basis_info: basis specification from Aaron input file
                     basis name and path to external basis file
                     should be separated by whitespace, e.g.
                     def2-svpd /home/CoolUser/gbs/def2-svpd.gbs
        """
        info = basis_info.split()

        basis_name = info[0]
        if len(info) > 1:
            user_defined = basis_info[:len(basis_name)].strip()
        else:
            user_defined = False

        aux_type = None
        finders = spec.split()
        # first thing is 'basis' - remove it
        finders.pop(0)
        i = 0
        # check for aux_type
        while i < len(finders):
            if finders[i].lower().startswith('aux'):
                aux_type = finders[i+1]
                finders.pop(i)
                finders.pop(i)
                break
            i += 1

        print("finders")
        print(finders)

        if len(finders) == 0:
            finders = None

        self.basis.add_basis(Basis(basis_name, elements=finders, aux_type=aux_type, user_defined=user_defined))
        
        return

    def set_ecp(self, spec, basis_info):
        """
        Stores ecp information as a dictionary keyed by atoms

        :spec: element specification string from Aaron input file
               may include element symbols or tm, possibly prefixed by !
               to indicate that that element or all transition metals are
               to be excluded
        
        :basis_info: basis specification from Aaron input file
                     basis name and path to external basis file
                     should be separated by whitespace, e.g.
                     def2-svpd /home/CoolUser/gbs/def2-svpd.gbs
        """

        print(spec, "<%s>" % basis_info)

        info = basis_info.split()

        basis_name = info[0]
        if len(info) > 1:
            user_defined = basis_info[:len(basis_name)].strip()
        else:
            user_defined = False

        finders = spec.split()
        #first thing is 'ecp' - remove it
        finders.pop(0)

        print(finders)

        if len(finders) == 0:
            finders = None

        self.basis.add_ecp(ECP(basis_name, elements=finders, user_defined=user_defined))
        
        return
