import glob
import hashlib
import json
import os
import pickle
import re
from copy import copy

from AaronTools.component import Component
from AaronTools.const import AARONLIB, ELEMENTS, QCHASM, TMETAL
from AaronTools.fileIO import FileWriter
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils import utils


class GeometryMetaData:
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
        stored in a GeometryMetaData object

        Returns: Geometry()
        """
        if self.pickle_update():
            return self.catalyst
        self.catalyst = Geometry(self.template_file)
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
        :conf_spec: the Geometry().conf_spec dictionary, uses the one
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

    def __init__(self, params=None):
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
        if params is None:
            return

        self.reaction_type = params["reaction_type"]
        self.template = params["template"]

        if "selectivity" in params:
            self.selectivity = params["selectivity"].split(";")
        if "ligand" in params:
            self.ligand = params["ligand"]
        if "substrate" in params:
            self.substrate = params["substrate"]
        if "con_thresh" in params:
            self.con_thresh = params["con_thresh"]

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
            path = "TS_geoms/{}/{}".format(self.reaction_type, self.template)
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
        combinations. See GeometryMetaData for data structure info

        Sets self.catalyst_data = [GeometryMetaData(), ...]
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
                        GeometryMetaData(
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
                        GeometryMetaData(
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
                    GeometryMetaData(
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


class Theory:
    """
    Class Attributes:
    :by_step:       dictionary sorted by step
    :nobasis:       list of methods for which no basis set is required
    :_skip_keys:    route_kwargs keys that are handled specially

    Attributes:
    :method:        method to use
    :basis:         basis set to use
    :ecp:           ecp to use
    :gen_basis:     path to gen basis files
    :route_kwargs:  other kwargs
    :_unique_basis: the basis set name, if for all atoms

    :top_dir:       local directory to store files
    :remote_dir:    directory to store computation files
    :host:          cluster host (where to submit jobs)
    :transfer_host: host to use to transfer files (defaults to :host:)
    :queue_type:    queuing software used by cluster
    :queue:         name of queue to submit to
    :procs:         number of cores per node
    :nodes:         number of nodes
    :wall:          wall time
    :memory:        memory requested (can be function)
    :exec_memory:   memory requested for computation (can be function)
    """

    nobasis = ["AM1", "PM3", "PM3MM", "PM6", "PDDG", "PM7"]
    by_step = {}  # key 0.0 corresponds to default
    _skip_keys = ["temperature", "charge", "mult"]

    # class methods
    @classmethod
    def read_params(cls, params):
        if "host" in params and "transfer_host" not in params:
            params["transfer_host"] = params["host"]
        for name, info in params.items():
            step = name.split("_")[0]
            try:
                step = float(step)
                name = "_".join(name.split("_")[1:])
            except ValueError:
                if step.lower() in ["low", "short"]:
                    step = 1.0
                    name = "_".join(name.split("_")[1:])
                elif step.lower() == "high":
                    step = 5.0
                    name = "_".join(name.split("_")[1:])
                else:
                    step = 0.0  # default
            # change to correct Theory instance
            if step not in cls.by_step:
                cls.by_step[step] = cls()
                cls.by_step[step].params = params
            obj = cls.by_step[step]
            obj.step = step

            # cluster options
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
                obj.ppnode = info
            elif re.search("^(n_?)?nodes$", name.strip(), re.I):
                obj.nodes = info
            elif re.search("^wall([ _]?time)?", name.strip(), re.I):
                obj.walltime = info
            elif name.lower() == "memory":
                obj.mem = info
            elif name.lower() == "exec_memory":
                obj.exec_mem = info

            # theory options
            elif name.lower() == "method":
                obj.method = info
            elif name.lower() == "basis":
                obj.set_basis(info)
            elif name.lower() == "ecp":
                obj.set_ecp(info)
            elif name.lower() == "temperature":
                obj.route_kwargs["temperature"] = float(info)
            elif name.lower() == "solvent_model":
                if "scrf" in obj.route_kwargs:
                    obj.route_kwargs["scrf"][info] = ""
                else:
                    obj.route_kwargs["scrf"] = {info: ""}
            elif name.lower() == "solvent":
                if "scrf" in obj.route_kwargs:
                    obj.route_kwargs["scrf"]["solvent"] = info
                else:
                    obj.route_kwargs["solvent"] = info
            elif name.lower() == "charge":
                obj.route_kwargs["charge"] = int(info)
            elif re.search("^mult(iplicity)?$", name.strip(), re.I):
                obj.route_kwargs["mult"] = int(info)
            elif re.search("^(int(egration)?_?)?grid$", name.strip(), re.I):
                obj.route_kwargs["int"] = {"grid": info}
            elif re.search(
                "^emp(erical_?)?disp(ersion)?$", name.strip(), re.I
            ):
                obj.route_kwargs["EmpericalDispersion"] = info
            elif re.search("^den(sity_?)?fit(ting)$", name.strip(), re.I):
                try:
                    denfit = bool(int(info))
                except ValueError:
                    denfit = bool(info)
                if denfit:
                    obj.route_kwargs["denfit"] = ""
            elif re.search("^add(itional)?_?opt(ions?)?$", name.strip(), re.I):
                obj.route_kwargs[""] = info

        # cascade defaults to steps without settings
        cls.cascade_defaults()
        # check for nobasis
        for obj in cls.by_step.values():
            if obj.method is None:
                raise RuntimeError("Theory().method is None")
            if obj.method.upper() in Theory.nobasis:
                obj.basis = {}
                obj.ecp = {}

    @classmethod
    def cascade_defaults(cls):
        # get default object
        if 0.0 not in cls.by_step:
            return
        else:
            defaults = cls.by_step[0.0]
            defaults.procs = int(re.findall("\d+", str(defaults.nodes))[0])
            defaults.procs *= int(re.findall("\d+", str(defaults.ppnode))[0])
            if not defaults.exec_mem:
                defaults.exec_mem = defaults.mem
        # add keyword defaults
        if "" not in defaults.route_kwargs:
            defaults.route_kwargs[""] = ""
        if "charge" not in defaults.route_kwargs:
            defaults.route_kwargs["charge"] = 0
        if "mult" not in defaults.route_kwargs:
            defaults.route_kwargs["mult"] = 1

        # update step objects with defaults
        defaults = defaults.__dict__
        for step, obj in cls.by_step.items():
            if step == 0.0:
                continue
            if not obj.exec_mem:
                obj.exec_mem = obj.mem
            for key, val in obj.__dict__.items():
                if val in ["route_kwargs", "procs"]:
                    # do these after
                    continue
                if val is not None:
                    # don't overwrite
                    continue
                obj.__dict__[key] = defaults[key]
            for key, val in defaults["route_kwargs"].items():
                if key in obj.route_kwargs:
                    # don't overwrite
                    continue
                obj.route_kwargs[key] = val
            obj.procs = int(re.findall("\d+", str(obj.nodes))[0])
            obj.procs *= int(re.findall("\d+", str(obj.ppnode))[0])

    @classmethod
    def get_step(cls, step):
        step = float(step)
        if step not in cls.by_step:
            step = 0.0
        return cls.by_step[step]

    @classmethod
    def set_top_dir(cls, top_dir):
        for theory in cls.by_step.values():
            theory.top_dir = top_dir

    @classmethod
    def write_com(cls, geometry, step=0.0, **kwargs):
        if float(step) in cls.by_step:
            obj = cls.by_step[float(step)]
        else:
            obj = cls.by_step[0.0]
        FileWriter.write_com(geometry, theory=obj, step=step, **kwargs)

    @classmethod
    def make_footer(cls, geometry, step):
        if not isinstance(geometry, Geometry):
            msg = "argument provided must be Geometry type; "
            msg += "provided type {}".format(type(geometry))
            raise ValueError(msg)
        footer = "\n"
        if float(step) in cls.by_step:
            theory = cls.by_step[float(step)]
        else:
            theory = cls.by_step[0.0]

        # constrained optimization
        if int(step) == 2:
            constraints = geometry.get_constraints()
            for con in constraints:
                footer += "B {} {} F\n".format(con[0] + 1, con[1] + 1)
            if len(constraints) > 0:
                footer += "\n"

        if theory.basis and (
            not theory.unique_basis(geometry) or theory.ecp or theory.gen_basis
        ):
            basis = {}
            for e in sorted(set(geometry.elements())):
                tmp = theory.basis[e]
                if tmp not in basis:
                    basis[tmp] = []
                basis[tmp] += [e]
            for b in sorted(basis):
                footer += ("{} " * len(basis[b])).format(*basis[b])
                footer += "0\n{}\n".format(b)
                footer += "*" * 4 + "\n"

        if theory.gen_basis:
            for g in sorted(set(theory.gen_basis.values())):
                footer += "@{}/N\n".format(g)

        if theory.ecp:
            footer += "\n"
            ecp = {}
            for e in sorted(set(geometry.elements())):
                if e not in theory.ecp:
                    continue
                tmp = theory.ecp[e]
                if tmp not in ecp:
                    ecp[tmp] = []
                ecp[tmp] += [e]
            for e in sorted(ecp):
                footer += ("{} " * len(ecp[e])).format(*ecp[e])
                footer += "0\n{}\n".format(e)

        if footer.strip() == "":
            return "\n\n\n"

        return footer + "\n\n\n"

    @classmethod
    def make_header(cls, geometry, step, **kwargs):
        """
        :geometry: Geometry()
        :step:
        :kwargs: additional keyword=value arguments for header
        """
        theory = cls.get_step(step)
        if "fname" in kwargs:
            fname = kwargs["fname"]
            del kwargs["fname"]
        else:
            fname = geometry.name
        # get comment
        if "comment" in kwargs:
            comment = kwargs["comment"]
            del kwargs["comment"]
        else:
            comment = "step {}".format(
                int(step) if int(step) == step else step
            )

        # checkfile
        if step > 1:
            header = "%chk={}.chk\n".format(fname)
        else:
            header = ""

        # route
        header += theory.make_route(geometry, step, **kwargs)

        # comment, charge and multiplicity
        header += "\n\n{}\n\n{},{}\n".format(
            comment, theory.route_kwargs["charge"], theory.route_kwargs["mult"]
        )
        return header

    @classmethod
    def make_route(cls, geometry, step, **kwargs):
        """
        Creates the route line for G09 com files
        Returns: str

        :geometry:  the geometry object or name string
        :step:
        :**kwargs:  additional Gaussian keywords to add
        """
        theory = cls.get_step(step)
        route_dict = {}

        # method and basis
        method = "#{}".format(theory.method)
        if theory.ecp:
            method += "/genecp"
        elif theory.unique_basis(geometry):
            method += "/{}".format(theory.unique_basis)
        elif theory.basis:
            method += "/gen"
        route_dict["method"] = method

        # calcuation type
        if step < 2:
            route_dict["opt"] = {}
            route_dict["nosym"] = {}
        elif step < 3:
            route_dict["opt"] = {"modredundant": "", "maxcyc": "1000"}
        elif step < 4:
            try:
                name = geometry.name
            except AttributeError:
                # assume it's a string then
                name = geometry
            route_dict["opt"] = {"ts": "", "maxcyc": "1000"}
            if os.access("{}.chk".format(name), os.R_OK):
                route_dict["opt"]["readfc"] = ""
            else:
                route_dict["opt"]["calcfc"] = ""
        elif step < 5:
            route_dict["freq"] = {"hpmodes": "", "noraman": ""}
            route_dict["freq"]["temperature"] = theory.route_kwargs[
                "temperature"
            ]

        # screen unneeded keywords
        skip_keys = Theory._skip_keys.copy()
        if step < 2:
            skip_keys += ["scrf"]
        # other keywords
        route_dict = utils.add_dict(
            route_dict, theory.route_kwargs, skip=skip_keys
        )
        # from **kwargs parameters
        route_dict = utils.add_dict(route_dict, kwargs, skip=Theory._skip_keys)

        # make the string from the dictionary
        route = route_dict["method"]
        del route_dict["method"]
        for key, val in sorted(route_dict.items()):
            if isinstance(val, dict):
                tmp = []
                for k, v in sorted(val.items()):
                    if k and v:
                        tmp += ["{}={}".format(k, v)]
                    elif k:
                        tmp += [k]
                    elif v:
                        tmp += [v]
                if len(tmp) > 0:
                    val = "({})".format(",".join(tmp))
                else:
                    val = ""
            if key and val:
                route += " {}={}".format(key, val)
            elif key:
                route += " {}".format(key)
            elif val:
                route += " {}".format(val)

        return route

    # instance methods
    def __init__(self, params=None):
        self.step = 0.0
        self.method = None
        self.basis = {}
        self.ecp = {}
        self.gen_basis = {}
        self.route_kwargs = {}
        self._unique_basis = None

        self.top_dir = None
        self.user = None
        self.host = None
        self.transfer_host = None
        self.remote_dir = None
        self.queue_type = None
        self.queue = None
        self.nodes = 1
        self.ppnode = None
        self.walltime = None
        self.mem = None
        self.exec_mem = None

        self.params = params

        if params is None:
            return
        if params is not None:
            Theory.read_params(params)

    def parse_function(self, func, params=None, as_int=False):
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

    def set_basis(self, spec):
        """
        Stores basis set information as a dictionary keyed by atoms

        :spec: list of basis specification strings from Aaron input file
            acceptable forms:
                "6-31G" will set basis for all atoms to 6-31G
                "tm SDD" will set basis for all transiton metals to SDD
                "C H O N def2tzvp" will set basis for C, H, O, and N atoms
        """
        for a in spec:
            info = a.split()
            basis = info.pop()
            for i in info:
                i = i.strip()
                if i in ELEMENTS:
                    self.basis[i] = basis
                elif i.lower() == "TM":
                    for tm in TMETAL:
                        self.basis[tm] = basis
            if len(info) == 0:
                for e in ELEMENTS:
                    if e not in self.basis:
                        self.basis[e] = basis
        return

    def set_ecp(self, args):
        """
        Stores ecp information as a dictionary keyed by atoms

        :spec: list of basis specification strings from Aaron input file
            acceptable forms:
                "tm SDD" will set basis for all transiton metals to SDD
                "Ru Pt SDD" will set basis for Ru and Pt to SDD
        """
        for a in args:
            info = a.strip().split()
            if len(info) < 2:
                msg = "ecp setting requires atom type: {}"
                msg = msg.format(" ".join(info))
                raise IOError(msg)
            ecp = info.pop()
            for i in info:
                if i in ELEMENTS:
                    self.ecp[i] = ecp
                elif i.lower() == "tm":
                    for t in TMETAL.keys():
                        self.ecp[i] = ecp
                else:
                    msg = "ecp setting requires atom type: {}"
                    msg = msg.format(" ".join(info))
                    raise IOError(msg)
        return

    def set_gen_basis(self, gen):
        """
        Creates gen basis specifications
        Checks to ensure basis set files exist

        :gen: str specifying the gen basis file directory
        """
        if self.gen_basis:
            return

        gen = gen.rstrip("/") + "/"
        for element in self.basis:
            if self.basis[element].lower().startswith("gen"):
                fname = self.basis[element]
                fname = gen + fname.strip("gen/")
                self.gen_basis[element] = fname
                del self.basis[element]
        if not self.gen_basis:
            return

        if not os.access(gen, os.R_OK):
            msg = "Cannot open directory {} to read basis set files."
            raise IOError(msg.format(gen))

        for gf in self.gen_basis.values():
            if not os.access(gf, os.R_OK):
                msg = "Cannot find basis set file {} in {}."
                raise IOError(msg.format(gf, gen))
        return

    def unique_basis(self, geometry):
        if self._unique_basis:
            return self._unique_basis
        if not self.basis:
            return False

        all_basis = set([])
        for a in geometry.atoms:
            all_basis.add(self.basis[a.element].upper())
        if len(all_basis) != 1:
            self._unique_basis = None
            return False
        else:
            self._unique_basis = self.basis[a.element]
            return self._unique_basis
