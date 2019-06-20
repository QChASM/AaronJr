import glob
import json
import os
import re
from copy import copy

import Aaron.json_extension as json_ext
from AaronTools.catalyst import Catalyst
from AaronTools.component import Component
from AaronTools.const import AARONLIB, ELEMENTS, QCHASM, TMETAL
from AaronTools.fileIO import FileWriter
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.utils import utils


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
        status=None,
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
        self.status = status
        self.catalyst_change = catalyst_change

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

    def get_basename(self, conf_spec=None):
        """
        :conf_spec: the Catalyst().conf_spec dictionary, uses the one
            associated with self.catalyst if None
        """
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
        if conf_spec is None:
            conf_spec = self._get_cf_dir_num()
        basename += "cf{}".format(conf_spec)
        return basename

    def get_relative_path(self):
        cat_change = os.path.join(
            self.ligand_change[0], self.substrate_change[0]
        )
        rv = self.ts_directory.split(cat_change)[-1][1:]
        return os.path.join(cat_change, rv)

    def get_catalyst_name(self, conf_spec=None):
        """
        :conf_spec: the Catalyst().conf_spec dictionary, uses the one
            associated with self.catalyst if None
        """
        basename = self.get_basename(conf_spec)
        cf_dir = "cf{}".format(self._get_cf_dir_num(conf_spec))
        return os.path.join(self.ts_directory, cf_dir, basename)

    def _get_cf_dir_num(self, conf_spec=None):
        """
        :conf_spec: the Catalyst().conf_spec dictionary, uses the one
            associated with self.catalyst if None
        """
        if conf_spec is None:
            conf_spec = self.catalyst.conf_spec
        match = re.search(".*/cf(\d+).xyz", self.template_file, flags=re.I)
        if match is None:
            s = ""
        else:
            s = match.group(1) + "-"
        for start, (conf_num, skip) in sorted(conf_spec.items()):
            s += str(conf_num)
        return s

    def _get_nconfs(self):
        nconfs = 1
        for start in self.catalyst.conf_spec:
            sub = self.catalyst.find_substituent(start)
            nconfs *= sub.conf_num
        return nconfs

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
                self.catalyst.substitute(sub, atom)
                requested_changes += [atom + ".1"]
        return requested_changes

    def _do_substrate_change(self):
        requested_changes = []
        if self.substrate_change[1] is not None:
            sub, change = self.substrate_change
            # substitutions
            for atom, sub in change.items():
                requested_changes += [atom + ".1"]
                self.catalyst.substitute(sub, atom)
        return requested_changes

    def generate_structure(self, old_aaron=False):
        """
        Generates a mapped/substituted catalyst structure from the information
        stored in a CatalystMetaData object

        Returns: Catalyst()

        :cat_data: CatalystMetaData() used to generate the structure
        """
        if self.catalyst is None:
            try:
                self._read_json()
                if self.catalyst:
                    return self.catalyst
            except BaseException:
                pass
            self.catalyst = Catalyst(self.template_file)
        if self.ligand_change is None and self.substrate_change is None:
            if old_aaron:
                self.catalyst.conf_spec = {}
            return self.catalyst
        # do mappings/substitution
        requested_changes = self._do_ligand_change()
        requested_changes += self._do_substrate_change()
        self.catalyst_change = requested_changes

        # try to get better starting structure
        self.catalyst.remove_clash()
        self.catalyst.minimize()

        # only do conformers for requested changes if old_aaron
        debug = False
        if old_aaron and requested_changes:
            if debug:
                for key in self.catalyst.conf_spec:
                    sub = self.catalyst.find_substituent(key)
                    print(key, sub.name, sub.atoms[0].name)
                print(requested_changes)
            to_delete = []
            try:
                tmp = []
                for c in requested_changes:
                    tmp += self.catalyst.find_exact(c)
                requested_changes = tmp
            except LookupError:
                raise LookupError(c)
            for key in self.catalyst.conf_spec:
                try:
                    sub = self.catalyst.find_substituent(key)
                except LookupError:
                    pass
                if sub.atoms[0] not in requested_changes:
                    to_delete += [key]
                elif sub.conf_num is not None and sub.conf_num <= 1:
                    to_delete += [key]
            for key in to_delete:
                del self.catalyst.conf_spec[key]
            if debug:
                for key in self.catalyst.conf_spec:
                    sub = self.catalyst.find_substituent(key)
                    print(key, sub.name, sub.atoms[0].name)
        self.catalyst.name = os.path.join(
            self.ts_directory, self.get_basename()
        )
        return self.catalyst

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
        with open(name + ".json", "w") as f:
            json.dump(self, f, cls=json_ext.JSONEncoder)

    def _read_json(self):
        name = os.path.join(
            self.ts_directory,
            os.path.basename(self.template_file).rsplit(".", maxsplit=1)[0],
        )
        with open(name + ".json") as f:
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
        self.top_dir = ""
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
        if "top_dir" in params:
            self.top_dir = params["top_dir"]

    def set_up(self, top_dir=None):
        self.make_directories(top_dir)

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
                path += "/{}".format(sel)
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

    def generate_structure_data(self):
        """
        Generates catalyst metadata for requested mapping/substitution
        combinations. See CatalystMetaData for data structure info

        Sets self.catalyst_data = [CatalystMetaData(), ...]
        """
        self.catalyst_data = []
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
                            substrate_change=("original_substrate", None),
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
                        ligand_change=("original_ligand", None),
                        substrate_change=("original_substrate", None),
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

    def make_directories(self, top_dir=None):
        """
        :top_dir:   the directory in which to put generated directories
        """
        if top_dir is None:
            top_dir = self.top_dir
        if os.path.isfile(top_dir):
            top_dir = os.path.dirname(os.path.abspath(top_dir))
        if not os.access(top_dir, os.W_OK):
            try:
                os.makedirs(top_dir)
            except OSError:
                msg = "Directory write permission denied:\n  {}"
                raise OSError(msg.format(top_dir))
        self.top_dir = top_dir

        if not self.catalyst_data:
            self.generate_structure_data()
        for cat_data in self.catalyst_data:
            # generate child directory name
            lig_str = cat_data.ligand_change[0]
            sub_str = cat_data.substrate_change[0]
            # child_dir is top/lig_spec/sub_spec...
            child_dir = os.path.join(top_dir, lig_str, sub_str)
            # ...[/selectivity]...
            if cat_data.selectivity:
                child_dir = os.path.join(child_dir, cat_data.selectivity)
            # .../ts#
            dir_up, tname = os.path.split(cat_data.template_file)
            dir_up = os.path.basename(dir_up)
            tname = os.path.splitext(tname)[0]
            if tname.lower().startswith("ts"):
                # template files are ts structures
                child_dir = os.path.join(child_dir, tname)
                try:
                    os.makedirs(child_dir, mode=0o755)
                except FileExistsError:
                    pass
                cat_data.ts_number = int(tname[2:])
            elif dir_up.lower().startswith("ts"):
                # template files are conformer structures within ts directory
                child_dir = os.path.join(child_dir, dir_up)
                try:
                    os.makedirs(child_dir, mode=0o755)
                except FileExistsError:
                    pass
                cat_data.ts_number = int(dir_up[2:])
            cat_data.ts_directory = child_dir
            try:
                os.makedirs(os.path.join(child_dir, "json"))
            except FileExistsError:
                pass

    def make_conformers(self, top_dir=None, old_aaron=False):
        # generate structure data and make directories
        if top_dir is None:
            top_dir = self.top_dir
        else:
            self.top_dir = top_dir
        if not self.catalyst_data:
            # make_directories calls generate_structure_data() if needed
            self.make_directories(top_dir=top_dir)
        else:
            for cat_data in self.catalyst_data:
                if cat_data.ts_directory is None:
                    self.make_directories(top_dir=top_dir)
                    break

        n = len(self.catalyst_data)
        for i, cat_data in enumerate(self.catalyst_data):
            cat_data.generate_structure(old_aaron=old_aaron)
            catalyst = cat_data.catalyst
            m = cat_data._get_nconfs()
            j = -1
            while True:
                j += 1
                utils.progress_bar(
                    (j / m) + i, n, name="Generating Conformers"
                )
                cf_num = cat_data._get_cf_dir_num()
                cat_name = (
                    os.path.join(
                        cat_data.ts_directory, "json/cf{}".format(cf_num)
                    )
                    + ".json"
                )
                with open(cat_name, "w") as f:
                    json.dump(cat_data, f, cls=json_ext.JSONEncoder)
                if not catalyst.next_conformer():
                    break
        utils.clean_progress_bar()

    def load_from_logs(self, top_dir=None, old_aaron=False):
        new_cat_data = []
        cat_change_done = []
        if top_dir is None:
            top_dir = self.top_dir
        if not self.template_XYZ:
            self.get_templates()
        if not self.catalyst_data:
            self.generate_structure_data()
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
                        tmp.substrate_change = "original_substrate", None
                new_cat_data += [tmp]
        self.catalyst_data = new_cat_data
        for cd in self.catalyst_data:
            print(cd.__dict__, end="\n\n")


class ClusterOpts:
    """
    Attributes:
        queue_type
        n_procs     number of processors for steps2-5
        wall        walltime in hours for steps2-5
        short_procs number of cores for step1
        short_wall  walltime in hours for step1
    """

    def __init__(self, params=None):
        self.top_dir = None
        self.host = None
        self.remote_dir = None
        self.queue_type = None
        self.queue = None
        self.n_procs = None
        self.short_procs = None
        self.wall = None
        self.short_wall = None
        self.n_nodes = 1
        self.short_nodes = 1
        self.memory = 0
        self.exec_memory = 0
        if params is None:
            return

        if "remote_dir" in params:
            self.remote_dir = params["remote_dir"]
            self.host = params["host"]

        self.queue_type = params["queue_type"]
        self.queue = params["queue"]
        self.n_procs = params["n_procs"]
        self.wall = params["wall"]

        if "short_procs" in params:
            self.short_procs = params["short_procs"]
        if "short_wall" in params:
            self.short_wall = params["short_wall"]
        if "short_nodes" in params:
            self.short_nodes = params["short_nodes"]
        if "n_nodes" in params:
            self.n_nodes = params["n_nodes"]
        if "memory" in params:
            memory = params["memory"]
            if "$" in memory:
                memory = self._parse_function(memory, params, as_int=True)
            self.memory = memory
        if "exec_memory" in params:
            memory = params["exec_memory"]
            if "{" in memory:
                memory = self._parse_function(memory, params, as_int=True)
            self.exec_memory = memory

    def _parse_function(self, func, params, as_int=False):
        original = func
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
                val = re.match("-?\d+\.?\d*", params[val]).group(0)
                new_func = new_func.replace(subst, val)
            if as_int:
                new_func = int(eval(new_func))
            else:
                new_func = eval(new_func)
            original = original.replace("{" + func + "}", str(new_func))
        return original


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
    """

    nobasis = ["AM1", "PM3", "PM3MM", "PM6", "PDDG", "PM7"]
    by_step = {}  # key 0.0 corresponds to default
    _skip_keys = ["temperature", "charge", "mult"]

    # class methods
    @classmethod
    def read_params(cls, params):
        for name, info in params.items():
            step = name.split("_")[0]
            try:
                step = float(step)
            except ValueError:
                if step.lower() == "low":
                    step = 1.0
                elif step.lower() == "high":
                    step = 5.0
                else:
                    step = 0.0  # default
            # change to correct Theory instance
            if step not in cls.by_step:
                cls.by_step[step] = cls()
            obj = cls.by_step[step]
            obj.step = step
            if "method" in name.lower():
                obj.method = info
            elif "basis" in name.lower():
                obj.set_basis(info)
            elif "ecp" in name.lower():
                obj.set_ecp(info)
            elif "temperature" in name.lower():
                obj.route_kwargs["temperature"] = float(info)
            elif "solvent_model" in name.lower():
                if "scrf" in obj.route_kwargs:
                    obj.route_kwargs["scrf"][info] = ""
                else:
                    obj.route_kwargs["scrf"] = {info: ""}
            elif "solvent" in name.lower():
                if "scrf" in obj.route_kwargs:
                    obj.route_kwargs["scrf"]["solvent"] = info
                else:
                    obj.route_kwargs["scrf"]["solvent"] = info
            elif "charge" in name.lower():
                obj.route_kwargs["charge"] = int(info)
            elif "mult" in name.lower():
                obj.route_kwargs["mult"] = int(info)
            elif "grid" in name.lower():
                obj.route_kwargs["int"] = {"grid": info}
            elif re.search("emp(erical_?)?disp(ersion)?", name, re.I):
                obj.route_kwargs["EmpericalDispersion"] = info
            elif re.search("den(sity_?)?fit(ting)", name, re.I):
                try:
                    denfit = bool(int(info))
                except ValueError:
                    denfit = bool(info)
                if denfit:
                    obj.route_kwargs["denfit"] = ""
            elif "additional_opt" in name.lower():
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
            for key, val in obj.__dict__.items():
                if val == "route_kwargs":
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

    @classmethod
    def get_step(cls, step):
        step = float(step)
        if step not in cls.by_step:
            step = 0.0
        return cls.by_step[step]

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
            header = "%chk={}.chk\n".format(geometry.name)
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

        # other keywords
        route_dict = utils.add_dict(
            route_dict, theory.route_kwargs, skip=Theory._skip_keys
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
                if len(tmp) > 1:
                    val = "({})".format(",".join(tmp))
                elif len(tmp) == 1:
                    val = tmp[0]
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

        if params is not None:
            Theory.read_params(params)

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
