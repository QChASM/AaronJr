import os

from AaronTools.const import ELEMENTS, TMETAL, QCHASM, AARONLIB
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.component import Component
from AaronTools.catalyst import Catalyst
from AaronTools.fileIO import FileWriter


class CatalystMetaData:
    """
    :template_file:     file location of template.xyz
    :reaction_type:     name of the reaction type
    :template_name:     name of the template subdirectory
    :selectivity:       selectivity of the template used
    :ligand_changes:    changes to the ligand made
    :substrate_changes: changes to the substrate made
    :catalyst_dir:      where the catalyst optimizations will be done
    """

    def __init__(self, template_file=None, reaction=None, selectivity=None,
                 ligand_change=None, substrate_change=None, catalyst_dir=None):
        self.template_file = template_file
        if reaction is not None:
            if isinstance(reaction, Reaction):
                self.reaction_type = reaction.reaction_type
                self.template_name = reaction.template
            else:
                self.reaction_type, self.template_name = reaction
        else:
            self.reaction_type = None
            self.template_name = None
        self.selectivity = selectivity
        self.ligand_change = ligand_change
        self.substrate_change = substrate_change
        self.catalyst_dir = catalyst_dir


class Reaction:
    """
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
    """

    def __init__(self, params=None):
        self.reaction_type = ''
        self.template = ''
        self.template_XYZ = []
        self.selectivity = []
        self.ligand = {}
        self.substrate = {}
        self.catalyst_data = []
        self.con_thresh = None
        self.top_dir = None
        if params is None:
            return

        self.reaction_type = params['reaction_type']
        self.template = params['template']

        if 'selectivity' in params:
            self.selectivity = params['selectivity'].split(';')
        if 'ligand' in params:
            self.ligand = params['ligand']
        if 'substrate' in params:
            self.substrate = params['substrate']
        if 'con_thresh' in params:
            self.con_thresh = params['con_thresh']
        if 'top_dir' in params:
            self.top_dir = params['top_dir']

    def get_templates(self):
        """
        Finds template files in either user's or built-in library
        """
        selectivity = self.selectivity
        if len(selectivity) == 0:
            # ensure we do the loop at least once
            selectivity += ['']

        templates = []
        for sel in selectivity:
            # get relative path
            path = "TS_geoms/{}/{}".format(self.reaction_type, self.template)
            if sel:
                path += '/{}'.format(sel)
            # try to find in user's library
            if os.access(os.path.join(AARONLIB, path), os.R_OK):
                path = os.path.join(AARONLIB, path)
            # or in built-in library
            elif os.access(os.path.join(QCHASM, path), os.R_OK):
                path = os.path.join(QCHASM, 'Aaron', path)
            else:
                msg = "Cannot find/access TS structure templates ({})"
                raise OSError(msg.format(path))
            # find template XYZ files in subdirectories
            for top, dirs, files in os.walk(path):
                if not files:
                    continue
                for f in files:
                    if f.endswith('.xyz'):
                        d = os.path.relpath(top, path)
                        # fix for path joining later
                        if d == '.':
                            d = ''
                        # autodetect R/S selectivity
                        elif d == 'R' or d == 'S':
                            path = os.path.join(top, d)
                            if d not in self.selectivity:
                                self.selectivity += [d]
                        templates += [(path, d, f)]
        self.template_XYZ = templates

    def load_lib_ligand(cls, lig_name):
        """
        Loads a Component() from a ligand XYZ file in the library

        :lig_name: the name of the ligand
        :returns: Component()
        """
        if not lig_name.endswith('.xyz'):
            name = lig_name + '.xyz'
        path = [os.path.join(AARONLIB, 'Ligands', name)]
        path += [os.path.join(QCHASM, 'AaronTools/Ligands', name)]
        for p in path:
            if os.path.isfile(p):
                return Component(p)
        else:
            raise FileNotFoundError("Could not find ligand: " + name)

    def _do_ligand_changes(self, template_file, lig_change):
        lig, change = lig_change
        catalyst = Catalyst(template_file)
        # mappings
        if 'map' in change:
            name, old_keys = change['map']
            if not old_keys:
                old_keys = []
                for lig in catalyst.components['ligand']:
                    old_keys += list(lig.key_atoms)
            new_lig = self.load_lib_ligand(name)
            catalyst.map_ligand(new_lig, old_keys)
        # substitutions
        for atom, sub in change.items():
            if atom == 'map':
                continue
            # convert to relative numbering
            if 'map' in change:
                atom = '*.' + atom
            else:
                atom = str(int(atom)
                           + len(catalyst.atoms)
                           - sum([len(i.atoms)
                                  for i in catalyst.components['ligand']]))
            catalyst.substitute(sub, atom)
        return catalyst

    def _do_substrate_changes(self, template_file, sub_change):
        sub, change = sub_change
        catalyst = Catalyst(template_file)
        # substitutions
        for atom, sub in change.items():
            catalyst.substitute(sub, atom)
        return catalyst

    def generate_structures(self):
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
            template_path = os.path.join(*template[:2]).rstrip('/')
            if os.path.basename(template_path) in self.selectivity:
                selectivity = os.path.basename(template_path)
            for lig_change in self.ligand.items():
                if not lig_change[1]:
                    lig_change = None
                for sub_change in self.substrate.items():
                    self.catalyst_data += [CatalystMetaData(template_file,
                                                            self, selectivity,
                                                            lig_change,
                                                            sub_change)]
                # if no ligand changes, no need to add another entry
                if lig_change is None:
                    continue
                # we still want ligand changes without substrate changes
                self.catalyst_data += [CatalystMetaData(template_file,
                                                        self, selectivity,
                                                        lig_change, None)]
            # even if no changes requested, still want to optimize templates
            if len(self.catalyst_data) <= i:
                self.catalyst_data += [CatalystMetaData(template_file,
                                                        self, selectivity,
                                                        None, None)]

    def _get_nconfs(self, cat_data):
        all_subs = []
        if cat_data.ligand_change is not None:
            for name, change in cat_data.ligand_change[1].items():
                if name != 'map':
                    all_subs += [change]
        if cat_data.substrate_change is not None:
            all_subs += cat_data.substrate_change[1].values()
        nconfs = 1
        for sub in all_subs:
            sub = Substituent(sub)
            nconfs *= sub.conf_num
        return nconfs

    def _get_next_cf_dir_number(self, ts_dir):
        pass

    def make_directories(self, top_dir=None):
        """
        :top_dir:   the directory in which to put generated directories
        """
        if top_dir is None:
            top_dir = self.top_dir
        if os.path.isfile(top_dir):
            top_dir = os.path.dirname(top_dir)
        if not os.access(top_dir, os.W_OK):
            msg = "Directory does not exist or write permission denied:\n  {}"
            raise OSError(msg.format(top_dir))
        self.top_dir = top_dir

        if not self.catalyst_data:
            self.generate_structures()
        for cat_data in self.catalyst_data:
            # generate child directory name
            if cat_data.ligand_change is None:
                lig_str = "none"
            else:
                lig_str = cat_data.ligand_change[0]
            if cat_data.substrate_change is None:
                sub_str = "none"
            else:
                sub_str = cat_data.substrate_change[0]
            # child_dir is top/lig_spec/sub_spec...
            child_dir = os.path.join(top_dir, lig_str, sub_str)
            # .../selectivity...
            if cat_data.selectivity:
                child_dir = os.path.join(child_dir, cat_data.selectivity)
            # .../ts#...
            dir_up, tname = os.path.split(cat_data.template_file)
            dir_up = os.path.basename(dir_up)
            tname = os.path.splitext(tname)[0]
            if tname.lower().startswith('ts'):
                # template files are ts structures
                child_dir = os.path.join(child_dir, tname)
                try:
                    os.makedirs(child_dir, mode=0o755)
                except FileExistsError:
                    pass
            elif dir_up.lower().startswith('ts'):
                # template files are conformer structures within ts directory
                child_dir = os.path.join(child_dir, dir_up, tname)
                try:
                    os.makedirs(child_dir, mode=0o755)
                except FileExistsError:
                    pass


class CompOpts:
    """
    Class Attributes:
        by_step         {step_number: CompOpts()}
    Attributes:
        theory          Theory()
        temperature	temperature in K
        denfit	        true|false.  Controls whether 'denfit' option is invoked
        charge	        integer value of charge
        mult	        integer value of spin multiplicity
        solvent_model	pcm, smd, etc
        solvent	        dichloromethane, water, etc
        grid	        integration grid keyword
        emp_dispersion	empirical dispersion keywords. e.g. emp_dispersion=GD3
        additional_opts dictionary of extra keywords
    """
    by_step = {}  # key 0.0 corresponds to default

    @classmethod
    def write_com(cls, geometry, step, *args, **kwargs):
        if float(step) in cls.by_step:
            options = cls.by_step[step]
        else:
            options = cls.by_step[0.0]
        FileWriter.write_com(geometry, float(step), options, *args, **kwargs)

    @classmethod
    def cascade_defaults(cls):
        if 0.0 not in cls.by_step:
            return
        else:
            defaults = cls.by_step[0.0].__dict__
        for step, compopt in cls.by_step.items():
            if step == 0.0:
                continue
            for key, val in compopt.__dict__.items():
                if val is not None:
                    # don't overwrite set settings
                    continue
                compopt.__dict__[key] = defaults[key]

    def __init__(self, params=None):
        self.theory = Theory()
        self.temperature = None
        self.denfit = None
        self.charge = None
        self.mult = None
        self.solvent_model = None
        self.solvent = None
        self.grid = None
        self.emp_dispersion = None
        self.additional_opts = None
        if params is None:
            return

        def store_settings(name, info):
            if 'method' in name:
                self.theory.method = info
            elif 'basis' in name:
                self.theory.set_basis(info)
            elif 'ecp' in name:
                self.theory.set_ecp(info)
            elif 'temperature' == name:
                self.temperature = float(info)
            elif 'denfit' == name:
                try:
                    self.denfit = bool(int(info))
                except ValueError:
                    self.denfit = bool(info)
            elif 'charge' == name:
                self.charge = int(info)
            elif 'mult' == name:
                self.mult = int(info)
            elif 'solvent_model' == name:
                self.solvent_model = info
            elif 'solvent' == name:
                self.solvent = info
            elif 'grid' == name:
                self.grid = info
            elif 'emp_dispersion' == name:
                self.emp_dispersion = info
            elif 'additional_opts' == name:
                self.additional_opts = info

        for name, info in params.items():
            # get step key
            key = name.split('_')
            if len(key) == 1:
                key = 0.0  # default
            else:
                key = key[0]
                try:
                    key = [float(key)]
                except ValueError:
                    if key not in ['low', 'high']:
                        key = 0.0  # default
                    elif key == 'low':
                        key = 1.0
                    elif key == 'high':
                        key = 5.0
            # change to correct CompOpt
            if key not in CompOpts.by_step:
                CompOpts.by_step[key] = CompOpts()
            else:
                self = CompOpts.by_step[key]
            self.theory.step = key
            # store info in appropriate attribute
            store_settings(name, info)
        # cascade defaults to steps without settings
        self.cascade_defaults()


class ClusterOpts:
    """
    Attributes:
        n_procs     number of processors for steps2-5
        wall        walltime in hours for steps2-5
        short_procs number of cores for step1
        short_wall  walltime in hours for step1
    """

    def __init__(self, params=None):
        self.n_procs = None
        self.short_procs = None
        self.wall = None
        self.short_wall = None
        if params is None:
            return

        self.n_procs = params['n_procs']
        self.wall = params['wall']

        if 'short_procs' in params:
            self.short_procs = params['short_procs']
        if 'short_wall' in params:
            self.short_wall = params['short_wall']


class Theory:
    """
    Attributes:
        method	    method to use
        basis	    basis set to use
        ecp	    ecp to use
        gen_basis   path to gen basis files
    """
    nobasis = ['AM1', 'PM3', 'PM3MM', 'PM6', 'PDDG', 'PM7']

    def __init__(self):
        self.method = None
        self.basis = {}
        self.ecp = {}
        self.gen_basis = {}
        self._unique_basis = None

    def set_basis(self, spec):
        """
        Stores basis set information as a dictionary keyed by atoms

        :spec: list of basis specification strings from Aaron input file
            acceptable forms:
                "6-31G" will set basis for all atoms to 6-31G
                "tm SDD" will set basis for all transiton metals to SDD
                "C H O N def2tzvp" will set basis for C, H, O, and N atoms
        """
        if self.method is None:
            raise RuntimeError('Theory().method is None')
        if self.method.upper() in Theory.nobasis:
            return
        for a in spec:
            info = a.split()
            basis = info.pop()
            for i in info:
                i = i.strip()
                if i in ELEMENTS:
                    self.basis[i] = basis
                elif i.lower() == 'TM':
                    for tm in TMETAL:
                        self.basis[tm] = basis
            if len(info) == 0:
                for e in ELEMENTS:
                    if e not in self.basis:
                        self.basis[e] = basis
        return

    def set_gen_basis(self, gen):
        """
        Creates gen basis specifications
        Checks to ensure basis set files exist

        :gen: str specifying the gen basis file directory
        """
        if self.gen_basis:
            return

        gen = gen.rstrip('/') + '/'
        for element in self.basis:
            if self.basis[element].lower().startswith('gen'):
                fname = self.basis[element]
                fname = gen + fname.strip('gen/')
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

    def set_ecp(self, args):
        """
        Stores ecp information as a dictionary keyed by atoms

        :spec: list of basis specification strings from Aaron input file
            acceptable forms:
                "tm SDD" will set basis for all transiton metals to SDD
                "Ru Pt SDD" will set basis for Ru and Pt to SDD
        """
        if self.method is None:
            raise RuntimeError('Theory().method is None')
        if self.method.upper() in Theory.nobasis:
            return
        for a in args:
            info = a.strip().split()
            if len(info) < 2:
                msg = "ecp setting requires atom type: {}"
                msg = msg.format(' '.join(info))
                raise IOError(msg)
            ecp = info.pop()
            for i in info:
                if i in ELEMENTS:
                    self.ecp[i] = ecp
                elif i.lower() == 'tm':
                    for t in TMETAL.keys():
                        self.ecp[i] = ecp
                else:
                    msg = "ecp setting requires atom type: {}"
                    msg = msg.format(' '.join(info))
                    raise IOError(msg)
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

    def make_footer(self, geometry, step):
        if not isinstance(geometry, Geometry):
            msg = "argument provided must be Geometry type; "
            msg += "provided type {}".format(type(geometry))
            raise ValueError(msg)
        footer = '\n'

        # constrained optimization
        if int(step) == 2:
            constraints = geometry.get_constraints()
            for con in constraints:
                footer += 'B {} {} F\n'.format(con[0] + 1, con[1] + 1)
            if len(constraints) > 0:
                footer += '\n'

        if self.basis and (not self.unique_basis(geometry)
                           or self.ecp
                           or self.gen_basis):
            basis = {}
            for e in sorted(set(geometry.elements())):
                tmp = self.basis[e]
                if tmp not in basis:
                    basis[tmp] = []
                basis[tmp] += [e]
            for b in sorted(basis):
                footer += ("{} "*len(basis[b])).format(*basis[b])
                footer += "0\n{}\n".format(b)
                footer += "*"*4 + "\n"

        if self.gen_basis:
            for g in sorted(set(self.gen_basis.values())):
                footer += "@{}/N\n".format(g)

        if self.ecp:
            footer += '\n'
            ecp = {}
            for e in sorted(set(geometry.elements())):
                if e not in self.ecp:
                    continue
                tmp = self.ecp[e]
                if tmp not in ecp:
                    ecp[tmp] = []
                ecp[tmp] += [e]
            for e in sorted(ecp):
                footer += ("{} "*len(ecp[e])).format(*ecp[e])
                footer += "0\n{}\n".format(e)

        if footer.strip() == '':
            return '\n\n\n'

        return footer + '\n\n\n'

    def make_header(self, geometry, step, options, *args, **kwargs):
        """
        :geometry: Geometry()
        :options: CompOpts()
        :args: additional arguments for header
        :kwargs: additional keyword=value arguments for header
        """
        # determine type of job
        if step < 2:
            job_type = 'relax'
        elif step < 3:
            job_type = 'constrained_opt'
        elif step < 4:
            job_type = 'ts_opt'
        elif step < 5:
            job_type = 'frequencies'
        else:
            job_type = 'single_point'

        # get comment
        if 'comment' in kwargs:
            comment = kwargs['comment']
            del kwargs['comment']
        else:
            comment = 'step {}'.format(
                int(step) if int(step) == step else step)

        # checkfile
        has_check = False
        if step > 1:
            header = "%chk={}.chk\n".format(geometry.name)
            if os.access("{}.chk".format(geometry.name), os.R_OK):
                has_check = True
        else:
            header = ''

        # method and basis
        header += '#{}'.format(self.method)
        if self.ecp:
            header += '/genecp'
        elif self.unique_basis(geometry):
            header += '/{}'.format(self.unique_basis)
        elif self.basis:
            header += '/gen'

        if job_type == 'relax':
            header += ' opt nosym'
        elif job_type == 'constrained_opt':
            header += ' opt=(modredundant,maxcyc=1000)'
        elif job_type == 'ts_opt':
            header += ' opt=({}fc,ts,maxcyc=1000)'.format(
                'read' if has_check else 'calc')
        elif job_type == 'frequencies':
            header += ' freq=(hpmodes,noraman,temperature={})'.format(
                options.temperature)
        elif job_type == 'opt':
            header += ' opt=({}fc,maxcyc=1000)'.format(
                'read' if has_check else 'calc')

        # solvent
        if options.solvent_model.lower != 'gas':
            header += ' scrf=({},solvent={})'.format(options.solvent_model,
                                                     options.solvent)
        # other keywords
        if options.additional_opts:
            header += ' ' + options.additional_opts
        # alternate form for other keywords
        for arg in args:
            header += ' {}'.format(arg)
        for kw, arg in kwargs.items():
            if hasattr(arg, '__iter__') and not isinstance(arg, str):
                tmp = ''
                for a in arg:
                    tmp += '{},'.format(a)
                arg = tmp[:-1]
            header += ' {}=({})'.format(kw, arg)

        # comment, charge and multiplicity
        header += '\n\n{}\n\n{},{}\n'.format(comment,
                                             options.charge,
                                             options.mult)
        return header
