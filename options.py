import re
import os

from AaronTools.const import ELEMENTS, TMETAL
from AaronTools.geometry import Geometry


class Reaction:
    """
    Attributes:
        reaction_type   templates stored in directories
        template            in Aaron library named as:
        selectivity         TS_geoms/rxn_type/template/selectivity
        ligand          ligand mapping/substitutions
        substrate       substrate substitutions
        con_thresh      connectivity threshold
    """

    def __init__(self, params=None):
        self.reaction_type = ''
        self.template = ''
        self.selectivity = []
        self.ligand = {}
        self.substrate = {}
        self.con_thresh = None
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
    by_step = {}

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

    def make_footer(self, geometry):
        if not isinstance(geometry, Geometry):
            msg = "argument provided must be Geometry type; "
            msg += "provided type {}".format(type(geometry))
            raise ValueError(msg)

        footer = '\n'
        if self.basis:
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
            return '\n'

        return footer + '\n'

    def make_header(self, geometry, options, *args, **kwargs):
        """
        :geometry: Geometry()
        :options: CompOpts()
        :args: additional arguments for header
        :kwargs: additional keyword=value arguments for header
        """
        if 'job_type' in kwargs:
            job_type = kwargs['job_type']
            del kwargs['job_type']
        if 'comment' in kwargs:
            comment = kwargs['comment']
            del kwargs['comment']
        else:
            comment = geometry.comment

        header = '#{}/'.format(self.method)
        if self.ecp:
            header += 'genecp'

        if job_type == 'min':
            header += ' opt'

        if options.solvent_model.lower != 'gas':
            header += ' scrf({},solvent={})'.format(options.solvent_model,
                                                    options.solvent)
        if options.additional_opts:
            header += ' ' + options.additional_opts

        for arg in args:
            header += ' {}'.format(arg)
        for kw, arg in kwargs.items():
            if hasattr(arg, '__iter__') and not isinstance(arg, str):
                tmp = ''
                for a in arg:
                    tmp += '{},'.format(a)
                arg = tmp[:-1]
            header += ' {}=({})'.format(kw, arg)

        header += '\n\n{}\n\n{},{}\n'.format(comment,
                                             options.charge,
                                             options.mult)
        return header
