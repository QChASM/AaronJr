import argparse
import os

from Aaron.options import Reaction, AaronTheory
from AaronTools.const import HOME, QCHASM

from configparser import ConfigParser


class AaronConfigParser(ConfigParser):
    """
    case of options is preserved
    """
    #make options case-sensitive
    optionxform = str

    def options(self, section, ignore_defaults=False, **kwargs):
        """
        get list of options
        ignore_defaults: bool - if True, section will not inherit defaults
        """
        if ignore_defaults:
            try:
                return list(self._sections[section].keys())
            except KeyError:
                raise NoSectionError(section)
        else:
            return super().options(section, **kwargs)

    def getlist(self, section, option, *args, delim=',', **kwargs):
        """returns a list of option values by splitting on the delimiter specified by delim"""
        raw = self.get(section, option, *args, **kwargs)
        out = [x.strip() for x in raw.split(delim) if len(x.strip()) > 0]
        return out


class AaronInit:
    """
    Attributes:
    :jobname:       str - name of aaron input file
    :args:          command line arguments
    :cluster_opts:  ClusterOpts() - contains resource settings
    :comp_opts:     {step: CompOpts()} - contains computation settings
    :reaction:      Reaction() - contains
    """

    def __init__(self, infile, args=None, quiet=False):
        self.jobname = infile
        self.args = args

        if not quiet:
            print("Reading input file...")
        # read the input file once to see if a custom default has been specified
        # a default section cannot be changed after instantiation 
        test_config = AaronConfigParser()
        read = test_config.read(infile)
        if not quiet:
            print("Successfully read files:", read)
        
        if test_config.has_section("options"):
            default_section = test_config.get("options", "default", fallback="default")
        else:
            default_section = "default"

        #default section names for different program-specific options
        gaussian_default = "Gaussian %s" % default_section
        orca_default = "ORCA %s" % default_section
        psi4_default = "Psi4 %s" % default_section

        #actually read in the input file and use the custom default section
        self.config = AaronConfigParser(default_section=default_section)
        read = self.config.read([infile, os.path.join(QCHASM, "Aaron", "aaron.ini"), os.path.join(HOME, "aaron.ini")])
        self.gaussian_config = AaronConfigParser(default_section=gaussian_default)
        self.gaussian_config.read([infile, os.path.join(QCHASM, "Aaron", "aaron.ini"), os.path.join(HOME, "aaron.ini")])
        self.orca_config = AaronConfigParser(default_section=orca_default)
        self.orca_config.read([infile, os.path.join(QCHASM, "Aaron", "aaron.ini"), os.path.join(HOME, "aaron.ini")])
        self.psi4_config = AaronConfigParser(default_section=psi4_default)
        self.psi4_config.read([infile, os.path.join(QCHASM, "Aaron", "aaron.ini"), os.path.join(HOME, "aaron.ini")])
        if not quiet:
            print("Attempting to read:", [infile, os.path.join(QCHASM, "aaron.ini"), os.path.join(HOME, "aaron.ini")])
            print("Successfully read files:", read)

        if not self.config.get("options", "top_dir", fallback=False):
            self.config.set("options", "top_dir", os.path.dirname(os.path.abspath(infile)))
        self.theory = AaronTheory(self.config, self.gaussian_config, self.orca_config, self.psi4_config)
        self.theory = AaronTheory.by_step[0.0]
        if not quiet:
            print("Setting up reaction...")
        self.reaction = Reaction(self.config)

        if not quiet:
            print("Starting workflow")


if __name__ == "__main__":
    """
    stores arguments from command line:
        input_file  str     fname.in
        --absthermo bool    use absolute thermo for Aaron output
        --sleep     int     sleep time between status checks (minutes)
        --record    bool    save failed log files
        --debug     bool    run all opts using low_method
        --nosub     bool    make job/input files but don't submit
        --short     bool    use short wall time for all computations
        --restart   bool    complete restart of workflow
    reads settings from aaronrc files and input file
    """
    args = argparse.ArgumentParser(
        description="AARON: An Automated Reaction Optimizer for New catalysts"
    )
    args.add_argument(
        "input_file",
        type=str,
        help="The Aaron input file (file extension: .ini)",
    )
    args.add_argument(
        "--record",
        "-r",
        action="store_true",
        help="Save failed .log files for further examination",
    )
    args.add_argument(
        "--sleep",
        "-s",
        type=int,
        help="Sleep interval (minutes) between status checks.",
    )
    args.add_argument(
        "--absthermo",
        "-a",
        action="store_true",
        help="Use absolute themochemistry for Aaron output",
    )
    args.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Runs all optimization using the low_method and"
        + " using the short wall time",
    )
    args.add_argument(
        "--nosub",
        "-n",
        action="store_true",
        help="Builds .com files for the current step without" + " submitting",
    )
    args.add_argument(
        "--short",
        action="store_true",
        help="Use short wall time for all computations",
    )
    args.add_argument(
        "--restart",
        action="store_true",
        help="Restart Aaron workflow. All failed and repeated"
        + " conformers will be recovered.",
    )

    args = args.parse_args()
    init = AaronInit(args.input_file, args)
