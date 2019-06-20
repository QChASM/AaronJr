import argparse

from Aaron.options import ClusterOpts, Reaction, Theory
from AaronTools.const import HOME, QCHASM


class AaronInit:
    """
    Attributes:
    :jobname:       str - name of aaron input file
    :args:          command line arguments
    :cluster_opts:  ClusterOpts() - contains resource settings
    :comp_opts:     {step: CompOpts()} - contains computation settings
    :reaction:      Reaction() - contains
    """

    def __init__(self, infile, args=None):
        self.jobname = infile
        self.args = args

        self.params = self.read_aaron_input(infile)
        self.cluster_opts = ClusterOpts(self.params)
        self.theory = Theory(self.params)
        self.theory = Theory.by_step[0.0]
        if "top_dir" not in self.params:
            self.params["top_dir"] = infile
        self.reaction = Reaction(self.params)

        if "gen" in self.params:
            for theory in self.theory.by_step.values():
                theory.set_gen_basis(self.params["gen"])

    def read_aaron_input(self, infile):
        def parse(f, params, custom=None):
            profile_found = False
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                # skip to correct profile
                if custom is not None:
                    if not profile_found and "=" in line:
                        pass
                    elif not profile_found and line != custom:
                        continue
                    elif not profile_found and line == custom:
                        profile_found = True
                        continue
                    elif profile_found and "=" not in line:
                        break

                # ligand mapping and substitution
                if line.lower() == "&ligands":
                    line = f.readline().strip()
                    while line != "&" and line != "":
                        line = line.split(":")
                        name = line[0].strip()
                        info = line[1].strip().split()
                        if "ligand" not in params:
                            params["ligand"] = {}
                        params["ligand"][name] = info
                        line = f.readline().strip()
                    continue
                # substrate substitutions
                if line.lower() == "&substrates":
                    line = f.readline().strip()
                    while line != "&" and line != "":
                        line = line.split(":")
                        name = line[0].strip()
                        info = line[1].strip().split()
                        if "substrate" not in params:
                            params["substrate"] = {}
                        params["substrate"][name] = info
                        line = f.readline().strip()
                    continue

                # store key=value setting pairs
                line = line.split("=", 1)
                name = line[0].strip().lower()
                info = line[1].strip()
                if name in ["basis", "ecp"]:
                    if name not in params:
                        params[name] = []
                    params[name] += [info]
                elif name not in params:
                    params[name] = info

            return params

        params = {"ligand": {}, "substrate": {}}
        with open(infile) as f:
            parse(f, params)
        if "custom" not in params:
            params["custom"] = "Default"

        # fill missing parameters from user profile
        try:
            with open(HOME + ".aaronrc") as f:
                parse(f, params, params["custom"])
        except FileNotFoundError:
            pass

        # fill missing parameters from QCHASM defaults
        try:
            with open(QCHASM + "Aaron/.aaronrc") as f:
                parse(f, params, params["custom"])
        except FileNotFoundError:
            pass

        # organize ligand and substrate
        ligand = {}
        for lig, info in params["ligand"].items():
            if lig not in ligand:
                ligand[lig] = {}
            for i in info:
                i = i.split("=")
                if len(i) == 1:
                    ligand[lig] = None
                    continue
                if i[0].strip().lower().startswith("ligand"):
                    old_keys = i[0].strip().lstrip("ligand.")
                    ligand[lig]["map"] = (i[1].strip(), old_keys)
                else:
                    ligand[lig][i[0].strip()] = i[1].strip()
        params["ligand"] = ligand
        substrate = {}
        for sub, info in params["substrate"].items():
            if sub not in substrate:
                substrate[sub] = {}
            for i in info:
                i = i.split("=")
                substrate[sub][i[0].strip()] = i[1].strip()
        params["substrate"] = substrate
        return params


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
        help="The Aaron input file (file extension: .in)",
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
    print(type(args))
    init = AaronInit(args.input_file, args)
