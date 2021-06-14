#! /usr/bin/env python3

from Aaron.job import get_steps
from AaronTools.config import Config

from exec_template_options import main as exec_options
from qadapter_options import main as qadapter_options


def main(config=None):
    if config is not None:
        config = Config(config)
    else:
        config = Config()
    opts_defined = {}
    opts_found = set([])
    for step in get_steps(config):
        step_config = config.for_step(step, parse_functions=False)
        opts_found.update(
            qadapter_options(step_config["HPC"]["queue_type"], verbose=False)
        )
        opts_found.update(
            exec_options(step_config["Job"]["exec_type"], verbose=False)
        )
        for name, section in step_config.items():
            opts_defined.setdefault(name, set([]))
            for option in section.keys():
                opts_defined[name].add(option)

    opts_undefined = opts_found - opts_defined["HPC"] - opts_defined["Job"]
    opts_optional = opts_undefined.intersection(set(["cmdline"]))
    opts_needed = opts_undefined.difference(opts_optional)

    print("All options found in queue adapter and executable template:")
    print("", *opts_found, sep="  ")
    if opts_needed:
        print("Necessary options undefined in configuration file(s):")
        print("", *opts_needed, sep="  ")
    if opts_optional:
        print("Optional options undefined in configuration file(s)")
        print("", *opts_optional, sep="  ")
    if not opts_undefined:
        print("No options left undefined in configuration file")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "For determining what options must be set in the configuration for a workflow"
    )
    parser.add_argument("config", nargs="?", default=None)

    args = parser.parse_args()
    main(args.config)
