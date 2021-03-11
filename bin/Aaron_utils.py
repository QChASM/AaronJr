#!/usr/bin/env python3
import os

from Aaron.job import LAUNCHPAD, Job
from AaronTools.config import Config
from AaronTools.geometry import Geometry



def main(args):
    config = Config(args.config)
    config.parse_functions()
    all_fws = set([])
    workflows = set([])
    try:
        template = config.get_template()
    except FileNotFoundError as e:
        raise e
    if isinstance(template, Geometry):
        all_fws = get_fws(args, template, config)
    else:
        for template, kind in config.get_template():
            all_fws = all_fws.union(
                get_fws(
                    args,
                    template,
                    config,
                    job_type=kind,
                )
            )
    for fw in all_fws:
        wf = LAUNCHPAD.get_wf_by_fw_id(fw.fw_id)
        workflows.add(wf)

    if args.command == "resources":
        resources(workflows)
    if args.command == "results":
        results(workflows)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Utilities for analyzing Aaron run. For help with a specific command, use 'Aaron_utils.py <command> --help'"
    )
    parser.add_argument(
        "config",
        default="config.ini",
        nargs="?",
        help="The configuration file used for the Aaron run",
    )
    command = parser.add_subparsers(title="commands", dest="command")
    # 'results' specific options
    results_parser = command.add_parser(
        "results", help="Show computational results summary"
    )
    results_parser.add_argument(
        "--thermo",
        default="free-energy",
        help="Which thermodynamic quantity to use (default is 'free-energy')",
        choices=["energy", "enthalpy", "free-energy"],
    )
    # 'resources' specific options
    resources_parser = command.add_parser(
        "resources", help="Show resource usage summary"
    )
    resources_parser.add_argument(
        "--include-failed", action="store_true", help="Include failed jobs"
    )
    # these options available for all COMMANDS
    for command_parser in command._name_parser_map.values():
        command_parser.add_argument(
            "-csv", "--csv", action="store_true", help="Use csv format"
        )
        command_parser.add_argument(
            "-of",
            "--outfile",
            nargs=1,
            help="Save to file instead of printing to STDOUT",
        )

    args = parser.parse_args()
    print(args)
    main(args)
