#!/usr/bin/env python3
import os
from warnings import warn

from AaronTools.config import Config
from AaronTools.const import AARONLIB
from AaronTools.geometry import Geometry

from one_job import get_template, one_job


def get_input(path, kind):
    names = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            names += [item]
    print("Choose {} directory to add templates to:".format(kind))
    for i, r in enumerate(names):
        print("{: 3d}. {}".format(i + 1, r))
    choice = input("{} number (enter nothing to add new): ".format(kind))
    if not choice:
        name = input("Enter {} name: ".format(kind))
    else:
        name = names[int(choice) - 1]
    return name


def move_to_library(args, jobs):
    path = os.path.join(AARONLIB, "TS_geoms")
    reaction = get_input(path, "reaction")
    path = os.path.join(path, reaction)
    template = get_input(path, "template")
    path = os.path.join(path, template)

    for job in jobs:
        job.step = job.get_steps()[-1]
        head, tail = os.path.split(job.structure.name)
        name = [tail]
        while not tail.startswith("TS") and not tail.startswith("INT"):
            head, tail = os.path.split(head)
            name = [tail] + name
        name = "_".join(name)
        name = os.path.join(path, head, name)
        if os.access(name + ".xyz", os.R_OK):
            ans = input(
                "Template file already exists at {} Do you want to overwrite it? (yes/NO) ".format(
                    name + ".xyz"
                )
            )
            if ans.lower() not in ["y", "yes"]:
                print("Skipping {}".format(job.structure.name))
                continue
        output = job.validate()
        if output.finished:
            geometry = output.geometry
        else:
            geometry = None
        try:
            geometry.name = name
            print("Saving {}".format(name))
            geometry.write()
        except AttributeError:
            warn(
                "Not adding structure for {}, error found in output file".format(
                    job.structure.name
                )
            )


def main(args):
    config = Config(args.config, quiet=args.show)
    config.parse_functions()
    template = get_template(config)
    all_fws = set([])
    template = get_template(config)
    if isinstance(template, Geometry):
        all_fws = one_job(args, template, config, submit=False)
    else:
        for template, kind in get_template(config):
            all_fws = all_fws.union(
                one_job(
                    args,
                    template,
                    config,
                    job_type=kind,
                    submit=False,
                )
            )
    move_to_library(args, all_fws)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Saves structures from jobs defined in Aaron configuration file to the template library",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        metavar="{path/to/file}",
        default="config.ini",
        help="computational configuration file",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="turn off status messages"
    )
    args = parser.parse_args()
    args.show = None
    main(args)
