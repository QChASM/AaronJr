#!/usr/bin/env python3
import itertools as it

import jinja2
from Aaron.job import ENVIRONMENT, find_exec_template


def main(exec_type, verbose=True):
    exec_template = "{}.template".format(exec_type)
    exec_template = find_exec_template(exec_template)
    if verbose:
        print("Using %s" % exec_template)

    exec_template = ENVIRONMENT.loader.get_source(
        ENVIRONMENT, exec_type + ".template"
    )
    exec_template = ENVIRONMENT.parse(exec_template).body
    exec_template = it.chain.from_iterable(
        [x.nodes for x in exec_template if hasattr(x, "nodes")]
    )
    options_found = set([])
    for item in exec_template:
        if not isinstance(item, jinja2.nodes.Name):
            continue
        if item.name in [
            "work_dir",
            "job_name",
        ]:
            continue
        options_found.add(item.name)
    if verbose:
        print("[Job] options found:")
        for item in options_found:
            print(" ", item)
    return options_found


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="For checking executable template location and listing options found in the queue adapter template that should be set in the configuration file's [Job] section"
    )
    parser.add_argument(
        "exec_type",
        help="What you will set `exec_type` to in the [Job] section of your configuration file",
    )

    args = parser.parse_args()
    main(args.exec_type)
