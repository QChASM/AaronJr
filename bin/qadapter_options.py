#!/usr/bin/env python3
import string

from Aaron.job import find_qadapter_template


def main(queue_type, verbose=True):
    qadapter_template = "{}_qadapter.template".format(queue_type)
    qadapter_template = find_qadapter_template(qadapter_template)
    if verbose:
        print("Using %s" % qadapter_template)

    with open(qadapter_template) as f:
        contents = f.read()
    template_keys = [
        k[1]
        for k in string.Formatter().parse(contents)
        if k[1] not in [None, "job_name", "launch_dir", "rocket_launch"]
    ]
    if verbose:
        print("[Job] options found:")
        for k in template_keys:
            print(" ", k)
    return template_keys


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="For checking queue adapter template location and listing options found in the queue adapter template that should be set in the configuration file's [Job] section"
    )
    parser.add_argument(
        "queue_type",
        help="What you will set `queue_type` to in the [HPC] section of your configuration file",
    )

    args = parser.parse_args()
    main(args.queue_type)
