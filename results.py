#!/usr/bin/env python
import itertools as it
import json
import os
import pickle
import re
import sys
import tkinter as tk

from AaronTools.comp_output import CompOutput
from AaronTools.config import Config
from AaronTools.const import UNIT
from AaronTools.utils.utils import progress_bar
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.path import Path


def get_curr_screen_geometry():
    """
    Workaround to get the size of the current screen in a multi-screen setup.

    Returns:
        geometry (str): The standard Tk geometry string.
            [width]x[height]+[left]+[top]
    """
    root = tk.Tk()
    dpi = root.winfo_fpixels("1i")
    root.destroy()
    return dpi


class Results:
    job_patt = re.compile("([^\s+/*-][^\s{}:]*)({([^\s{}:]+)})?(:(\S+))?")

    def __init__(self, args, job_dict):
        self.config = Config(args.config, quiet=True)
        self.config.parse_functions()
        self.thermo = args.thermo
        if args.thermo in ["RRHO", "QRRHO", "QHARM"]:
            self.thermo = "free_energy"
        self.data = {}
        self.relative = {}
        try:
            with open(args.cache, "rb") as f:
                self.data = pickle.load(f)
        except Exception:
            pass
        if args.reload or args.step or not self.data:
            self.load_jobs(job_dict, args=args)
            if not args.step:
                with open(args.cache, "wb") as f:
                    pickle.dump(self.data, f)

        self.apply_corrections(args)
        self.parse_functions(args)
        self.get_relative()
        if args.command == "results":
            self.print_results(args)
        if args.command == "plot":
            self.make_plots(args)

    def get_relative(self):
        names, changes = set([]), set([])
        for key in self.data.keys():
            names.add(key[0])
            changes.add(key[2])
        for change in changes:
            if "relative" not in names:
                for key in self.keys_where(change=change, strict_change=False):
                    try:
                        output, job = self.data[key]
                    except KeyError:
                        continue
                    val = getattr(output, self.thermo)
                    if val is None:
                        continue
                    if (
                        change not in self.relative
                        or self.relative[change][0] > val
                    ):
                        self.relative[change] = val, key
            else:
                key = ("relative", "", change)
                try:
                    output, job = self.data[key]
                except KeyError:
                    continue
                self.relative[change] = getattr(output, self.thermo), key
        if len(self.relative) == 1:
            val = list(self.relative.values()).pop()
        else:
            val = None, None
        for change in changes:
            if change not in self.relative:
                self.relative[change] = val

    def print_results(self, args):
        results = []
        names, templates, changes = set([]), set([]), set([])
        for n, t, c in self.data.keys():
            if (
                not c
                and args.change
                and "none" not in [a.lower() for a in args.change]
            ):
                continue
            if c and args.change and c not in args.change:
                continue
            names.add(n)
            templates.add(t)
            changes.add(c)
        line = "{:>5} {:<%d} {:<%d} {: 15.1f}" % (
            max([len(n) for n in names] + [4]),
            max([len(t) for t in templates] + [8]),
        )
        header = "{:>5} {:<%d} {:<%d} {:>15}" % (
            max([len(n) for n in names] + [4]),
            max([len(t) for t in templates] + [8]),
        )

        last_change = None
        for name, template, change in sorted(
            self.data.keys(), key=lambda x: (x[2], x[0], x[1])
        ):
            if (
                args.change
                and not change
                and "none" not in [a.lower() for a in args.change]
            ):
                continue
            if args.change and change and change not in args.change:
                continue
            if name == "relative" or name.startswith("&"):
                continue
            output, job = self.data[(name, template, change)]
            val = getattr(output, self.thermo)
            if val is None:
                continue
            minimum, min_key = self.relative[change]
            if minimum is None or args.absolute:
                minimum = 0
            if args.unit == "kcal/mol":
                val = (val - minimum) * UNIT.HART_TO_KCAL
            elif args.unit == "J/mol":
                val = (val - minimum) * UNIT.HART_TO_JOULE
            else:
                val = val - minimum
            results += [(change, name, template, val)]
            if not args.csv:
                if change != last_change:
                    print()
                    print(change if change else "Original", end=" ")
                    if minimum:
                        print("{:.1f}".format(minimum))
                    else:
                        print()
                    print(
                        header.format("wf_id", "name", "template", "kcal/mol")
                    )
                print(
                    line.format(
                        (job.root_fw_id if job and job.root_fw_id else ""),
                        name,
                        template,
                        val,
                    )
                )
                last_change = change
        with open("results.csv", "w") as f:
            f.write(
                "\n".join(
                    [",".join([str(l) for l in line]) for line in results]
                )
            )

    def keys_where(self, name="", template="", change="", strict_change=False):
        rv = []
        for key in self.data.keys():
            if name and key[0] and key[0] != name:
                continue
            if template and key[1] and key[1] != template:
                continue
            if change and key[2] != change:
                continue
            if strict_change and key[2] != change:
                continue
            rv += [key]
        return rv

    def parse_functions(self, args):
        def add_data(data_key, to_parse):
            data = CompOutput()
            for i, t in enumerate(to_parse):
                # use original data for calculation when updating
                if t in original_data:
                    to_parse[i] = str(
                        getattr(original_data[t][0], self.thermo)
                    )
                elif t in self.data:
                    to_parse[i] = str(getattr(self.data[t][0], self.thermo))
            to_parse = " ".join(to_parse)
            try:
                setattr(data, self.thermo, eval(to_parse, {}))
            except (TypeError, NameError):
                setattr(data, self.thermo, None)
            if data_key in self.data:
                job = self.data[data_key][1]
            else:
                job = None
            self.data[data_key] = data, job

        def parse_groups():
            # get changes that should be grouped together
            groups = {}
            val = [v for v in self.config["Results"]["group"].split("\n") if v]
            key = None
            for v in val:
                if "=" in v:
                    v = [i.strip() for i in v.split("=")]
                    key = v[0]
                    groups.setdefault(key, set([]))
                    if not v[1]:
                        continue
                    v = v[1]
                v = [i.strip() for i in v.split(",") if i.strip()]
                for tmp in v:
                    name, change = tmp.split(":")
                    if change.lower() == "none":
                        change = ""
                    groups[key] |= set(
                        self.keys_where(
                            name=name, change=change, strict_change=True
                        )
                    )
            # update data dict to use new change identification
            for change, key_list in groups.items():
                for select in key_list:
                    for key in self.keys_where(*select, strict_change=True):
                        new_key = (
                            key[0],
                            "{}/{}".format(
                                key[1], key[2] if key[2] else "original"
                            ),
                            change,
                        )
                        print(change, new_key, key)
                        self.data[new_key] = self.data[key]
                        del self.data[key]

        if "Results" not in self.config:
            return
        if "group" in self.config["Results"]:
            parse_groups()
        job_patt = Results.job_patt
        original_names = set([key for key in self.data.keys()])
        original_data = self.data.copy()
        done = set([])

        for key, val in self.config["Results"].items():
            if key in self.config["DEFAULT"]:
                continue
            if key in ["group"]:
                continue
            if not (
                key == "relative"
                or key.startswith("&")
                or any(key.startswith(n) for n in original_names)
            ):
                continue
            # determine what data keys to use
            changes = {}
            for i, v in enumerate(val.split()):
                match = job_patt.search(v)
                if match is None:
                    continue
                name, _, template, _, change = match.groups()
                for data_key in self.keys_where(
                    name=name, template=template, change=change
                ):
                    changes.setdefault(i, [])
                    changes[i] += [(i, data_key)]
            # match up data_key possibilities for each variable in function
            # empty strings act like wildcards so we can use templates/changes
            # as filters relative to a base case (ie: original structure)
            for change_list in it.product(*changes.values()):
                new_val = val.split()
                template, changed, not_changed = set([]), set([]), set([])
                for i, c in change_list:
                    new_val[i] = c
                    if c[1]:
                        template.add(c[1])
                    if c[2]:
                        changed.add(c[2])
                    if not c[2]:
                        not_changed.add(i)
                if len(changed) > 1:
                    continue
                if changed:
                    changed = changed.pop()
                else:
                    changed = ""
                # if change is an option for the unchanged species,
                # skip b/c we want to match changes if we can
                skip = False
                for i in not_changed:
                    if key.startswith("&"):
                        tmp = [
                            c[2]
                            for c in self.keys_where(
                                name=new_val[i][0],
                                template=new_val[i][1],
                            )
                        ]
                    else:
                        tmp = [
                            c[2]
                            for c in original_names
                            if new_val[i][0] == c[0] and new_val[i][1] == c[1]
                        ]
                    if changed and changed in tmp:
                        skip = True
                if skip:
                    continue
                if key.startswith("&") or key == "relative":
                    data_key = (
                        key,
                        "",
                        changed,
                    )
                else:
                    if "group" in self.config["Results"]:
                        tmp = self.keys_where(key, "", changed)
                        template &= {t[1] for t in tmp}
                    if len(template) > 1:
                        continue
                    data_key = (
                        key,
                        template.pop() if template else "",
                        changed,
                    )
                if data_key in done:
                    continue
                done.add(data_key)
                add_data(data_key, new_val)

    def load_jobs(self, job_dict, args=None):
        self.data = {}
        pb_i = 0
        pb_max = len(list(it.chain.from_iterable(job_dict.values())))
        for config_name, job_list in job_dict.items():
            for job in job_list:
                pb_i += 1
                progress_bar(pb_i, pb_max, name="Loading job output")
                step_list = job.get_steps()
                if not step_list:
                    step_list = [0]
                step_list.reverse()
                keep_step = None
                for step in step_list:
                    job.step = step
                    if args.step and job.step not in args.step:
                        continue
                    job.set_fw_id()
                    if not job.fw_id:
                        if job.step != 1:
                            print(job.get_query_spec(job._get_metadata()))
                        continue
                    change, template = os.path.split(job.config["Job"]["name"])
                    key = config_name, template, change
                    output = job.get_output()
                    if keep_step is None:
                        # keep step the first time we successfully find a job
                        keep_step = step
                    job.step = keep_step
                    if output is None:
                        continue
                    if not output.finished:
                        continue
                    # add missing output values from earlier steps
                    # allows us to use SP energies and lower-level frequencies
                    if key not in self.data:
                        self.data[key] = output, job
                    else:
                        for attr in output.__dict__:
                            if getattr(self.data[key][0], attr) is None:
                                setattr(
                                    self.data[key][0],
                                    attr,
                                    getattr(output, attr),
                                )

    def apply_corrections(self, args):
        for key, (output, job) in self.data.items():
            try:
                corr = None
                if args.thermo in ["energy", "enthalpy"]:
                    dE, dH, s = output.therm_corr(temperature=args.temp)
                    output.energy += output.ZPVE
                    output.enthalpy += dH
                elif args.thermo in ["free_energy", "RRHO"]:
                    corr = output.calc_G_corr(
                        v0=0, temperature=args.temp, method="RRHO"
                    )
                elif args.thermo in ["QRRHO"]:
                    corr = output.calc_G_corr(
                        v0=args.w0, temperature=args.temp, method="QRRHO"
                    )
                elif args.thermo in ["QHARM"]:
                    corr = output.calc_G_corr(
                        v0=args.w0, temperature=args.temp, method="QHARM"
                    )
                if corr:
                    output.free_energy = output.energy + corr
            except AttributeError:
                print(
                    "Warning: frequencies not found for {} {} {}".format(*key),
                    file=sys.stderr,
                )

    def make_plots(self, args):
        def get_vertices(path, change):
            label_patt = re.compile("\[([^]]*)\]")
            spacing = self.config.get("Plot", "spacing")
            tags, verts = [], []
            for s in selectivity:
                vals = []
                tag = []
                for i, p in enumerate(path):
                    for q in p:
                        match = label_patt.search(q)
                        if match:
                            label = match.group(1)
                        else:
                            label = ""
                        match = Results.job_patt.search(q)
                        key = match.group(1), match.group(3), change
                        skip = False
                        for t in selectivity:
                            if t != s and t in key[1]:
                                skip = True
                        if skip:
                            continue
                        val = getattr(self.data[key][0], self.thermo)
                        if not val:
                            continue
                        minimum = self.relative[change][0]
                        if not minimum or args.absolute:
                            minimum = 0
                        if args.unit == "kcal/mol":
                            val = (val - minimum) * UNIT.HART_TO_KCAL
                        elif args.unit == "J/mol":
                            val = (val - minimum) * UNIT.HART_TO_JOULE
                        else:
                            val = val - minimum
                        vals += [
                            (i * float(spacing), val),
                            (i * float(spacing) + 1, val),
                        ]
                        tag += [label, label]
                tags += [tag]
                verts += [vals]
            return tags, verts

        def get_padding(xpt, ypt):
            x_range = max(xpt) - min(xpt)
            y_range = max(ypt) - min(ypt)
            xpad = (x_range) * 0.025
            ypad = (y_range) * 0.05
            tmp = [
                float(t) / 100
                for t in self.config.get(
                    "Plot", "padding", fallback=""
                ).split()
            ]
            if len(tmp) == 1:
                xpad, ypad = tmp[0] * x_range, tmp[0] * y_range
            elif len(tmp) == 2:
                xpad, ypad = tmp[0] * x_range, tmp[1] * y_range
            elif len(tmp) == 4:
                tmp = [
                    a * b
                    for a, b in zip(tmp, [x_range, x_range, y_range, y_range])
                ]
                xpad, ypad = tmp[0:2], tmp[2:4]
            return xpad, ypad

        def get_labels(tag, vert, ax):
            spacing = float(self.config.get("Plot", "spacing"))
            labelsize = float(
                self.config.get("Plot", "labelsize", fallback="8")
            )
            done = set([])
            for tag, vert in zip(tags, verts):
                for i, v in enumerate(vert):
                    if v[0] % spacing == 0:
                        if i > 0 and v[1] < vert[i - 1][1]:
                            lpad = -labelsize - 2
                        else:
                            lpad = labelsize / 2
                        loc = (v[0] + vert[i + 1][0]) / 2, v[1]
                        if loc in done:
                            continue
                        done.add(loc)
                        ax.annotate(
                            tag[i],
                            loc,
                            textcoords="offset points",
                            xytext=(0, lpad),
                            ha="center",
                            fontsize=labelsize,
                        )

        def one_plot(tags, verts, title):
            colors = self.config.get(
                "Plot",
                "colors",
                fallback=self.config.get("Plot", "color", fallback=""),
            ).split()
            linewidth = float(
                self.config.get("Plot", "linewidth", fallback="1.5")
            )
            labelsize = float(self.config.get("Plot", "labelsize"))
            titlesize = float(self.config.get("Plot", "titlesize"))
            dpi = get_curr_screen_geometry()
            size = [
                float(s)
                for s in self.config.get("Plot", "size", fallback="").split()
            ]
            if not size:
                size = [3.25]
                size += [size[0] * 3 / 4]
            size = [s * 300 / dpi for s in size]

            fig = plt.figure(figsize=size, dpi=dpi, tight_layout=True)
            ax = fig.add_subplot(frameon=False)
            ax.set_title(title, fontsize=titlesize)
            ax.set_xticks([], minor=[])
            ax.tick_params(labelsize=labelsize)
            xpt = []
            ypt = []
            # get paths between stationary points
            for i, vert in enumerate(verts):
                xpt += [v[0] for v in vert]
                ypt += [v[1] for v in vert]
                if colors:
                    color = colors[i]
                else:
                    color = None
                patch = patches.PathPatch(
                    Path(vert),
                    facecolor="none",
                    edgecolor=color,
                    linewidth=linewidth,
                    joinstyle="bevel",
                )
                ax.add_patch(patch)
            # label stationary points
            get_labels(tags, verts, ax)
            # plot formatting
            xpad, ypad = get_padding(xpt, ypt)
            xlim = []
            ylim = []
            if self.config.get("Plot", "xlim", fallback=""):
                xlim = [
                    float(x) for x in self.config.get("Plot", "xlim").split()
                ]
            else:
                xlim = [min(xpt), max(xpt)]
            if self.config.get("Plot", "ylim", fallback=""):
                ylim = [
                    float(y) for y in self.config.get("Plot", "ylim").split()
                ]
            else:
                ylim = [min(ypt), max(ypt)]
            try:
                ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
                ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
            except TypeError:
                ax.set_xlim(xlim[0] - xpad[0], xlim[1] + xpad[1])
                ax.set_ylim(ylim[0] - ypad[0], ylim[1] + ypad[1])
            return fig

        spacing = self.config.get("Plot", "spacing", fallback="2")
        self.config.set("Plot", "spacing", spacing)
        labelsize = self.config.get("Plot", "labelsize", fallback="16")
        self.config.set("Plot", "labelsize", labelsize)
        titlesize = self.config.get("Plot", "titlesize", fallback="24")
        self.config.set("Plot", "titlesize", titlesize)
        if args.change is None:
            args.change = sorted(
                set([c if c else "None" for n, t, c in self.data.keys()])
            )
        if args.title is None:
            args.title = []
        for i, c in enumerate(args.change):
            if i == len(args.title):
                args.title += [""]
        plot_list = [(c, t) for c, t in zip(args.change, args.title)]
        if not plot_list:
            plot_list = [
                p.split(",")
                for p in self.config.get("Plot", "plot", fallback="").split(
                    "\n"
                )
                if p
            ]
        for plot in plot_list:
            change, title = [p.strip() for p in plot]
            if change.lower() == "none":
                change = ""
            path = [p for p in self.config["Plot"]["path"].split("\n") if p]
            selectivity = self.config.get(
                "Plot",
                "selectivity",
                fallback=self.config.get(
                    "Results", "selectivity", fallback=""
                ),
            ).split()
            for i, p in enumerate(path):
                if "*" in p:
                    path[i] = [p.replace("*", s) for s in selectivity]
                else:
                    path[i] = [p]
            tags, verts = get_vertices(path, change)
            fig = one_plot(tags, verts, title)
            if args.save:
                fig.savefig(
                    "{}.{}".format(
                        change if change else "original",
                        self.config.get("Plot", "format", fallback="tiff"),
                    ),
                    dpi=300,
                    transparent=True,
                )
            else:
                plt.show()
