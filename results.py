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
    # job_patt groups: full match, name, template, change
    job_patt = re.compile("((&?[\w-]+)(?:{([\w-]+)})?(?::([\w-]+))?)")

    def __init__(self, args, job_dict):
        self.config = Config(args.config, quiet=True)
        self.config.parse_functions()
        self.thermo = args.thermo
        if args.thermo in ["RRHO", "QRRHO", "QHARM"]:
            self.thermo = "free_energy"
        self.data = {}
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
        if self.config.get("Results", "group", fallback=""):
            self.parse_groups()
        self.parse_functions(args)
        if args.command == "results":
            self.print_results(args)
        if args.command == "plot":
            Plot(self, args)

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
                    job.fw_id = job.find_fw().fw_id
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
        """
        Applies the thermodynamic correction requested with args.thermo
        """
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
                if args.thermo.lower() not in ["energy", "enthalpy"]:
                    print(
                        "Warning: frequencies not found for {} {} {}".format(
                            *key
                        ),
                        file=sys.stderr,
                    )

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
            except (TypeError, NameError, SyntaxError):
                setattr(data, self.thermo, None)
            if data_key in self.data:
                job = self.data[data_key][1]
            else:
                job = None
            self.data[data_key] = data, job

        def split_function(val):
            """
            split function into words and determine what data keys to use
            we will replace jobspec-words with data values before eval

            Returns: val, jobspec
                val ([str]): the value split into words
                jobspec (dict): keys are indexes in val, values are keys for self.data
            """
            matches = [m[0] for m in job_patt.findall(val) if m[0] != "-"]
            hash_match = {}
            for m in sorted(matches, key=lambda x: len(x), reverse=True):
                hash_match[hash(m)] = m
                val.replace(m, " {} ".format(hash(m)))
            for h, m in hash_match.items():
                val.replace(str(h), m)
            val = val.split()
            # parse datakey and store in jobspec keyed by new_val index
            jobspec = {}
            for i, v in enumerate(val):
                match = job_patt.search(v)
                if match is None:
                    continue
                match, name, template, change = match.groups()
                for data_key in self.keys_where(
                    name=name, template=template, change=change
                ):
                    jobspec.setdefault(i, [])
                    jobspec[i] += [(i, data_key)]
            return val, jobspec

        if "Results" not in self.config:
            # nothing to parse
            return
        job_patt = Results.job_patt
        original_names = set([key for key in self.data.keys()])
        original_data = self.data.copy()
        done = set([])

        for key, val in self.config["Results"].items():
            if key in self.config["DEFAULT"]:
                continue
            if key in ["group", "hide"]:
                continue
            new_val, jobspec = split_function(val)
            # match up data_key possibilities for each variable in function
            # empty strings act like wildcards so we can use templates/changes
            # as filters relative to a base case (ie: original structure)
            for key_list in it.product(*jobspec.values()):
                template, changed, not_changed = set([]), set([]), set([])
                for i, data_key in key_list:
                    new_val[i] = data_key
                    if data_key[1]:
                        template.add(data_key[1])
                    if data_key[2]:
                        changed.add(data_key[2])
                    if not data_key[2]:
                        not_changed.add(i)
                # skip non-matching changes
                if len(changed) > 1:
                    continue
                # now we're just looking at a single change
                if changed:
                    changed = changed.pop()
                else:
                    changed = ""
                # if change is an option for the unchanged species,
                # skip b/c we want to match changes if we can
                skip = False
                for i in not_changed:
                    if key.startswith("&"):
                        # new data keys start with '&'
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
                    # these are new data keys, not associated with a template file
                    data_key = (
                        key,
                        "",
                        changed,
                    )
                else:
                    #  get correct template name
                    if "group" in self.config["Results"]:
                        tmp = self.keys_where(key, "", changed)
                        template &= {t[1] for t in tmp}
                    # if we have a template mis-match, skip
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

    def get_relative(self):
        """
        For each change made to structure, find the lowest energy datapoint
        Save data in self.relative keyed by change
        """
        relative = {}
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
                    if change not in relative or relative[change][0] > val:
                        relative[change] = val, key
            else:
                key = ("relative", "", change)
                try:
                    output, job = self.data[key]
                except KeyError:
                    continue
                relative[change] = getattr(output, self.thermo), key
        if len(relative) == 1:
            val = list(relative.values()).pop()
        else:
            val = None, None
        for change in changes:
            if change not in relative:
                relative[change] = val
        return relative

    def print_results(self, args):
        results = []
        relative = self.get_relative()
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
        hidden = self.parse_hidden()
        for name, template, change in sorted(
            self.data.keys(), key=lambda x: (x[2], x[0], x[1])
        ):
            if (name, template, change) in hidden:
                continue
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
            minimum, min_key = relative[change]
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
                    [",".join([str(cell) for cell in row]) for row in results]
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

    def parse_groups(self):
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
                        "{}:{}".format(
                            key[1], key[2] if key[2] else "original"
                        ),
                        change,
                    )
                    self.data[new_key] = self.data[key]
                    del self.data[key]

    def parse_hidden(self):
        hidden = set([])
        for h in self.config.get("Results", "hide", fallback="").split(","):
            if not h:
                continue
            match, name, template, change = Results.job_patt.search(h).groups()
            strict_change = False
            if change and change.lower() == "none":
                change = "original"
                strict_change = True
            if template and "group" in self.config["Results"]:
                keys = self.keys_where(
                    name=name,
                    change=change if change != "original" else "",
                    strict_change=strict_change,
                )
                keys = [k for k in keys if template == k[1].split(":")[0]]
            else:
                keys = self.keys_where(
                    name=name,
                    template=template,
                    change=change if change != "original" else "",
                    strict_change=strict_change,
                )
            hidden |= set(keys)
        return hidden


class Plot:
    """
    Attributes:
    :args: the command line arguments given to Aaron
    :results: the Results() object containg the data to plot
    :selectivity: list of selectivity categories (eg: [cis, trans])
    :spacing: x-dist between rxn steps
    :labelsize: font size for labels
    :titlesize: font size for title
    """

    dpi = get_curr_screen_geometry()

    def __init__(self, results, args):
        self.args = args
        self.results = results
        self.selectivity = [
            s.strip()
            for s in self.results.config.get(
                "Plot",
                "selectivity",
                fallback=results.config.get(
                    "Results", "selectivity", fallback=""
                ),
            ).split(",")
        ]
        self.path = [
            p.strip()
            for p in self.results.config["Plot"]["path"].split("\n")
            if p
        ]
        for i, p in enumerate(self.path):
            if "*" in p:
                self.path[i] = [p.replace("*", s) for s in self.selectivity]
            else:
                self.path[i] = [p]
        self.spacing = float(
            results.config.get("Plot", "spacing", fallback="2")
        )
        self.labelsize = float(
            results.config.get("Plot", "labelsize", fallback="16")
        )
        self.titlesize = float(
            results.config.get("Plot", "titlesize", fallback="24")
        )
        self.colors = [
            c.strip()
            for c in self.results.config.get(
                "Plot",
                "colors",
                fallback=self.results.config.get("Plot", "color", fallback=""),
            ).split(",")
            if c
        ]
        self.linewidth = float(
            self.results.config.get("Plot", "linewidth", fallback="1.5")
        )
        self.xlim = [
            float(x)
            for x in self.results.config.get(
                "Plot", "xlim", fallback=""
            ).split(",")
            if x
        ]
        self.ylim = [
            float(y)
            for y in self.results.config.get(
                "Plot", "ylim", fallback=""
            ).split(",")
            if y
        ]
        self.size = [
            float(s.strip())
            for s in self.results.config.get(
                "Plot", "size", fallback=""
            ).split(",")
            if s
        ]
        if not self.size:
            self.size = [3.25]
            self.size += [self.size[0] * 3 / 4]
        self.size = [s * 300 / Plot.dpi for s in self.size]

        for change, title in self.get_plot_list():
            if change.lower() == "none":
                change = ""
            fig = self.one_plot(change, title)
            if self.args.save:
                fig.savefig(
                    "{}.{}".format(
                        change if change else "original",
                        results.config.get("Plot", "format", fallback="tiff"),
                    ),
                    dpi=300,
                    transparent=True,
                )
            else:
                plt.show()

    def get_plot_list(self):
        if self.args.change is None:
            change = sorted(
                set(c if c else "None" for n, t, c in self.results.data.keys())
            )
        else:
            change = self.args.change
        if self.args.title is None:
            title = []
        else:
            title = self.args.title
        for i, c in enumerate(change):
            if i == len(title):
                title += [""]
        plot_list = [(c, t) for c, t in zip(change, title)]
        if not plot_list:
            for plot in self.results.config.get(
                "Plot", "plot", fallback=""
            ).split("\n"):
                if not plot:
                    continue
                plot_list += [p.strip() for p in plot.split(",")]
        return plot_list

    def one_plot(self, change, title):
        fig = plt.figure(figsize=self.size, dpi=Plot.dpi, tight_layout=True)
        ax = fig.add_subplot(frameon=False)

        # get paths between stationary points
        xpt = []
        ypt = []
        skip = None
        tags, verts = self.get_vertices(change)
        for i, (tag, vert) in enumerate(zip(tags, verts)):
            xpt += [v[0] for v in vert]
            ypt += [v[1] for v in vert]
            try:
                color = self.colors[i]
            except IndexError:
                color = None
            patch = patches.PathPatch(
                Path(vert),
                facecolor="none",
                edgecolor=color,
                linewidth=self.linewidth,
                joinstyle="bevel",
            )
            ax.add_patch(patch)

            # label stationary points
            skip = self.set_labels(tag, vert, ax, skip=skip)

        # plot formatting
        ax.set_title(title, fontsize=self.titlesize)
        ax.set_xticks([], minor=[])
        ax.tick_params(labelsize=self.labelsize)
        xpad, ypad = self.get_padding(xpt, ypt)
        if not self.xlim:
            xlim = [min(xpt), max(xpt)]
        else:
            xlim = self.xlim
        if not self.ylim:
            ylim = [min(ypt), max(ypt)]
        else:
            ylim = self.ylim
        try:
            ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
            ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
        except TypeError:
            ax.set_xlim(xlim[0] - xpad[0], xlim[1] + xpad[1])
            ax.set_ylim(ylim[0] - ypad[0], ylim[1] + ypad[1])
        return fig

    def get_vertices(self, change):
        label_patt = re.compile("\[([^]]*)\]")
        relative = self.results.get_relative()
        tags = []
        verts = []
        for s in self.selectivity:
            vals = []
            tag = []
            for i, p in enumerate(self.path):
                for q in p:
                    match = label_patt.search(q)
                    if match:
                        label = match.group(1)
                    else:
                        label = ""
                    match = Results.job_patt.search(q)
                    key = match.group(2), match.group(3), change
                    skip = False
                    for t in self.selectivity:
                        if t != s and t in key[1]:
                            skip = True
                    if skip:
                        continue
                    val = getattr(
                        self.results.data[key][0], self.results.thermo
                    )
                    if not val:
                        continue
                    minimum = relative[change][0]
                    if not minimum or self.args.absolute:
                        minimum = 0
                    if self.args.unit == "kcal/mol":
                        val = (val - minimum) * UNIT.HART_TO_KCAL
                    elif self.args.unit == "J/mol":
                        val = (val - minimum) * UNIT.HART_TO_JOULE
                    else:
                        val = val - minimum
                    vals += [
                        (i * float(self.spacing), val),
                        (i * float(self.spacing) + 1, val),
                    ]
                    tag += [label, label]
            tags += [tag]
            verts += [vals]
        return tags, verts

    def get_padding(self, xpt, ypt):
        x_range = max(xpt) - min(xpt)
        y_range = max(ypt) - min(ypt)
        xpad = (x_range) * 0.025
        ypad = (y_range) * 0.05
        tmp = [
            float(t) / 100
            for t in self.results.config.get(
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

    def set_labels(self, tag, vert, ax, skip=None):
        if not skip:
            skip = set([])
        for i, v in enumerate(vert):
            if v[0] % self.spacing != 0:
                continue
            if i > 0 and v[1] < vert[i - 1][1]:
                lpad = -self.labelsize - 2
            else:
                lpad = self.labelsize / 2
            loc = (v[0] + vert[i + 1][0]) / 2, v[1]
            if (loc, tag[i]) in skip:
                continue
            skip.add((loc, tag[i]))
            ax.annotate(
                tag[i],
                loc,
                textcoords="offset points",
                xytext=(0, lpad),
                ha="center",
                fontsize=self.labelsize,
            )
        return skip
