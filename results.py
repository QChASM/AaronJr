#!/usr/bin/env python
import io
import itertools as it
import json
import os
import pickle
import re
import subprocess
import sys
import tkinter as tk

import numpy as np
import pandas as pd
from AaronTools.comp_output import CompOutput
from AaronTools.config import Config
from AaronTools.const import PHYSICAL, UNIT
from AaronTools.utils import utils
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


def row2output(row):
    output = CompOutput()
    other = {}
    for col, val in row.iteritems():
        try:
            if np.isnan(val):
                continue
        except (TypeError, ValueError):
            pass
        if col in [
            "name",
            "change",
            "template",
            "selectivity",
            "conformer",
            "step",
        ]:
            continue
        if col.startswith("other"):
            key = col.split(".")[1]
            other[key] = val
            continue
        setattr(output, col, val)
    if other:
        setattr(output, "other", other)
    return output


def boltzmann_weight(energy, temperature):
    return np.exp(energy / (PHYSICAL.R * temperature))


class Results:
    # job_patt groups: full match, name, template, change
    job_patt = re.compile("((&?[\w-]+)(?:{([\w-]+)})?(?::([\w-]+))?)")

    def __init__(self, args, job_dict=None):
        self.config = Config(args.config, quiet=True)
        self.config.parse_functions()
        self.args = args
        self.thermo = self.args.thermo.lower()
        if self.thermo in ["rrho", "qrrho", "qharm"]:
            self.thermo = "free_energy"
        self.data = pd.DataFrame()
        try:
            self.data = pd.read_pickle(self.args.cache)
        except (FileNotFoundError, EOFError, OSError):
            if job_dict is None:
                raise OSError(
                    "There is something wrong with the results cache file. Please use the --reload option to fix this"
                )
            pass
        if self.args.reload or self.data.empty:
            self.load_jobs(job_dict)
            pd.to_pickle(self.data, self.args.cache)

        self.apply_corrections()
        if self.args.command == "results":
            self.print_results(self.args)
        if self.args.command == "plot":
            Plot(self, self.args)

    def load_jobs(self, job_dict):
        data = []
        for name, job_list in job_dict.items():
            for job in job_list:
                change, template = os.path.split(job.config["Job"]["name"])
                change, selectivity = os.path.split(change)
                if not change:
                    change = selectivity
                    selectivity = None
                conformer = job.conformer
                data_row = {
                    "name": name,
                    "change": change,
                    "template": template,
                    "selectivity": selectivity,
                    "conformer": conformer,
                }
                for step in reversed(job.step_list):
                    if "conformer" in job.config.for_step(step).get(
                        "Job", "type"
                    ):
                        continue
                    fw = job.set_fw(step=step)
                    if fw is None:
                        continue
                    output = job.get_output(load_geom=True)
                    if output is None:
                        continue
                    if "fw_id" not in data_row:
                        data_row["fw_id"] = int(fw.fw_id)
                        data_row["step"] = step
                    for attr in output.__dict__:
                        if attr == "other" and output.other:
                            for key, val in output.other.items():
                                key = "other.{}".format(key)
                                if key not in data_row:
                                    data_row[key] = val
                        if attr in ["opts", "conformers", "archive", "other"]:
                            continue
                        val = getattr(output, attr)
                        if not val:
                            continue
                        if attr not in data_row:
                            data_row[attr] = val
                if "fw_id" in data_row and data_row["fw_id"]:
                    data_row = pd.Series(data_row)
                    data.append(data_row)
        self.data = pd.DataFrame(data)
        for column in [
            "charge",
            "conformer",
            "fw_id",
            "multiplicity",
            "opt_steps",
        ]:
            try:
                self.data[column] = self.data[column].astype(
                    int, errors="ignore"
                )
            except KeyError:
                pass
        self.data.set_index("fw_id", inplace=True)

    def apply_corrections(self):
        """
        Applies the thermodynamic correction requested with args.thermo
        """
        for key, data in self.data.groupby(
            ["name", "change", "template", "selectivity", "conformer"]
        ):
            for index, row in data.iterrows():
                output = row2output(row)
                try:
                    corr = None
                    if self.thermo in ["energy", "enthalpy"]:
                        dE, dH, s = output.therm_corr(
                            temperature=self.args.temp
                        )
                        self.data.loc[index, "energy"] = (
                            output.energy + output.ZPVE
                        )
                        self.data.loc[index, "enthalpy"] = output.enthalpy + dH
                    elif self.args.thermo.lower() in ["free_energy", "rrho"]:
                        corr = output.calc_G_corr(
                            v0=0, temperature=self.args.temp, method="RRHO"
                        )
                    elif self.args.thermo.upper() in ["QRRHO"]:
                        corr = output.calc_G_corr(
                            v0=self.args.w0,
                            temperature=self.args.temp,
                            method="QRRHO",
                        )
                    elif self.args.thermo.upper() in ["QHARM"]:
                        corr = output.calc_G_corr(
                            v0=self.args.w0,
                            temperature=self.args.temp,
                            method="QHARM",
                        )
                    if corr:
                        self.data.loc[index, "free_energy"] = (
                            output.energy + corr
                        )
                except (TypeError, AttributeError):
                    pass

    def print_results(self, args):
        thermo_unit = "{} ({})".format(self.args.thermo, self.args.unit)
        cols = [
            "name",
            "change",
            "selectivity",
            "template",
            "conformer",
        ]
        data = self.data.copy()
        data.sort_values(by=cols, inplace=True)
        geoms = data["geometry"]
        if args.script:
            data = data[cols]
            for index, geom in geoms.iteritems():
                with subprocess.Popen(
                    args.script,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                ) as proc:
                    out, err = proc.communicate(
                        geom.write(style="xyz", outfile=False).encode()
                    )
                if err:
                    raise RuntimeError(err.decode())
                data.loc[index, args.script] = out.decode().strip()
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(data)
            return

        data = self.get_relative(data)

        for d in data:
            tmp_cols = cols.copy()
            if (
                "conformer" not in d.columns
                or len(d.groupby("conformer")) == 1
            ):
                tmp_cols.remove("conformer")
            if (
                "selectivity" not in d.columns
                or len(d.groupby("selectivity")) == 1
            ):
                tmp_cols.remove("selectivity")
            d = d[tmp_cols + [self.thermo]]
            if args.unit == "kcal/mol":
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            else:
                d[self.thermo] = d[self.thermo] / UNIT.HART_TO_KCAL
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(d)
            print()

        header = True
        for d in self.boltzmann_average(data, cols[:-1], self.thermo):
            if len(d.groupby("conformer")) == 1:
                continue
            d = d[cols[:-1] + [self.thermo]]
            if args.unit == "kcal/mol":
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            else:
                d[self.thermo] = d[self.thermo] / UNIT.HART_TO_KCAL
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            if not header:
                print()
                print("Boltzmann averaged over conformers for template")
                header = False
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(d)
            print()

        header = True
        for d in self.boltzmann_average(data, cols[:-2], self.thermo):
            if len(d.groupby("selectivity")) == 1:
                continue
            temperature = d["temperature"].tolist()[0]
            p = d[self.thermo].map(
                lambda x: np.exp(-x / (PHYSICAL.R * temperature))
            )
            p["selectivity"] = d["selectivity"]
            p = p.groupby("selectivity").sum()
            p = 100 * p / sum(p)
            d = d[cols[:-2] + [self.thermo]]
            if args.unit == "kcal/mol":
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            else:
                d[self.thermo] = d[self.thermo] / UNIT.HART_TO_KCAL
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            if header:
                print()
                print("Boltzmann averaged over selectivity")
                header = False
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(d)
            print(", ".join(["{:0.1f}% {}".format(p[i], i) for i in p.index]))
            print()

    def get_relative(self, data):
        data.dropna(subset=[self.thermo], inplace=True)
        data[self.thermo] = data[self.thermo] * UNIT.HART_TO_KCAL
        if "Results" in self.config:
            data = self.parse_functions(data, self.config, self.thermo)
        elif not self.args.absolute:
            relative = []
            for atoms, group in data.groupby(["name", "change"]):
                group[self.thermo] -= group[self.thermo].min()
                relative.append(group)
            data = relative
        else:
            data = [data]
        return data

    @staticmethod
    def boltzmann_average(data, cols, thermo):
        boltzmann_avg = []
        for d in data:
            new_d = []
            try:
                data_grouped = d.groupby(cols + ["temperature"])
            except KeyError:
                continue
            for key, group in data_grouped:
                try:
                    temperature = group["temperature"].tolist()[0]
                except KeyError:
                    continue
                avg = utils.boltzmann_average(
                    group[thermo].to_numpy(),
                    group[thermo].to_numpy(),
                    temperature,
                    absolute=False,
                )
                tmp = group.iloc[0]
                tmp[thermo] = avg
                new_d.append(tmp)
            if len(new_d):
                boltzmann_avg.append(pd.concat(new_d, axis=1).T)
        return boltzmann_avg

    @staticmethod
    def parse_functions(data, config, thermo):
        data = data.copy()[["name", "change", "template", thermo]]
        relative = None
        for key, val in config["Results"].items():
            if not key.startswith("&") and key != "relative":
                continue
            subst = []
            for match in Results.job_patt.findall(val):
                if match[0] == "-":
                    continue
                tmp = data.copy()
                if match[1]:
                    tmp = tmp[tmp["name"] == match[1]]
                if match[2]:
                    tmp = tmp[tmp["template"] == match[2]]
                if match[3].lower() == "none":
                    tmp = tmp[tmp["change"] == ""]
                elif match[3]:
                    tmp = tmp[tmp["change"] == match[3]]
                tmp.set_index("change", inplace=True)
                if tmp.shape[0] == 1:
                    subst.append(tmp.iloc[0])
                else:
                    subst.append(tmp)
                val = val.replace(
                    match[0],
                    "subst[{}][{}]".format(len(subst) - 1, "thermo"),
                    1,
                )
            val = eval(val, {"subst": subst, "thermo": thermo})
            val = pd.concat([val, subst[0]["template"]], axis=1)
            val["name"] = key.lstrip("&")
            val.reset_index(inplace=True)
            if key == "relative":
                relative = val
            else:
                data = data.append(val)
                data.drop_duplicates(
                    subset=["name", "template", "change"],
                    keep="last",
                    inplace=True,
                )
        if relative.shape[0] == 1:
            relative = relative.iloc[0]
        tmp = []
        for change, group in data.groupby("change"):
            rel = relative[relative["change"] == change][thermo].to_list()
            if len(rel) < 1:
                rel = relative[relative["change"] == ""][thermo].to_list()
            if len(rel) != 1:
                raise RuntimeError("[Results] relative setting is ambiguious")
            rel = rel[0]
            group[thermo] = group[thermo] - rel
            tmp.append(group)
        if tmp:
            data = tmp
        else:
            data = [data]
        return data


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
                    "Results",
                    "selectivity",
                    fallback=results.config.get(
                        "Reaction", "selectivity", fallback=""
                    ),
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
                set(
                    c if c else "None"
                    for c in self.results.data["change"].unique()
                )
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
        relative = self.results.get_relative(self.results.data)
        tmp = []
        for r in relative:
            if len(r[r["change"] == change]) > 0:
                tmp.append(r)
        relative = pd.concat(tmp)
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
                            val = relative[
                                relative["template"].str.contains(s)
                            ]
                            break
                    else:
                        val = relative
                    if key[0]:
                        val = val[val["name"] == key[0]]
                    if key[1]:
                        val = val[val["template"] == key[1]]
                    if key[2]:
                        val = val[val["change"] == key[2]]
                    if val.empty:
                        continue
                    val = val.pop(self.results.thermo).tolist()[0]
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
