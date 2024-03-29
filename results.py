#!/usr/bin/env python
import os
import re
import subprocess

import matplotlib
import numpy as np
import pandas as pd
from AaronTools import addlogger
from AaronTools.comp_output import CompOutput
from AaronTools.config import Config
from AaronTools.const import PHYSICAL, UNIT
from AaronTools.utils import utils
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.path import Path


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


@addlogger
class Results:
    # job_patt groups: full match, name, template, change
    job_patt = re.compile("((&?[\w\.-]+)(?:{([\w\.-]+)})?(?::([\w\.-]+))?)")
    LOG = None
    LEVEL = "info"

    def __init__(self, args, job_dict=None):
        self.config = Config(args.config, quiet=True)
        self.config.parse_functions()
        self._calc_attr = []

        self.args = args
        self.thermo = self.args.thermo.lower()
        if self.thermo in ["rrho", "qrrho", "qharm"]:
            self.thermo = "free_energy"
        self.thermo_unit = "{} ({})".format(self.args.thermo, self.args.unit)

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

        if not self.args.temp and self.config.get(
            "Results", "temperature", fallback=""
        ):
            self.args.temp = self.config.getfloat("Results", "temperature")
        if self.args.temp:
            self.data["temperature"] = self.args.temp

        if self.thermo not in self.data:
            self.LOG.error(
                "`%s` not in data. Please adjust your `--thermo` selection, ensure necessary computations were run (e.g. frequency computation for free_energy), or ensure the correct step numbers are loaded (if using the `load` keyword in [Results] section). Then use `AaronJr results --reload` to update the data cache.",
                self.thermo,
            )
            exit(1)

        self._parse_calc_attr(job_dict)
        self.apply_corrections()
        if self.args.command == "results":
            self.print_results(self.args)
        if self.args.command == "plot":
            Plot(self, self.args)

    def _parse_calc_attr(self, job_dict):
        calc = self.config.get("Results", "calc", fallback="")
        if not calc:
            return
        calc = [c.strip() for c in calc.split("\n") if c.strip()]
        for i, c in enumerate(calc):
            calc[i] = [
                x.strip() for x in re.split("[:=]", c.strip(), maxsplit=1)
            ]
        calc = dict(calc)
        # groups: full_match, step, attr/method[, arguments]
        func_patt = re.compile("((\d+\.?\d*)\.(\w+(?:\((.*?)\))?))")

        for name, job_list in job_dict.items():
            for job in job_list:
                step_list = self._get_job_step_list(job)
                fw_id = job
                # find index for self.data row
                for step in step_list:
                    if "conformer" in job.config.for_step(step).get(
                        "Job", "type"
                    ):
                        continue
                    fw = job.set_fw(step=step)
                    if fw is None:
                        continue
                    fw_id = int(fw.fw_id)
                    break
                # parse function and eval
                for attr, func in calc.items():
                    self._calc_attr.append(attr)
                    parsed_func = func
                    step_out = {}
                    val = np.nan
                    for match in func_patt.findall(func):
                        step = match[1]
                        old = match[0]
                        new = re.sub(
                            "^{}\.".format(step),
                            "step_out['{}'].".format(step),
                            old,
                        )
                        parsed_func = parsed_func.replace(old, new, 1)
                        fw = job.set_fw(step=step)
                        if fw is None:
                            break
                        output = job.get_output()
                        if output is None:
                            break
                        step_out[step] = output
                    else:
                        val = eval(parsed_func, {"step_out": step_out})
                    self.data.loc[fw_id, attr] = val

    def _get_job_step_list(self, job):
        step_list = self.config.get(
            "Results",
            "load",
            fallback=job.config.get("Results", "load", fallback=""),
        )
        if step_list:
            step_list = [float(s.strip()) for s in step_list if s.strip()]
        else:
            step_list = reversed(job.step_list)
        return step_list

    def load_jobs(self, job_dict):
        data = []
        for name, job_list in job_dict.items():
            for job in job_list:
                step_list = self._get_job_step_list(job)
                change, template = os.path.split(job.jobname)
                if job.conformer:
                    template = template.replace(
                        "_{}".format(job.conformer), "", 1
                    )
                change, selectivity = os.path.split(change)
                if not change:
                    change = selectivity
                    selectivity = None
                if selectivity is None:
                    for s in self.config.get(
                        "Results", "selectivity", fallback=""
                    ).split(","):
                        s = s.strip()
                        if s and s in template:
                            selectivity = s
                            break
                conformer = job.conformer
                data_row = {
                    "name": name,
                    "change": change,
                    "template": template,
                    "selectivity": selectivity,
                    "conformer": conformer,
                }
                for step in step_list:
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
        try:
            self.data.set_index("fw_id", inplace=True)
        except KeyError:
            self.LOG.error(
                "No fireworks found for these jobs. Run `AaronJr update` and try again"
            )
            exit(1)

    def apply_corrections(self):
        """
        Applies the thermodynamic correction requested with args.thermo
        """
        if self.thermo in self._calc_attr:
            return
        for index, row in self.data.iterrows():
            output = row2output(row)
            try:
                corr = None
                if self.thermo in ["energy", "enthalpy"]:
                    dE, dH, s = output.therm_corr(temperature=self.args.temp)
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
                    self.data.loc[index, "free_energy"] = output.energy + corr
            except (TypeError, AttributeError):
                pass

    def print_results(self, args):
        thermo_unit = self.thermo_unit
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
            if len(data["selectivity"].unique()):
                del data["selectivity"]
            if len(data["conformer"].unique()):
                del data["conformer"]
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(data)
            return

        data = self.get_relative(data)

        for d in data:
            if args.change:
                args_change = set(args.change)
                if "None" in args_change:
                    args_change.discard("None")
                    args_change.add("")
                if "none" in args.change:
                    args_change.discard("none")
                    args_change.add("")
                if set(d["change"].unique()) - args_change:
                    continue
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
                d[self.thermo] = d[self.thermo] * UNIT.HART_TO_KCAL
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            else:
                d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(d)
            print()

        # skip boltzmann summary if [Results] boltzmann = False
        # or if --absolute flag used
        if self.args.absolute or not self.config.getboolean(
            "Results", "boltzmann", fallback=True
        ):
            return
        header = True
        for d in self.boltzmann_average(data, cols[:-1], self.thermo):
            d = d[cols[:-1] + [self.thermo]]
            # if args.unit == "kcal/mol":
            #     d[self.thermo] = d[self.thermo] * UNIT.HART_TO_KCAL
            #     d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            # else:
            #     d.rename(columns={self.thermo: thermo_unit}, inplace=True)
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
            temperature = d["temperature"].tolist()[0]
            p = d[self.thermo].map(
                lambda x: np.exp(-x / (PHYSICAL.R * temperature))
            )
            p["selectivity"] = d["selectivity"]
            p = p.groupby("selectivity").sum()
            p = 100 * p / sum(p)
            d = d[cols[:-2] + [self.thermo]]
            # if args.unit == "kcal/mol":
            #     d[self.thermo] = d[self.thermo] * UNIT.HART_TO_KCAL
            #     d.rename(columns={self.thermo: thermo_unit}, inplace=True)
            # else:
            #     d.rename(columns={self.thermo: thermo_unit}, inplace=True)
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

    def get_relative(self, data, change=None):
        data = data.dropna(subset=[self.thermo])
        if "Results" in self.config:
            data = self.parse_functions(
                data, self.config, self.thermo, absolute=self.args.absolute
            )
        elif not self.args.absolute:
            relative = []
            for atoms, group in data.groupby(["name", "change"]):
                group[self.thermo] -= group[self.thermo].min()
                relative.append(group)
            data = relative
        else:
            data = [data]
        if change is not None:
            for d in data:
                d = d[d["change"] == change]
                if d.empty:
                    continue
                return d
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
    def parse_functions(data, config, thermo, absolute=False):
        data = data.copy()
        relative = None
        drop = None
        for key, val in config["Results"].items():
            if key == "drop":
                drop = val
            if not key.startswith("&") and key.lower() != "relative":
                continue
            subst = []
            orig_val = val
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
            if not isinstance(val, pd.Series):
                val = pd.Series(
                    {
                        thermo: val,
                        "name": key.lstrip("&"),
                        "change": "",
                        "template": "",
                    },
                    name=-1,
                )
            else:
                val = pd.DataFrame(val)
                val["name"] = key.lstrip("&")
                if len(subst[0]["template"]) == len(val):
                    val["template"] = subst[0]["template"]
                elif len(subst[1]["template"]) == len(val):
                    val["template"] = subst[1]["template"]
                else:
                    # this shouldn't happen, but if it does...
                    raise NotImplementedError(
                        "Size mismatch in {}".format(orig_val)
                    )
                val.reset_index(inplace=True)
                val.index = pd.Index([-1] * len(val.index))
            val.dropna(inplace=True)
            if key == "relative":
                relative = val
            else:
                data = data.append(val)
                data.drop_duplicates(
                    subset=["name", "template", "change"],
                    keep="last",
                    inplace=True,
                )

        if drop:
            for _, name, template, change in Results.job_patt.findall(drop):
                selection = data["name"] != name
                if template:
                    selection = selection | data["template"] != template
                if change and change.lower() != "none":
                    selection = selection | data["change"] != change
                elif change.lower() == "none":
                    selection = selection | data["change"] != ""
                data = data[selection]
        if not absolute and relative is not None:
            if relative.shape[0] == 1:
                relative = relative.iloc[0]
            tmp = []
            for change, group in data.groupby("change"):
                rel = relative[relative["change"] == change]
                rel = rel[thermo].to_list()
                if len(rel) < 1:
                    rel = relative[relative["change"] == ""][thermo].to_list()
                if len(rel) != 1:
                    raise RuntimeError(
                        "[Results] relative setting is ambiguious"
                    )
                rel = rel[0]
                group[thermo] = group[thermo] - rel
                tmp.append(group)
            if tmp:
                data = tmp
            else:
                data = [data]
        elif not absolute:
            relative = []
            for atoms, group in data.groupby(["name", "change"]):
                group[thermo] -= group[thermo].min()
                relative.append(group)
            data = relative
        else:
            tmp = []
            for _, group in data.groupby(["change"]):
                tmp += [group]
            data = tmp
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

    def __init__(self, results, args):
        self.args = args
        self.results = results

        # AaronJr settings
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
        self.path = None
        self.smooth = None
        if "path" in self.results.config["Plot"]:
            self.path = [
                p.strip()
                for p in self.results.config["Plot"]["path"].split("\n")
                if p
            ]
            for i, p in enumerate(self.path):
                if "*" in p:
                    self.path[i] = [
                        p.replace("*", s) for s in self.selectivity
                    ]
                else:
                    self.path[i] = [p]
        elif "smooth" in self.results.config["Plot"]:
            self.smooth = self.results.config["Plot"]["smooth"]
        else:
            raise Exception(
                "Plot requires either `path` or `smooth` option to be defined"
            )
        self.spacing = float(
            results.config.get("Plot", "spacing", fallback="2")
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

        # matplotlib settings
        self.figure_kwargs = self._get_figure_kwargs()
        self.path_kwargs = self._get_path_kwargs()
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

        # make and save/show plots
        for change, title in self.get_plot_list():
            if change.lower() == "none":
                change = ""
            if self.path is not None:
                fig = self.rxn_energy_diagram(change, title)
            if self.smooth is not None:
                fig = self.smooth_plot(change, title)
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

    def _get_size(self):
        size = [
            float(s.strip())
            for s in self.results.config.get(
                "Plot", "size", fallback=""
            ).split(",")
            if s
        ]
        if not size:
            size = [3.25]
            size += [size[0] * 3 / 4]
        return size

    def _get_figure_kwargs(self):
        rv = {}
        rv["figsize"] = self._get_size()
        rv["dpi"] = float(
            self.results.config.get("Plot", "dpi", fallback="300")
        )
        # str or None
        rv["facecolor"] = self.results.config.get(
            "Plot", "facecolor", fallback=None
        )
        rv["edgecolor"] = self.results.config.get(
            "Plot", "edgecolor", fallback=None
        )
        rv["linewidth"] = float(
            self.results.config.get("Plot", "linewidth", fallback="1.5")
        )
        rv["frameon"] = self.results.config.getboolean(
            "Plot", "frameon", fallback=None
        )
        rv["subplotpars"] = self._get_subplot_params(
            self.results.config.get("Plot", "subplotpars", fallback=None)
        )
        for kw in ["tight_layout", "constrained_layout"]:
            try:
                rv[kw] = self.results.config.getboolean(
                    "Plot", kw, fallback=True
                )
            except ValueError:
                for k, v in [
                    t.split("=")
                    for t in self.results.config.get(
                        "Plot", kw, fallback=""
                    ).split(",")
                ]:
                    rv[kw][k.strip()] = float(v.strip())

        return rv

    def _get_subplot_params(self, val=None):
        """
        val = None or "left=float, right=float, ..."
        see matplotlib.figure.SubplotParams doc for details
        """
        if val is None:
            return None
        val = [v.strip() for v in val.split(",")]
        kwargs = {}
        for v in val:
            k, v = v.split("=")
            k = k.strip()
            v = v.strip()
            kwargs[k] = float(k)
        return matplotlib.figure.SubplotParams(**kwargs)

    def _get_path_kwargs(self):
        rv = {}
        # scalar or None
        for kw in ["alpha"]:
            rv[kw] = self.results.config.getfloat("Plot", kw, fallback=None)
        # str
        for kw in [
            "facecolor",
            "fc",
            "edgecolor",
            "ec",
            "hatch",
            "linestyle",
            "ls",
        ]:
            if kw in self.results.config["Plot"]:
                rv[kw] = self.results.config["Plot"][kw]
        rv["linewidth"] = self.results.config.getfloat(
            "Plot", "linewidth", fallback="1.5"
        )
        rv["joinstyle"] = self.results.config.get("Plot", "joinstyle", "bevel")
        return rv

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
            title = [
                t.strip()
                for t in self.results.config.get(
                    "Plot", "title", fallback=""
                ).split(",")
            ]
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

    def smooth_plot(self, change, title):
        if not title:
            title = change
        job_patt = re.compile(
            "((&?[\w\.\*-]+)(?:{([\w\.\*-]+)})?(?::([\w\.\*-]+))?)"
        )

        fig = plt.figure(**self.figure_kwargs)
        ax = fig.add_subplot(frameon=False)

        match = job_patt.search(self.smooth)
        groups = list(match.groups())
        x_val = None
        for i, tmp in enumerate(groups):
            if tmp is None:
                continue
            groups[i] = tmp.replace(".", "\.").replace("*", "(.*)")
            if "*" in tmp:
                if i == 1:
                    x_val = re.compile("^" + groups[i] + "$"), "name"
                else:
                    x_val = re.compile("^" + groups[i] + "$"), "template"
        _, name, template, _ = groups
        data = self.results.get_relative(self.results.data, change=change)
        data = data[data["name"].str.match(name)]
        data = data[data["template"].str.match(template)]
        data["x"] = data[x_val[1]].map(lambda x: x_val[0].match(x).group(1))
        try:
            data["x"] = data["x"].map(lambda x: float(x))
        except ValueError:
            ax.tick_params(axis="x", labelrotation=60)
            pass
        xpt = data["x"]
        ypt = data[self.results.thermo]
        plt.scatter(xpt, ypt)
        plt.plot(xpt, ypt)

        # plot formatting
        ax.set_title(title, fontsize=self.titlesize)
        ax.tick_params(labelsize=self.labelsize)
        if "xlabel" in self.results.config["Plot"]:
            ax.set_xlabel(self.results.config["Plot"]["xlabel"])
        else:
            ax.set_xlabel(groups[1])
        ax.set_ylabel(self.results.thermo_unit)
        return fig

    def rxn_energy_diagram(self, change, title):
        fig = plt.figure(self.figure_kwargs)
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
                Path(vert), color=color, **self.path_kwargs
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
