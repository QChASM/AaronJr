import glob
import json
import os
import re
import subprocess
import warnings

import numpy as np
from fabric.connection import Connection
from fireworks import Firework, LaunchPad, PyTask, TemplateWriterTask
from jinja2 import Template

import Aaron.json_extension as json_ext
from AaronTools.const import AARONLIB

warnings.filterwarnings("ignore", message=".*EllipticCurve.*")


def qsub(cmd, regex, host=None):
    if host is not None:
        with Connection(host) as c:
            result = c.run(cmd)
    else:
        result = subprocess.run(cmd)
    return re.search(regex, result.stdout, re.M).group(1), result.stderr


def remote_run(host, cmd):
    with Connection(host) as c:
        result = c.run(cmd)
    return result.stdout, result.stderr


def remote_put(host, source, target):
    with Connection(host) as c:
        c.put(source, remote=target)


class Job:
    """
    Attributes
    :cluster_opts:
    :catalyst_data: CatalystMetaData() associated with ts structure
    :catalyst:      Catalyst() of the current conformer
    :name:          str with the full path and base name (without 3.log, etc)
    :step:          the current step
    :status:        job status
    :attempt:       the step2 attempt number for the current cycle
    :cycle:         the number of step2/3 cycles
    :conv_attempt:  the convergence attempt number for opt steps
    """

    MAX_CONV_ATTEMPT = 7
    MAX_ATTEMPT = 5
    MAX_CYCLE = 5
    all_jobs = {}
    status_stop = ["done", "skipped", "failed", "repeat"]

    @classmethod
    def update_json(cls, fname):
        """
        updates json tracking of all jobs
        """
        with open(fname, "w") as f:
            json.dump(cls.all_jobs, f, cls=json_ext.JSONEncoder, indent=2)

    @classmethod
    def read_json(cls, fname):
        """
        loads all_jobs from json
        """
        try:
            with open(fname) as f:
                cls.all_jobs = json.load(f, cls=json_ext.JSONDecoder)
        except (FileNotFoundError, OSError):
            pass

    def __init__(
        self, catalyst_data, theory, cluster_opts, step=None, overwrite=False
    ):
        self.catalyst_data = catalyst_data
        if self.catalyst_data.catalyst is None:
            self.catalyst_data.generate_structure()
        self.theory = theory
        self.cluster_opts = cluster_opts

        job_key = self.catalyst_data.get_basename()
        if not overwrite and job_key in Job.all_jobs:
            for key, val in Job.all_jobs[job_key].__dict__.items():
                if key in ["catalyst_data", "theory", "cluster_opts"]:
                    continue
                self.__dict__[key] = val
            Job.all_jobs[job_key] = self
            return

        self.name = self.catalyst_data.get_catalyst_name()
        if step is None:
            self.step = 1
        else:
            self.step = step
        self.attempt = 1
        self.cycle = 1
        self.conv_attempt = 1
        self.status = "start"
        self.msg = []

        Job.all_jobs[self.catalyst_data.get_basename()] = self

    def get_theory(self, step=None):
        """
        Returns the theory object associated with a particular step

        :step: the step to return the theory of (defaults to self.step)
        """
        if step is None:
            return self.theory.get_step(self.step)
        else:
            return self.theory.get_step(step)

    def fix_convergence_error(self, error_code, progress):
        """
        Determines what should be done for a geometry convergence error, and
        updates route_kwargs for resubmission.

        Returns: True if simple restart with updated route_kwargs is needed,
        Flase if reverting to step 2 or over convergence attempt limit.

        :error_code: error code defined by Comp_Output(logfile)
        :progress: list of YES/NO corresponding to convergence progress
        """
        theory = self.get_theory()
        route_kwargs = theory.route_kwargs
        if "opt" in route_kwargs:
            opt = route_kwargs["opt"]
        else:
            opt = {}
        if "guess" in route_kwargs:
            guess = route_kwargs["guess"]
        else:
            guess = {}
        change_maxstep = True

        # clean up old stuff
        if error_code != "CALDSU" and "INDO" in guess:
            del guess["INDO"]
        if progress.count("YES") != 3 and "nolinear" in opt:
            del opt["nolinear"]

        # make changes
        if self.conv_attempt > self.MAX_CONV_ATTEMPT:
            # determine if we should revert to previous step or give up
            if self.step >= 3:
                self.status = "revert"
            else:
                self.status = "failed"
        else:
            # determine if we should add nolinear or guess=INDO
            if error_code == "CALDSU":
                if "INDO" not in guess:
                    guess["INDO"] = ""
                    change_maxstep = False
            elif progress.count("YES") == 3:
                if "nolinear" not in opt:
                    opt["nolinear"] = ""
                    change_maxstep = False
            elif error_code == "LINK":
                change_maxstep = False
            if change_maxstep:
                if "maxstep" not in opt or opt["maxstep"] > 5:
                    if int(self.step) == 2 and self.conv_attempt < 2:
                        pass
                    else:
                        opt["maxstep"] = 5
                elif opt["maxstep"] > 2:
                    opt["maxstep"] = 2
                elif opt["maxstep"] <= 2:
                    if self.step >= 3:
                        self.status = "revert"
                    else:
                        self.status = "failed"

        # update
        if self.status == "revert":
            self.get_theory().route_kwargs = {}
            route_kwargs = self.get_theory().route_kwargs
            if "guess" in route_kwargs:
                del route_kwargs["guess"]
            route_kwargs["opt"] = {"maxstep": 15}
            return False
        elif self.status == "failed":
            self.get_theory().route_kwargs = {}
            return False
        else:
            if opt:
                route_kwargs["opt"] = opt
            elif "opt" in route_kwargs:
                del route_kwargs["opt"]
            if guess:
                route_kwargs["guess"] = guess
            elif "guess" in route_kwargs:
                del route_kwargs["guess"]

        if self.status in ["revert", "failed"]:
            return False
        self.conv_attempt += 1
        return True

    def fix_error(self, error_code, progress):
        """
        Handles errors classified by error_code. Determines changes to
        route_kwargs and whether a job should be restarted, reverted to
        previous step, or skipped.

        Raises: RuntimeError if the error detected must be fixed by the user
        (eg: Aaron input file errors, no disk space, too little memory)

        :error_code: Attribute detected by CompOutput() for job's logfile
        :progress: Also detected by CompOutput(), the convergence progress
        """
        theory = self.get_theory()
        route_kwargs = theory.route_kwargs
        # clean route_kwargs
        try:
            del route_kwargs["scf"]["xqc"]
        except KeyError:
            pass
        try:
            del route_kwargs["opt"]["noeigen"]
        except KeyError:
            pass
        try:
            del route_kwargs["opt"]["calcfc"]
        except KeyError:
            pass
        try:
            del route_kwargs["opt"]["readfc"]
        except KeyError:
            pass

        # Geometry convergence or unknown error
        if error_code in ["CONV_CDS", "CONV_LINK", "UNKNOWN"]:
            if error_code == "UNKNOWN":
                msg = "Unknown error."
            else:
                msg = "Geometry convergence error."
            if self.fix_convergence_error(error_code, progress):
                self.msg += [msg + " Restarting."]
                self.restart()
            elif self.status == "revert" and self.cycle < Job.MAX_CYCLE:
                self.msg += [msg + " Reverting to step 2"]
                self.update_geometry()
                self.attempt = 1
                self.cycle += 1
                self.revert(step=2)
            elif self.status == "revert":
                self.msg += [msg + " Too many cycles, skipping."]
                self.status == "failed"
            else:
                self.msg += [msg + " Too many attempts, skipping."]
            return

        # SCF convergence error
        if error_code == "SCF_CONV":
            if self.step < 3 or self.cycle >= 2:
                self.msg += ["SCF convergence error. Restarting with scf=xqc"]
                route_kwargs["scf"] = {"xqc": ""}
                if "opt" in route_kwargs and "maxstep" in route_kwargs["opt"]:
                    route_kwargs["opt"]["maxstep"] = 15
                self.attempt += 1
                self.restart()
            else:
                self.msg += [
                    "SCF convergence error. Reverting to step 2 with current geometry."
                ]
                self.update_geometry()
                self.cycle += 1
                self.attempt = 1
                self.revert(step=2)
            return

        # Wrong number of eigen values
        if error_code == "EIGEN":
            self.msg += ["Wrong number of eigen values. Restarting"]
            if "opt" in route_kwargs:
                route_kwargs["opt"]["noeigen"] = ""
            else:
                route_kwargs["opt"] = {"noeigen": ""}
            if "maxstep" in route_kwargs["opt"]:
                route_kwargs["opt"]["maxstep"] -= 2
                if route_kwargs["opt"]["maxstep"] < 1:
                    route_kwargs["opt"]["maxstep"] = 1
            else:
                route_kwargs["opt"]["maxstep"] = 15
            self.attempt += 1
            self.restart()
            return

        # problem with checkpoint file
        if error_code == "CHK":
            fname = self.catalyst_data.get_catalyst_name() + ".{}.chk".format(
                self.step
            )
            if os.path.isfile(fname):
                # corrupted checkpoint file
                self.msg += [
                    "Problem with checkpoint file, recalculating force constants."
                ]
                os.remove(fname)
                if "opt" in route_kwargs:
                    if "readfc" in route_kwargs["opt"]:
                        del route_kwargs["opt"]["readfc"]
                    route_kwargs["opt"]["calcfc"] = ""
                else:
                    route_kwargs["opt"] = {"calcfc": ""}
                self.restart()
                self.attempt += 0.5
            else:
                # some other problem... just try step from scratch
                self.msg += [
                    "FileIO error, restarting step with fresh *.com file"
                ]
                self.revert(self.step)
                self.attempt += 2
            return

        # Geometry related errors
        if error_code == "CLASH":
            self.msg += ["Atoms too close. Fixing and restarting."]
            self.update_geometry()
            bad_subs = self.catalyst_data.catalyst.remove_clash()
            if bad_subs:
                self.status = "skipped"
                self.msg += [
                    "Atoms too close, failed to fix automatically. Skipping."
                ]
                return
            self.attempt += 1
            self.restart(update_geometry=False)
            return
        if error_code == "REDUND":
            self.msg += ["Bend failed for angle. Restarting"]
            self.attempt += 1
            self.restart()
            return
        if error_code == "FBX":
            self.msg += ["Flat angle/dihedral error. Attempting to fix."]
            self.attempt += 1
            self.restart()
            return
        if error_code == "CONSTR":
            self.msg += ["Error imposing constraints. Attempting to fix."]
            self.update_geometry()
            cat = self.catalyst_data.catalyst
            for i, j, dist in cat.get_constraints():
                cat.atoms[i].coords = np.round(cat.atoms[i].coords, 4)
                cat.atoms[j].coords = np.round(cat.atoms[j].coords, 4)
            self.attempt += 1
            self.restart(update_geometry=False)
            return

        # Error is with input file or a system error that Aaron cannot fix
        if error_code == "GALLOC":
            # this happens sometimes and is generally a node problem.
            # restarting is usually fine, may not be if job is picked up
            # on the same node as before (TODO: node-specific submission)
            self.msg += ["Error allocating memory on node. Restarting."]
            self.attempt += 0.5
            self.restart()
            return
        if error_code == "QUOTA":
            raise RuntimeError(
                "Erroneous write. Check quota or disk space, then restart Aaron."
            )
        if error_code == "CHARGEMULT":
            raise RuntimeError(
                "Bad charge/multiplicity provided. Please fix you Aaron input file or template geometries."
            )
        if error_code == "BASIS":
            raise RuntimeError(
                "Error reading basis set. Confirm that gen=/path/to/basis/ is correct in your .aaronrc and that the basis set file requested exists, or switch to an internally provided basis set."
            )
        if error_code == "ATOM":
            raise RuntimeError("Bad atomic symbol, check template XYZ files.")
        if error_code == "MEM":
            raise RuntimeError(
                "Node(s) out of memory. Increase memory requested in Aaron input file"
            )

    def examine_reaction(self, update_from="log"):
        """
        Determines if constrained atoms are too close/ too far apart. If so,
        fixes the issue and restarts from step 2 with the adjusted structure

        Returns:
            None if constraints are all fine
            set(tuple(atom1, atom2, flag)) if bad constraints detected
        """
        cat = self.catalyst_data.catalyst
        # examine constraints
        result = cat.examine_constraints()
        if len(result) == 0:
            return None
        # fix geometry
        if update_from == "log":
            cat.update_geometry(cat.name + ".2.log")
        elif update_from == "xyz":
            cat.update_geometry(cat.name + ".2.xyz")
        rv = set([])
        msg = (
            "{}: Bad distances in reaction center detected, "
            "updating structure\n".format(self.catalyst_data.get_basename())
        )
        for i, j, d in result:
            rv.add((i, j, d))
            d *= 0.1
            cat.change_distance(
                cat.atoms[i], cat.atoms[j], dist=d, adjust=True
            )
            msg += "...Adjusting distance of {}-{} bond by {} A\n".format(
                i + 1, j + 1, d
            )
        self.msg += [msg[:-1]]
        # restart from step 2
        if self.cycle >= Job.MAX_CYCLE:
            self.status = "killed"
            self.msg += ["WARN: Too many cycles, job killed"]
        else:
            self.cycle += 1
            self.attempt = 1
            self.msg += ["Reverting to step 2"]
            self.revert(step=2)
        return rv

    def update_geometry(self, update_from=None):
        cat = self.catalyst_data.catalyst
        if update_from is None:
            fname = cat.name + ".{}.log".format(self.step)
        else:
            fname = update_from
        try:
            cat.update_geometry(fname)
        except (RuntimeError, FileNotFoundError) as err:
            self.msg += [
                "WARN: Cannot update geometry from {} ({}). Keeping current"
                " geometry as-is".format(os.path.basename(fname), err)
            ]

    def revert(self, step, update_geometry=True, update_from=None):
        if update_geometry:
            self.update_geometry(update_from)
        self.step = step
        self.remove_after(step)
        self.status = "2submit"

    def restart(self, update_geometry=True, update_from=None):
        if update_geometry:
            self.update_geometry(update_from)
        self.remove_after(self.step)
        self.write()
        self.status = "2submit"

    def remove_after(self, step):
        for fname in glob.glob(self.catalyst_data.catalyst.name + ".*"):
            fsplit = fname.split(".")
            if fsplit[-1] not in ["com", "job", "log"]:
                continue
            ftype = fsplit.pop()
            fstep = fsplit.pop()
            tmp = fsplit.pop()
            if not tmp.startswith("cf") and tmp.isnumeric():
                fstep = tmp + "." + fstep
            fstep = float(fstep)

            if fstep >= step and ftype == "log":
                os.rename(fname, fname + ".bkp")
            elif fstep >= step:
                os.remove(fname)

    def write(self, style="com", step=None, **kwargs):
        """
        Writes the job's current geometry to a file.
        """
        if step is None:
            step = self.step
        self.catalyst_data.catalyst.write(
            style=style, step=step, theory=self.theory, **kwargs
        )

    def submit(self, step=None, **kwargs):
        self.status = "pending"
        if step is not None:
            self.step = step
        catalyst = self.catalyst_data.catalyst
        # freeze/relax everything but substitutions for step 1
        if int(self.step) == 1:
            skip_step1 = True
            catalyst.freeze()
            # relax substitutions
            for atom in catalyst.atoms:
                if "changed" in atom.tags:
                    catalyst.relax(atom)
                    skip_step1 = False
        # if no substitutions were made, we can go straight to step 2
        if skip_step1:
            self.step = 2
            catalyst.relax()
        # build com file
        self.write()
        # make and launch firework
        if self.cluster_opts.remote_dir:
            tasks = self.transfer_task()
        else:
            tasks = []
        tasks += self.qprep_task()
        fw = Firework(tasks, name=self.catalyst_data.get_basename())
        LaunchPad().add_wf(fw)

    def qadapter_task(self):
        opts = self.cluster_opts
        params = opts.__dict__.copy()
        if self.step < 2:
            if "n_procs" in params and "short_procs" in params:
                params["n_procs"] = params["short_procs"]
            if "wall" in params and "short_wall" in params:
                params["wall"] = params["short_wall"]
        if self.step > 4:
            if "n_procs" in params and "long_procs" in params:
                params["n_procs"] = params["long_procs"]
            if "wall" in params and "long_wall" in params:
                params["wall"] = params["long_wall"]
        params["n_procs"] = re.search("\d+", params["n_procs"]).group(0)

        if "{" in opts.exec_memory:
            params["exec_memory"] = opts._parse_function(
                opts.exec_memory, params, as_int=True
            )
        if "{" in opts.memory:
            params["memory"] = opts._parse_function(
                opts.memory, params, as_int=True
            )
        params["job_name"] = "{}.{}".format(
            self.catalyst_data.get_basename(), self.step
        )

        template_task = TemplateWriterTask(
            template_file="G09_template.txt",
            template_dir=AARONLIB,
            context=params,
            output_file=os.path.join(
                opts.top_dir,
                self.catalyst_data.get_relative_path(),
                "G09_exec",
            ),
        )

        return [template_task]

    def transfer_task(self):
        remote_dir = os.path.join(
            self.cluster_opts.remote_dir,
            self.catalyst_data.get_relative_path(),
        )
        local_com = "{}.{}.com".format(
            self.catalyst_data.get_catalyst_name(), self.step
        )
        tasks = []
        tasks.append(
            PyTask(
                func="Aaron.job.remote_run",
                args=[
                    self.cluster_opts.transfer_host,
                    "mkdir -p {}".format(remote_dir),
                ],
            )
        )
        tasks.append(
            PyTask(
                func="Aaron.job.remote_put",
                args=[self.cluster_opts.transfer_host, local_com, remote_dir],
            )
        )
        return tasks

    def qprep_task(self):
        opts = self.cluster_opts
        if opts.remote_dir:
            work_dir = opts.remote_dir
        else:
            work_dir = opts.top_dir
        work_dir = os.path.join(
            work_dir, self.catalyst_data.get_relative_path()
        )

        qcmd = self.cluster_opts.qcmd.format(
            queue=opts.queue,
            wall=opts.short_wall if self.step < 2 else opts.wall,
            procs=opts.short_nodes if self.step < 2 else opts.n_procs,
            job_name="{}.{}.com".format(
                self.catalyst_data.get_basename(), self.step
            ),
        )
        qcmd = 'cd "{}" && {}'.format(work_dir, qcmd)

        if opts.remote_dir:
            task = PyTask(
                func="Aaron.job.qsub",
                args=[qcmd, opts.qcmd_regex, opts.host],
                stored_data_varname="submit_result",
            )
        else:
            task = PyTask(
                func="Aaron.job.qsub",
                args=[qcmd, opts.qcmd_regex],
                stored_data_varname="submit_result",
            )
        return [task]
