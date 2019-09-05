import glob
import json
import os
import pickle
import warnings

import numpy as np
from fabric.connection import Connection
from fireworks import Firework, LaunchPad, Workflow
from fireworks.core.rocket_launcher import launch_rocket, rapidfire
from invoke.exceptions import UnexpectedExit

import Aaron.json_extension as json_ext
from Aaron.firetasks import workflow
from AaronTools.comp_output import CompOutput
from AaronTools.const import AARONLIB

warnings.filterwarnings("ignore", message=".*EllipticCurve.*")
warnings.filterwarnings("ignore", module=".*queue.*")
LAUNCHPAD = LaunchPad.auto_load()


class Job:
    """
    Attributes
    :theory:
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
    def update_json(cls, fname="jobs.json"):
        """
        updates json tracking of all jobs
        """
        with open(fname, "w") as f:
            json.dump(cls.all_jobs, f, cls=json_ext.JSONEncoder, indent=2)

    @classmethod
    def read_json(cls, fname="jobs.json"):
        """
        loads all_jobs from json
        """
        try:
            with open(fname) as f:
                cls.all_jobs = json.load(f, cls=json_ext.JSONDecoder)
        except (FileNotFoundError, OSError):
            pass

    def __init__(self, catalyst_data, theory, step=None, overwrite=False):
        self.wf = None
        self.catalyst_data = catalyst_data
        if self.catalyst_data.catalyst is None:
            self.catalyst_data.generate_structure()
        self.theory = theory

        job_key = self.catalyst_data.get_basename()
        if not overwrite and job_key in Job.all_jobs:
            for key, val in Job.all_jobs[job_key].__dict__.items():
                if key in ["catalyst_data", "theory"]:
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
        self.other = {}

        Job.all_jobs[self.catalyst_data.get_basename()] = self

    # private utilities
    def _step(self, step=None):
        if step is None:
            step = self.step
        if int(step) == step:
            return int(step)
        return step

    def _step_list(self):
        rv = sorted(
            list(self.theory.by_step.keys())
            + [
                float(i)
                for i in range(1, 5)
                if float(i) not in self.theory.by_step.keys()
            ]
        )
        return rv

    def _get_submit_dir(self, step=None):
        theory = self.get_theory(step)
        submit_dir = os.path.join(
            theory.top_dir, self.catalyst_data.get_relative_path()
        )
        if theory.remote_dir:
            submit_dir = os.path.join(
                theory.remote_dir, self.catalyst_data.get_relative_path()
            )
        return submit_dir

    def _get_local_dir(self, step=None):
        return os.path.join(
            self.get_theory().top_dir, self.catalyst_data.get_relative_path()
        )

    def _revert_to_step_2(self, update_geometry=False, update_from=None):
        if self.cycle >= Job.MAX_CYCLE:
            self.status = "killed"
            self.msg += ["WARN: Too many cycles, job killed"]
        else:
            self.cycle += 1
            self.attempt = 1
            self.msg += ["Reverting to step 2"]
            self.revert(
                step=2,
                update_geometry=update_geometry,
                update_from=update_from,
            )
            self.write()

    # utilities
    def basename_with_step(self, step=None):
        return "{}.{}".format(
            self.catalyst_data.get_basename(), self._step(step)
        )

    def get_dirs(self, step=None):
        submit_dir = self._get_submit_dir(step)
        local_dir = submit_dir
        theory = self.get_theory(step)
        if theory.remote_dir:
            local_dir = self._get_local_dir(step)
        return submit_dir, local_dir

    def get_theory(self, step=None):
        """
        Returns the theory object associated with a particular step

        :step: the step to return the theory of (defaults to self.step)
        """
        if step is None:
            return self.theory.get_step(self.step)
        return self.theory.get_step(step)

    def write(self, style="com", step=None, **kwargs):
        """
        Writes the job's current geometry to a file.
        """
        if step is None:
            step = self.step
        theory = self.get_theory()
        kwargs["fname"] = os.path.join(
            self._get_submit_dir(), self.basename_with_step()
        )
        self.catalyst_data.catalyst.write(
            style=style, step=step, theory=theory, **kwargs
        )

    def examine_reaction(self, update_from="log"):
        """
        Determines if constrained atoms are too close/ too far apart. If so,
        fixes the issue and restarts from step 2 with the adjusted structure

        Returns:
            True if constraints are all fine, false otherwise
        """
        cat = self.catalyst_data.catalyst
        # examine constraints
        result = cat.examine_constraints()
        if not result:
            return True
        # fix geometry
        if update_from == "log":
            update_from = cat.name + ".2.log"
        elif update_from == "xyz":
            update_from = cat.name + ".2.xyz"
        cat.update_geometry(update_from)
        bad_con = set([])
        msg = (
            "{}: Bad distances in reaction center detected, "
            "updating structure\n".format(self.basename_with_step())
        )
        for i, j, d in result:
            bad_con.add((i, j, d))
            d *= 0.1
            cat.change_distance(
                cat.atoms[i], cat.atoms[j], dist=d, adjust=True
            )
            msg += "...Adjusting distance of {}-{} bond by {} A\n".format(
                i + 1, j + 1, d
            )
        self.msg += [msg[:-1]]
        # restart from step 2
        self._revert_to_step_2(update_geometry=False)
        return False

    def examine_connectivity(self, log):
        com = os.path.join(
            self.get_theory(2).top_dir,
            self.catalyst_data.get_relative_path(),
            "{}.2.com".format(self.catalyst_data.get_basename()),
        )
        formed, broken = log.geometry.compare_connectivity(com)
        if not formed and not broken:
            return True
        self.update_geometry(update_from=com)
        catalyst = self.catalyst_data.catalyst
        if "original_constraints" not in self.other:
            self.other["original_constraints"] = catalyst.get_constraints()
        for i, j in formed.union(broken):
            a = catalyst.find_exact(i)[0]
            b = catalyst.find_exact(j)[0]
            a.constraint.add(b, a.dist(b))
            b.constraint.add(a, b.dist(a))
        self._revert_to_step_2(update_geometry=False)
        return False

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

    def remote_run(self, cmd, hide=None):
        host = self.get_theory().host
        with Connection(host) as c:
            result = c.run(cmd, hide=hide)
        return result.stdout, result.stderr

    def remote_put(self, source, target):
        host = self.get_theory().transfer_host
        with Connection(host) as c:
            c.put(source, remote=target)

    def remote_get(self, source, target):
        host = self.get_theory().transfer_host
        with Connection(host) as c:
            c.get(source, target)
        return target

    def get_spec(self, step=None):
        """
        Generates spec dict for storing in FW or for lookup
        """
        if step is None:
            step = self.step
        rv = {"step": step}
        theory = self.get_theory(step)
        cat_data = self.catalyst_data
        for key, val in theory.__dict__.items():
            if not val:
                continue
            if key in [
                "top_dir",
                "host",
                "transfer_host",
                "remote_dir",
                "queue_type",
                "queue",
                "params",
            ]:
                continue
            if key == "basis":
                tmp = {}
                for element in [a.element for a in cat_data.catalyst.atoms]:
                    tmp[element] = val[element]
                val = tmp
            if isinstance(val, str) and "{" in val:
                val = theory.parse_function(val, theory.params, as_int=True)
            rv[key] = val

        for key, val in cat_data.__dict__.items():
            if key in ["catalyst_change", "ts_directory"]:
                continue
            if key == "template_file":
                if AARONLIB in val:
                    val = os.path.relpath(val, start=AARONLIB)
                    val = os.path.join("$AARONLIB", val)
            rv[key] = val
        return rv

    def next_step(self):
        """
        Returns the next step greater than the current step in user-defined steps
        and steps 1-4
        """
        for step in self._step_list():
            if step <= self.step:
                continue
            return int(step) if int(step) == step else step

    # error handling
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
        if progress.count(True) != 3 and "nolinear" in opt:
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
            elif progress.count(True) == 3:
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
        elif self.status == "failed":
            self.get_theory().route_kwargs = {}
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

    def fix_error(self, log):
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
        error_code = log.error
        progress = [v["converged"] for v in log.gradient.values()]
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
                self.status = "failed"
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
            fname = os.path.join(
                self.catalyst_data.get_relative_path(),
                "{}.chk".format(self.basename_with_step()),
            )
            found = False
            if theory.remote_dir:
                fname = os.path.join(theory.remote_dir, fname)
                try:
                    self.remote_run("ls {}".format(fname), hide="both")
                    found = True
                except UnexpectedExit:
                    pass
            else:
                fname = os.path.join(theory.top_dir, fname)
                found = os.path.isfile(fname)
            if found:
                # corrupted checkpoint file
                self.msg += [
                    "Problem with checkpoint file, recalculating force constants."
                ]
                if theory.remote_dir:
                    self.remote_run("rm -f {}".format(fname), hide="out")
                else:
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
            for i, j, _ in cat.get_constraints():
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

    # workflow and updates
    def update_geometry(self, update_from=None):
        cat = self.catalyst_data.catalyst
        if update_from is None:
            update_from = cat.name + ".{}.log".format(self._step)
        elif isinstance(update_from, CompOutput):
            update_from = update_from.geometry
        try:
            cat.update_geometry(update_from)
        except (RuntimeError, FileNotFoundError) as err:
            if isinstance(update_from, str):
                fname = update_from
            else:
                fname = update_from.name
            self.msg += [
                "WARN: Cannot update geometry from {} ({}). Keeping current"
                " geometry as-is".format(os.path.basename(fname), err)
            ]

    def revert(self, step, update_geometry=True, update_from=None):
        if update_geometry:
            self.update_geometry(update_from)
            self.write()
        self.step = step
        self.remove_after(step)
        self.status = "2submit"

    def restart(self, update_geometry=True, update_from=None):
        if update_geometry:
            self.update_geometry(update_from)
        self.remove_after(self.step)
        self.write()
        self.status = "2submit"

    def find_fw_ids(self, step=None):
        if step is None:
            step = self.step
        spec_query = {"name": self.basename_with_step(step)}
        for key, val in self.get_spec(step).items():
            if key in ["status", "catalyst"]:
                continue
            spec_query["spec." + key] = val
        rv = LAUNCHPAD.get_fw_ids(query=spec_query)
        return rv

    def check_status(self):
        """
        Finds current fw with latest step for this job
        """
        fw = None
        for step in self._step_list():
            if step < self.step:
                continue
            submit_dir, local_dir = self.get_dirs(step)
            if not os.access(local_dir, os.W_OK):
                os.makedirs(local_dir)
            if self.get_theory(step).remote_dir:
                try:
                    # get com then log,
                    # in case still queued, will fail after com has been got
                    com = "{}.com".format(self.basename_with_step(step))
                    log = "{}.log".format(self.basename_with_step(step))
                    self.remote_get(
                        os.path.join(submit_dir, com),
                        os.path.join(local_dir, com),
                    )
                    self.remote_get(
                        os.path.join(submit_dir, log),
                        os.path.join(local_dir, log),
                    )
                except FileNotFoundError:
                    pass
            try:
                rv = self.find_fw_ids(step)
            except IndexError:
                # if we are here, fw has same value as before try block
                break
            for fw_id in rv:
                fw = LAUNCHPAD.get_fw_by_id(fw_id)
                if fw.state == "DEFUSED":
                    fw = None
                    continue
                break
        if fw:
            self.step = fw.spec["step"]
            self.wf = LAUNCHPAD.get_wf_by_fw_id(fw.fw_id)
        if fw and fw.state == "FIZZLED":
            raise RuntimeError(
                "Issue running job. Please check executable and qadapter templates for errors."
            )
        return fw

    def submit(self, parent_fw_id=None, step=None, **kwargs):
        """
        Submits job and starts monitoring. Jobs for new steps appended to Job.wf

        :parent_fw_id: the fw_id for the parent step (default: last fw in the workflow)
        :step: the step to do (default: self.step)

        **kwargs: for passing extra route_kwargs
        """
        self.status = "pending"
        # find
        fw = self.check_status()
        fname = os.path.join(self._get_local_dir(step), "job.pkl")
        if not fw:
            fw = Firework(
                workflow.SubmitTask(job=fname, step=step, kwargs=kwargs),
                name=self.basename_with_step(),
                spec=self.get_spec(),
            )
            if self.wf is None:
                self.wf = Workflow(
                    [fw], name=self.catalyst_data.get_basename()
                )
                LAUNCHPAD.add_wf(self.wf)
            else:
                if parent_fw_id is None:
                    parent_fw_id = self.wf.fws[-1].fw_id
                LAUNCHPAD.append_wf(Workflow([fw]), [parent_fw_id])
        # launch
        with open(fname, "wb") as f:
            pickle.dump(self, f)
        rapidfire(LAUNCHPAD, nlaunches=0)

    def launch_job(self, fw_id):
        print("  Submitting...")
        submit_dir = self._get_submit_dir()
        qlaunch_cmd = "qlaunch -r -q {} --launch_dir {} singleshot --fw_id {}".format(
            os.path.join(submit_dir, "qadapter.yaml"), submit_dir, fw_id
        )
        if self.get_theory().remote_dir:
            out, err = self.remote_run(qlaunch_cmd, hide="both")
            if out:
                print("out:", out)
            for e in err.split("\n"):
                if (
                    "specified in qadapter but it is not present in template"
                    in e
                    or ".format(subs_key, self.template_file)" in e
                    or e.strip() == ""
                ):
                    continue
                print("err", e.strip())
        else:
            os.system(qlaunch_cmd)
