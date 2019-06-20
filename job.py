import json
import os
import warnings

import fireworks
import yaml
from fabric.connection import Connection

import Aaron.json_extension as json_ext
from AaronTools.const import AARONLIB, CONNECTIVITY_THRESHOLD
from AaronTools.geometry import Geometry

# hopefully fixed after next paramiko update
warnings.filterwarnings("ignore", message=".*EllipticCurve.*")


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
        self.attempt = 0
        self.cycle = 0
        self.conv_attempt = 0
        self.status = "start"

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
        Returns updated route_kwargs for resubmission

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
            if self.step >= 3:
                self.status = "revert"
            else:
                self.status = "failed"
        else:
            if error_code == "CALDSU":
                if "INDO" not in guess:
                    guess["INDO"] = ""
                    change_maxstep = False
            elif error_code == "LINK":
                change_maxstep = False
            elif progress.count("YES"):
                if "nolinear" not in opt:
                    opt["nolinear"] = ""
                    change_maxstep = False
            if change_maxstep:
                if "maxstep" not in opt or opt["maxstep"] > 5:
                    if int(self.step) == 2 and self.conv_attempt < 2:
                        pass
                    else:
                        opt["maxstep"] = 5
                elif opt["maxstep"] == 5:
                    opt["maxstep"] = 2
                elif opt["maxstep"] == 2:
                    if self.step >= 3:
                        self.status = "revert"
                    else:
                        self.status = "failed"

        # update
        if self.status == "revert":
            self.get_theory().route_kwargs = {}
            self.step = 2
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

    def examine_connectivity(self):
        """
        Determines formed/broken bonds relative to step 2 com file
        """

        def within_thresh(i, j):
            dist = self.catalyst.atoms[i].dist(self.catalyst.atoms[j])
            dist -= compare.atoms[i].dist(compare.atoms[j])
            dist = abs(dist)
            if self.reaction and self.reaction.con_thresh:
                return dist < self.reaction.con_thresh
            else:
                return dist < CONNECTIVITY_THRESHOLD

        formed = []
        broken = []
        if self.step < 2:
            return formed, broken
        compare = Geometry("{}.2.com".format(self.name))
        connectivities = zip(
            self.catalyst.connectivity(), compare.connectivity()
        )
        for i, con in enumerate(connectivities):
            a = set(con[0])
            b = set(con[1])
            if len(a) > len(b):
                tmp = a - b
                for j in a - b:
                    if within_thresh(i, j):
                        tmp.remove(j)
                if len(tmp) != 0:
                    formed += [(i, sorted(tmp))]
            if len(b) > len(a):
                tmp = b - a
                for j in b - a:
                    if within_thresh(i, j):
                        tmp.remove(j)
                if len(tmp) != 0:
                    broken += [(i, sorted(tmp))]
        return formed, broken

    def submit(self, step=None, **kwargs):
        if step is not None:
            self.step = step
        catalyst = self.catalyst_data.catalyst
        if int(self.step) == 1:
            catalyst.freeze()
            for change in self.catalyst_data.catalyst_change:
                sub = catalyst.find_substituent(change)
                catalyst.relax(sub.atoms)
        self.theory.write_com(catalyst, step=self.step, **kwargs)
        self.make_firework()
        if self.cluster_opts.remote_dir:
            with Connection(self.cluster_opts.host) as remote:
                remote_dir = os.path.join(
                    self.cluster_opts.remote_dir,
                    self.catalyst_data.get_relative_path(),
                )
                remote.run("mkdir -p {}".format(remote_dir))
                remote.put(
                    "{}.{}.com".format(catalyst.name, self.step),
                    remote=remote_dir,
                )
                remote.put()

    def make_qadapter(self):
        execute = "rlaunch -w {} -l {} singleshot".format(
            os.path.join(None), os.path.join(AARONLIB, "launchpad.yaml")
        )
        qadapter = {}
        qadapter["rocket_launch"] = execute
        qadapter["queue"] = self.cluster_opts.queue

        qadapter["job_name"] = self.catalyst_data.get_basename()

        if self.step < 2:
            nnodes = self.cluster_opts.short_nodes
            ppnode = self.cluster_opts.short_procs
            walltime = self.cluster_opts.short_wall
        else:
            nnodes = self.cluster_opts.n_nodes
            ppnode = self.cluster_opts.n_procs
            walltime = self.cluster_opts.wall

        qadapter["nnodes"] = nnodes
        qadapter["ppnode"] = ppnode
        qadapter["walltime"] = "{}:00:00".format(walltime)
        qadapter["mem"] = self.cluster_opts.memory
        qtemplate = os.path.join(
            AARONLIB, "{}_qadapter.txt".format(self.cluster_opts.queue_type)
        )
        if not os.access(qtemplate, os.R_OK):
            qtemplate = None
        from fireworks.user_objects.queue_adapters.common_adapter import (
            CommonAdapter,
        )

        qadapter = CommonAdapter(
            self.cluster_opts.queue_type,
            self.cluster_opts.queue,
            template_file=qtemplate,
            **qadapter
        )
        print(qadapter.to_dict())

    def make_firework(self):
        template = {}
        template["exec_memory"] = self.cluster_opts.exec_memory
        template["parallel"] = (
            self.cluster_opts.n_nodes * self.cluster_opts.n_procs
        )
        template["job_name"] = self.catalyst_data.get_basename()

        name = self.catalyst_data.get_basename() + "{}.tm".format(self.step)
        template_task = fireworks.TemplateWriterTask(
            template_file=os.path.join(AARONLIB, "G09_template.txt"),
            context=template,
            output_file=name,
        )
        script_task = fireworks.ScriptTask.from_str(
            "echo {}".format(name, name)
        )
        fw = fireworks.Firework([template_task, script_task])
        with open(self.catalyst_data.name + ".yaml", "w") as f:
            yaml.dump(fw.to_dict(), f)
        return
