import io
import json
import os
import pickle
from time import localtime, sleep

from fireworks import FireTaskBase, Firework, FWAction, LaunchPad, ScriptTask
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from jinja2 import Environment, FileSystemLoader

import Aaron.json_extension as json_ext
from AaronTools.comp_output import CompOutput
from AaronTools.const import AARONLIB

LAUNCHPAD = LaunchPad.auto_load()


class SubmitTask(FireTaskBase):
    _fw_name = "SubmitTask"
    required_params = ["job"]
    optional_params = ["step", "kwargs"]

    def _make_com(self, job, step, **kwargs):
        """
        Makes com file and transfers if necessary

        :job: the job to use
        :step: the step to set job to before building
        :**kwargs: route_kwargs to add to com file
        """
        if step is not None:
            job.step = step
        "Generating job: step {} for {}".format(
            job.step, job.catalyst_data.get_basename()
        )
        theory = job.get_theory()
        catalyst = job.catalyst_data.catalyst
        # freeze/relax everything but substitutions for step 1
        if int(job.step) == 1:
            skip_step1 = True
            catalyst.freeze()
            # relax substitutions
            for atom in catalyst.atoms:
                if "changed" in atom.tags:
                    catalyst.relax(atom)
                    skip_step1 = False
            # if no substitutions were made, we can go straight to step 2
            if skip_step1:
                job.step = 2
                catalyst.relax()
        # relax after step 1 done
        if job.step >= 2:
            catalyst.relax()
        # build com file
        job.write(**kwargs)
        # transfer if remote
        submit_dir, local_dir = job.get_dirs()
        if theory.remote_dir:
            # make conf dir and transfer com file
            job.remote_run("mkdir -p {}".format(submit_dir))
            job.remote_put(
                "{}.com".format(
                    os.path.join(local_dir, job.basename_with_step())
                ),
                submit_dir,
            )

    def _make_qadapter(self, job):
        """
        Makes and transfers (if needed) qadapter for job

        Returns:
            dictionary of parsed theory options
        """
        submit_dir, local_dir = job.get_dirs()
        theory = job.get_theory()
        opts = theory.__dict__.copy()
        #if "{" in opts["mem"]:
        #    opts["mem"] = theory.parse_function(
        #        opts["mem"], opts["params"], as_int=True
        #    )
        #if "{" in opts["exec_mem"]:
        #    opts["exec_mem"] = theory.parse_function(
        #        opts["exec_mem"], opts["params"], as_int=True
        #    )
        opts["job_name"] = job.basename_with_step()
        opts["rocket_launch"] = "rlaunch singleshot"

        # create qadapter
        if theory.remote_dir:
            AARONLIB_REMOTE, _ = job.remote_run(
                "echo -n $AARONLIB", hide="out"
            )
            qadapter_template = os.path.join(
                AARONLIB_REMOTE, "{}_qadapter.txt".format(theory.queue_type)
            )
        else:
            qadapter_template = os.path.join(
                AARONLIB, "{}_qadapter.txt".format(theory.queue_type)
            )
        qadapter = CommonAdapter(
            theory.queue_type,
            theory.queue,
            template_file=qadapter_template,
            **opts
        )
        # transfer if remote
        if theory.remote_dir:
            job.remote_put(
                io.StringIO(qadapter.to_format(f_format="yaml")),
                os.path.join(submit_dir, "qadapter.yaml"),
            )
        else:
            qadapter.to_file(os.path.join(submit_dir, "qadapter.yaml"))

        return opts

    def run_task(self, fw_spec):
        """
        Sets up the queue-adapter for FireWorks use and creates a corresponding
        FireTask for reserved qlaunching
        """
        with open(self.get("job"), "rb") as f:
            job = pickle.load(f)
        step = self.get("step", None)
        kwargs = self.get("kwargs", {})

        # make (and optionally transfer) com file
        self._make_com(job, step, **kwargs)
        # make (and optionally transfer) qadapter
        opts = self._make_qadapter(job)
        # make script task
        environment = Environment(loader=FileSystemLoader(AARONLIB))
        template = environment.get_template("G09_template.txt")
        script_fw = Firework(
            ScriptTask(script=template.render(**opts).split("\n"))
        )
        # make monitor task
        monitor_fw = Firework(
            MonitorTask(job=self.get("job"), fw_id=script_fw.fw_id)
        )
        return FWAction(additions=[script_fw, monitor_fw])


class MonitorTask(FireTaskBase):
    _fw_name = "MonitorTask"
    required_params = ["job", "fw_id"]
    optional_params = ["refresh_sec"]

    def run_task(self, fw_spec):
        """
        Creates a monitoring task for attaching to the sumitted job
        Possible FW states:
            'ARCHIVED', 'FIZZLED', 'DEFUSED', 'PAUSED', 'WAITING',
            'READY', 'RESERVED', 'RUNNING', 'COMPLETED'
        """
        with open(self.get("job"), "rb") as f:
            job = pickle.load(f)
        fw_id = self.get("fw_id")
        refresh_sec = self.get("refresh_sec", 300)
        theory = job.get_theory()
        submit_dir, local_dir = job.get_dirs()

        print("  Monitoring...")
        log_name = "{}.log".format(job.basename_with_step())
        while True:
            fw = LAUNCHPAD.get_fw_by_id(fw_id)
            if fw.state == "READY":
                job.launch_job(fw.fw_id)
            if fw.state == "RESERVED":
                sleep(refresh_sec)
                continue
            # update log for reading
            if theory.remote_dir:
                log = job.remote_get(
                    os.path.join(submit_dir, log_name),
                    os.path.join(local_dir, log_name),
                )
            else:
                log = os.path.join(local_dir, log_name)
            print(localtime(), log_name)
            log = CompOutput(log)
            job.wf.apply_action(
                FWAction(
                    stored_data=json.loads(
                        json.dumps(log, cls=json_ext.ATEncoder)
                    )
                )
            )
            print(log.get_progress())
            print("Finished:", log.finished)
            if job.step >= 2 and job.step < 3:
                if not job.examine_connectivity(log):
                    LAUNCHPAD.defuse_fw(fw.fw_id)
                    job.submit(parent_fw_id=fw.fw_id)
                    break
            if job.step > 2:
                if not job.examine_reaction():
                    LAUNCHPAD.defuse_fw(fw.fw_id)
                    job.submit(parent_fw_id=fw.fw_id)
                    break
            if fw.state == "COMPLETED" and log.finished:
                job.step = job.next_step()
                job.submit(parent_fw_id=fw.fw_id)
                break
            if fw.state == "COMPLETED" and not log.finished:
                job.fix_error(log)
                job.submit(parent_fw_id=fw.fw_id)
                break
            # restart job that's hit walltime limit
            if LAUNCHPAD.detect_lostruns(
                query={"fw_id": fw.fw_id},
                expiration_secs=int(job.get_theory().walltime) * 3600 + 300,
                rerun=True,
            )[1]:
                job.attempt += 0.1
                continue
            sleep(refresh_sec)
        job.wf.apply_action(FWAction(update_spec=job.get_spec()))
