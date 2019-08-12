#!/usr/bin/env python3
import os
import unittest

import numpy as np
from fireworks import Firework, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket

from Aaron.aaron_init import AaronInit
from Aaron.job import Job
from AaronTools.test import TestWithTimer, prefix

prefix = prefix.replace("AaronTools", "Aaron", 1)


class TestJob(TestWithTimer):
    def touch_files(self, name, steps=[1, 4]):
        for i in range(steps[0], steps[1]):
            for s in ["com", "log", "job"]:
                tmp = name + ".{}.{}".format(i, s)
                with open(tmp, "w") as f:
                    f.close()

    def setUp(self):
        super().setUp()
        self.init = AaronInit(prefix + "test_files/simple.in")
        self.reaction = self.init.reaction
        self.theory = self.init.theory

        self.reaction.generate_structure_data()
        self.catalyst_data = self.reaction.catalyst_data
        self.catalyst_data[0].generate_structure()
        self.catalyst = self.catalyst_data[0].catalyst

    def test_init(self):
        test_job = Job(
            self.catalyst_data[0], self.theory, self.init.cluster_opts
        )
        test_str = ""
        for key, val in test_job.__dict__.items():
            if key in ["catalyst_data", "theory", "cluster_opts"]:
                test_str += "{}:\n".format(key)
                for k, v in val.__dict__.items():
                    test_str += "  {}: {}\n".format(k, v)
                continue
            test_str += "{}: {}\n".format(key, val)
        with open("tmp", "w") as f:
            f.write(test_str)
        with open(prefix + "ref_files/job_init.txt") as f:
            self.assertEqual(test_str, f.read())

    def test_fix_conv_error(self):
        def get_route(error_code=None, progress=[]):
            theory = test_job.get_theory()
            success = True
            if error_code is not None:
                success = test_job.fix_convergence_error(error_code, progress)
            if success:
                route = theory.make_route(
                    test_job.catalyst_data.catalyst, test_job.step
                )
            else:
                route = ""
            route = "step {} ({}){}{}\n".format(
                int(test_job.step)
                if test_job.step == int(test_job.step)
                else test_job.step,
                test_job.status,
                ": " if route else "",
                route,
            )
            print(route, end="")

            if test_job.status == "revert":
                test_job.conv_attempt = 0
                test_job.status = "start"

        three_yes = ["YES"] * 3 + ["NO"]

        # step 2
        test_job = Job(
            self.catalyst_data[0], self.theory, self.init.cluster_opt, step=2
        )
        get_route()
        get_route("LINK", progress=three_yes)
        get_route("CALDSU")
        get_route("UNK")
        get_route("UNK", progress=three_yes)
        get_route("UNK", progress=three_yes)
        get_route("UNK")
        get_route("UNK")
        get_route("UNK")
        get_route("UNK")
        get_route("LINK")
        print()

        # step 3
        test_job = Job(self.catalyst_data[0], self.theory, step=3)
        get_route()
        get_route("LINK", progress=three_yes)
        get_route("CALDSU")
        get_route("UNK")
        get_route("UNK")
        get_route("UNK")
        get_route("UNK")
        get_route("UNK")
        get_route("UNK")
        get_route("UNK", progress=three_yes)
        get_route("UNK")
        get_route("UNK")
        get_route("LINK")
        get_route("UNK")
        get_route("UNK")
        print()

    def test_examine_reaction(self):
        """
        tests examine_reaction, revert, and remove_after
        """

        def check_files(name, removed_after=None):
            for i in range(1, 4):
                tmp = name + ".{}.".format(i)
                if removed_after is not None and i > removed_after:
                    self.assertFalse(os.access(tmp + "com", os.W_OK))
                    self.assertFalse(os.access(tmp + "job", os.W_OK))
                    self.assertFalse(os.access(tmp + "log", os.W_OK))
                    self.assertTrue(os.access(tmp + "log.bkp", os.W_OK))
                elif removed_after is not None and i == removed_after:
                    self.assertTrue(os.access(tmp + "com", os.W_OK))
                    self.assertFalse(os.access(tmp + "job", os.W_OK))
                    self.assertFalse(os.access(tmp + "log", os.W_OK))
                    self.assertTrue(os.access(tmp + "log.bkp", os.W_OK))
                else:
                    self.assertTrue(os.access(tmp + "com", os.W_OK))
                    self.assertTrue(os.access(tmp + "job", os.W_OK))
                    self.assertTrue(os.access(tmp + "log", os.W_OK))

        test_job = Job(
            self.catalyst_data[0], self.theory, self.init.cluster_opts
        )
        cat = test_job.catalyst_data.catalyst
        cat.write(cat.name + ".2")

        # no changes
        rv = test_job.examine_reaction(update_from="xyz")
        self.assertTrue(rv is None)

        # bad constraints and below max cycle
        touch_files(cat.name)
        start_dist = cat.atoms[0].dist(cat.atoms[5])
        ref_dist = start_dist - 0.1
        cat.change_distance(cat.atoms[0], cat.atoms[5], dist=0.5, adjust=True)
        rv = test_job.examine_reaction(update_from="xyz")
        test_dist = cat.atoms[0].dist(cat.atoms[5])
        self.assertSetEqual(rv, {(0, 5, -1)})
        self.assertEqual(ref_dist, test_dist)
        self.assertEqual(test_job.msg[-1], "Reverting to step 2")
        check_files(cat.name, removed_after=2)

        # bad constraints and over max cycle
        touch_files(cat.name)
        test_job.cycle = 5
        test_job.msg = []
        cat.change_distance(cat.atoms[0], cat.atoms[5], dist=0.5, adjust=True)
        rv = test_job.examine_reaction(update_from="xyz")
        test_dist = cat.atoms[0].dist(cat.atoms[5])
        self.assertSetEqual(rv, {(0, 5, -1)})
        self.assertEqual(ref_dist, test_dist)
        self.assertEqual(test_job.msg[-1], "WARN: Too many cycles, job killed")
        check_files(cat.name)

    def test_restart(self):
        test_job = Job(
            self.catalyst_data[0], self.theory, self.init.cluster_opts
        )
        cat = test_job.catalyst_data.catalyst
        self.touch_files(cat.name)

        test_job.step = 1
        test_job.restart()
        self.assertTrue(len(test_job.msg) == 1)
        self.assertEqual(
            test_job.msg[-1],
            "WARN: Cannot update geometry from "
            "S_binap.TMEN_Ph.R.ts1.cf11111.1.log (Updated geometry has "
            "different number of atoms). Using current geometry for restart",
        )

        test_job.msg = []
        test_job.step = 4
        test_job.restart()
        self.assertTrue(len(test_job.msg) == 1)
        self.assertEqual(
            test_job.msg[-1],
            "WARN: Cannot update geometry from "
            "S_binap.TMEN_Ph.R.ts1.cf11111.4.log ([Errno 2] No such file or "
            "directory: '/home/deo/pyChASM/Aaron/test/test_files/S_binap/"
            "TMEN_Ph/R/ts1/947/S_binap.TMEN_Ph.R.ts1.cf11111.4.log'). Using "
            "current geometry for restart",
        )

        test_job.msg = []
        test_job.step = 1
        cat.write(cat.name + ".1")
        cat.coord_shift([1, 0, 0])
        old_coords = cat._stack_coords()
        test_job.restart(update_from=cat.name + ".1.xyz")
        self.assertTrue(len(test_job.msg) == 0)
        test_diff = old_coords - cat._stack_coords()
        ref_diff = np.zeros((len(cat.atoms), 3))
        ref_diff[:, 0] = 1
        self.assertLess(np.linalg.norm(test_diff - ref_diff), 10 ** -4)

    def test_qadapter(self):
        test_job = Job(
            self.catalyst_data[0], self.theory, self.init.cluster_opts
        )
        print(test_job.step)
        print(test_job.catalyst_data.catalyst.name)

        fw = Firework(test_job.qadapter_task())

        launchpad = LaunchPad(name="test")
        launchpad.reset("", require_password=False)
        launchpad.add_wf(fw)
        launch_rocket(launchpad)


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestJob("test_init"))
    # suite.addTest(TestJob("test_fix_conv_error"))
    # suite.addTest(TestJob("test_examine_reaction"))
    # suite.addTest(TestJob("test_restart"))
    suite.addTest(TestJob("test_qadapter"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
