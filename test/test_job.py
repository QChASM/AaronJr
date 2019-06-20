#!/usr/bin/env python3
import json
import os
import unittest

from Aaron.aaron_init import AaronInit
from Aaron.job import Job
from Aaron.json_extension import JSONDecoder, JSONEncoder
from AaronTools.test import TestWithTimer, prefix

prefix = prefix.replace("AaronTools", "Aaron", 1)


class TestReaction(TestWithTimer):
    def setUp(self):
        super().setUp()
        self.init = AaronInit(prefix + "test_files/simple.in")
        self.reaction = self.init.reaction
        self.theory = self.init.theory

        self.reaction.get_templates()
        self.reaction.make_directories(prefix + "test_files/job_test")
        self.catalyst_data = self.reaction.catalyst_data

    def test_init(self):
        test_job = Job(self.catalyst_data[0], self.theory)
        test_str = ""
        for key, val in test_job.__dict__.items():
            if key in ["catalyst_data", "theory"]:
                test_str += "{}:\n".format(key)
                for k, v in val.__dict__.items():
                    test_str += "  {}: {}\n".format(k, v)
                continue
            test_str += "{}: {}\n".format(key, val)
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
        test_job = Job(self.catalyst_data[0], self.theory, step=2)
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


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestReaction("test_init"))
    suite.addTest(TestReaction("test_fix_conv_error"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
