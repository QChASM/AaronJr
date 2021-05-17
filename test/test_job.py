#!/usr/bin/env python3
import json
import os
import unittest

from Aaron.job import Job
from AaronTools.config import Config
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, validate

prefix = prefix.replace("AaronTools", "Aaron", 1)
spec_skip = [
    "infile",
    "metadata",
    "Job/remote_dir",
    "HPC/user",
    "HPC/scratch_dir",
    "HPC/work_dir",
]


class TestJob(TestWithTimer):
    def test_spec(self):
        self.maxDiff = None
        config = Config(
            os.path.join(prefix, "test_files", "substitution.ini"), quiet=True
        )
        for change in config._changes:
            geom = Geometry.from_string(config["Geometry"]["structure"])
            this_config = config.for_change(change, structure=geom)
            job = Job(geom, this_config, testing=True)
            ref = job.get_spec()
            with open(os.path.join(prefix, "ref_files", "job_spec.json")) as f:
                ref = json.load(f)
            test = json.loads(json.dumps(job.get_spec()))
            for key in spec_skip:
                del ref[key]
                del test[key]
            self.assertDictEqual(ref, test)

    def test_substitution(self):
        self.maxDiff = None
        # substitute atom and fuse ring
        config = Config(
            os.path.join(prefix, "test_files", "substitution.ini"), quiet=True
        )
        geom = Geometry.from_string(config["Geometry"]["structure"])
        job = Job(geom, config, testing=True)
        test = job.structure
        ref = Geometry(
            os.path.join(prefix, "ref_files", "substitution_with_rings.xyz")
        )
        self.assertTrue(validate(test, ref, sort=True))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestJob("test_spec"))
    suite.addTest(TestJob("test_substitution"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
