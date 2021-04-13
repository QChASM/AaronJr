#!/usr/bin/env python3
import json
import os
import unittest

from Aaron.job import Job
from AaronTools.config import Config
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, validate

prefix = prefix.replace("AaronTools", "Aaron", 1)


class TestJob(TestWithTimer):
    def test_spec(self):
        config = Config(os.path.join(prefix, "test_files", "substitution.ini"))
        for change in config._changes:
            geom = Geometry.from_string(config["Geometry"]["structure"])
            this_config = config.for_change(change, structure=geom)
            job = Job(geom, this_config, set_root=False)
            ref = job.get_spec()
            with open(os.path.join(prefix, "ref_files", "job_spec.json")) as f:
                ref = json.load(f)
            test = json.loads(json.dumps(job.get_spec()))
            self.assertDictEqual(ref, test)

    def test_substitution(self):
        # substitute atom and fuse ring
        config = Config(os.path.join(prefix, "test_files", "substitution.ini"))
        geom = Geometry.from_string(config["Geometry"]["structure"])
        job = Job(geom, config, set_root=False)
        test = job.structure
        ref = Geometry(
            os.path.join(prefix, "ref_files", "substitution_with_rings.xyz")
        )
        self.assertTrue(validate(test, ref))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestJob("test_spec"))
    # suite.addTest(TestJob("test_substitution"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
