#!/usr/bin/env python3
import os
import unittest

from Aaron.job import Job
from AaronTools.config import Config
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, validate

prefix = prefix.replace("AaronTools", "Aaron", 1)


class TestJob(TestWithTimer):
    def test_substitution(self):
        # substitute atom and fuse ring
        config = Config(os.path.join(prefix, "test_files/substitution.ini"))
        geom = Geometry.from_string(config["Geometry"]["structure"])
        job = Job(geom, config, set_root=False)
        test = job.structure
        ref = Geometry(
            os.path.join(prefix, "ref_files/substitution_with_rings.xyz")
        )
        self.assertTrue(validate(test, ref))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestJob("test_substitution"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
