#! /usr/bin/env python3
import unittest

from AaronTools.catalyst import Catalyst
from Aaron.aaron_init import AaronInit
from Aaron.job import Job


class TestJob(unittest.TestCase):
    init = AaronInit({'input_file': 'test_files/S-binap.in'})

    cat1 = Catalyst(
        'test_files/wS_binap/wS_binap-RR_dpen_napth/R/ts1/Cf1/'
        + 'wS_binap-RR_dpen_napth.R.ts1.Cf1.1.log')
    cat2 = Catalyst(
        'test_files/wS_binap/wS_binap-RR_dpen_napth/R/ts1/Cf1/'
        + 'wS_binap-RR_dpen_napth.R.ts1.Cf1.2.log')
    cat3 = Catalyst(
        'test_files/wS_binap/wS_binap-RR_dpen_napth/R/ts1/Cf1/'
        + 'wS_binap-RR_dpen_napth.R.ts1.Cf1.3.log')
    cat4 = Catalyst(
        'test_files/wS_binap/wS_binap-RR_dpen_napth/R/ts1/Cf1/'
        + 'wS_binap-RR_dpen_napth.R.ts1.Cf1.4.log')

    def test_init(self):
        job = Job(TestJob.cat2)
        self.assertEqual(job.path, 'test_files/wS_binap/wS_binap-RR_dpen_napth/R/ts1/Cf1')
        self.assertEqual(job.name, 'wS_binap-RR_dpen_napth.R.ts1.Cf1')
        self.assertEqual(job.step, 2)

    def test_examine_connectivity(self):
        job = Job(TestJob.cat2)

        # no change
        formed, broken = job.examine_connectivity()
        self.assertTrue(len(formed) == 0 and len(broken) == 0)

        # broken
        job.catalyst.change_distance('116', '113', dist=3, adjust=True, fix=2)
        formed, broken = job.examine_connectivity()
        self.assertTrue(len(broken) == 1 and len(formed) == 0)
        job.catalyst.change_distance('116', '113', dist=-3, adjust=True, fix=2)

        # formed
        job.catalyst.change_distance('8', '58', dist=-1, adjust=True, fix=2)
        formed, broken = job.examine_connectivity()
        self.assertTrue(len(formed) == 1 and len(broken) == 0)
        job.catalyst.change_distance('8', '58', dist=1, adjust=True, fix=2)

        # formed and broken
        job.catalyst.change_distance('14', '4', dist=-1, adjust=True, fix=2, as_group=False)
        formed, broken = job.examine_connectivity()
        self.assertTrue(len(formed) == 1 and len(broken) == 1)
        job.catalyst.change_distance('14', '4', dist=1, adjust=True, fix=2, as_group=False)


if __name__ == '__main__':
    unittest.main()
