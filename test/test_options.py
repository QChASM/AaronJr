#!/usr/bin/env python3
import os
import unittest

from Aaron.aaron_init import AaronInit
from AaronTools.test import TestWithTimer, prefix

prefix = prefix.replace("AaronTools", "Aaron", 1)


class TestReaction(TestWithTimer):
    def setUp(self):
        super().setUp()
        self.init = AaronInit(
            os.path.join(prefix, "ref_files/S-binap.in"), quiet=True
        )
        self.reaction = self.init.reaction
        self.theory = self.init.theory

    def test_get_templates(self):
        reaction = self.init.reaction
        reaction.get_templates()
        test_str = ""
        for t in sorted(reaction.template_XYZ):
            test_str += os.path.join(*t)
            test_str += " {}\n".format(str(t))
        with open(os.path.join(prefix, "ref_files/templates.txt")) as f:
            self.assertEqual(test_str, f.read())

    def test_generate_structure_data(self):
        self.reaction.generate_structure_data(self.theory.top_dir)
        test_str = ""
        for cat in sorted(self.reaction.catalyst_data):
            test_str += "{} {} {}\n".format(
                cat.template_file, cat.ligand_change, cat.substrate_change
            )
        with open("tmp.txt", "w") as f:
            f.write(test_str)
        with open(os.path.join(prefix, "ref_files/gen_struct.txt")) as f:
            self.assertEqual(test_str, f.read())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestReaction("test_get_templates"))
    suite.addTest(TestReaction("test_generate_structure_data"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
