#!/usr/bin/env python3
import os
import unittest

from Aaron.aaron_init import AaronInit
from AaronTools.test import TestWithTimer, prefix
from AaronTools.utils import utils

prefix = prefix.replace("AaronTools", "Aaron", 1)


class TestReaction(TestWithTimer):
    def setUp(self):
        super().setUp()
        self.init = AaronInit(prefix + "test_files/S-binap.in")
        self.simple = AaronInit(prefix + "test_files/simple.in")

    def test_get_templates(self):
        reaction = self.init.reaction
        reaction.get_templates()
        test_str = ""
        for t in sorted(reaction.template_XYZ):
            test_str += os.path.join(*t)
            test_str += " {}\n".format(str(t))
        with open(prefix + "ref_files/templates.txt") as f:
            self.assertEqual(test_str, f.read())

    def test_generate_structure_data(self):
        self.init.reaction.generate_structure_data()
        test_str = ""
        for cat in sorted(self.init.reaction.catalyst_data):
            test_str += "{} {} {}\n".format(
                cat.template_file, cat.ligand_change, cat.substrate_change
            )
        with open(prefix + "ref_files/gen_struct.txt") as f:
            self.assertEqual(test_str, f.read())

    def test_make_conformers(self):
        """
        Tests make_directories, generate_structure,
        uses Catalyst().next_conformer() and Geometry().write()
        """
        top_dir = prefix + "test_files/make_dirs"
        self.init.reaction.make_directories(top_dir)

        big_count = 0
        big_n = len(self.init.reaction.catalyst_data)
        test_str = ""
        for cat_data in self.init.reaction.catalyst_data:
            utils.progress_bar(big_count, big_n)
            big_count += 1
            test_str += "{}: {}\n".format(big_count, cat_data.template_file)
            test_str += "{}, {}, {}\n".format(
                cat_data.reaction_type,
                cat_data.template_name,
                cat_data.ts_number,
            )
            test_str += "lig: {}\n".format(cat_data.ligand_change)
            test_str += "sub: {}\n".format(cat_data.substrate_change)
            # make catalyst
            cat_data.generate_structure()
            # make conformers
            while True:
                conf_spec = cat_data.catalyst.conf_spec
                # get conf_directory
                next_dir_num = cat_data._get_cf_dir_num()
                conf_dir = os.path.join(
                    cat_data.ts_directory, "Cf{}".format(next_dir_num)
                )
                test_str += "Cf #{}: {}\n".format(next_dir_num, conf_dir)
                # make conf_dir
                try:
                    os.makedirs(conf_dir, mode=0o755)
                except FileExistsError:
                    pass
                for a, i in sorted(conf_spec.items()):
                    sub = cat_data.catalyst.find_substituent(a)
                    test_str += "\t{} ({}): cf {}\n".format(
                        a.name, sub.name, i[0]
                    )
                if not cat_data.catalyst.next_conformer():
                    break
            test_str += "\n"
        utils.clean_progress_bar()

        with open("tmp.txt", "w") as f:
            f.write(test_str)
        with open(prefix + "ref_files/make_dirs.txt") as f:
            self.assertEqual(test_str, f.read())

        for root, dirs, files in os.walk(prefix + "ref_files/make_dirs"):
            for d in dirs:
                test_path = os.path.join(root, d)
                test_path.replace("ref_files", "test_files")
                self.assertTrue(os.access(test_path, os.W_OK))

        for root, dirs, files in os.walk(top_dir, topdown=False):
            for d in dirs:
                try:
                    os.removedirs(os.path.join(root, d))
                except FileNotFoundError:
                    pass

    def test_brute_force(self):
        reaction = self.simple.reaction
        reaction.make_conformers(top_dir=prefix + "test_files/brute_force")


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestReaction("test_get_templates"))
    # suite.addTest(TestReaction("test_generate_structure_data"))
    # suite.addTest(TestReaction("test_make_conformers"))
    suite.addTest(TestReaction("test_brute_force"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
