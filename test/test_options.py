#!/usr/bin/env python3
import unittest
import os

from AaronTools.test import prefix, TestWithTimer
from Aaron.aaron_init import AaronInit


prefix = prefix.replace('AaronTools', 'Aaron', 1)


class TestReaction(TestWithTimer):
    def setUp(self):
        super().setUp()
        self.init = AaronInit(prefix + "test_files/S-binap.in")

    def test_get_templates(self):
        reaction = self.init.reaction
        reaction.get_templates()
        test_str = ''
        for t in sorted(reaction.template_XYZ):
            test_str += os.path.join(*t)
            test_str += ' {}\n'.format(str(t))
        with open('tmp.txt', 'w') as f:
            f.write(test_str)
        with open(prefix + "ref_files/templates.txt") as f:
            self.assertEqual(test_str, f.read())

    def test_generate_structures(self):
        self.init.reaction.generate_structures()
        test_str = ''
        for cat in self.init.reaction.catalyst_data:
            test_str += '{} {} {}\n'.format(cat.template_file,
                                            cat.ligand_change,
                                            cat.substrate_change)
        with open(prefix + 'ref_files/gen_struct.txt') as f:
            self.assertEqual(test_str, f.read())

    def test_make_directories(self):
        top_dir = prefix + 'test_files/make_dirs'
        try:
            os.makedirs(top_dir, mode=0o755)
        except FileExistsError:
            pass
        self.init.reaction.make_directories(top_dir)
        for root, dirs, files in os.walk(prefix + 'ref_files/make_dirs'):
            for d in dirs:
                test_path = os.path.join(root, d)
                test_path.replace('ref_files', 'test_files')
                self.assertTrue(os.access(test_path, os.W_OK))
        return
        for root, dirs, files in os.walk(top_dir, topdown=False):
            for d in dirs:
                try:
                    os.removedirs(os.path.join(root, d))
                except FileNotFoundError:
                    pass


if __name__ == '__main__':
    unittest.main()
