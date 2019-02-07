#! /usr/bin/env python3
import unittest

from Aaron.aaron_init import AaronInit
from AaronTools.geometry import Geometry


class TestAaronInit(unittest.TestCase):
    def setUp(self):
        """
        initializes aaron instance
        """
        self.init = AaronInit('test_files/S-binap.in')

    def test_read_aaron_input(self):
        def make_str(obj):
            rv = []
            for k in sorted(obj.__dict__):
                if k == 'theory':
                    continue
                rv += [str((k, obj.__dict__[k]))]
            return ", ".join(rv) + '\n'
        test_str = ''
        for d in sorted(self.init.__dict__):
            name, value = d, self.init.__dict__[d]
            if name in ['comp_opts', 'cluster_opts', 'reaction']:
                test_str += name + ": " + make_str(value)
            elif name == 'theory':
                for t in value:
                    tmp = t
                    if t == '':
                        tmp = 'normal'
                    tmp = 'theory (' + tmp + ')'
                    test_str += tmp + ": " + make_str(value[t])
            else:
                test_str += name + ": " + str(value) + '\n'

        with open('ref_files/test_init.txt') as f:
            self.assertEqual(test_str, f.read())

    def test_footer(self):
        geom = Geometry('test_files/wS_binap-RR_dpen_napth.R.ts1.Cf1.4.log')
        test_str = ''
        for t in self.init.theory:
            test_str += self.init.theory[t].make_footer(geom) + '\n'
            test_str += "\n" + "="*12 + "\n"
        with open('ref_files/footer.txt') as f:
            self.assertEqual(test_str, f.read())


if __name__ == '__main__':
    unittest.main()
