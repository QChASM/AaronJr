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
            for key in sorted(obj.__dict__):
                value = obj.__dict__[key]
                if key == 'theory':
                    tmp = "{}=(".format(key)
                    for k in sorted(value.__dict__):
                        tmp += "{}={}, ".format(k, value.__dict__[k])
                    rv += [tmp[:-2] + ")"]
                else:
                    rv += ["{}={}".format(key, value)]
            return ", ".join(rv) + '\n'

        test_str = ''
        for d in sorted(self.init.__dict__):
            name, value = d, self.init.__dict__[d]
            if name == 'comp_opts':
                for step in sorted(value.keys()):
                    val = value[step]
                    test_str += "{} ({}): {}".format(name, step, make_str(val))
            elif name in ['cluster_opts', 'reaction']:
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
        geom = Geometry('test_files/S_binap-TMEN_Ph.R.ts1.Cf1.4.log')
        test_str = ''
        for step in sorted(self.init.comp_opts):
            compopt = self.init.comp_opts[step]
            test_str += compopt.theory.make_footer(geom) + '\n'
            test_str += "\n" + "="*12 + "\n"
        with open('tmp.txt', 'w') as f:
            f.write(test_str)
        with open('ref_files/footer.txt') as f:
            self.assertEqual(test_str, f.read())


if __name__ == '__main__':
    unittest.main()
