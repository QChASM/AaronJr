#! /usr/bin/env python3
import os
import unittest

from Aaron.aaron_init import AaronInit
from Aaron.test import prefix
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer



class TestAaronInit(TestWithTimer):
    def setUp(self):
        super().setUp()
        self.init = AaronInit(
            os.path.join(prefix, "ref_files/simple.ini"), quiet=True
        )
        self.test_geom = Geometry(
            os.path.join(prefix, "ref_files/S_binap-TMEN_Ph.R.ts1.Cf1.2.log")
        )
        self.test_geom.comment = "F:44-8-3-4"
        self.test_geom.parse_comment()

    def test_read_aaron_input(self):
        def make_str(obj):
            rv = []
            for key in sorted(obj):
                value = obj[key]
                if isinstance(value, dict):
                    value = ", ".join(
                        [
                            "=".join(
                                [
                                    str(k) if k else "additional_opts",
                                    str(
                                        sorted(v.items())
                                        if isinstance(v, dict)
                                        else str(v)
                                    ),
                                ]
                            )
                            for k, v in sorted(value.items())
                        ]
                    )
                    value = "({})".format(value)
                if value:
                    rv += ["\t{}={}".format(key, value)]
                else:
                    rv += ["\t{}".format(key)]
            return ",\n".join(rv) + "\n"

        test_str = ""
        for d in sorted(self.init.__dict__):
            name, value = d, self.init.__dict__[d]
            if name == "theory":
                for step, theory in sorted(value.by_step.items()):
                    test_str += "{} ({}): \n{}".format(
                        name, step, make_str(theory.__dict__)
                    )
            elif name == "reaction":
                test_str += "{}: \n{}".format(name, make_str(value.__dict__))
            elif isinstance(value, dict):
                test_str += "{}: \n{}".format(name, make_str(value))
            else:
                test_str += name + ": " + str(value) + "\n"
        with open(os.path.join(prefix, "ref_files/test_init.txt")) as f:
            self.assertEqual(test_str, f.read())

    def test_footer(self):
        test_str = ""
        for step in range(1, 5):
            if float(step) in self.init.theory.by_step:
                theory = self.init.theory.by_step[float(step)]
            else:
                theory = self.init.theory.by_step[0.0]
            test_str += theory.make_footer(self.test_geom, step=float(step))
            test_str += "=" * 12 + "\n"
        with open(os.path.join(prefix, "ref_files/footer.txt")) as f:
            self.assertEqual(test_str, f.read())

    def test_header(self):
        test_str = ""
        for step in range(1, 5):
            if float(step) in self.init.theory.by_step:
                theory = self.init.theory.by_step[float(step)]
            else:
                theory = self.init.theory.by_step[0.0]
            test_str += theory.make_header(self.test_geom, step)
            test_str += "=" * 12 + "\n"
        with open("tmp.txt", "w") as f:
            f.write(test_str)
        with open(os.path.join(prefix, "ref_files/header.txt")) as f:
            self.assertEqual(test_str, f.read())

    def test_write_com(self):
        for step in range(1, 5):
            self.init.theory.write_com(self.test_geom, float(step))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestAaronInit("test_read_aaron_input"))
    suite.addTest(TestAaronInit("test_footer"))
    suite.addTest(TestAaronInit("test_header"))
    suite.addTest(TestAaronInit("test_write_com"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
