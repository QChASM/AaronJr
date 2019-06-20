#!/usr/bin/env python

import json
import unittest

import numpy as np

from Aaron.aaron_init import AaronInit
from Aaron.job import Job
from Aaron.json_extension import JSONDecoder, JSONEncoder
from AaronTools.test import TestWithTimer, prefix

prefix = prefix.replace("AaronTools", "Aaron", 1)


class TestJSON(TestWithTimer):
    init = AaronInit(prefix + "test_files/simple.in")
    init.reaction.set_up()
    cat_data = init.reaction.catalyst_data
    theory = init.theory
    job = Job(cat_data[0], theory)

    def json_tester(
        self, obj, ref_json, equality_function, as_iter=False, **kwargs
    ):
        # test to json
        test = json.dumps(obj, cls=JSONEncoder, indent=2)
        with open(ref_json) as f:
            self.assertEqual(test, f.read())
        # test from json
        with open(ref_json) as f:
            test = json.load(f, cls=JSONDecoder)
        if as_iter:
            for r, t in zip(obj, test):
                equality_function(r, t, **kwargs)
        else:
            equality_function(obj, test, **kwargs)

    def atom_equal(self, ref, test, skip=[]):
        for key, val in ref.__dict__.items():
            if key in skip + ["_rank"]:
                continue
            try:
                if key == "coords":
                    self.assertTrue(
                        np.linalg.norm(ref.coords - test.coords) < 10 ** -15
                    )
                elif key in ["connected", "constraint"]:
                    self.assertEqual(
                        len(ref.__dict__[key]), len(test.__dict__[key])
                    )
                    for r, t in zip(
                        sorted(ref.__dict__[key]), sorted(test.__dict__[key])
                    ):
                        self.atom_equal(
                            r, t, skip=skip + ["connected", "constraint"]
                        )
                elif key == "tags":
                    ref_set = set(
                        [t for t in ref.tags if not t.startswith("sub-")]
                    )
                    test_set = set(
                        [t for t in test.tags if not t.startswith("sub-")]
                    )
                    if len(test_set) > len(ref_set):
                        tmp = test_set - ref_set
                        for t in tmp:
                            if t not in [
                                "substrate",
                                "backbone",
                                "ligand",
                                "key",
                            ]:
                                test_set.remove(t)
                    self.assertSetEqual(ref_set, test_set)
                else:
                    self.assertEqual(ref.__dict__[key], test.__dict__[key])
            except AssertionError as e:
                print(ref, ref.__dict__[key])
                print(test, test.__dict__[key])
                raise AssertionError("{}\t(key={})".format(str(e), key))

    def geom_equal(self, ref, test, skip=[]):
        if "comment" not in skip:
            self.assertEqual(ref.comment, test.comment)
        for a in ref.atoms:
            b = test.find_exact(a.name)[0]
            self.atom_equal(a, b, skip)
        else:
            self.assertEqual(len(ref.atoms), len(test.atoms))

    def sub_equal(self, ref, test):
        self.assertEqual(ref.name, test.name)
        for a, b in zip(ref.atoms, test.atoms):
            self.atom_equal(a, b)
        self.assertEqual(ref.end, test.end)
        self.assertEqual(ref.conf_num, test.conf_num)
        self.assertEqual(ref.conf_angle, test.conf_angle)

    def component_equal(self, ref, test):
        self.geom_equal(ref, test)
        self.assertEqual(len(ref.substituents), len(test.substituents))
        for r, t in zip(ref.substituents, test.substituents):
            self.geom_equal(r, t, skip=["comment"])
        self.assertEqual(len(ref.backbone), len(test.backbone))
        for r, t in zip(ref.backbone, test.backbone):
            self.atom_equal(r, t)
        self.assertEqual(len(ref.key_atoms), len(test.key_atoms))
        for r, t in zip(ref.key_atoms, test.key_atoms):
            self.atom_equal(r, t)

    def catalyst_equal(self, ref, test, skip=[]):
        self.geom_equal(ref, test, skip)
        for r, t in zip(ref.center, test.center):
            self.atom_equal(r, t, skip)
        for key in ref.components:
            for r, t in zip(ref.components[key], test.components[key]):
                self.component_equal(r, t)
        self.assertEqual(len(ref.conf_spec), len(test.conf_spec))
        try:
            for key, val in ref.conf_spec.items():
                test_key = test.find_exact(key.name)[0]
                self.assertTrue(test_key in test.conf_spec)
                self.assertListEqual(val, test.conf_spec[test_key])
        except AssertionError as e:
            raise AssertionError("{}\t(key={})".format(str(e), key))

    def comp_out_equal(self, ref, test):
        keys = [
            "geometry",
            "opts",
            "frequency",
            "archive",
            "gradient",
            "E_ZPVE",
            "ZPVE",
            "energy",
            "enthalpy",
            "free_energy",
            "grimme_g",
            "charge",
            "multiplicity",
            "mass",
            "temperature",
            "rotational_temperature",
            "rotational_symmetry_number",
            "error",
            "error_msg",
            "finished",
        ]
        for key in keys:
            rval = ref.__dict__[key]
            tval = ref.__dict__[key]
            if key == "geometry":
                self.geom_equal(rval, tval)
            elif key == "opts":
                if rval is None:
                    continue
                for r, t in zip(rval, tval):
                    self.geom_equal(r, t)
            elif key == "frequency":
                self.freq_equal(rval, tval)
            elif key == "gradient":
                self.assertDictEqual(rval, tval)
            elif key == "rotational_temperature":
                self.assertListEqual(rval, tval)
            else:
                self.assertEqual(rval, tval)

    def freq_equal(self, ref, test):
        for r, t in zip(ref.data, test.data):
            self.assertEqual(r.frequency, t.frequency)
            self.assertEqual(r.intensity, t.intensity)
            self.assertTrue(np.linalg.norm(r.vector - t.vector) < 10 ** -12)
        self.assertListEqual(
            ref.imaginary_frequencies, test.imaginary_frequencies
        )
        self.assertListEqual(ref.real_frequencies, test.real_frequencies)
        self.assertEqual(ref.is_TS, test.is_TS)
        self.assertEqual(
            len(ref.by_frequency.keys()), len(test.by_frequency.keys())
        )
        for key in ref.by_frequency.keys():
            self.assertTrue(key in test.by_frequency)
            rv = ref.by_frequency[key]
            tv = test.by_frequency[key]
            self.assertEqual(rv["intensity"], tv["intensity"])
            self.assertTrue(
                np.linalg.norm(rv["vector"] - tv["vector"]) < 10 ** -12
            )

    def theory_equal(self, ref, test):
        for key, val in ref.__dict__.items():
            self.assertTrue(key in test.__dict__)
            self.assertEqual(val, test.__dict__[key])
        return True

    def job_equal(self, ref, test):
        pass

    def cat_data_equal(self, ref, test):
        for key, val in ref.__dict__.items():
            if key == "catalyst":
                if ref.catalyst is None:
                    continue
                self.catalyst_equal(
                    ref.catalyst, test.catalyst, skip=["comment", "_rank"]
                )
                continue
            try:
                self.assertEqual(val, test.__dict__[key])
            except AssertionError as e:
                raise AssertionError("{}\t(key={})".format(str(e), key))

    def test_theory(self):
        test = json.dumps(self.theory, cls=JSONEncoder)
        test = json.loads(test, cls=JSONDecoder)
        self.theory_equal(self.theory, test)

    def test_catalyst_metadata(self):
        for data in self.cat_data:
            test = json.dumps(data, cls=JSONEncoder)
            test = json.loads(test, cls=JSONDecoder)
            if data.catalyst is not None:
                data.catalyst.write("ref")
                test.catalyst.write("test")
            self.cat_data_equal(data, test)

    def test_job(self):
        test = json.dumps(self.job, cls=JSONEncoder)


if __name__ == "__main__":
    unittest.main()
