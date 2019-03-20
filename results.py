#!/usr/bin/env python
import os
import numpy as np

from AaronTools.const import UNIT, PHYSICAL
from AaronTools.comp_output import CompOutput
from Aaron.aaron_init import AaronInit


THERMO = 'energy', 'enthalpy', 'free_energy', 'grimme_g'


class Data:
    def __init__(self, log, directory):
        info = log.split('.')[:-2]
        if len(info) == 4:
            self.name, self.sel, self.ts, self.cf = info
        else:
            self.name, self.ts, self.cf = info
            self.sel = None
        self.output = CompOutput(os.path.join(directory, log))
        for key in THERMO:
            if key not in self.output.__dict__:
                continue
            self.__dict__[key] = self.output.__dict__[key]


class Results:
    def __init__(self, aaron_in):
        directory = os.path.dirname(aaron_in)
        self.aaron = AaronInit(aaron_in)
        self.data = {}
        self.min = {}
        for dirpath, dirnames, filenames in os.walk(directory):
            log = None
            for f in filenames:
                if f.endswith('.4.log'):
                    log = f
            if not log:
                continue
            tmp = Data(log, dirpath)
            if tmp.name in self.data:
                self.data[tmp.name] += [tmp]
            else:
                self.data[tmp.name] = [tmp]
                self.min[tmp.name] = {key: None for key in THERMO}
            for key in THERMO:
                if tmp.__dict__[key] is None:
                    continue
                if (self.min[tmp.name][key] is None
                        or self.min[tmp.name][key] > tmp.__dict__[key]):
                    self.min[tmp.name][key] = tmp.__dict__[key]

        for name in self.data:
            self.data[name] = self.sort_data(self.data[name])

    def sort_data(self, data):
        rv = {}
        for d in data:
            if d.sel not in rv:
                rv[d.sel] = {}
            if d.ts not in rv[d.sel]:
                rv[d.sel][d.ts] = {}
            rv[d.sel][d.ts][d.cf] = d
        return rv

    def print_ee(self, ofile=None):
        all_output = ''
        per_str = '{:<8s}' + ' {:>6.1f}% '*4 + '\n'
        num_str = '{:<8s}' + ' {:>7.1f} '*4 + '\n'
        head_str = '{:>8s} '*5 + '\n'
        head_str = head_str.format('', 'E  ', 'H  ', 'G(RRHO)', 'G(qRRHO)')

        def print_ts(data):
            rv = ''
            bw_thermo = {key: 0 for key in THERMO}
            temperature = None
            for cf in sorted(data.keys()):
                d = data[cf]
                thermo = []
                bw = self.boltzmann_weight(d, name=d.name)
                for key in THERMO:
                    thermo += [d.__dict__[key] - self.min[d.name][key]]
                    bw_thermo[key] += bw[key]
                thermo = [t * UNIT.HART_TO_KCAL for t in thermo]
                rv += num_str.format('    ' + cf, *thermo)
            else:
                if d.output.temperature is None:
                    temperature = 298
                else:
                    temperature = d.output.temperature

            tmp = []
            for key in THERMO:
                val = np.log(bw_thermo[key])
                tmp += [-PHYSICAL.BOLTZMANN * temperature * val]
            rv = num_str.format('  ' + d.ts, *tmp) + rv
            return rv, bw_thermo

        def print_sel(data):
            rv = ''
            bw_thermo = {key: 0 for key in THERMO}
            for ts in data:
                s, bw = print_ts(data[ts])
                rv += s
                for key in bw:
                    bw_thermo[key] += bw[key]
            return rv, bw_thermo

        for name, data in self.data.items():
            output = ''
            bw_thermo = {}
            for sel in data:
                s, bw = print_sel(data[sel])
                output += sel + '\n' + s
                bw_thermo[sel] = bw

            ee = []
            sel = list(data.keys())[0]
            not_sel = [s for s in data.keys() if s != sel]
            for key in THERMO:
                val = bw_thermo[sel][key]
                val -= np.sum([bw_thermo[s][key] for s in not_sel])
                val /= np.sum([bw_thermo[s][key] for s in bw_thermo])
                ee += [val * 100]
            header = '{}: Relative thermochemistry (kcal/mol)\n'.format(name)
            header += head_str
            header += per_str.format('%ee {}'.format(sel), *ee)
            all_output += header + output + "\n"
        print(all_output)

    def boltzmann_weight(self, data, temperature=None, name=None):
        """
        :data: an entry in self.data to bolzmann weight or a dictionary
        :temperature: required if :data: is a dictionary
        """
        rv = {}
        if isinstance(data, Data):
            temperature = data.output.temperature
            data = data.__dict__
        if temperature is None:
            temperature = 298
        for key in THERMO:
            if data[key] is None:
                data[key] = np.nan
            if name is None:
                val = data[key]
            else:
                val = data[key] - self.min[name][key]
            val *= UNIT.HART_TO_KCAL
            val /= PHYSICAL.BOLTZMANN * temperature
            if key not in rv:
                rv[key] = np.exp(-val)
            else:
                rv[key] += np.exp(-val)
        return rv

    def follow_save(self):
        key_str = '{}.{}.int_{}'
        for name in self.data:
            for sel in self.data[name]:
                sel = self.data[name][sel]
                ts = sel[sorted(sel.keys())[0]]
                cf = ts[sorted(ts.keys())[0]]
                output = cf.output
                for direction in ['r', 'p']:
                    key = key_str.format(name, cf.sel, direction)
                    if direction == 'r':
                        followed = output.follow(reverse=True)
                    else:
                        followed = output.follow()
                    followed.write(style='com', name=key,
                                   options=self.aaron.comp_opts,
                                   job_type='min', freq='noraman')
        return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('aaron_input')
    parser.add_argument(
        '--follow', '-f', action='store_true',
        help="""
            Follow imaginary mode for each selectivity in the +/- direction.
        """)

    args = parser.parse_args()
    res = Results(args.aaron_input)
    res.print_ee()
    if args.follow:
        res.follow_save()
