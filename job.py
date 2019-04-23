import os
import glob

from AaronTools.const import CONNECTIVITY_THRESHOLD
from AaronTools.geometry import Geometry
from AaronTools.comp_output import CompOutput
from AaronTools.job_control import JobControl
from AaronTools.fileIO import str2step


class Job:
    """
    Attributes:
        catalyst
        conformers
        step
        cycle
        path
        output
        status
    """
    name_template = "{}/{}.{}.{}"

    def __init__(self, catalyst=None, reaction=None):
        self.catalyst = catalyst
        self.reaction = reaction
        self.step = 1.0
        self.cycle = 1
        self.name = ''
        self.path = ''
        self.output = None
        self.status = ''

        if self.catalyst:
            tmp = self.catalyst.name.split('/')
            self.path = '/'.join(tmp[:-1])
            tmp = tmp[-1].split('.')
            self.name = '.'.join(tmp[:-1])
            self.step = float(tmp[-1])

        self.name_template = Job.name_template.format(
            self.path, self.name, '{}', '{}')

    def examine_connectivity(self):
        """
        Determines formed/broken bonds relative to step 2 com file
        """
        def within_thresh(i, j):
            dist = self.catalyst.atoms[i].dist(self.catalyst.atoms[j])
            dist -= compare.atoms[i].dist(compare.atoms[j])
            dist = abs(dist)
            if self.reaction and self.reaction.con_thresh:
                return dist < self.reaction.con_thresh
            else:
                return dist < CONNECTIVITY_THRESHOLD

        formed = []
        broken = []
        if self.step < 2:
            return formed, broken
        compare = Geometry('{}/{}.2.com'.format(self.path, self.name))
        connectivities = zip(self.catalyst.connectivity(),
                             compare.connectivity())
        for i, con in enumerate(connectivities):
            a = set(con[0])
            b = set(con[1])
            if len(a) > len(b):
                tmp = a - b
                for j in a - b:
                    if within_thresh(i, j):
                        tmp.remove(j)
                if len(tmp) != 0:
                    formed += [(i, sorted(tmp))]
            if len(b) > len(a):
                tmp = b - a
                for j in b - a:
                    if within_thresh(i, j):
                        tmp.remove(j)
                if len(tmp) != 0:
                    broken += [(i, sorted(tmp))]
        return formed, broken

    def check_step(self):
        """
        Determines the status of a jobs of the same path and name as self
        """
        if self.status == 'finished' and self.output:
            return

        for com_name in glob.glob(self.name_template.format('*', 'com')):
            step = com_name.split('.')[-2]
            log_name = self.name_template.format(step, 'log')

            step = str2step(step)
            if os.access(log_name, os.R_OK):
                self.output = CompOutput(log_name, get_all=False)
                self.step = step
                if self.output.error:
                    self.status = 'failed'
                elif self.output.finished:
                    self.status = 'done'
                    self.catalyst.update_geometry(self.output.geometry)
                elif JobControl.findJob(self.path):
                    self.status = 'running'
                    self.catalyst.update_geometry(self.output.geometry)
                break
            elif os.access(com_name, os.R_OK):
                self.step = step
                if JobControl.findJob(self.path):
                    self.status = 'pending'
                else:
                    self.status = '2submit'
