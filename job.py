from AaronTools.const import CONNECTIVITY_THRESHOLD
from AaronTools.geometry import Geometry


class Job:
    """
    Attributes:
        catalyst
        step
        cycle
        path
        status
    """

    def __init__(self, catalyst=None, reaction=None):
        self.catalyst = catalyst
        self.reaction = reaction
        self.step = 1
        self.cycle = 1
        self.name = ''
        self.path = ''
        self.status = ''

        if self.catalyst:
            tmp = self.catalyst.name.split('/')
            self.path = '/'.join(tmp[:-1])
            tmp = tmp[-1].split('.')
            self.name = '.'.join(tmp[:-1])
            self.step = int(tmp[-1])

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
        compare = Geometry(fname='{}/{}.2.com'.format(self.path, self.name))
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
