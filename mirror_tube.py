from mirror import Mirror, make_iso_mirrors, make_mirror_box

import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from geom import lineseg_dist, CLOSEST_MIRROR



def _same_pt(p0, p1):
    return np.all(p0 == p1)

class MirrorTube(object):
    """
    Ordered set of mirrors forming a tube (nonconvex prism) around the optical axis.
    """   
    def __init__(self, mirrors):
        self.mirrors = self._sort_and_check(mirrors)
        self._check_contains_center()
        
    def _sort_and_check(self, mirrors):
        """
        Make sure the endpoints line up in order and form a loop.
        """
        m_list = [mirrors.pop(0)]
        while len(mirrors) > 0:
            next_m = [m for m in mirrors if _same_pt(m.p0, m_list[-1].p1)]
            if len(next_m) == 0:
                raise ValueError("Mirrors do not form a loop.")
            m_list.append(next_m[0])
            mirrors.remove(next_m[0])
        return m_list
    
    def _check_contains_center(self):
        """
        Make sure the mirrors form a polygon that contains the origin, otherwise it won't look right.
        Make sure none is too close to the origin.
        """
        # Check contains center:
        polygon_verts = np.array([m.p0[:2] for m in self.mirrors])
        path = mpath.Path(polygon_verts)
        if not path.contains_point((0, 0)):
            raise ValueError("Mirrors do not form a polygon that contains the origin.")
        
        # Check not too close to center:
        for m in self.mirrors:
            if lineseg_dist(m.p0, m.p1, (0, 0, 0)) < CLOSEST_MIRROR:
                raise ValueError("Mirrors are too close to the origin.")
        
    @staticmethod
    def make_isoceles(iso_angle_deg = 30, radius=.6, rotation_rad=0):
        mirrors = make_iso_mirrors(iso_angle_deg, radius, rotation_rad)
        return MirrorTube(mirrors)
    
    @staticmethod
    def make_box(radius = .6, rotation_rad=0.0):
        mirrors = make_mirror_box(radius, rotation_rad)
        return MirrorTube(mirrors)

    @staticmethod
    def make_reg_n_gon(n, radius=.9, rotation_rad=0):
        if n<3:
            raise ValueError("Must have at least 3 mirrors.")
        angle = 360.0 / n
        points = []
        for i in range(n):
            theta = np.radians(rotation_rad + i * angle)
            points.append((radius * np.cos(theta), radius * np.sin(theta)))
        mirrors = []
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            mirrors.append(Mirror(p0, p1))
        return MirrorTube(mirrors)