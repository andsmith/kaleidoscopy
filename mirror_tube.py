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
    def __init__(self, mirrors, name=None, require_center_containment=True, require_min_mirror_dist=True):
        self.mirrors = self._sort_and_check(mirrors)
        self._require_center_containment = require_center_containment
        self._require_min_mirror_dist = require_min_mirror_dist
        self._check_contains_center()
        self.name = name
        
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
        Optionally enforce center containment and minimum mirror distance constraints.
        """
        if self._require_center_containment:
            polygon_verts = np.array([m.p0[:2] for m in self.mirrors])
            path = mpath.Path(polygon_verts)
            if not path.contains_point((0, 0)):
                raise ValueError("Mirrors do not form a polygon that contains the origin.")

        if self._require_min_mirror_dist:
            for m in self.mirrors:
                if lineseg_dist(m.p0, m.p1, (0, 0, 0)) < CLOSEST_MIRROR:
                    raise ValueError("Mirrors are too close to the origin.")
        
    @staticmethod
    def make_isoceles(iso_angle_deg = 30, radius=.6, rotation_rad=0):
        mirrors = make_iso_mirrors(iso_angle_deg, radius, rotation_rad)
        name="Isoceles (%.1f degrees)" % iso_angle_deg if iso_angle_deg != 60 else "Equilateral Triangle"
        return MirrorTube(mirrors, name=name)
    
    @staticmethod
    def make_box(radius = .6, rotation_rad=0.0):
        mirrors = make_mirror_box(radius, rotation_rad)
        return MirrorTube(mirrors, name="Square")

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
            
        name = "%d-gon" % n 
        if n == 3: name = "Triangle"
        elif n==4: name = "Square"
        elif n==5: name = "Pentagon"
        elif n==6: name = "Hexagon"
        elif n==8: name = "Octagon"
        elif n==10: name = "Decagon"
        elif n==12: name = "Dodecagon"
        return MirrorTube(mirrors, name=name)
    
    @staticmethod
    def make_n_star(n_points, rad_outer=.667, rad_inner=.333, rotation_rad=0):
        if n_points < 2:
            raise ValueError("Must have at least 2 points for a star.")
        angle = 360.0 / n_points
        points = []
        for i in range(n_points):
            theta = np.radians(rotation_rad + i * angle)
            points.append((rad_outer * np.cos(theta), rad_outer * np.sin(theta)))
            theta_inner = np.radians(rotation_rad + i * angle + angle / 2)
            points.append((rad_inner * np.cos(theta_inner), rad_inner * np.sin(theta_inner)))
        mirrors = []
        for i in range(len(points)):
            p0 = points[i]
            p1 = points[(i + 1) % len(points)]
            mirrors.append(Mirror(p0, p1))
        return MirrorTube(mirrors, name="%d-star" % n_points)