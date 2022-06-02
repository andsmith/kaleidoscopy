"""
Geometry utilities, based on scikit-geometry.
Recommend:   pip3 install --upgrade https://github.com/scikit-geometry/scikit-geometry/tarball/0.1.2 --user
This may require installing CGAL first (linux:  libcgal-dev, windows:  vcpkg install cgal)
"""

import logging
import time
import os
import numpy as np
import cv2
from scipy.interpolate import griddata
import matplotlib.pylab as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize


# from sympy import geometry as geo


def make_random_shape():
    n_verts = 5
    vertex_distances = np.random.rand(n_verts)
    return get_polygon(vertex_distances)


class Point(object):
    def __init__(self, x, y, abs_tol=1e-10):
        self.abs_tol = abs_tol
        self.x, self.y = float(x), float(y)

    def plot(self, *args, **kwargs):
        plt.plot(self.x, self.y, *args, **kwargs)

    def __str__(self):
        return "Point(%f, %f)" % (self.x, self.y)

    def near(self, other):
        d = self.distance(other)
        val = d < self.abs_tol
        # print("Comparing %s to %s -> %s" % (self, other, val))
        return val

    def distance(self, point):
        return np.sqrt((self.x - point.x) ** 2. + (self.y - point.y) ** 2)


#####
## https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def lines_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


####


class Segment(object):
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.length = p0.distance(p1)

    def plot(self, *args, **kwargs):
        plt.plot([self.p0.x, self.p1.x], [self.p0.y, self.p1.y], *args, **kwargs)

    def __str__(self):
        return "Line(start=%s, end=%s)" % (self.p0, self.p1)

    def interpolate(self, t, check_bounds=True):
        """
        linearly interpolate, between p0 at t=0 and p1 at t=1
        """
        if check_bounds and not 0. <= t <= 1.:
            raise Exception("Line parameter must be in [0,1].")
        x = self.p0.x * (1.0 - t) + self.p1.x * t
        y = self.p0.y * (1.0 - t) + self.p1.y * t
        return Point(x, y)

    def colinear(self, p):
        return False

    def intersects_point(self, p):
        """
           Return true, if within point._rel_tol of the line

           Math:  using parametric form for segment parameterized by t,
             solve for p (if possible)

           line: [x(t), y(t)] for x(t) = x0 * (1-t) + x1 * (t) and 0 <= t <=1, similar for y

           intersection:
            x0 * (1-t) + x1 * t = p.x    (x1 - x0) * t + x0 = p.x
            y0 * (1-t) + y1 * t = p.y    (y1 - y0) * t + y0 = p.y

            t = (p.x - x0)/ (x1-x0) =?= (p.y-y0)/(y1-y0)
           """
        x0, x1 = self.p0.x, self.p1.x
        y0, y1 = self.p0.y, self.p1.y

        # lines are aligned with axes, handle explicitly to avoid divide by 0 in the general case
        if x1 == x0 and y1 == y0:  # no line segment
            return p.near(self.p0)
        elif x1 == x0:  # vertical line, point must be "close"
            return np.abs(x1 - p.x) < p.abs_tol and y0 <= p.y <= y1
        elif y1 == y0:  # horizontal line
            return np.abs(y1 - p.y) < p.abs_tol and x0 <= p.x <= x1

        # the general case
        t0 = (p.x - x0) / (x1 - x0)
        t1 = (p.y - y0) / (y1 - y0)

        if not 0. - p.abs_tol < t0 < 1. + p.abs_tol:
            # parameter t is not in unit interval, no intersection
            return False

        p0 = Point(x0 * (1. - t0) + x1 * t0,
                   y0 * (1. - t0) + x1 * t0, abs_tol=p.abs_tol)
        p1 = Point(x0 * (1. - t1) + x1 * t1,
                   y0 * (1. - t1) + x1 * t1, abs_tol=p.abs_tol)

        return p0.near(p1)

    def intersect_line(self, other):
        """
        Return Point of intersection, or None if non-intersecting/crossing segments.

        Math:  using parametric form for segments a and b, parameterized by s and t,
          solve simultaneously for s and t

        line a: [xa(s), ya(s)] for xa(s) = xa0 * (1-s) + xa1 * (s) and 0 <= s <=1, similar for ya(s)
        line b: [xb(t), yb(t)] for xb(t) = xb0 * (1-t) + xb1 * (t) and 0 <= t <=1, etc.
        intersection:
            xa0 * (1-s) + xa1 * (s) = xb0 * (1-t) + xb1 * (t)
            ya0 * (1-s) + ya1 * (s) = yb0 * (1-t) + yb1 * (t)

            (-xa0 + xa1) * s + (xb0 -xb1) * t = (xb0 - xa0)   the "A" matrix for simult. eqns.
            (-ya0 + ya1) * s + (yb0 -yb1) * t = (yb0 - ya0)
        """
        tol = np.max([self.p0.abs_tol, self.p1.abs_tol, other.p0.abs_tol, other.p1.abs_tol])

        # special cases:
        def _check_colinearity(line, point):
            if line.intersects_point(point):
                return point
            return None

        colinearities = list()
        colinearities.append(_check_colinearity(self, other.p0))
        colinearities.append(_check_colinearity(self, other.p1))
        colinearities.append(_check_colinearity(other, self.p0))
        colinearities.append(_check_colinearity(other, self.p1))
        colinearities = [c for c in colinearities if c is not None]
        if len(colinearities) > 0:
            return colinearities[0]

        if self.colinear(other.p0):
            if self.intersects_point(other.p0):
                return other.p0
        if self.colinear(other.p1):
            if self.intersects_point(other.p1):
                return other.p1

        if not lines_intersect(self.p0, self.p1, other.p0, other.p1):
            return None
        xa0, ya0 = self.p0.x, self.p0.y
        xa1, ya1 = self.p1.x, self.p1.y

        xb0, yb0 = other.p0.x, other.p0.y
        xb1, yb1 = other.p1.x, other.p1.y

        a = np.array([[-xa0 + xa1, xb0 - xb1],
                      [-ya0 + ya1, yb0 - yb1]])
        b = np.array([[xb0 - xa0],
                      [yb0 - ya0]])

        st = np.linalg.solve(a, b)
        p0 = self.interpolate(st[0])
        p1 = self.interpolate(st[1])
        dist = Segment(p0, p1).length
        mean_length = np.mean([self.length, other.length])
        assert (dist / mean_length < tol), "Error calculating intersection"
        return p0


class Polygon(object):
    def __init__(self, corners):
        if not isinstance(corners[0], Point):
            self.vertices = [Point(corner[0], corner[1]) for corner in corners]
        else:
            self.vertices = corners
        self.n = len(self.vertices)
        verts = np.array([(c.x, c.y) for c in corners])
        self.bounds = [np.min(verts[:, 0]), np.min(verts[:, 1]),
                       np.max(verts[:, 0]), np.max(verts[:, 1])]
        self.sides = [Segment(self.vertices[i], self.vertices[i + 1]) for i in range(self.n - 1)]
        self.sides.append(Segment(self.vertices[0], self.vertices[-1]))

    def get_sides(self):
        sides = [np.hstack([self.vertices[i], self.vertices[i + 1]]) for i in range(self.n - 1)]
        sides.append(np.hstack([self.vertices[-1], self.vertices[0]]))
        return sides


class Circle(object):
    def __init__(self, center, radius):
        if not isinstance(center, Point):
            center = Point(center[0], center[1])
        self.x = center.x
        self.y = center.y
        self.center = center
        self.r = radius

    def __str__(self):
        return "Circle(center=%s, radius=%f)" % (self.center, self.r)

    def plot(self, *args, **kwargs):
        n_samples = 100
        theta = np.linspace(0.0, np.pi * 2.0, n_samples)
        x, y = self.x + np.cos(theta) * self.r, self.y + np.sin(theta) * self.r
        plt.plot(x, y, *args, **kwargs)

    def intersect_line(self, seg):
        """
        Return Point, Segment, or None depending on intersection, etc.
        math, using parametric form of line & implicit form of circle:
            line: [x(t), y(t)] for x(t) = x0 (1-t) + x1 (t) and 0 <= t <=1, similarly for y
            circle: (x-xc)^2 + (y-yc)^2 = r^2
            intersection: (x(t)-xc)^2 + (y(t)-yc)^2 = r^2, (substituting) solve for t:
                (x0 + (x1 - x0) t - xc) ^ 2 +(y0 + (y1 - y0 ) t - yc) ^ 2 = r^2
                (x0-xc)^2 + x0(x1-x0) t +(x1-x0)^2 t^2 + (y0-yc)^2 + y0(y1-y0) t + (y1-y0)^2 t^2 = r^2
                ((x1-x0)^2 + (y1-y0)^2) t^2 + (x0(x1-x0) + y0(y1-y0)) * t + (x0-xc)^2 + (y0-yc)^2 - r^2 = 0
                -----------------------       -----------------------       ---------------------------
                a                             b                             c

            Use quadratic formula:
                if b^2-4ac is negative, no intersection,
                if zero, single intersection,
                otherwise, line segment defined by the two solutions (plugged into line equation)

        :param seg:  Segment() object which may intersect circle
        :param extend_interior_segments:  if one or both endpoints are inside the circle, extend the
            segment until it intersects the circle.
        """

        x0, x1 = seg.p0.x, seg.p1.x
        y0, y1 = seg.p0.y, seg.p1.y
        xc, yc = self.x, self.y
        r = self.r

        a = (x0 ** 2. - 2. * x0 * x1 + x1 ** 2. + y0 ** 2. - 2. * y0 * y1 + y1 ** 2.)
        b = (-2. * x0 ** 2. - 2. * x1 * xc - 2. * y0 ** 2. - 2. * y1 * yc)
        c = (x0 ** 2. + 2. * x0 * xc + xc ** 2. + y0 ** 2. + 2. * y0 * yc + yc ** 2. - r ** 2.)
        t = -b / (2. * a)

        disc = b ** 2. - 4. * a * c

        if disc < 0:  ### should use tol here?
            # no solutions, line does not intersect circle
            return None
        elif disc == 0:
            # single solution, line is tangent to circle, intersection may be in circle
            if 0 < t < 1:
                return seg.interpolate(t)
            return None
        else:
            # two solutions, line intersects circle in 2 places
            t0 = t - np.sqrt(disc)
            t1 = t + np.sqrt(disc)

        # The line crosses the circle, but maybe not all of the line, so the intersection
        # may need to be shortened. Circle is the (  ) and the line is 0----1, i.e. p0 to p1.

        if t0 > 1 and t1 > 1:  # 0--1 (    )        no intersection t0, t1 > 1
            return None
        elif t0 > 0 and t1 > 0:  # 0----(-1  )        intersection t1 > 1
            return Segment(seg.interpolate(t0), seg.p1)
        elif t0 > 0 and t1 > 0:  # 0----(----)-1      intersection
            return Segment(seg.interpolate(t0), seg.interpolate(t1))
        elif t0 > 0 and t1 > 0:  # ( 0--1 )     interior "intersection" 0 < t0, t1 < 1
            return Segment(seg.p0, seg.p1)
        elif t0 > 0 and t1 > 0:  # ( 0--)-1      intersection t0 < 0
            return Segment(seg.p0, seg.interpolate(t1))
        else:  # t0 > 0 and t1 > 0:  # (    ) 0--1   no intersection, t0, t1 < 0
            return None


def polygon_encloses_point(polygon, point, max_tries=10000):
    """
    Randomized algorithm, using winding number. (

    Connect the point to a point outside the polygon, count the number
    of polygon edges it intersects, and if it's odd, the point is inside.

    Edge cases:  ray goes through a vertex, point is on polygon edge

    :returns:  True if point is in interior, False if exterior or boundary.
    """
    if any([point.near(vertex) for vertex in polygon.vertices]):
        return False

    if any([side.intersects_point(point) for side in polygon.sides]):
        return False

    for _ in range(max_tries):
        p_far = Point(polygon.bounds[0] - 1.0 - np.random.rand(1)[0],
                      polygon.bounds[1] - 1.0 - np.random.rand(1)[0], abs_tol=point.abs_tol)  # outside bounding box
        ray = Segment(point, p_far)
        vertex_intersections = [ray.intersects_point(v) for v in polygon.vertices]
        if any(vertex_intersections):
            continue

    intersections = [side.intersect_line(ray) for side in polygon.sides]

    count = np.sum([intersection is not None for intersection in intersections])
    return np.mod(count, 2) == 1


def rand_interior_point(polygon, max_tries=1000000):
    """
    Return a point inside the polygon (at random), or raises exception
    :param polygon:  sympy.geometry.Polygon object
    :returns: Point object
    """
    x_span = polygon.bounds[2] - polygon.bounds[0]
    y_span = polygon.bounds[3] - polygon.bounds[1]
    for _ in range(max_tries):
        p = Point((np.random.rand(1)[0] - polygon.bounds[0]) * x_span + polygon.bounds[0],
                  (np.random.rand(1)[0] - polygon.bounds[1]) * y_span + polygon.bounds[1], )
        if polygon_encloses_point(polygon, p):
            return p
    raise Exception("Couldn't find interior point!  Does polygon have small area?")


def max_inscribed_circle(corners):
    """
    Find the center and radius of the largest inscribed circle.
    :param polygon:  Nx2  of polygon corners
    :returns:  (x,y) center, r radius
    """
    poly = Polygon(corners)
    interior_point = rand_interior_point(poly)
    exterior_penalty = 100.0

    def _inscribed_rad_err(circle_params):
        """
        Error function of candidate circle:
            1. center must be interior, else error+= 100*distance to an interior point
            2. vertices must be at least r from (x,y), else error = Sum of squared extra distances
            3. edges must not cross circle, error = sum of chord lengths of intersecting edges

            (4). error is -radius
        :param circle_params:  3-tuple, (x, y, r) of circle
        """
        x, y, r = circle_params
        e = 0.0
        point = Point(x, y)
        circle = Circle(point, r)
        p = np.array([x, y])

        # condition 1
        if not polygon_encloses_point(poly, point):
            e += exterior_penalty * float(point.distance(interior_point))

        # condition 2
        segs = (corners - p.reshape(1, 2))
        sq_dists = np.sum(segs ** 2, axis=1)

        error = sq_dists - r
        error[error < 0] = 0
        e += np.sum(error)

        # condition 3
        chords = [circle.intersect_line(side) for side in poly.sides]
        e += np.sum([chord.length for chord in chords])

        return e - r

    small_spread = np.sum(np.sqrt(np.diag(np.cov(corners.T)))) / 50.0  # sum of x & y st. dev.
    diag = float(Segment(Point(poly.bounds[0], poly.bounds[1]),
                         Point(poly.bounds[2], poly.bounds[3])).length)
    initial_guess = (float(interior_point.x),
                     float(interior_point.y),
                     small_spread)
    bounds = [(poly.bounds[0], poly.bounds[2]),
              (poly.bounds[1], poly.bounds[3]),
              (0.0, diag)]
    # print(bounds)
    # print(initial_guess, bounds)
    best = minimize(_inscribed_rad_err, x0=initial_guess, method='Powell')

    center, radius = np.array((best.x[0], best.x[1])), best.x[2]
    return center, radius
