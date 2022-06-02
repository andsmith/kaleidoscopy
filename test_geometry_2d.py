import matplotlib.pyplot as plt
import numpy as np
from geometry_2d import make_random_shape, max_inscribed_circle, Point, Circle, Segment, Polygon
import logging


def plot_polygon(corners, *args, **kwargs):
    x = np.hstack([corners[:, 0], corners[0, 0]])
    y = np.hstack([corners[:, 1], corners[0, 1]])
    plt.plot(x, y, *args, **kwargs)


def test_inscribed_circles():
    n = 1
    for r in range(5):
        for c in range(5):
            # plt.subplot(5,5,n)
            shape = make_random_shape()
            plot_polygon(shape, "-k")
            center, radius = max_inscribed_circle(shape)
            # print(center, radius)
            inscribed_circle = get_polygon(np.ones(100) * radius)
            plot_polygon(inscribed_circle)
            plt.show()
            # plt.fill(center.reshape(1,2)+get_polygon(np.ones(100)*radius))
            n += 1
    plt.show()


def get_polygon(radii):
    theta = np.linspace(0.0, 2. * np.pi, radii.size + 1)[:-1]
    points = np.vstack([radii * np.cos(theta),
                        radii * np.sin(theta)]).T
    return points


def test_line_intersects_point(plot=False):
    tol = 1e-4
    big = 1e-3
    small = 1e-5

    p0 = Point(0., 0., abs_tol=tol)
    p1 = Point(1., 1., abs_tol=tol)
    p2 = Point(0., 1., abs_tol=tol)

    line1 = Segment(p0, p1)
    line2 = Segment(p0, p2)

    points_near_line1 = [Point(.2, .2, abs_tol=tol),
                         Point(.8 - small, .8, abs_tol=tol),
                         Point(.2, .2 + small, abs_tol=tol),
                         Point(.5 + small, .5 + small, abs_tol=tol)]

    points_near_line2 = [Point(0, .2, abs_tol=tol),
                         Point(0 - small, .8, abs_tol=tol),
                         Point(0, .2 + small, abs_tol=tol),
                         Point(0 + small, .5 + small, abs_tol=tol)]

    colinear_intersecting_line1 = [Point(.1, .1, abs_tol=tol),  # just inside endpoint
                                   Point(.9, .9, abs_tol=tol),
                                   Point(0., 0., abs_tol=tol),  # just touches endpoint
                                   Point(1., 1., abs_tol=tol)]  # other endpoint

    colinear_non_intersecting_line1 = [Point(1.1, 1.1, abs_tol=tol),
                                       Point(-.1, -.1, abs_tol=tol)]

    points_near_both = [Point(small, small, abs_tol=tol)]
    points_near_neither = [Point(big, 0., abs_tol=tol)]

    def _check_intersecting(line, point, intersects):
        test_str = "intersection of %s with %s:  %s" % (line, point, intersects)
        logging.info(test_str)
        intersection = line.intersects_point(point)
        if plot:
            line.plot('k-')
            if intersects and intersection:
                chr = "go"
            elif intersects and not intersection:
                chr = 'ro'
            elif not intersects and intersection:
                chr = 'rx'  # if we're here, we will hit an AssertionError, but oh well.
            else:
                chr = 'gx'
            point.plot(chr)
        assert intersects == intersection, test_str

    for p in colinear_non_intersecting_line1:
        _check_intersecting(line1, p, False)

    for p in colinear_intersecting_line1:
        _check_intersecting(line1, p, True)

    for p in points_near_both:
        _check_intersecting(line1, p, True)
        _check_intersecting(line2, p, True)

    for p in points_near_neither:
        _check_intersecting(line1, p, False)
        _check_intersecting(line2, p, False)

    for p in points_near_line1:
        _check_intersecting(line1, p, True)
        _check_intersecting(line2, p, False)

    for p in points_near_line2:
        _check_intersecting(line1, p, False)
        _check_intersecting(line2, p, True)
    if plot:
        plt.show()


def test_line_intersect_line():
    """
    Check: inteserecting, parallel, lines that would intersect but are too short, also parallel to axes
    """

    tests = [(Segment(Point(0., 0.), Point(1., 1.)),  # line 1
              Segment(Point(1., 0.), Point(0., 1.)),  # line 2
              Point(0.5, 0.5)),  # expected intersection
             (Segment(Point(0., 0.), Point(1., 1.)),
              Segment(Point(0.1, 0.2), Point(1.1, 1.2)),
              None),  # parallel, no expected intersection.
             (Segment(Point(0., 0.), Point(1., 1.)),
              Segment(Point(1., 0.), Point(0.6, 0.4)),
              None),  # not parallel, but not intersecting
             (Segment(Point(0., 0.), Point(1., 1.)),
              Segment(Point(1., 1.), Point(2., 0.)),
              Point(1., 1.)),  # shared endpoint intersection
             (Segment(Point(0., 0.), Point(1., 1.)),
              Segment(Point(1., 0.), Point(0.5, 0.5)),
              Point(0.5, 0.5)),  # endpoint of one, middle of other
             (Segment(Point(0., 0.), Point(1., 1.)),
              Segment(Point(1., 1.), Point(2., 2.)),
              Point(1., 1.)),  # co-linear  intersecting endpoint
             (Segment(Point(0., 0.), Point(1., 1.)),
              Segment(Point(1.1, 1.1), Point(2., 2.)),
              None),  # co-linear not intersecting
             ]

    for line1, line2, intersection in tests:
        if intersection is not None:
            test = "lines should intersect:  %s intersects %s at (%s)" % (line1, line2, intersection)
            logging.info(test)
            assert line1.intersect_line(line2).near(intersection), test
        else:
            test = "lines shouldn't intersect:  %s and %s" % (line1, line2)
            logging.info(test)
            i_section = line1.intersect_line(line2)
            assert i_section is None, "%s but do at %s" % (test, i_section)


def test_circle_intersect_line(plot=False):
    circle = Circle((0.0, 0.0), 1.0)
    angle = np.pi / 6.0
    tests = dict(tangent_angled=(Segment(Point(np.sqrt(2), 0.0),
                                         Point(0.0, np.sqrt(2))),  # line
                                 Point(np.sqrt(2.) / 2., np.sqrt(2.) / 2.)),  # intersection
                 too_high_angled=(Segment(Point(np.sqrt(2), 0.1),
                                          Point(0.0, np.sqrt(2) + .1)),
                                  None),
                 crossing_angled=(Segment(Point(-2., -2.),
                                          Point(2., 2.)),
                                  Segment(Point(-np.sqrt(2.) / 2., -np.sqrt(2.0) / 2.),
                                          Point(np.sqrt(2.) / 2., np.sqrt(2.0) / 2.)),),

                 tangent_horizontal=(Segment(Point(-1, -1),
                                             Point(1, -1)),
                                     Point(0.0, -1.0)),
                 midline_horizontal=(Segment(Point(-2, 0),
                                             Point(2, 0)),
                                     Segment(Point(-1, 0),
                                             Point(1, 0)),),
                 below_horizontal=(Segment(Point(-1, -2),
                                           Point(1, -2)),
                                   None),
                 tangent_vertical=(Segment(Point(-1, -1),
                                           Point(-1, 1)),
                                   Point(-1.0, 0.0)),
                 midline_vertical=(Segment(Point(0, -2),
                                           Point(0, 2)),
                                   Segment(Point(0, -1),
                                           Point(0, 1)),),
                 left_vertical=(Segment(Point(-2, -1),
                                        Point(-2, 1)),
                                None),
                 interior_angled=(Segment(Point(0., 0.),
                                          Point(.3 * np.cos(angle), .3 * np.sin(angle))),
                                  Segment(Point(.3 * np.cos(angle), .3 * np.sin(angle)),
                                          Point(.3 * np.cos(angle), .3 * np.sin(angle)))),
                 interior_angled_big=(Segment(Point(-1.2 * np.cos(angle * 2), -1.2 * np.sin(angle * 2)),
                                              Point(1.2 * np.cos(angle * 2), 1.2 * np.sin(angle * 2))),
                                      Segment(Point(.3 * np.cos(angle * 2), .3 * np.sin(angle * 2)),
                                              Point(.3 * np.cos(angle * 2), .3 * np.sin(angle * 2)))))

    tests = {'interior_angled': tests['interior_angled']}

    '''
    test_name = 'interior_angled'
    test_line, intersection = tests[test_name]
    test_str = "circle/line test '%s':  %s with %s should be %s" % (test_name,
                                                                    circle,
                                                                    test_line,
                                                                    intersection)

    import ipdb; ipdb.set_trace()
    result = circle.intersect_line(test_line)
    print(result)
    import sys
    sys.exit()
    '''

    for test_name in tests:
        test_line, intersection = tests[test_name]
        print(type(test_line), type(intersection))
        test_str = "circle/line test '%s':  %s with %s should be %s" % (test_name,
                                                                        circle,
                                                                        test_line,
                                                                        intersection)
        logging.info(test_str)
        if plot:
            circle.plot('-k', linewidth=2)
            if intersection is None:
                test_line.plot('-', linewidth=2)
            elif isinstance(intersection, Point):
                test_line.plot('-b', linewidth=1)

                intersection.plot('go', markersize=6)
            elif isinstance(intersection, Segment):
                test_line.plot('-b', linewidth=1)

                intersection.plot('r:', linewidth=3)
            result = circle.intersect_line(test_line)
            print("-----> " , result)
            if isinstance(result, Point):
                result.plot("rx", markersize=8)
            if isinstance(result, Segment):
                result.plot("g-.", linewidth=4)
            # assert isinstance(result, Point), "%s  -> %s" % (test_str, type(result))

    plt.show()


def test_points():
    tol = 1e-3
    big = tol * 100
    small = tol / 100

    p0 = Point(1, 1, abs_tol=tol)
    near = [Point(1, 1, abs_tol=tol),
            Point(1 + small, 1, abs_tol=tol),
            Point(1, 1 + small, abs_tol=tol),
            Point(1 + small, 1 + small, abs_tol=tol)]
    far = [Point(1 + big, 1, abs_tol=tol),
           Point(1, 1 + big, abs_tol=tol),
           Point(1 + big, 1 + big, abs_tol=tol)]

    for far_point in far:
        test = "points should not be near:  %s and %s (+/- %s)" % (p0, far_point, tol)
        logging.info(test)
        assert not far_point.near(p0), test

    for near_point in near:
        test = "points should be near:  %s and %s (+/- %s)" % (p0, near_point, tol)
        logging.info(test)
        assert near_point.near(p0), test


def test__geometry():
    test_points()
    test_line_intersects_point()
    test_line_intersect_line()
    test_circle_intersect_line(plot=True)
    logging.info("Passed all tests")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test__geometry()
