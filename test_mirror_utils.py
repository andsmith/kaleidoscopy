from mirror_utils import unitize_shape
import numpy as np


def test_unitize_shape():
    TOL=1.e-12
    tests = [np.array([[-1.0, -1.0], [1.0, 1.0]]),
             np.array([[-1.0, 0.0], [1.0, 0.0]]),
             np.array([[0.0, -1.0], [0.0, 1.0]]),
             np.array([[0.5,0.5], [0.6,0.5]]),
             np.array([[0.5,0.5], [0.5,0.6]]),
             np.array([[0.5,0.5], [0.6,0.6]]),
             ]
    for points in tests:
        u_points = unitize_shape(points)
        span = np.max(u_points,axis=0)-np.min(u_points,axis=0)
        assert np.abs(np.max(span)-1.0)< TOL, "Failed:  %s --> span:  %s" % (points, span)


if __name__=="__main__":
    test_unitize_shape()
    print("All tests pass.")