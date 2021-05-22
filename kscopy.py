import numpy as np
import cv2
from util import make_bounds


class MirrorAssembly(object):
    """
    Define a ortho-prism (shape) assembly of mirrors, i.e. all perpendicular to flat, facing indwards.
    Input is an arbitrary list of 2-d polygon vertices.
    """

    def __init__(self, corners):
        """
        Define mirror prism.
        :param corners:  list of 2-d coordinates (numpy arrays), i.e. mirror closes loop of vertices
        """
        n = len(corners)

        # mirror centers are midpoints between corners, and in 3d
        centers = [(self._corners[i] + self._corners[i + 1]) / 2.0 for i in range(n - 1)]
        centers.append((self._corners[0] + self._corners[-1]) / 2.0)
        centers = [np.hstack(c, [0]) for c in centers]

        # Need two non-parallel, co-planar vectors whose cross-product will give us the normal for each mirror
        # First will connect each mirror's centers to a corner.
        co_planar_a = [centers[i] - corners[i] for i in range(self._n)]
        # Second will connect each mirror's first corner to a point 1cm above that corner.
        co_planar_b = [centers[i] - np.hstack((corners[i:2], [1.0])) for i in range(n)]
        normals = np.cross(co_planar_a, co_planar_b)

        self._bounds = make_bounds(corners)
        self._n = n
        self._centers = centers
        self._corners = corners
        self._normals = normals

    def get_bounds(self):
        return self._bounds


class IsosceleseMirrorAssembly(MirrorAssembly):
    def __init__(self, theta_deg, h_cm):
        theta = np.deg2rad(theta_deg)
        corners = [np.array([-np.sin(theta), 0]),
                   np.array([0, h_cm]),
                   np.array([np.sin(theta), 0]), ]
        super(IsosceleseMirrorAssembly, self).__init__(corners=corners)


class Kalleidoscope(object):
    def __init__(self, resolution):
        self._resolution = resolution
        self._mirrors = IsosceleseMirrorAssembly(theta_deg=15.0, h_cm=5.0)
        self._image = None
        self._image_bounds = None
        self._eye_scope_cm = 4.0
        self._eye_image_plane_cm = 2.0
        self._scope_img_cm = 20.0

    def _set_img_bounds(self, dpi):
        width = self._image.shape[1] / dpi
        height = self._image.shape[0] / dpi
        self._image_bounds = make_bounds([[-width / 2.0, -height / 2.0, ],
                                          [width / 2.0, height / 2.0, ], ])

    def show(self, img, dpi=100.0):
        self._image = img
        self._set_img_bounds(dpi)
        self._make_ray_mask()

    def _make_ray_mask(self):

        # calculate dimensions of image plane & it's bounds of the rays.
        origin = np.ones(3).reshape(1, 3)
        scope_extent = self._mirrors.get_bounds()
        height_px_per_cm = self._resolution[0] / (scope_extent['top'] - scope_extent['bottom'])
        width_px_per_cm = self._resolution[1] / (scope_extent['right'] - scope_extent['left'])

        if height_px_per_cm > width_px_per_cm:
            if scope_extent['top'] > -scope_extent['bottom']:
                scale = scope_extent['top'] / self._eye_scope_cm
            else:
                scale = -scope_extent['bottom'] / self._eye_scope_cm
        else:
            if scope_extent['left'] > -scope_extent['right']:
                scale = scope_extent['left'] / self._eye_scope_cm
            else:
                scale = -scope_extent['right'] / self._eye_scope_cm

                
        image_plane_scope = make_bounds([[]])



if __name__ == "__main__":
    scope = Kalleidoscope()
    img = cv2.imread('test_img.jpg')
    scope.show(img)
