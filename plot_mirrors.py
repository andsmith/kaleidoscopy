"""
Debugging tool for making sure mirrors have correct geometry.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from mirrors import MirrorPrism
from mirrors_rectangle import RectangularPrism
from mirrors_isosceles import IsoscelesPrism
from mirrors_n_gon import NGonPrism
from mirrors_circle import CirclePrism
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def close(points):
    return np.vstack([points, points[0, :]])


def _plot_mirror_orientations_2d(m):
    """
    Plots layout of mirrors in 2d (from above).
    Draws a line from the midpoint of each mirror to a dot in the direction of that mirror's normal.
    """

    def _plot_shape(pts, plt_chr="-"):
        plt.plot(pts[:, 0], pts[:, 1], plt_chr)

    m._aperture_scale = 0.666
    points = close(m.get_scaled_shape())
    u_points = close(m.get_unscaled_shape())
    _plot_shape(u_points, 'g:')
    _plot_shape(points, 'b-')
    if points.shape[0] < 50:
        for i in range(points.shape[0] - 1):
            midpoint = (points[i, :] + points[i + 1, :]) / 2.
            print(midpoint)
            # plt.plot(midpoint[0],midpoint[1], 'o')
            plt.text(midpoint[0], midpoint[1], "%i" % (i,))

    # Plot normal info
    if not m.is_planar():
        return
    planes = m.get_surfaces()
    for p in planes:
        origin, direction = p.get_params()
        opd = origin + direction * 0.05
        plt.plot(opd[0], opd[1], 'ok')
        plt.plot([origin[0], opd[0]],
                 [origin[1], opd[1]], 'k-')


def make_side_quad(c1, c2, h):
    """
    Get coordinates of the corners of the corners of the rectangle above two points on the z=0 plane.
    """
    return np.array([[c1[0], c1[1], 0],
                     [c1[0], c1[1], h],
                     [c2[0], c2[1], h],
                     [c2[0], c2[1], 0]])


def draw_raytrace_state(mirrors, rays, z_height, ax=None):
    """
    Plot 3d mirrors & rays
    :param mirrors:  MirrorPrism object
    :param rays:  dict(ray_origins: Nx3 array, ray_unit_directions: Nx3 array)
    :param z_height:  eye-distance, ray origin, (0,0, z), also height of mirrors
    :param ax: 3x axes, or None for new figures
    """
    corner_coords = close(mirrors.get_unscaled_shape())

    surf_coord_list = [make_side_quad(corner_coords[i, :],
                                      corner_coords[i + 1, :],
                                      z_height) for i in range(corner_coords.shape[0] - 1)]
    mirror_handles = []
    for face_coords in surf_coord_list:
        mh, ax = draw_3d_polygon(face_coords, ax=ax)
        mirror_handles.append(mh)

    # plot mirrors
    # draw eye
    eye_handle = [ax.scatter(0.0, 0.0, 0.0, color='b')]

    return ax


def draw_3d_polygon(corners, ax=None, color=(0.1, .15, 1.0, .25)):
    """
    Add a quadrilateral to the axes or make new.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    x = corners[:, 0]
    y = corners[:, 1]
    z = corners[:, 2]
    verts = [list(zip(x, y, z))]  # list necessary python 2/3?

    poly = Poly3DCollection(verts)
    poly.set_color(color)
    handle = ax.add_collection3d(poly)

    return handle, ax


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    draw_raytrace_state(RectangularPrism(), None,z_height=2.0)
    plt.show()
