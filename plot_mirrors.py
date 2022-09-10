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


def close(points):
    return np.vstack([points, points[0, :]])


def _plot_mirror(m):
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
                 [origin[1], opd[1]],'k-')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _plot_mirror(RectangularPrism())
    plt.show()
    mps = [IsoscelesPrism(), RectangularPrism(), NGonPrism(), CirclePrism()]
    for i, mp in enumerate(mps):
        plt.subplot(2, 2, i + 1)

        _plot_mirror(mp)
        plt.axis('equal')

    plt.show()
