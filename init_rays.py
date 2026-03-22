"""
Compute FOV angles and image-plane Z coordinate for the kaleidoscope ray grid.

The image plane is at z = IMG_Z (between the eye at z=0 and the target at z=TARG_Z).
The pixel grid at z=IMG_Z must stay entirely inside the mirror tube cross-section,
with each corner at least CLOSEST_PIXEL_PX pixel-widths away from any mirror segment.

Binary search finds the largest valid IMG_Z (farthest from the eye).
"""

import numpy as np
import matplotlib.path as mpath
from geom import TARG_Z, CLOSEST_PIXEL_PX, lineseg_dist

_N_BINARY_SEARCH_ITERS = 64


def compute_fov(w_out, h_out):
    """
    Compute the half-extents of the FOV rectangle in natural coords at TARG_Z.

    The wider dimension spans [-1, 1]; the narrower is scaled to maintain aspect ratio.

    :param w_out: output image width in pixels
    :param h_out: output image height in pixels
    :returns: (x_max, y_max)  -- half-widths in natural coords at TARG_Z
    """
    a = w_out / h_out
    if a >= 1.0:
        return 1.0, 1.0 / a
    else:
        return a, 1.0


def _pixel_size(s, x_max, y_max, w_out, h_out):
    """Return the smaller pixel dimension in natural coords at scale factor s."""
    return min(2 * x_max * s / w_out, 2 * y_max * s / h_out)


def _check_valid(s, x_max, y_max, w_out, h_out, mirrors, polygon):
    """
    Return True if the pixel grid at scale s=IMG_Z/TARG_Z satisfies the clearance constraint.

    Grid corners at s: (±x_max*s, ±y_max*s).
    Constraint: each corner must be inside the mirror polygon AND at least
    CLOSEST_PIXEL_PX * pixel_size from every mirror segment.
    """
    min_dist_req = CLOSEST_PIXEL_PX * _pixel_size(s, x_max, y_max, w_out, h_out)
    corners_2d = [
        ( x_max * s,  y_max * s),
        (-x_max * s,  y_max * s),
        ( x_max * s, -y_max * s),
        (-x_max * s, -y_max * s),
    ]
    for cx, cy in corners_2d:
        if not polygon.contains_point((cx, cy)):
            return False
        p3 = np.array([cx, cy, 0.0])
        for m in mirrors:
            if lineseg_dist(p3, m.p0, m.p1) < min_dist_req:
                return False
    return True


def find_img_z(w_out, h_out, mirrors, x_max, y_max, targ_z=TARG_Z):
    """
    Binary search for the largest IMG_Z such that every pixel-grid corner is
    inside the mirror polygon and at least MIN_DIST from every mirror segment,
    where MIN_DIST = CLOSEST_PIXEL_PX * pixel_size_in_natural_coords.

    Mirrors are vertical planes (same XY cross-section at all Z values).
    At z=IMG_Z the grid spans [-x_max*s, x_max*s] x [-y_max*s, y_max*s],
    where s = IMG_Z / targ_z.

    :param w_out: output image width in pixels
    :param h_out: output image height in pixels
    :param mirrors: list of Mirror objects (from MirrorTube.mirrors)
    :param x_max: FOV half-width in natural coords (from compute_fov)
    :param y_max: FOV half-height in natural coords (from compute_fov)
    :param targ_z: Z coordinate of the target plane (default TARG_Z)
    :returns: img_z (float) -- largest valid image plane Z distance
    :raises ValueError: if no valid img_z exists
    """
    # Build closed path for polygon containment tests
    verts = [m.p0[:2] for m in mirrors]
    verts.append(verts[0])  # close the polygon
    polygon = mpath.Path(np.array(verts))

    lo = 1e-10
    hi = targ_z

    # Feasibility check: at infinitesimally small z the corners are near origin,
    # well inside the polygon. If not, mirrors are arranged pathologically.
    if not _check_valid(lo, x_max, y_max, w_out, h_out, mirrors, polygon):
        raise ValueError(
            "No valid img_z: mirror tube does not contain the origin with sufficient clearance."
        )

    for _ in range(_N_BINARY_SEARCH_ITERS):
        mid = (lo + hi) / 2.0
        if _check_valid(mid, x_max, y_max, w_out, h_out, mirrors, polygon):
            lo = mid   # valid -- try larger
        else:
            hi = mid   # too close -- shrink

    return lo


def make_ray_grid(w_out, h_out, img_z, x_max, y_max, targ_z=TARG_Z, start_at_eye=False,
                  add_noise=True):
    """
    Create the initial ray grid through the image plane at z=img_z,
    one ray per output pixel.

    Grid spans [-x_max*s, x_max*s] x [-y_max*s, y_max*s] at z=img_z,
    where s = img_z / targ_z.

    By default ray origins are placed on the image plane itself (so the first
    mirror intersection is computed forward from there).  Pass
    ``start_at_eye=True`` to originate rays at the eye (0, 0, 0) instead.

    :param w_out: output image width in pixels
    :param h_out: output image height in pixels
    :param img_z: image plane Z coordinate
    :param x_max: FOV half-width in natural coords at targ_z
    :param y_max: FOV half-height in natural coords at targ_z
    :param targ_z: target plane Z (default TARG_Z)
    :param start_at_eye: if True, origins are all (0,0,0); if False (default),
                         origins are the pixel positions on the image plane
    :param add_noise: if True (default), apply a single uniform random XY offset
                      to all grid points.  The offset magnitude is drawn independently
                      for X and Y from Uniform[-spacing/100, +spacing/100], where
                      spacing is the grid point separation in that dimension.  This
                      prevents rays from hitting mirror-mirror corners simultaneously
                      (a degenerate case for centred rectangular mirror tubes).
    :returns: (origins, directions) -- each shape (w_out*h_out, 3), float64
        directions: unit vectors from eye through each pixel (same either way)
    """
    s = img_z / targ_z
    x_vals = np.linspace(-x_max * s, x_max * s, w_out)
    y_vals = np.linspace(-y_max * s, y_max * s, h_out)

    if add_noise:
        dx_spacing = 2 * x_max * s / max(w_out - 1, 1)
        dy_spacing = 2 * y_max * s / max(h_out - 1, 1)
        x_vals = x_vals + np.random.uniform(-dx_spacing / 100, dx_spacing / 100)
        y_vals = y_vals + np.random.uniform(-dy_spacing / 100, dy_spacing / 100)

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)  # shape (h, w)
    z_grid = np.full_like(x_grid, img_z)

    img_plane = np.stack([x_grid, y_grid, z_grid], axis=-1)  # (h, w, 3)
    directions = img_plane / np.linalg.norm(img_plane, axis=-1, keepdims=True)

    if start_at_eye:
        origins = np.zeros_like(img_plane)
    else:
        origins = img_plane

    return origins.reshape(-1, 3), directions.reshape(-1, 3)


if __name__ == "__main__":
    import logging
    from mirror_configs import PresetFactory

    logging.basicConfig(level=logging.DEBUG)

    for preset_name in ['equilateral triangle', 'square', 'hexagon']:
        tube = PresetFactory.make_preset(preset_name, r=0.4)
        w, h = 1280, 720
        x_max, y_max = compute_fov(w, h)
        img_z = find_img_z(w, h, tube.mirrors, x_max, y_max)
        print(f"{preset_name}: img_z={img_z:.6f}  x_max={x_max:.3f}  y_max={y_max:.3f}")

    logging.info("init_rays tests passed.")
