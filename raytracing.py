"""
Create the kaleidoscope's image map by raytracing from the eye, through
the image plane, bouncing off the mirrors and hitting the target.

The image map can then be applied live to video to simulate the kaleidoscope effect.

Coordinate conventions:
    - eye is at (0, 0, 0), and it looks in the positive z direction
    - image plane is at z = EYE_DIST
    - target plane is at z = TARGET_DIST (TARGET_DIST > EYE_DIST)
    - mirrors are vertical, and are touching the target plane (no ray will go under the mirror to hit the target)
    - mirrors are defined by two points, p0 and p1 in the XY plane, extend indfinitely in both z directions
    - The field of view is set so image would fill the target plane if there were no mirrors:
      - X-axis of the target (the screen) is normalized to [-1, 1], 
      - Y-axis is normalized to [-1, 1] / aspect ratio

"""
import numpy as np
import cv2
import json

from threading import Thread, Lock
import logging
from init_rays import make_ray_grid



class Raytracer(object):
    def __init__(self, size, mirrors, targ_z, x_max, y_max, img_z, threaded=False, bounce_file=None):
        """
        :param size: (w_out, h_out) output image size in pixels
        :param mirrors: list of Mirror objects
        :param targ_z: z-coordinate of the target plane
        :param x_max: FOV half-width at targ_z in natural coords (from compute_fov)
        :param y_max: FOV half-height at targ_z in natural coords (from compute_fov)
        :param img_z: image plane Z coordinate (from find_img_z)
        :param threaded: if True, run the raytracing in a separate thread.
           If running in a separate thread, the map can be accessed as it is being
           raytraced.
        :param bounce_file: if not None, path to a JSON file where the full bounce
           history will be written (updated after each iteration).
        """
        self._size = size
        self._mirrors = mirrors
        self._targ_z = targ_z
        self._x_max = x_max
        self._y_max = y_max
        self._img_z = img_z
        self._map = None
        self._bounce_count = None
        self._map_lock = Lock()
        self._threaded = threaded
        self._bounce_file = bounce_file

    def _init_rays(self):
        """
        Create the initial ray grid through the image plane using make_ray_grid.
        Origins are placed on the image plane at z=img_z (not the eye).
        Track the output pixel (x_ind, y_ind) each ray corresponds to.
        Also initialises the K map, bounce count, and step-mode state.
        """
        w_out, h_out = self._size
        self._origins, self._directions = make_ray_grid(
            w_out, h_out, self._img_z, self._x_max, self._y_max, self._targ_z
        )
        # Pixel index arrays in row-major order matching make_ray_grid's reshape(-1, 3)
        x_inds, y_inds = np.meshgrid(np.arange(w_out, dtype=np.int32),
                                      np.arange(h_out, dtype=np.int32))
        self._x_inds = x_inds.reshape(-1)
        self._y_inds = y_inds.reshape(-1)

        # K map: identity (each output pixel maps to itself) until rays fill it in
        self._map = [x_inds.copy(), y_inds.copy()]
        self._bounce_count = np.full((h_out, w_out), -1, dtype=np.int32)

        # Step-mode state (used by step() and get_bounces/get_initial_rays)
        N = len(self._origins)
        self._initial_origins    = self._origins.copy()
        self._initial_directions = self._directions.copy()
        self._step_origins = self._origins.copy()   # current position of ALL N rays
        self._step_dirs    = self._directions.copy()
        self._step_active  = np.ones(N, dtype=bool) # False once ray has hit target
        self._bounces      = []
        self._display_mask = None  # optional bool mask for which rays to visualise
        self._step_last_mirror_inds = np.full((N, 2), -1, dtype=np.int32)
        self._step_index = 0
        self._step_last_summary = None
        self._bounce_log = self._build_bounce_header() if self._bounce_file is not None else None

    def get_map(self):
        with self._map_lock:
            return self._map, self._bounce_count

    # ------------------------------------------------------------------
    # Side-view / debug interface  (same API as FakeRaytracer)
    # ------------------------------------------------------------------

    def get_ray_params(self):
        """Return geometric parameters needed by side_view.render_side_view."""
        w_out, h_out = self._size
        return {
            'x_max':          self._x_max,
            'y_max':          self._y_max,
            'img_z':          self._img_z,
            'targ_z':         self._targ_z,
            'mirrors':        self._mirrors,
            'ray_grid_shape': (h_out, w_out),
            'display_mask':   self._display_mask,
        }

    def get_initial_rays(self):
        """Return (origins Nx3, directions Nx3) of the initial ray grid, or (None, None)."""
        if self._map is None:
            return None, None
        return self._initial_origins, self._initial_directions

    def get_bounces(self):
        """Return the list of per-step bounce dicts accumulated by step()."""
        return self._bounces

    def step(self, record_bounces=True, verbose=None):
        """
        Run one bounce iteration for all currently-active rays.

        If record_bounces is True, store full per-step arrays in the same
        format used by FakeRaytracer so side_view can display the result.
        If False, only keep a lightweight per-step summary.

        :param verbose:
            None - current behavior (step summary logging only)
            1    - also print per-surface hit counts (target + each mirror)
            2    - also print per-ray distances to each mirror and target,
                   marking the hit surface with '*'

        Must call _init_rays() (or start()) before the first step().

        :returns: True if rays still remain to be traced, False when complete.
        """
        if self._map is None:
            self._init_rays()

        if not np.any(self._step_active):
            return False

        N      = len(self._step_origins)
        act    = np.where(self._step_active)[0]   # indices of active rays in [0, N)
        n_step = self._step_index

        if verbose not in (None, 1, 2):
            raise ValueError("verbose must be one of None, 1, or 2.")

        debug_mirror_dists = None
        debug_target_dists = None
        if verbose in (1, 2):
            ao = self._step_origins[act]
            ad = self._step_dirs[act]
            debug_mirror_dists = np.stack(
                [mirror.get_dist(ao, ad) for mirror in self._mirrors],
                axis=1,
            )
            lm_act = self._step_last_mirror_inds[act]
            for slot in range(lm_act.shape[1]):
                valid = lm_act[:, slot] >= 0
                if np.any(valid):
                    debug_mirror_dists[valid, lm_act[valid, slot]] = np.inf
            debug_target_dists = (self._targ_z - ao[:, 2]) / ad[:, 2]

        # Snapshot positions/directions at the *start* of this step.
        # Only needed when recording full per-step traces.
        if record_bounces:
            step_origins_snap = self._step_origins.copy()
            step_dirs_snap    = self._step_dirs.copy()

        # Run one bounce for the active subset, excluding each ray's last mirror
        bounce = _bounce(self._step_origins[act], self._step_dirs[act],
                         self._mirrors, self._targ_z,
                         last_mirror_inds=self._step_last_mirror_inds[act])

        t_local = bounce['target_hit_inds']  # indices into act[]
        m_local = bounce['mirror_hit_inds']  # indices into act[]

        if verbose in (1, 2):
            mirror_choice_local = bounce['closest_mirror_inds']
            print(f"Step {n_step} surface hits:")
            print(f"  target: {len(t_local)}")
            for m in range(len(self._mirrors)):
                print(f"  mirror {m}: {int(np.sum(mirror_choice_local == m))}")

            if verbose == 2:
                target_choice_local = np.zeros(len(act), dtype=bool)
                target_choice_local[t_local] = True
                print("  Per-ray distances (active rays):")
                for local_i, global_i in enumerate(act):
                    row = [f"ray {global_i:06d}"]
                    for m in range(len(self._mirrors)):
                        hit_mark = '*' if (not target_choice_local[local_i] and mirror_choice_local[local_i] == m) else ' '
                        d = debug_mirror_dists[local_i, m]
                        d_txt = "inf" if not np.isfinite(d) else f"{d:.6f}"
                        row.append(f"M{m}{hit_mark}:{d_txt}")
                    t_mark = '*' if target_choice_local[local_i] else ' '
                    td = debug_target_dists[local_i]
                    td_txt = "inf" if not np.isfinite(td) else f"{td:.6f}"
                    row.append(f"T{t_mark}:{td_txt}")
                    print("    " + "  ".join(row))

        # Build full-N hit arrays (inactive rays carry their last position forward)
        hit_origins = self._step_origins.copy()
        hit_mirror  = np.zeros(N, dtype=bool)
        hit_target  = np.zeros(N, dtype=bool)

        hit_origins[act]         = bounce['new_origins']
        hit_target[act[t_local]] = True
        hit_mirror[act[m_local]] = True

        # hit_dirs: direction AFTER this step
        # (reflected for mirror-hits, 0 for target-hits, carried forward for inactive)
        hit_dirs      = self._step_dirs.copy()
        hit_dirs[act] = bounce['new_directions']   # 0 for target-hits, reflected for mirrors

        # Update K map for rays that reached the target this step
        if len(t_local):
            global_t = act[t_local]
            target_xy = bounce['new_origins'][t_local, :2]
            targ_px, targ_py = self._unscale_coords(target_xy[:, 0], target_xy[:, 1])
            with self._map_lock:
                self._map[0][self._y_inds[global_t], self._x_inds[global_t]] = targ_px
                self._map[1][self._y_inds[global_t], self._x_inds[global_t]] = targ_py
                self._bounce_count[self._y_inds[global_t], self._x_inds[global_t]] = n_step
        else:
            global_t = np.empty(0, dtype=np.int32)

        if len(m_local):
            global_m = act[m_local]
        else:
            global_m = np.empty(0, dtype=np.int32)

        # Update last-mirror tracking: shift slot 0 → slot 1, record new mirror in slot 0.
        new_lm = self._step_last_mirror_inds.copy()
        new_lm[:, 1] = self._step_last_mirror_inds[:, 0]
        new_lm[act[m_local], 0] = bounce['closest_mirror_inds'][m_local]
        new_lm[act[t_local], :] = -1
        self._step_last_mirror_inds = new_lm

        # Deactivate target-hit rays; advance mirror-hit rays
        self._step_active[act[t_local]] = False
        self._step_origins = hit_origins
        new_dirs = self._step_dirs.copy()
        new_dirs[act[m_local]] = bounce['new_directions'][m_local]
        self._step_dirs = new_dirs

        if record_bounces:
            self._bounces.append({
                'step_origins': step_origins_snap,
                'step_dirs':    step_dirs_snap,
                'hit_origins':  hit_origins,
                'hit_dirs':     hit_dirs,
                'hit_mirror':   hit_mirror,
                'hit_target':   hit_target,
            })

        self._step_last_summary = {
            'step': n_step,
            'target_global_inds': global_t,
            'mirror_global_inds': global_m,
            'mirror_inds': bounce['closest_mirror_inds'][m_local].copy(),
            'mirror_hit_origins': bounce['new_origins'][m_local].copy(),
            'mirror_new_directions': bounce['new_directions'][m_local].copy(),
            'remaining_active': int(np.sum(self._step_active)),
        }
        self._append_bounce_log_step()

        logging.info("Step %i: %i hit target, %i bounced, %i still active.",
                     n_step, len(t_local), len(m_local), int(np.sum(self._step_active)))
        self._step_index += 1
        return bool(np.any(self._step_active))

    # ------------------------------------------------------------------

    def _unscale_coords(self, x, y):
        """
        Convert natural target-plane coords to output image pixel indices (int32).

        Natural coords: x in [-x_max, +x_max], y in [-y_max, +y_max].
        Pixel convention (from README): left column = x=-1, right = x=+1,
          top row = y=+1, bottom row = y=-1 (y axis is flipped).

        Returns int32 arrays suitable for the remap() C extension's x_map / y_map.
        Values are clamped to valid pixel range.
        """
        w_out, h_out = self._size
        px = (x + self._x_max) / (2.0 * self._x_max) * w_out
        py = (self._y_max - y) / (2.0 * self._y_max) * h_out
        px = np.clip(px, 0, w_out - 1).astype(np.int32)
        py = np.clip(py, 0, h_out - 1).astype(np.int32)
        return px, py

    def _build_bounce_header(self):
        """Build the header section of the bounce log from the current initial ray state."""
        w_out, h_out = self._size
        return {
            'header': {
                'grid_shape': [h_out, w_out],
                'image_z': float(self._img_z),
                'target_z': float(self._targ_z),
                'x_max': float(self._x_max),
                'y_max': float(self._y_max),
                'mirrors': [
                    {
                        'id': f'MIRROR_{i}',
                        'p0_xy': [float(m.p0[0]), float(m.p0[1])],
                        'p1_xy': [float(m.p1[0]), float(m.p1[1])],
                    }
                    for i, m in enumerate(self._mirrors)
                ],
                'ray_origins_xy': self._initial_origins[:, :2].tolist(),
            },
            'iterations': [],
        }

    def _write_bounce_log(self, log):
        """Write the bounce log dict to self._bounce_file as JSON."""
        with open(self._bounce_file, 'w') as f:
            json.dump(log, f, indent=2)
        logging.info("Wrote bounce log with %i iterations to %s", len(log['iterations']), self._bounce_file)

    def _append_bounce_log_step(self):
        """Append the latest step summary to the bounce log and write JSON."""
        if self._bounce_file is None or self._step_last_summary is None:
            return
        if self._bounce_log is None:
            self._bounce_log = self._build_bounce_header()

        step_summary = self._step_last_summary
        t_inds = step_summary['target_global_inds']
        m_inds = step_summary['mirror_global_inds']
        iter_record = {
            'step': step_summary['step'],
            'target_hits': [f'RAY_{i:06d}' for i in t_inds],
            'mirror_hits': [
                {
                    'ray': f'RAY_{m_inds[j]:06d}',
                    'mirror': f'MIRROR_{step_summary["mirror_inds"][j]}',
                    'hit_xyz': step_summary['mirror_hit_origins'][j].tolist(),
                    'new_direction_xyz': step_summary['mirror_new_directions'][j].tolist(),
                }
                for j in range(len(m_inds))
            ],
        }
        self._bounce_log['iterations'].append(iter_record)
        self._write_bounce_log(self._bounce_log)

    def start(self):

        def _trace():
            """
             Trace the image map through the mirrors to the target plane:
            1. For each ray, see which mirror (or the target) it hits first.
            2. Separate the rays into groups depending on what they hit,
                2.a. if they hit a mirror, keep bouncing them.
                2.b. if they hit the target, assign the target xy locations as the map values for those rays.
                map_x, map_y = np.zeros((h_px, w_px)), np.zeros((h_px, w_px))
            3. Repeat until all rays have hit the target.
            """
            self._init_rays()

            while True:
                has_more = self.step(record_bounces=False)

                if not has_more:
                    break

            logging.info("Raytracing complete in %i iterations.", self._step_index)

        if self._threaded:
            # Thread will update self._map as results are computed.
            self._thread = Thread(target=_trace)
            self._thread.start()
        else:
            _trace()


def _bounce(origins, directions, mirrors, targ_z, last_mirror_inds=None):
    """
    Advance each ray to its next surface.
    :param origins: Nx3 array of ray origins
    :param directions: Nx3 array of ray directions
    :param mirrors: list of Mirror objects
    :param targ_z: z-coordinate of the target plane
    :param last_mirror_inds: (N, 2) int array of the last two mirror indices each ray hit
        (-1 means unused slot).  Both mirrors are excluded from intersection tests so a
        ray cannot re-hit either of the two surfaces at a corner (d=0 oscillation fix).
    :return: dict {'target_hit_inds': list of T indices of rays that hit the target,
                   'mirror_hit_inds': list of N-T indices of rays that hit a mirror,
                   'new_origins': Nx3 array of new ray origins (including those in the target plane),
                   'new_directions': Nx3 array of new ray directions,
                   'closest_mirror_inds': N-length array of the mirror index closest to each ray}
    """
    n = origins.shape[0]
    mirror_dists = np.stack([mirror.get_dist(origins, directions) for mirror in mirrors],
                                  axis=1)

    # Exclude only the most recent mirror each ray bounced off to prevent re-hits
    # at d≈0 corners.  Rays can hit the same mirror again after moving away.
    if last_mirror_inds is not None:
        valid = last_mirror_inds[:, 0] >= 0
        if np.any(valid):
            mirror_dists[valid, last_mirror_inds[valid, 0]] = np.inf
    target_dists = (targ_z - origins[:, 2]) / directions[:, 2]

    closest_mirrors = np.argmin(mirror_dists, axis=1)
    closest_m_dists = mirror_dists[np.arange(n), closest_mirrors]
    target_hits = np.where(target_dists < closest_m_dists)[0]
    mirror_hits = np.where(target_dists >= closest_m_dists)[0]

    new_origins = origins.copy()
    new_directions = directions.copy()

    # calculate the hit locations for the targets:
    target_origins = origins[target_hits] + directions[target_hits] * target_dists[target_hits, None]
    new_origins[target_hits] = target_origins
    new_directions[target_hits] = 0.0

    # exclude rays that hit the target from the mirror hit list
    closest_mirrors[target_hits] = -1

    # Reflect each ray off its closest mirror (single-hit per step).
    # Corner cases (two mirrors at equal distance) are handled by sequential
    # single-hits across two steps: after step 1 the second mirror is at d=0,
    # and both mirrors end up in the 2-slot last_mirror_inds exclusion list
    # so the ray escapes the corner cleanly on the following step.
    for m in range(len(mirrors)):
        hits_m = closest_mirrors == m  # target hits have closest_mirrors=-1, so excluded
        new_directions[hits_m] = mirrors[m].reflect(directions[hits_m])
        new_origins[hits_m] = origins[hits_m] + directions[hits_m] * closest_m_dists[hits_m, None]


    rv = {'target_hit_inds': target_hits,
          'mirror_hit_inds': mirror_hits,
          'new_origins': new_origins,
          'new_directions': new_directions,
          'closest_mirror_inds': closest_mirrors}

    return rv


class FakeRaytracer:
    """
    Stub raytracer that returns a placeholder image of the correct output size.
    Can optionally be configured with mirror geometry to generate synthetic ray
    bounce data for side_view.py queries.
    """

    _FAKE_GRID_LONG = 20   # rays along the longer dimension for fake bounce grid
    _MAX_BOUNCES = 8

    def __init__(self, size, mirrors=None, x_max=1.0, y_max=None, img_z=None, targ_z=None):
        """
        :param size: (width, height) of the output image
        :param mirrors: list of Mirror objects (optional; enables ray data generation)
        :param x_max: FOV half-width in natural coords at targ_z (default 1.0)
        :param y_max: FOV half-height in natural coords (default x_max * h/w)
        :param img_z: image plane Z coordinate (required if mirrors provided)
        :param targ_z: target plane Z coordinate (default TARG_Z from geom)
        """
        from geom import TARG_Z as _TARG_Z
        self._size = size
        self._mirrors = mirrors
        w, h = size
        self._x_max = x_max
        self._y_max = y_max if y_max is not None else x_max * h / w
        self._targ_z = targ_z if targ_z is not None else _TARG_Z
        self._img_z = img_z

        self._origins = None
        self._directions = None
        self._ray_grid_shape = None
        self._bounces = []

        if mirrors is not None and img_z is not None:
            self._generate_fake_rays()

    def render(self):
        """Return a gray placeholder image with a 'rendering not implemented yet' message."""
        w, h = self._size
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        text = "rendering not implemented yet"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(img, text, (x, y), font, scale, (180, 180, 180), thickness)
        return img

    def _generate_fake_rays(self):
        """
        Generate a coarse ray grid and simulate real bounces for side_view queries.
        Uses actual mirror geometry so the paths are geometrically correct.
        """
        w, h = self._size
        aspect = w / h
        if aspect >= 1.0:
            n_x = self._FAKE_GRID_LONG
            n_y = max(2, round(self._FAKE_GRID_LONG / aspect))
        else:
            n_y = self._FAKE_GRID_LONG
            n_x = max(2, round(self._FAKE_GRID_LONG * aspect))

        self._ray_grid_shape = (n_y, n_x)

        s = self._img_z / self._targ_z
        x_vals = np.linspace(-self._x_max * s, self._x_max * s, n_x)
        y_vals = np.linspace(-self._y_max * s, self._y_max * s, n_y)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        z_grid = np.full_like(x_grid, self._img_z)

        img_plane = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
        dir3 = img_plane / np.linalg.norm(img_plane, axis=-1, keepdims=True)
        N = dir3.shape[0]

        self._origins = img_plane.copy()   # rays start at image plane, not the eye
        self._directions = dir3.copy()
        self._bounces = []

        cur_origins = self._origins.copy()
        cur_dirs = dir3.copy()
        still_active = np.ones(N, dtype=bool)
        last_mirror_per_ray = np.full((N, 2), -1, dtype=np.int32)

        for _ in range(self._MAX_BOUNCES):
            if not np.any(still_active):
                break

            step_origins = cur_origins.copy()
            step_dirs = cur_dirs.copy()
            hit_origins = cur_origins.copy()
            hit_mirror = np.zeros(N, dtype=bool)
            hit_target = np.zeros(N, dtype=bool)
            new_origins = cur_origins.copy()
            new_dirs = cur_dirs.copy()

            act = np.where(still_active)[0]
            ao = cur_origins[act]
            ad = cur_dirs[act]

            m_dists = np.stack(
                [m.get_dist(ao, ad) for m in self._mirrors], axis=1
            )   # (len(act), n_mirrors)

            # Exclude only the most recent mirror (slot 0) to prevent d≈0 re-hits.
            # After moving away, rays can hit mirrors again, even the same one.
            act_last_slot = last_mirror_per_ray[act, 0]
            valid = act_last_slot >= 0
            if np.any(valid):
                m_dists[valid, act_last_slot[valid]] = np.inf

            t_dists = (self._targ_z - ao[:, 2]) / ad[:, 2]

            best_m_idx = np.argmin(m_dists, axis=1)
            best_m_dist = m_dists[np.arange(len(act)), best_m_idx]

            for local_i, global_i in enumerate(act):
                td = t_dists[local_i]
                md = best_m_dist[local_i]

                if td <= md or not np.isfinite(md):
                    # hits target
                    hit_target[global_i] = True
                    new_origins[global_i] = ao[local_i] + ad[local_i] * td
                    still_active[global_i] = False
                    last_mirror_per_ray[global_i, :] = -1
                else:
                    # hits a mirror
                    hit_mirror[global_i] = True
                    m = self._mirrors[best_m_idx[local_i]]
                    new_origins[global_i] = ao[local_i] + ad[local_i] * md
                    new_dirs[global_i] = m.reflect(ad[local_i:local_i + 1])[0]
                    last_mirror_per_ray[global_i, 1] = last_mirror_per_ray[global_i, 0]
                    last_mirror_per_ray[global_i, 0] = best_m_idx[local_i]

            hit_origins[act] = new_origins[act]

            # hit_dirs: direction AFTER this step (reflected for mirrors, 0 for target-hits)
            hit_dirs = new_dirs.copy()
            hit_dirs[hit_target] = 0.0

            self._bounces.append({
                'step_origins': step_origins,
                'step_dirs':    step_dirs,
                'hit_origins':  hit_origins,
                'hit_dirs':     hit_dirs,
                'hit_mirror':   hit_mirror,
                'hit_target':   hit_target,
            })

            cur_origins = new_origins.copy()
            cur_dirs = new_dirs.copy()

    def get_ray_params(self):
        """
        Return geometric parameters needed by side_view.

        :returns: dict with keys x_max, y_max, img_z, targ_z, mirrors, ray_grid_shape
        """
        return {
            'x_max':          self._x_max,
            'y_max':          self._y_max,
            'img_z':          self._img_z,
            'targ_z':         self._targ_z,
            'mirrors':        self._mirrors,
            'ray_grid_shape': self._ray_grid_shape,
        }

    def get_map(self):
        """
        Return a coarse bounce map derived from the pre-simulated ray data.

        :returns: (None, bounce_count) where bounce_count is (n_y, n_x) int32
                  with the mirror-bounce count per ray (-1 = active/not-yet-hit),
                  or (None, None) if no ray data is available.
        """
        if not self._bounces or self._ray_grid_shape is None:
            return None, None
        n_y, n_x = self._ray_grid_shape
        bounce_count = np.full(n_y * n_x, -1, dtype=np.int32)
        for k, step in enumerate(self._bounces):
            newly_hit = step['hit_target'] & (bounce_count == -1)
            bounce_count[newly_hit] = k
        return None, bounce_count.reshape(n_y, n_x)

    def get_initial_rays(self):
        """
        Return the initial ray origins and directions.

        :returns: (origins Nx3, directions Nx3), or (None, None) if not configured
        """
        return self._origins, self._directions

    def get_bounces(self):
        """
        Return the list of per-step bounce dicts.

        Each dict has keys: step_origins, step_dirs, hit_origins, hit_mirror, hit_target.
        Empty list if no mirrors configured.
        """
        return self._bounces


def test_raytracing(test_kind='corner-1'):
    """
    Interactive step-through test.  Uses a 10×10 square mirror tube.
    Press SPACE to advance one bounce step; Q or ESC to quit.
    Side-view panels update after each step so you can inspect the geometry.

    :param test_kind: which rays to trace:
        'grid'    - all rays (default)
        'border'  - only rays on the perimeter of the pixel grid
        'corners' - only the 4 corner rays of the pixel grid
        'corner-i' - where i can be one of [1,2,3,4] for top-left,top-right,bottom-left,bottom-right corner ray
    """
    from mirror_tube import MirrorTube
    from init_rays import compute_fov, find_img_z
    from side_view import SideViewState, render_side_view
    from geom import TARG_Z

    size = (20, 20)
    tube = MirrorTube.make_reg_n_gon(n=3,radius=0.2)
    x_max, y_max = compute_fov(*size)
    img_z = find_img_z(*size, tube.mirrors, x_max, y_max)

    rt = Raytracer(size, tube.mirrors, TARG_Z, x_max, y_max, img_z, bounce_file=None)
    rt._init_rays()

    w, h = size
    if test_kind == 'border':
        mask = (
            (rt._x_inds == 0) | (rt._x_inds == w - 1) |
            (rt._y_inds == 0) | (rt._y_inds == h - 1)
        )
        rt._step_active &= mask
        rt._display_mask = mask
    elif test_kind == 'corners':
        mask = (
            ((rt._x_inds == 0) | (rt._x_inds == w - 1)) &
            ((rt._y_inds == 0) | (rt._y_inds == h - 1))
        )
        rt._step_active &= mask
        rt._display_mask = mask
    elif test_kind.startswith('corner-'):
        try:
            corner_i = int(test_kind.split('-', 1)[1])
        except (IndexError, ValueError):
            raise ValueError("test_kind must be 'corner-1'..'corner-4'.")

        corner_xy = {
            1: (0, 0),
            2: (w - 1, 0),
            3: (0, h - 1),
            4: (w - 1, h - 1),
        }
        if corner_i not in corner_xy:
            raise ValueError("test_kind must be 'corner-1'..'corner-4'.")

        cx, cy = corner_xy[corner_i]
        mask = (rt._x_inds == cx) & (rt._y_inds == cy)
        rt._step_active &= mask
        rt._display_mask = mask

    state    = SideViewState(mode='start')
    win_name = "Raytracer Step Test  [SPACE=step  b/h/s/t=mode  r=reset zoom  Q=quit]"
    win_size = (1200, 800)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, *win_size)
    cv2.setMouseCallback(win_name, state.mouse_callback)

    done = False
    step_verbose = 2 if test_kind in ('corner-1', 'corners') else None
    print("Press SPACE to advance one step, Q/ESC to quit.")
    while True:
        rect = cv2.getWindowImageRect(win_name)
        if rect[2] > 10 and rect[3] > 10:
            win_size = (rect[2], rect[3])

        sv_img = render_side_view(rt, win_size, state=state)
        cv2.imshow(win_name, sv_img)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            if done:
                print("Raytracing already complete.")
            else:
                has_more = rt.step(verbose=step_verbose)
                if not has_more:
                    done = True
                    print(f"All rays traced in {len(rt.get_bounces())} steps.")
        elif k == ord('b'):
            state.mode = 'bounce'
        elif k == ord('h'):
            state.mode = 'hit'
        elif k == ord('s'):
            state.mode = 'start'
        elif k == ord('t'):
            state.mode = 'trails'
        elif k == ord('r'):
            state.reset_zoom()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_raytracing()
