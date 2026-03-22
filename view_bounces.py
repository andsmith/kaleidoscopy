"""
view_bounces.py - Replay a Raytracer bounce-history JSON in the side_view.

Contains:
  FakeRaytracer - moved from raytracing.py; supports space-key step-through.
  JsonRaytracer - loads a bounce JSON and replays with the same interface.
  run_viewer()  - interactive side-view window (SPACE = advance one step).

Usage:
    python view_bounces.py bounces.json
"""
import json
import sys

import numpy as np
import cv2
import logging

from mirror import Mirror
from geom import TARG_Z as _DEFAULT_TARG_Z
from side_view import SideViewState, render_side_view


# ---------------------------------------------------------------------------
# FakeRaytracer (moved from raytracing.py)
# ---------------------------------------------------------------------------

class FakeRaytracer:
    """
    Stub raytracer that returns a placeholder image of the correct output size.
    Can optionally be configured with mirror geometry to generate synthetic ray
    bounce data for side_view.py queries.
    """

    _FAKE_GRID_LONG = 20   # rays along the longer dimension for fake bounce grid
    _MAX_BOUNCES = 8

    def __init__(self, size, mirrors=None, x_max=1.0, y_max=None, img_z=None,
                 targ_z=None, step_through=False):
        """
        :param size: (width, height) of the output image
        :param mirrors: list of Mirror objects (optional; enables ray data generation)
        :param x_max: FOV half-width in natural coords at targ_z (default 1.0)
        :param y_max: FOV half-height in natural coords (default x_max * h/w)
        :param img_z: image plane Z coordinate (required if mirrors provided)
        :param targ_z: target plane Z coordinate (default TARG_Z from geom)
        :param step_through: if True, start with no bounces revealed; call step() to
               advance one at a time.  If False (default), all bounces are visible
               immediately (backward-compatible behaviour).
        """
        self._size = size
        self._mirrors = mirrors
        w, h = size
        self._x_max = x_max
        self._y_max = y_max if y_max is not None else x_max * h / w
        self._targ_z = targ_z if targ_z is not None else _DEFAULT_TARG_Z
        self._img_z = img_z

        self._origins = None
        self._directions = None
        self._ray_grid_shape = None
        self._bounces = []

        if mirrors is not None and img_z is not None:
            self._generate_fake_rays()

        # step_through=False reveals all pre-computed bounces immediately
        self._step_idx = 0 if step_through else len(self._bounces)

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

            # Exclude the last two mirrors each ray bounced off.
            for slot in range(2):
                act_last_slot = last_mirror_per_ray[act, slot]
                valid_slot = act_last_slot >= 0
                if np.any(valid_slot):
                    m_dists[valid_slot, act_last_slot[valid_slot]] = np.inf

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

    # ------------------------------------------------------------------
    # Step-through interface
    # ------------------------------------------------------------------

    def step(self):
        """Reveal the next bounce step.  Returns True if more steps remain."""
        if self._step_idx < len(self._bounces):
            self._step_idx += 1
        return self._step_idx < len(self._bounces)

    # ------------------------------------------------------------------
    # Side-view interface
    # ------------------------------------------------------------------

    def get_ray_params(self):
        """Return geometric parameters needed by side_view.render_side_view."""
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

        Always reflects the full pre-computed result regardless of step_idx,
        so the heatmap is available from the start.
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
        """Return (origins Nx3, directions Nx3), or (None, None) if not configured."""
        return self._origins, self._directions

    def get_bounces(self):
        """Return bounce dicts revealed so far (all of them if step_through=False)."""
        return self._bounces[:self._step_idx]


# ---------------------------------------------------------------------------
# JsonRaytracer
# ---------------------------------------------------------------------------

class JsonRaytracer:
    """
    Loads a Raytracer bounce-history JSON (written with bounce_file=...) and
    replays it through the side_view.  Supports SPACE-key step-through via
    the same step() / get_bounces() interface as Raytracer and FakeRaytracer.
    """

    def __init__(self, bounce_file):
        """
        :param bounce_file: path to a JSON file produced by Raytracer(bounce_file=...)
        """
        logging.info("Loading bounce file: %s", bounce_file)
        with open(bounce_file) as f:
            data = json.load(f)
        self._load(data)
        self._step_idx = 0
        logging.info("Loaded %d iterations, %d rays.", len(self._bounces), self._N)

    def _load(self, data, filter_rays=True):
        """
        Load bounce data from the given JSON dict.  If filter_rays=True, only keep
        rays that show up in the bounce history (not just the init rays).
        """
        
        header = data['header']

        self._mirrors = [
            Mirror(m['p0_xy'], m['p1_xy'])
            for m in header['mirrors']
        ]
        self._img_z  = header['image_z']
        self._targ_z = header['target_z']
        self._x_max  = header['x_max']
        self._y_max  = header['y_max']
        h, w = header['grid_shape']
        self._grid_shape = (h, w)
        N = h * w
        self._N = N

        # Reconstruct initial origins (from stored XY + image_z) and directions
        # (directions are implicit: rays travel from the eye at (0,0,0))
        origins_xy = np.array(header['ray_origins_xy'], dtype=np.float64)  # (N, 2)
        origins_z  = np.full((N, 1), self._img_z)
        self._initial_origins    = np.hstack([origins_xy, origins_z])
        norms                    = np.linalg.norm(self._initial_origins, axis=1, keepdims=True)
        self._initial_directions = self._initial_origins / norms

        # Build ray-ID string -> flat index map
        ray_idx = {f'RAY_{i:06d}': i for i in range(N)}

        # Optional display filter: only show rays explicitly referenced in history.
        # This is useful for logs containing a subset of rays.
        if filter_rays:
            display_mask = np.zeros(N, dtype=bool)
            for iter_record in data['iterations']:
                for ray_id in iter_record.get('target_hits', []):
                    i = ray_idx.get(ray_id)
                    if i is not None:
                        display_mask[i] = True
                for mh in iter_record.get('mirror_hits', []):
                    i = ray_idx.get(mh.get('ray'))
                    if i is not None:
                        display_mask[i] = True
            self._display_mask = display_mask
            logging.info("filter_rays enabled: %d/%d rays referenced in history.",
                         int(np.count_nonzero(display_mask)), N)
        else:
            self._display_mask = np.ones(N, dtype=bool)

        # Reconstruct per-step bounce dicts from the JSON iteration records.
        # cur_pos / cur_dir track each ray's live position and direction as we
        # replay.  Inactive rays (already hit target) keep their last values so
        # that step_origins / hit_origins stay consistent with the Raytracer output.
        cur_pos = self._initial_origins.copy()
        cur_dir = self._initial_directions.copy()
        bounce_count_flat = np.full(N, -1, dtype=np.int32)

        self._bounces = []
        for iter_record in data['iterations']:
            step_num    = iter_record['step']
            step_origins = cur_pos.copy()   # snapshot before any updates this step
            step_dirs    = cur_dir.copy()
            hit_origins  = cur_pos.copy()   # inactive rays: stay in place
            hit_dirs     = cur_dir.copy()   # inactive rays: carry direction forward
            hit_mirror   = np.zeros(N, dtype=bool)
            hit_target   = np.zeros(N, dtype=bool)

            # --- target hits ---
            t_ids_raw = iter_record.get('target_hits', [])
            if t_ids_raw:
                t_ids = np.array([ray_idx[r] for r in t_ids_raw], dtype=np.int32)
                to = cur_pos[t_ids]
                td = cur_dir[t_ids]
                ts = (self._targ_z - to[:, 2]) / td[:, 2]
                hit_origins[t_ids]        = to + td * ts[:, None]
                hit_dirs[t_ids]           = 0.0
                hit_target[t_ids]         = True
                bounce_count_flat[t_ids]  = step_num
                cur_pos[t_ids] = hit_origins[t_ids]
                cur_dir[t_ids] = 0.0

            # --- mirror hits ---
            m_hits_raw = iter_record.get('mirror_hits', [])
            if m_hits_raw:
                m_ids    = np.array([ray_idx[mh['ray']]            for mh in m_hits_raw], dtype=np.int32)
                m_xyz    = np.array([mh['hit_xyz']                 for mh in m_hits_raw], dtype=np.float64)
                m_newdir = np.array([mh['new_direction_xyz']       for mh in m_hits_raw], dtype=np.float64)
                hit_origins[m_ids] = m_xyz
                hit_dirs[m_ids]    = m_newdir
                hit_mirror[m_ids]  = True
                cur_pos[m_ids] = m_xyz
                cur_dir[m_ids] = m_newdir

            self._bounces.append({
                'step_origins': step_origins,
                'step_dirs':    step_dirs,
                'hit_origins':  hit_origins,
                'hit_dirs':     hit_dirs,
                'hit_mirror':   hit_mirror,
                'hit_target':   hit_target,
            })

        self._bounce_count = bounce_count_flat.reshape(h, w)

    # ------------------------------------------------------------------
    # Step-through interface
    # ------------------------------------------------------------------

    def step(self):
        """Reveal the next iteration.  Returns True if more iterations remain."""
        if self._step_idx < len(self._bounces):
            self._step_idx += 1
        return self._step_idx < len(self._bounces)

    # ------------------------------------------------------------------
    # Side-view interface
    # ------------------------------------------------------------------

    def get_ray_params(self):
        return {
            'x_max':          self._x_max,
            'y_max':          self._y_max,
            'img_z':          self._img_z,
            'targ_z':         self._targ_z,
            'mirrors':        self._mirrors,
            'ray_grid_shape': self._grid_shape,
            'display_mask':   self._display_mask,
        }

    def get_initial_rays(self):
        return self._initial_origins, self._initial_directions

    def get_bounces(self):
        """Return only the iterations revealed so far."""
        return self._bounces[:self._step_idx]

    def get_map(self):
        """Return (None, bounce_count) where bounce_count reflects the full trace."""
        return None, self._bounce_count


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def run_viewer(raytracer, title=None):
    """
    Open an interactive side-view window for stepping through a bounce replay.

    Keyboard controls:
      SPACE      - advance one bounce step
      b/h/s/t    - switch mode (bounce / hit / start / trails)
      r          - reset zoom
      Q / ESC    - quit
    """
    n_total  = len(getattr(raytracer, '_bounces', []))
    win_name = (title or "view_bounces") + \
               "  [SPACE=step  b/h/s/t=mode  r=reset zoom  Q=quit]"
    win_size = (1400, 900)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, *win_size)

    state = SideViewState(mode='bounce')
    cv2.setMouseCallback(win_name, state.mouse_callback)

    print(f"Press SPACE to step through bounces.  Total: {n_total} iterations.")

    done = False
    while True:
        rect = cv2.getWindowImageRect(win_name)
        if rect[2] > 10 and rect[3] > 10:
            win_size = (rect[2], rect[3])

        sv_img = render_side_view(raytracer, win_size, state=state)

        # Overlay step counter in the bottom-left corner
        step_now = getattr(raytracer, '_step_idx', '?')
        cv2.putText(sv_img, f"step {step_now} / {n_total}",
                    (8, sv_img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)

        cv2.imshow(win_name, sv_img)
        k = cv2.waitKey(30) & 0xFF

        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            if done:
                print("All steps already revealed.")
            else:
                has_more = raytracer.step()
                step_now = getattr(raytracer, '_step_idx', '?')
                print(f"  step {step_now} / {n_total}")
                if not has_more:
                    done = True
                    print("All steps revealed.")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python view_bounces.py <bounce_file.json>")
        sys.exit(1)
    rt = JsonRaytracer(sys.argv[1])
    run_viewer(rt, title=sys.argv[1])
