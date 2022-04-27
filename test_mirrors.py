import logging
import numpy as np
import matplotlib.pyplot as plt
from mirrors_rectangle import RectangularPrism
from mirrors_isosceles import IsoscelesPrism
from mirrors_n_gon import NGonPrism
from mirrors_circle import CirclePrism

TESTS = [{'type': RectangularPrism,
          'mods': [{'_aspect': 1.0, '_aperture_scale': 1.0},
                   {'_aspect': 3.0, '_aperture_scale': 0.8},
                   {'_aspect': 1.0 / 3.0, '_aperture_scale': 0.2}]},
         {'type': IsoscelesPrism,
          'mods': [{'_theta': np.deg2rad(10.0), '_aperture_scale': 1.0},
                   {'_theta': np.deg2rad(30.0), '_aperture_scale': 0.8},
                   {'_theta': np.deg2rad(40.0), '_aperture_scale': 0.2}]},
         {'type': CirclePrism,
          'mods': [{'_aperture_scale': 1.0},
                   {'_aperture_scale': 0.8},
                   {'_aperture_scale': 0.2}]},
         {'type': NGonPrism,
          'mods': [{'_n': 6, '_aperture_scale': 1.0},
                   {'_n': 5, '_aperture_scale': 0.8},
                   {'_n': 7, '_aperture_scale': 0.2}]}]


def _test_mirrors_inscribed_rectangles():  # interactive

    aspect = 640.0 / 480.0

    for test in TESTS:
        logging.info("Testing:  %s" % (test['type'].__name__,))
        for mod_ind, mod in enumerate(test['mods']):
            mod_change_strs = []
            mirrors = test['type']()
            for mod_param in mod:
                mod_change_strs.append("%s:  %s" % (mod_param, mod[mod_param]))
                setattr(mirrors, mod_param, mod[mod_param])
            logging.info("   ... mod:  %s" % (", ".join(mod_change_strs),))
            box = mirrors.get_inscribed_rectangle_size(aspect=aspect)
            verts = mirrors.get_rel_shape()
            verts = np.vstack([verts, verts[0, :].reshape(1, 2)])
            plt.subplot(1,len(test['mods']), mod_ind+1)
            plt.plot(verts[:, 0], verts[:, 1], '-o', markersize=10)
            center = np.array([0.5, 0.5])
            box_coords = np.array([[center[0] - box[0], center[1] - box[1]],
                                   [center[0] - box[0], center[1] + box[1]],
                                   [center[0] + box[0], center[1] + box[1]],
                                   [center[0] + box[0], center[1] - box[1]],
                                   [center[0] - box[0], center[1] - box[1]]])
            plt.plot(box_coords[:, 0], box_coords[:, 1], '-')
            plt.gca().set_aspect('equal')
            plt.gca().set_xlim((0,1))
            plt.gca().set_ylim((0, 1))
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_mirrors_inscribed_rectangles()
