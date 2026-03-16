"""
Pre-set mirror configurations for the kaleidoscope menu.

1. equilateral triangle (r=.4)
2. isoceleses triangle, actute (r=.4, angle=20)
3. isoceleses triangle, obtuse (r=.4, angle=75)
4. square (r=.4)
5. hexagon (r=.4) 

"""
from mirror_tube import MirrorTube


class PresetFactory:
    PRESET_NAMES = ['equilateral triangle', 'acute isoceles triangle', 'obtuse isoceles triangle', 'square', 'hexagon']
    @staticmethod
    def make_preset(name, r=0.4):
        if name == 'equilateral triangle':
            return MirrorTube.make_isoceles(radius=r, iso_angle_deg=30)
        elif name == 'acute isoceles triangle':
            return MirrorTube.make_isoceles(radius=r, iso_angle_deg=20)
        elif name == 'obtuse isoceles triangle':
            return MirrorTube.make_isoceles(radius=r, iso_angle_deg=75)
        elif name == 'square':
            return MirrorTube.make_box(radius=r)
        elif name == 'hexagon':
            return MirrorTube.make_reg_n_gon(6, radius=r)
        else:
            raise ValueError(f'Unknown preset name: {name}')


def test_configs():
    for name in PresetFactory.PRESET_NAMES:
        print(f"Testing preset: {name}")
        tube = PresetFactory.make_preset(name, r=0.4)
        print(f"Preset {name} has {len(tube.mirrors)} mirrors.")


if __name__ == "__main__":
    test_configs()