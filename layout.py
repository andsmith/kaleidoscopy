import numpy as np

LAYOUT = {'icons': {'size': 200,
                    'colors': {'background': (254, 250, 245),
                               'foreground': (10, 10, 15)},
                    'fig_top_y': .1,  # draw up to here
                    'fig_bottom_y': .70,  # draw down to here
                    'text_bottom_y': 0.90},  # write on this baseline
          'mouse_controls': {'v_adjust_divisor': 400.0,
                             'h_adjust_divisor': 400.0},  # field of view, ~ 1 / mouse sensitivity
          'osd': {'text_color': (255, 254, 250),
                  'bkg_color': (118, 100, 90),
                  'osd_alpha': 0.65}}
