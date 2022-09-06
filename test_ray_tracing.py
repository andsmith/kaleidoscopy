from ray_tracing import ScopeTracer
from mirrors_rectangle import RectangularPrism
import logging


def test_raytracing():
    mirrors = RectangularPrism()
    mirrors._aspect = 2.0
    input_shape = [5, 5]
    output_shape = [4, 6]

    rt = ScopeTracer(mirrors, output_shape, lambda x: None, n_cores=1)
    rt.run()
    m = rt.get_current_result()  # gets final result when completed
    import pprint
    pprint.pprint(m)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_raytracing()
