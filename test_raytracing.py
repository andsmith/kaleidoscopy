import logging
from ray_tracing import RayTracer
from mirrors_rectangle import RectangularPrism









def test_ray_tracing():
    def _update_test(*args, **kwargs):
        print("Ray tracer updated.")
        pass
    plot_no = 1

    shape = (480,640)
    rt = RayTracer(mirrors=mirrors, img_shape=shape, update_callback=_update_test, n_cores=1)

    rt.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ray_tracing()
