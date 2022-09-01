import logging
from ray_tracing import make_unit_rays
from mirrors_rectangle import RectangularPrism









def test_make_unit_rays():
    #import ipdb ;ipdb.set_trace()
    origins, directions = make_unit_rays((11,20))  # landscape
    print(origins.shape)
    print(directions)
    origins, directions = make_unit_rays((15,10))  # portrait
    print(origins.shape)
    print(directions)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_make_unit_rays()
    print("All tests passed.")
