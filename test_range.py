from lib import CMAES

def obj_func(x):
    return x * x + 1

def parametric_func(x, y, a=1, b=1):
        return a * (x - 1)**2 + b * (y - 2)**2

def test_obj_func():
    for _ in range(100):
        lower = -5
        higher = 5
        bound_1 = [(lower, higher)]

        _loss, value = CMAES.opt(
                obj_func, 
                bound_1, 
                ["x"],
        )

        assert lower <= value[0] <= higher

def test_parametric_func():
    for _ in range(100):
        lower = -5
        higher = 5
        bound_2 = [(lower, higher), (lower, higher)]

        _loss, value = CMAES.opt(
                parametric_func, 
                bound_2, 
                ["x", "y"], 
        )

        assert lower <= value[0] <= higher and lower <= value[1] <= higher
