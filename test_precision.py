from lib import CMAES

def obj_func(x):
    return x * x + 1

def parametric_func(x, y, a=1, b=1):
        return a * (x - 1)**2 + b * (y - 2)**2

def test_obj_func():
    for _ in range(10):
        lower = -5
        higher = 5
        eps = 0.1
        ans = 1.0
        bound_1 = [(lower, higher)]

        _loss, value = CMAES.opt(
                obj_func, 
                bound_1, 
                ["x"],
        )

        assert abs(ans - value[0]) <= eps

def test_parametric_func():
    for _ in range(10):
        lower = -5
        higher = 5
        eps = 0.1
        ans = [1, 2]
        bound_2 = [(lower, higher), (lower, higher)]

        _loss, value = CMAES.opt(
                parametric_func, 
                bound_2, 
                ["x", "y"], 
        )

        assert abs(ans[0] - value[0]) <= eps and abs(ans[1] - value[1]) <= eps