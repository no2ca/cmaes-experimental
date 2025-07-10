from lib import CMAES

def obj_func(x):
        return x * x + 1

def parametric_func(x, y, a=1, b=1):
        return a * (x - 5)**2 + b * (y - 4)**2

def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

def test_obj_func_easy():
    for _ in range(100):
        lower = -5
        higher = 5
        eps = 0.1
        ans = 0
        bound_1 = [(lower, higher)]

        _loss, value = CMAES.opt(
                obj_func, 
                bound_1, 
                ["x"],
        )

        assert abs(ans - value[0]) <= eps

def test_parametric_func_easy():
    for _ in range(100):
        lower = -5
        higher = 5
        eps = 0.1
        ans = [5, 4]
        bound_2 = [(lower, higher), (lower, higher)]

        _loss, value = CMAES.opt(
                parametric_func, 
                bound_2, 
                ["x", "y"], 
        )

        assert abs(ans[0] - value[0]) <= eps and abs(ans[1] - value[1]) <= eps

def test_rosenbrock_easy():
    ans = [1, 1]
    eps = 0.1
    for i in range(100): 
        bounds = [(-2, 2), (-2, 2)]
        loss, value = CMAES.opt(rosenbrock, bounds, ["x", "y"], max_iter=500)
        print(f"count: {i}")
        assert abs(ans[0] - value[0]) <= eps and abs(ans[1] - value[1]) <= eps


def test_obj_strict():
        lower = -5
        higher = 5
        eps = 1e-4
        ans = 0
        for _ in range(50):
                bound_1 = [(lower, higher)]

                _loss, value = CMAES.opt(
                        obj_func, 
                        bound_1, 
                        ["x"],
                        max_iter=500,
                )

                assert abs(ans - value[0]) <= eps

def test_parametric_strict():
        lower = -5
        higher = 5
        eps = 1e-4
        ans = [5, 4]
        for _ in range(50):
                bound_2 = [(lower, higher), (lower, higher)]

                _loss, value = CMAES.opt(
                        parametric_func, 
                        bound_2, 
                        ["x", "y"], 
                        max_iter=500,
                )

                assert abs(ans[0] - value[0]) <= eps and abs(ans[1] - value[1]) <= eps
