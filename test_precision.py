from lib import CMAES

def obj_func(x):
        return x * x + 1

def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

def test_obj_func_easy():
        eps = 0.1
        ans = 0
        test_points = [
                [2],
                [-2],     # 負の値から
                [0],      # 原点から
        ]

        for init_point in test_points:
                cmaes = CMAES(arg_names=["x"], ave_vec=init_point, max_iter=200, sigma=0.8)
                loss, value = cmaes.opt(obj_func)
                print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")
                assert abs(ans - value[0]) <= eps

def test_rosenbrock_easy():
        eps = 0.1
        ans = [1, 1]
        test_points = [
                [0, 0],
        ]

        for init_point in test_points:
                cmaes = CMAES(arg_names=["x", "y"], ave_vec=init_point, max_iter=1000, sigma=0.8)
                loss, value = cmaes.opt(rosenbrock)
                print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")
                assert abs(ans[0] - value[0]) <= eps and abs(ans[1] - value[1]) <= eps


def test_obj_strict():
        eps = 1e-4
        ans = 0
        test_points = [
                [3],
                [1],
                [-2],
                [0], 
                [3]  
        ]

        for init_point in test_points:
                cmaes = CMAES(arg_names=["x"], ave_vec=init_point, max_iter=200, sigma=0.8)
                loss, value = cmaes.opt(obj_func)
                print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")
                assert abs(ans - value[0]) <= eps

def test_rosenbrock_strict():
        eps = 1e-3
        ans = [1, 1]
        test_points = [
                [1, -1],
                [1, 2], 
                [-2, 0],
                [0, 0], 
                [-1, -1]  
        ]

        for init_point in test_points:
                cmaes = CMAES(arg_names=["x", "y"], ave_vec=init_point, max_iter=300, sigma=0.6)
                loss, value = cmaes.opt(rosenbrock)
                print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")
                assert abs(ans[0] - value[0]) <= eps and abs(ans[1] - value[1]) <= eps