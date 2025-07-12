from lib import CMAES

def f(x):
        return x * x + 1

def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

def test_f():
        eps = 1e-2
        ans = 0
        c = 0
        N = 30
        test_points = [
                [3],
                [1],
                [-2],
                [0], 
        ]
        for i in range(N):
                for init_point in test_points:
                        cmaes = CMAES(arg_names=["x"], ave_vec=init_point, max_iter=200, sigma=0.8)
                        loss, value = cmaes.opt(f)
                        print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")
                        if abs(ans - value[0]) <= eps:
                                c += 1
        assert c >= N * 0.7 * len(test_points)

def test_rosenbrock():
        eps = 1e-2
        eps2 = 1e-1
        ans = [1, 1]
        init_point = [0.5, 1.5]
        count = 0
        N = 30
        for i in range(N):
                cmaes = CMAES(arg_names=["x", "y"], ave_vec=init_point, max_iter=200)
                loss, value = cmaes.opt(rosenbrock)
                print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")
                if abs(ans[0] - value[0]) <= eps and abs(ans[1] - value[1]) <= eps:
                        count += 1
                elif abs(ans[0] - value[0]) <= eps2 and abs(ans[1] - value[1]) <= eps2:
                        count += 0.3
        print(f"count: {count}")
        assert count >= N * 0.7
