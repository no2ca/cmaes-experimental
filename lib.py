import random
import numpy as np
from typing import Callable, List, Tuple

class CMAES():
    def __init__(self) -> None:
         pass
    def opt(
            f: Callable,
            bounds: List[Tuple[float, float]],
            arg_names: List[str],
            population_size = 10,
            elite_size = 5,
            max_iter = 100,
            tolerance = 1e-3,
            mut_rate = 0.1,
        ) -> Tuple[float, List[float]]:
        dim = len(bounds)
        loss = float('inf')
        best_val = None
        step_size = mut_rate
        sigma = 1.0
        c_mu = 0.5
        
        # 平均値ベクトルと共分散行列を初期化
        m = np.zeros(dim)
        C = np.identity(dim)
        
        # 選抜を行うループ
        for _ in range(max_iter):
            # 個体集合を生成
            population: List[List[float]] = []
            for _ in range(population_size):
                sampled_array = np.random.multivariate_normal(mean=m, cov=C, size=dim)
                i = sampled_array.tolist()
                population.append(i[0])
            
            # 関数に通して並べ替える
            scores: List[Tuple[float, List[float]]] = []
            for x in population:
                arg_dict = {name: val for name, val in zip(arg_names, x)}
                current_loss = f(**arg_dict)
                scores.append((current_loss, x))
            
            scores.sort(key=lambda x: x[0])

            # 暫定出力値の更新
            if loss > scores[0][0]:
                loss = scores[0][0]
                best_val = scores[0][1]

            # elite_sizeの個体を取り出す
            elites = scores[:5]

            # 平均値ベクトルの更新処理
            next_m = np.zeros(dim)
            weight = float(elite_size)
            for x in elites:
                x = np.array(x[1]) # dim次元のリストである値を取り出す
                next_m = next_m + x * weight / float((elite_size) * (elite_size + 1) / 2)
                weight -= 1

            # 共分散行列のランクmu更新
            C_tmp = np.zeros((dim, dim))
            for x in elites:
                mu = float(elite_size)
                x = np.array(x[1])
                # 列ベクトルに変換
                x_col = x.reshape(-1, 1)
                m_col = m.reshape(-1, 1)
                C_tmp = C_tmp + ((x_col - m_col) @ (x_col - m_col).T / mu)
            
            # print(f"[DEBUG] C_tmp: \n{C_tmp}")
            C_tmp /= sigma ** 2
            C = (1 - c_mu) * C + c_mu * C_tmp

            # 平均値ベクトルの更新
            m = next_m

        # print(f"[DEBUG] m: {m}")
        return (loss, best_val)

if __name__ == "__main__":
    def parametric_func(x, y, a=1, b=1):
        return a * (x - 1)**2 + b * (y - 2)**2
    
    lower = -5
    higher = 5
    bound_1 = [(lower, higher), (lower, higher)]

    loss, value = CMAES.opt(
            parametric_func, 
            bound_1, 
            ["x", "y"], 
            max_iter = 10,
    )

    print(f"最適化結果: \n解={loss}, 値={value}")

    def rosenbrock(x, y, z):
        return (1 - x)**2 + 100*(y - x**2)**2 + (z - 1)**2
    
    bounds = [(-2, 2), (-2, 2), (-2, 2)]
    loss, value = CMAES.opt(rosenbrock, bounds, ["x", "y", "z"], max_iter=20)

    print(f"Rosenbrock関数の最適化結果: \n解={loss}, 値={value}")