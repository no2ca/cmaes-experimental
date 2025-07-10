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
        sigma = 1
        
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
            
            # 比較して選抜する
            scores: List[Tuple[float, List[float]]] = []
            for x in population:
                arg_dict = {name: val for name, val in zip(arg_names, x)}
                current_loss = f(**arg_dict)
                scores.append((current_loss, x))
            
            scores.sort(key=lambda x: x[0])
            # elite_sizeの個体を取り出す
            # コピーは不要か？
            elites = scores[:5]
            # print(f"elites: {elites}")
            # 平均値ベクトルの更新
            next_m = np.zeros(dim)
            weight = float(elite_size)
            for x in elites:
                x = np.array(x[1]) # dim次元のリストである値を取り出す
                next_m = next_m + x * weight / float((elite_size) * (elite_size + 1) / 2)
                weight -= 1
            
            m = next_m
            print(f"m: {m}")

        loss = scores[0][0]
        best_val = scores[0][1]

        return (loss, best_val)

if __name__ == "__main__":
    def parametric_func(x, y, a=1, b=1):
        return a * (x - 1)**2 + b * (y - 2)**2
    
    lower = -5
    higher = 5
    bound_2 = [(lower, higher), (lower, higher)]

    _loss, value = CMAES.opt(
            parametric_func, 
            bound_2, 
            ["x", "y"], 
            max_iter = 10,
    )