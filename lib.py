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
        
        # 平均値ベクトルと共分散行列を初期化
        m = [0 for _ in range(dim)]
        C = np.identity(dim)

        # 初期個体集合を生成
        population: List[List[float]] = []
        for _ in range(population_size):
            sampled_array = np.random.multivariate_normal(mean=m, cov=C, size=dim)
            i = sampled_array.tolist()
            population.append(i[0])
        
        # 関数に入力して選抜する
        for _ in range(max_iter):
            scores: List[Tuple[float, List[float]]] = []
            for x in population:
                arg_dict = {name: val for name, val in zip(arg_names, x)}
                current_loss = f(**arg_dict)
                scores.append((x, current_loss))
            
            scores.sort(key=lambda x: x[1])

        best_val = scores[0][0]
        loss = scores[0][1]

        return (loss, best_val)
