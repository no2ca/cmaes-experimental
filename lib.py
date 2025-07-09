import random
import numpy as np
from typing import Callable, List, Tuple

def cma_es(
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
    population = []
    for _ in range(population_size):
        i = [random.uniform(low, high) for low, high in bounds]
        population.append(i)
    
    # 関数に入力して選抜する
    for _ in range(max_iter):
        scores = []
        for x in population:
            arg_dict = {name: val for name, val in zip(arg_names, x)}
            score = f(**arg_dict)
            scores.append((x, score))
        
        scores.sort(key=lambda x: x[1])
        current_best = scores[0]
        
        if current_best[1] < loss:
            step_size = abs(loss - current_best[1])
            loss = current_best[1]
            best_val = current_best[0]
        if step_size < 10e-6:
            break
    
    return (loss, best_val)