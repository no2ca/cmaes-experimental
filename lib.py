import numpy as np
import math
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
        c_mu = 0.3
        c_1 = 0.3
        chi = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * (dim ** 2)))
        w = 0
        wari = elite_size * (elite_size + 1) / 2
        for i in range(1, elite_size+1):
            w += (i / wari) ** 2
        mu_eff = 1.0 / w
        print(f"mu_eff: {mu_eff}")
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        print(f"c1: {c_c}")
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # pathは加重平均で計算されるベクトル
        p_c = np.zeros(dim)
        p_sigma = np.zeros(dim)

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
            elites = scores[:elite_size]

            # 平均値ベクトルの更新処理
            m_next = np.zeros(dim)
            weight = float(elite_size)
            for x in elites:
                x = np.array(x[1]) # dim次元のリストである値を取り出す
                m_next = m_next + x * weight / float((elite_size) * (elite_size + 1) / 2)
                weight -= 1

            # ステップサイズの更新処理
            y = (m_next - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(1 - (1 - c_sigma) ** 2) * mu_eff * matrix_inverse_sqrt(C) * y
            p_sigma_norm = np.linalg.norm(p_sigma)
            sigma_next = sigma * math.exp(c_sigma / compute_d_sigma(c_sigma, mu_eff, dim) * (p_sigma_norm / chi - 1))

            # 共分散行列のランクmu更新
            C_mu = np.zeros((dim, dim))
            for x in elites:
                mu = float(elite_size)
                x = np.array(x[1])
                # 列ベクトルに変換
                x_col = x.reshape(-1, 1)
                m_col = m.reshape(-1, 1)
                C_mu = C_mu + ((x_col - m_col) @ (x_col - m_col).T / mu)
            
            # print(f"[DEBUG] C_mu: \n{C_mu}")
            C_mu /= sigma ** 2

            # 共分散行列のランク1更新
            p_next = (1 - c_c) * p_c + math.sqrt(1 - (1 - c_c) ** 2) * mu_eff * y
            p_next_col = p_next.reshape(-1, 1)
            C_1 = p_next_col @ p_next_col.T

            # 平均値ベクトルと共分散行列の更新
            m = m_next
            C = (1 - c_mu - c_1) * C + c_mu * C_mu + c_1 * C_1
            p_c = p_next
            sigma = sigma_next

        # print(f"[DEBUG] m: {m}")
        return (loss, best_val)
    
def matrix_inverse_sqrt(C):
    # 固有値分解
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # 数値安定性のために微小値で下限をつける
    eigvals = np.maximum(eigvals, 1e-20)
    
    # Λ^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    
    # C^{-1/2} = Q Λ^{-1/2} Q^T
    C_inv_sqrt = eigvecs @ D_inv_sqrt @ eigvecs.T
    return C_inv_sqrt

def compute_d_sigma(c_sigma, mu_eff, n):
    return 1 + c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)

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
            max_iter = 500,
    )

    print(f"最適化結果: \n解={loss}, 値={value}")

    def rosenbrock(x, y):
        return (5 - x)**2 + 100*(y - x**2)**2
    
    bounds = [(-2, 2), (-2, 2)]
    loss, value = CMAES.opt(rosenbrock, bounds, ["x", "y"], max_iter=500)
    print(f"Rosenbrock関数の最適化結果: \n解={loss}, 値={value}")