import numpy as np
from typing import Callable, List, Tuple

class CMAES():
    def __init__(self, arg_names: List[str], ave_vec, sigma=1.0, max_iter=100):
        self.arg_names = arg_names
        self.dim = len(arg_names)
        self.population = int(4 + 3 * np.log(self.dim))
        self.mu = int(np.floor(self.population / 2))
        self.max_iter = max_iter
        self.m = np.array(ave_vec, dtype=np.float64)
        self.weights = self.calc_weights()
        self.mu_eff = 1.0 / (self.weights**2).sum()
        self.loss = float('inf')
        self.best_val = None
        self.sigma = float(sigma)
        self.c_1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
        1 - self.c_1,
        2.0 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff)
        )
        self.chi = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * (self.dim ** 2)))
        self.C = np.identity(self.dim)
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.p_c = np.zeros(self.dim)
        self.p_sigma = np.zeros(self.dim)

    def sample(self) -> List[float]:
        """多次元正規分布からサンプリングをする"""
        arr = np.random.multivariate_normal(mean=self.m, cov=self.C, size=self.dim)
        arr = arr.tolist()[0]
        return arr
    
    def calc_weights(self):
        """対数重みを計算する"""
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        return raw_weights / raw_weights.sum()
    
    def matrix_inverse_sqrt(self):
        # 固有値分解
        eigvals, eigvecs = np.linalg.eigh(self.C)
        
        # 数値安定性のために微小値で下限をつける
        eigvals = np.maximum(eigvals, 1e-20)
        
        # Λ^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        
        # C^{-1/2} = Q Λ^{-1/2} Q^T
        C_inv_sqrt = eigvecs @ D_inv_sqrt @ eigvecs.T
        return C_inv_sqrt

    def opt(self, f: Callable) -> Tuple[float, List[float]]:
        dim = self.dim
        mu_eff = self.mu_eff
        
        # 選抜を行うループ
        for _ in range(self.max_iter):
            # 個体集合を生成
            group: List[List[float]] = []
            for _ in range(self.population):
                group.append(self.sample())
            
            # 関数に入力する
            scores: List[Tuple[float, List[float]]] = []
            for x in group:
                arg_dict = {name: val for name, val in zip(self.arg_names, x)}
                current_loss = f(**arg_dict)
                scores.append((current_loss, x))
            
            # 損失で昇順に並べ替える
            scores.sort(key=lambda x: x[0])

            # 暫定出力値の更新
            if self.loss > scores[0][0]:
                self.loss = scores[0][0]
                self.best_val = scores[0][1]

            # self.muの個体を取り出す
            elites = scores[:self.mu]
            elites = np.array([i[1] for i in elites])

            # 平均値ベクトルの更新
            m_prev = self.m
            self.m = self.weights @ elites

            # ステップサイズσの更新処理
            y = (self.m - m_prev) / self.sigma
            p_sigma = (1 - self.c_sigma) * self.p_sigma
            p_sigma += np.sqrt(1 - (1 - self.c_sigma) ** 2) * mu_eff * (self.matrix_inverse_sqrt() @ y)
            self.p_sigma = p_sigma

            p_sigma_norm = np.linalg.norm(p_sigma)
            sigma_next = self.sigma * np.exp(
                (self.c_sigma / self.compute_d_sigma())
                * (p_sigma_norm / self.chi - 1)
            )
            self.sigma = sigma_next

            # 共分散行列のランクmu更新
            C_mu = np.zeros((dim, dim))
            for i in range(self.mu):
                x = np.array(x)
                # 列ベクトルに変換
                x_col = x.reshape(-1, 1)
                m_col = m_prev.reshape(-1, 1)
                C_mu = C_mu + self.weights[i] * ((x_col - m_col) @ (x_col - m_col).T / self.mu)

            # print(f"[DEBUG] C_mu: \n{C_mu}")
            C_mu /= self.sigma ** 2

            # 共分散行列のランク1更新
            p_c_next = (1 - self.c_c) * self.p_c + np.sqrt(1 - (1 - self.c_c) ** 2) * np.sqrt(mu_eff) * y
            self.p_c = p_c_next
            p_next_col = p_c_next.reshape(-1, 1)
            C_1 = p_next_col @ p_next_col.T

            # 共分散行列の更新
            C_new = (1 - self.c_mu - self.c_1) * self.C + self.c_mu * C_mu + self.c_1 * C_1
            self.C = C_new

        # print(f"[DEBUG] m: {m}")
        return (self.loss, self.best_val)

    def compute_d_sigma(self):
        return 1 + self.c_sigma + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1)

if __name__ == "__main__":
    def parametric_func(x, y):
        return (x - 1)**2 + (y - 2)**2

    def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

    # より困難な初期点での検証
    test_points = [
        [3, -1],
        [5, 5],      # 両方とも遠い
        [-2, 3],     # 負の値から
        [0, 0],      # 原点から
        [10, -5]     # 極端な点
    ]

    for init_point in test_points:
        cmaes = CMAES(arg_names=["x", "y"], ave_vec=init_point, max_iter=200, sigma=0.8)
        loss, value = cmaes.opt(rosenbrock)
        print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")