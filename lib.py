import numpy as np
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class CMAES():
    def __init__(self, arg_names: List[str], ave_vec: List[float], sigma=1.0, max_iter=100, population=None, mu=None):
        self.arg_names = arg_names
        self.dim = len(ave_vec)
        self.max_iter = max_iter
        # 個体数と選抜数
        self.population = population if population else int(4 + 3 * np.log(self.dim))
        self.mu = mu if mu else int(np.floor(self.population / 2))
        # 平均値ベクトル
        self.m = np.array(ave_vec, dtype=np.float64)
        # 重み行列の計算(muを定義した後)
        self.weights = self.calc_weights()
        self.mu_eff = 1.0 / (self.weights**2).sum()
        self.sigma = float(sigma)
        self.C = np.identity(self.dim)
        self.c_1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
        1 - self.c_1,
        2.0 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff)
        )
        self.chi = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * (self.dim ** 2)))
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.p_c = np.zeros(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.loss = float('inf')
        self.best_val = None

        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'mean_vector': [],
            'sigma': [],
            'eigenvalues': [],
            'populations': []  # 各世代の全個体
        }

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

    def compute_d_sigma(self):
        return 1 + self.c_sigma + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1)
    
    def debug(self):
        print(f"weights: {self.weights}")
        print(f"")
    
    def record_history(self, fitness_values, population):
        self.history['best_fitness'].append(np.min(fitness_values))
        self.history['mean_fitness'].append(np.mean(fitness_values))
        self.history['worst_fitness'].append(np.max(fitness_values))
        self.history['mean_vector'].append(self.m.copy())
        self.history['sigma'].append(self.sigma)
        eigenvals, _ = np.linalg.eigh(self.C)
        self.history['eigenvalues'].append(eigenvals.copy())
        self.history['populations'].append(population.copy())

    def opt(self, f: Callable) -> Tuple[float, List[float]]:
        dim = self.dim
        mu_eff = self.mu_eff
        
        # 選抜を行うループ
        for gen in range(self.max_iter):
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
                # print(f"DEBUG loss: {scores[0][0]}")
                self.loss = scores[0][0]
                self.best_val = scores[0][1]
            
            fitness_values = np.array([i[0] for i in scores])
            population = np.array([i[1] for i in scores])
            # print(f"min(fitness_values): {np.min(fitness_values)}")
            self.record_history(fitness_values, population)

            # self.muの個体を取り出す
            elites = scores[:self.mu]
            elites = np.array([i[1] for i in elites])

            # 平均値ベクトルの更新
            m_old = self.m
            self.m = self.weights @ elites
            # print(f"m: {self.m}")

            # 共分散行列のランクmu更新
            C_mu = np.zeros((dim, dim))
            for i in range(self.mu):
                x = np.array(elites[i])
                y_i = x - m_old
                C_mu = C_mu + self.weights[i] * (np.outer(y_i, y_i) / self.mu)

            # print(f"[DEBUG] C_mu: \n{C_mu}")
            C_mu /= self.sigma ** 2

            # ステップサイズσの更新処理
            y = (self.m - m_old) / self.sigma
            p_sigma = (1 - self.c_sigma) * self.p_sigma
            p_sigma += np.sqrt(1 - (1 - self.c_sigma) ** 2) * mu_eff * (self.matrix_inverse_sqrt() @ y)

            p_sigma_norm = np.linalg.norm(p_sigma)
            self.sigma = self.sigma * np.exp(
                (self.c_sigma / self.compute_d_sigma())
                * (p_sigma_norm / self.chi - 1)
            )
            self.p_sigma = p_sigma

            """
            # ステップサイズが多すぎるときにCの更新を止める
            left = np.sqrt((self.p_sigma ** 2).sum()) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (gen+1)))
            right = (1.4 + 2 / (self.dim + 1)) * self.chi
            hsigma = 1 if left < right else 0
            d_hsigma = (1 - hsigma) * self.c_c * (2 - self.c_c)
            """

            # 共分散行列のランク1更新
            self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(1 - (1 - self.c_c) ** 2) * np.sqrt(mu_eff) * y
            C_1 = np.outer(self.p_c, self.p_c)

            # 共分散行列の更新
            C_new = (1 - self.c_mu - self.c_1) * self.C + self.c_mu * C_mu + self.c_1 * C_1
            self.C = C_new

        # print(f"[DEBUG] m: {m}")
        return (self.loss, self.best_val)
    
    def plot_convergence(self, figsize=(12, 8), ans=None):
        """収束履歴をプロット"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 適応度の履歴
        generations = range(len(self.history['best_fitness']))
        axes[0, 0].semilogy(generations, self.history['best_fitness'], 'b-', label='Best')
        axes[0, 0].semilogy(generations, self.history['mean_fitness'], 'g-', label='Mean')
        axes[0, 0].semilogy(generations, self.history['worst_fitness'], 'r-', label='Worst')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # ステップサイズの履歴
        axes[0, 1].semilogy(generations, self.history['sigma'], 'purple')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Step Size (σ)')
        axes[0, 1].set_title('Step Size Evolution')
        axes[0, 1].grid(True)

        if self.dim == 2:
            mean_vectors = np.array(self.history['mean_vector'])
            axes[1, 0].plot(mean_vectors[:, 0], mean_vectors[:, 1], 'o-', markersize=3)
            axes[1, 0].plot(mean_vectors[0, 0], mean_vectors[0, 1], 'go', markersize=8, label='Start')
            axes[1, 0].plot(mean_vectors[-1, 0], mean_vectors[-1, 1], 'ro', markersize=8, label='End')
            axes[1, 0].set_xlabel(self.arg_names[0])
            axes[1, 0].set_ylabel(self.arg_names[1])
            axes[1, 0].set_title('Mean Vector Trajectory')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            if ans:
                axes[1, 0].plot(ans[0], ans[1], 'r*', markersize=8, label='Answer')

        eigenvalues = np.array(self.history['eigenvalues'])
        for i in range(self.dim):
            axes[1, 1].semilogy(generations, eigenvalues[:, i], label=f'λ{i+1}')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Eigenvalues')
        axes[1, 1].set_title('Covariance Matrix Eigenvalues')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_2d_optimization(self, objective_func, xlim=(-3, 3), ylim=(-3, 3), figsize=(10, 8)):
        """2次元最適化の可視化"""
        if self.dim != 2:
            print("2次元問題のみ対応")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 等高線プロット
        x = np.linspace(xlim[0], xlim[1], 100)
        y = np.linspace(ylim[0], ylim[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = objective_func(X[i, j], Y[i, j])
        
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # 最適化軌跡
        mean_vectors = np.array(self.history['mean_vector'])
        ax.plot(mean_vectors[:, 0], mean_vectors[:, 1], 'r-o', markersize=4, linewidth=2, label='Mean trajectory')
        
        # 最終世代の個体群と分散楕円
        if self.history['populations']:
            final_pop = self.history['populations'][-1]
            ax.scatter(final_pop[:, 0], final_pop[:, 1], alpha=0.6, s=20, label='Final population')
            
            # 分散楕円
            mean = mean_vectors[-1]
            cov = self.C * (self.sigma ** 2)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # 95%信頼楕円
            angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvals) * 2.448  # 95%信頼区間
            
            ellipse = Ellipse(mean, width, height, angle=angle, 
                            facecolor='none', edgecolor='red', linewidth=2, alpha=0.8)
            ax.add_patch(ellipse)
        
        ax.set_xlabel(self.arg_names[0])
        ax.set_ylabel(self.arg_names[1])
        ax.set_title('2D Optimization Visualization')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        plt.show()

if __name__ == "__main__":
    def parametric_func(x, y):
        return (x - 1)**2 + (y - 2)**2

    def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2

    def rastrigin_func(x, y):
        args = [x, y]
        k = 0
        for n in args:
            k += 10 + (n*n - 10 * np.cos(2*np.pi*n))
        return k

    test_points_dim_1 = [
        [2],
        [1],
        [-2],
        [-1],
    ]
        
    test_points_dim_2 = [
        [1, 1],
        [3, -1],
        [5, 5], 
        [-2, 3],
        [5, -5],
    ]

    for init_point in test_points_dim_2:
        cmaes = CMAES(arg_names=["x", "y"], ave_vec=init_point, max_iter=100, sigma=0.3)
        loss, value = cmaes.opt(rastrigin_func)
        print(f"初期点{init_point}: 値={loss:.2e}, 解={value}")