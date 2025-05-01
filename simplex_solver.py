# simplex_solver.py
import numpy as np
import matplotlib.pyplot as plt

class SimplexSolver:
    """
    Resolve problemas de Programação Linear do tipo:

        max  c^T x
        s.a. Ax ≤ b,  x ≥ 0

    usando Simplex primal (forma padrão + variáveis de folga)
    """
    def __init__(self, c, A, b, maximize=True):
        c, A, b = map(np.array, (c, A, b))
        self.maximize = maximize

        self.c = c if maximize else -c
        self.m, self.n = A.shape

        self.tableau = self._build_tableau(A, b)
        self.basis = list(range(self.n, self.n + self.m))
        self.iter_history = []

    def _build_tableau(self, A, b):
        """Monta o tableau inicial em forma canônica."""
        I = np.eye(self.m)
        tableau = np.hstack((A, I, b.reshape(-1, 1))).astype(float)
        # Linha de custos: [-c  0(slack)  0(b)]
        cost_row = np.hstack((-self.c, np.zeros(self.m + 1)))
        return np.vstack((tableau, cost_row))

    def _pivot(self, row, col):
        """Efetua pivoteamento em (row, col)."""
        self.tableau[row] /= self.tableau[row, col]
        for r in range(len(self.tableau)):
            if r != row:
                self.tableau[r] -= self.tableau[r, col] * self.tableau[row]
        self.basis[row] = col

    def _choose_entering(self):
        """Variável que entra: coef. mais negativo na linha de custos (se houver)."""
        last_row = self.tableau[-1, :-1]
        if np.all(last_row >= 0):
            return None
        return np.argmin(last_row)

    def _choose_leaving(self, col):
        """Regra do quociente mínimo (Evita números negativos ou zero)."""
        rhs = self.tableau[:-1, -1]
        col_vals = self.tableau[:-1, col]
        mask = col_vals > 1e-12
        if not np.any(mask):
            return None
        ratios = rhs[mask] / col_vals[mask]
        return np.where(mask)[0][np.argmin(ratios)]

    def solve(self, verbose=False, max_iter=10_000):
        """Roda o Simplex, retorna (status, valor ótimo, vetor solução)."""
        for it in range(max_iter):
            self.iter_history.append(self.tableau.copy())
            col = self._choose_entering()
            if col is None:
                status = "Ótimo"
                break
            row = self._choose_leaving(col)
            if row is None:
                status = "Ilimitado"
                break
            if verbose:
                print(f"Iter {it:3d}: entra x{col}, sai x{self.basis[row]}")
            self._pivot(row, col)
        else:
            status = "Max_iter"

        # Constrói solução
        x = np.zeros(self.n + self.m)
        for r, basic_var in enumerate(self.basis):
            x[basic_var] = self.tableau[r, -1]
        value = self.tableau[-1, -1] * (1 if self.maximize else -1)
        return status, value, x[:self.n]

    def plot_objective_progress(self):
        """Desenha a evolução do valor da função-objetivo ao longo das iterações."""
        vals = [-tab[-1, -1] if self.maximize else tab[-1, -1]
                for tab in self.iter_history]
        plt.figure()
        plt.plot(range(len(vals)), vals, marker='o')
        plt.title("Convergência do Simplex")
        plt.xlabel("Iteração")
        plt.ylabel("Valor da função-objetivo")
        plt.grid(True)
        plt.show()

    @staticmethod
    def from_user():
        """Pergunta dados ao usuário via console e devolve um solver pronto."""
        n = int(input("Nº de variáveis? "))
        m = int(input("Nº de restrições (≤)? "))
        print("Coeficientes da função-objetivo (separados por espaço):")
        c = list(map(float, input().split()))
        A, b = [], []
        for i in range(m):
            print(f"Restrição {i+1}:  coeficientes de x1..x{n} (≤) e lado direito:")
            *row, rhs = map(float, input().split())
            A.append(row)
            b.append(rhs)
        return SimplexSolver(c, np.array(A), np.array(b))

