import numpy as np
from tqdm import tqdm


def l2_diff(f1: np.array, f2: np.array) -> float:
    l2_diff = np.sqrt(np.sum((f1 - f2) ** 2)) / f1.shape[0]
    return l2_diff


class GaussSeidelGCS:
    def __init__(self, lambda_value: float, mu_value: float) -> None:
        self._lambda = lambda_value  # for constraints
        self._mu = mu_value  # for energy

    def compute(
            self,
            u: np.array,
            r: np.array,
            b: np.array,
            d: np.array
            ) -> tuple[np.array, list]:
        u = u.astype(np.float64)
        dx = d[:, :, 0]
        dy = d[:, :, 1]
        bx = b[:, :, 0]
        by = b[:, :, 1]
        alpha = np.zeros_like(u)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                alpha[i, j] = dx[i - 1, j] - dx[i, j] - bx[i-1, j] + bx[i, j]
                + dy[i, j-1] - dy[i, j] - by[i, j-1] + by[i, j]
        unew, hist = GaussSeidel().compute(
            p=u,
            b=(self._mu/self._lambda) * r - alpha,
            dx=1)
        return unew, hist


class GaussSeidel:
    """ Extracted from:
    https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html#gauss-seidel-method"""

    def __init__(
            self,
            tolerance: float = 1e-10,
            max_iter: int = 10000
            ) -> None:
        self.tolerance = tolerance
        self.max_iter = max_iter

    def compute(self, p, b, dx):
        it = 0  # iteration counter
        diff = 1.0
        tol_hist_gs = []
        pnew = p.copy()
        nx = p.shape[0]
        ny = p.shape[1]
        with tqdm(total=self.max_iter, desc='Gauss Seidel Iteration') as pbar:
            while (diff > self.tolerance):
                if it > self.max_iter:
                    print(f'Solution did not converged within the maximum'
                          f' number of iterations.'
                          f' Last l2_diff was: {diff:.5e}')
                    break

                np.copyto(p, pnew)

                # We only modify interior nodes. The boundary nodes
                # remain equal to zero and the Dirichlet boundary
                # conditions are therefore automatically enforced.
                for j in range(1, ny-1):
                    for i in range(1, nx-1):
                        pnew[i, j] = (0.25 * (pnew[i - 1, j] + p[i + 1, j]
                                              + pnew[i, j - 1]
                                              + p[i, j+1]-b[i, j]*dx**2))

                diff = l2_diff(pnew, p)
                tol_hist_gs.append(diff)
                it += 1
                pbar.update(1)
            else:
                print(f'The solution converged after {it} iterations')
        return pnew, tol_hist_gs


def create_test_data():
    # Grid parameters.
    nx = 101
    ny = 101
    xmin, xmax = 0.0, 1.0
    ymin, ymax = -0.5, 0.5
    lx = xmax - xmin
    ly = ymax - ymin
    dx = lx / (nx-1)
    dy = ly / (ny-1)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    b = (np.sin(np.pi * X) * np.cos(np.pi * Y)
         + np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))
    p0 = np.zeros((nx, ny))
    return p0, b, dx, dy


if __name__ == '__main__':
    p0, b, dx, _ = create_test_data()
    pnew, hist = GaussSeidel().compute(p0, b, dx)
    print()
