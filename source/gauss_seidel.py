import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from .utils import normalization


def l2_diff(f1: np.array, f2: np.array) -> float:
    l2_diff = np.sqrt(np.sum((f1 - f2) ** 2)) / f1.shape[0]
    return l2_diff


# FIXME: Add exception when the algorithm do not find a convergence
class GaussSeidelGCS:
    def __init__(
            self,
            nu_value: float,
            lambda_value: float,
            error: float,
            debug: bool = False,
            ) -> None:
        self._nu = nu_value
        self._lambda = lambda_value
        self._error = error
        self._debug = debug

    def compute(
            self,
            u: np.array,
            r: np.array,
            b: np.array,
            d: np.array,
            ) -> tuple[np.array, list]:
        u = u.astype(np.float64)

        # Create once the auxiliary vector alpha
        dx = d[:, :, 0]
        dy = d[:, :, 1]
        bx = b[:, :, 0]
        by = b[:, :, 1]
        alpha = np.zeros_like(u)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                alpha[i, j] = dx[i - 1, j] - dx[i, j] - bx[i-1, j] + bx[i, j]
                + dy[i, j-1] - dy[i, j] - by[i, j-1] + by[i, j]
        b = (self._lambda/self._nu) * r - alpha
        unew, hist = GaussSeidel(tolerance=self._error).compute(
            p=u,
            b=b,
            dx=1)
        # debug
        if self._debug:
            plt.title("Gauss Seidel Convergence of every iteration")
            plt.plot(np.arange(len(hist)), hist)
            plt.ylim(0, 1)
            plt.savefig('results/results_error_GS.png')

            cv.imwrite(f'results/r.tif', r)
        if np.all(unew == 1) or np.all(unew == 0):
            raise ValueError("Gauss Seidel GCS returning a level set of ones or zeros")
        return unew, hist


class GaussSeidel:
    """ Extracted from:
    https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html#gauss-seidel-method"""

    def __init__(
            self,
            tolerance: float = 1e-3,
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
        self.p_evolution = [p]
        if np.any(np.isnan(b)):
            raise ValueError("Gauss Seidel receive a b parameter with NaN value")
        if np.any(np.isnan(p)):
            raise ValueError("Gauss Seidel receive a p parameter with NaN value")
        with tqdm(total=self.max_iter, desc='Gauss Seidel Iteration') as pbar:
            while (diff > self.tolerance):
                if it > self.max_iter:
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
                self.p_evolution.append(pnew)
                it += 1
                pbar.update(1)
        if it > self.max_iter:
            print(f'Solution did not converged within the maximum'
                  f' number of iterations. Last l2_diff was: {diff:.5e}')
        else:
            print(f'The solution converged after {it} iterations')
        
        return pnew, tol_hist_gs

    def plot_per_iteration(self, iterations_list=None):
        if iterations_list is None:
            n_iter = len(self.p_evolution)
            c = 100
            n_graph = n_iter // c
            index = 1
            plt.figure(figsize=(12, 12))
            for i in range(n_iter):
                if i % c == 0:
                    p = self.p_evolution[i]
                    plt.subplot(n_graph//4, 4, index)
                    plt.imshow(p)
                    plt.title('iter = %s' % i)
                    index += 1
                if index > (n_graph//4) * 4:
                    break
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(12, 12))
            n_graph = len(iterations_list)
            index = 1
            for i in iterations_list:
                p = self.p_evolution[i]
                plt.subplot(n_graph//4, 4, index)
                plt.imshow(p)
                plt.title('iter = %s' % i)
                index += 1
                if index > (n_graph//4) * 4:
                    break
            plt.tight_layout()
            plt.show()


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
    gauss_seidel = GaussSeidel(max_iter=1000)
    pnew, hist = gauss_seidel.compute(p0, b, dx)
    gauss_seidel.plot_per_iteration([1,2,3,4,5,6,8,8,9])
    print()
