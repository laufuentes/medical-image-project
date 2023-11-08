import numpy as np
from tqdm import tqdm


def l2_diff(f1, f2):
    """
    Computes the l2-norm of the difference
    between a function f1 and a function f2
    
    Parameters
    ----------
    f1 : array of floats
        function 1
    f2 : array of floats
        function 2
    
    Returns
    -------
    diff : float
        The l2-norm of the difference.
    """
    l2_diff = np.sqrt(np.sum((f1 - f2)**2))/f1.shape[0]
    
    return l2_diff

class GaussSeidelGCS:
    def __init__(self, lambda_value, mu_value) -> None:
        self._lambda = lambda_value  # for constraints
        self._mu = mu_value  # for energy

    def compute(self, u, r, b, d):
        u = u.astype(np.float64)
        dx = d[:, :, 0]
        dy = d[:, :, 1]
        bx = b[:, :, 0]
        by = b[:, :, 1]
        alpha = np.zeros_like(u)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                alpha[i, j] = dx[i - 1, j] - dx[i, j] - bx[i-1, j] + bx[i, j] + dy[i, j-1] - dy[i, j] - by[i, j-1] + by[i, j]
        unew, hist = GaussSeidel().compute(u, - alpha * (self._mu/self._lambda) * r , dx = 1)
        return unew, hist

class GaussSeidelGCSOld:
    def __init__(self, lambda_value, mu_value) -> None:
        self._lambda = lambda_value  # for constraints
        self._mu = mu_value  # for energy
        self.tolerance = 1e-10
        self.max_iteration = 3000

    #  Solución iterativa de:
    #   laplaciano de p(x,y) = b(x,y)
    def compute(self, u, r, b, d):
        u = u.astype(np.float64)
        diff = 1000
        difference_values = []
        it = 0
        dx = d[:, :, 0]
        dy = d[:, :, 1]
        bx = b[:, :, 0]
        by = b[:, :, 1]
        alpha = np.zeros_like(u)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                alpha[i,j] = dx[i-1, j] - dx[i, j] - bx[i-1, j] + bx[i, j] + dy[i, j-1] - dy[i, j] - by[i, j-1] + by[i, j]

        
        with tqdm(total=self.max_iteration, desc='Gauss Seidel iteration algorithm') as pbar:
            
            while (diff > self.tolerance):
                if it > self.max_iteration:
                    print(f'Solution did not converge, last diference is {diff}')
                    break
                beta = u.copy()

                for i in range(1, u.shape[0] - 1):
                    for j in range(1, u.shape[1] - 1):
                        beta[i, j] = 0.25 * (beta[i - 1, j] + u[i + 1, j] + beta[i, j - 1] + u[i, j+1] - (self._mu/self._lambda) * r[i, j] + alpha[i,j])
                
                unew = beta.copy()
                unew[beta < 0] = 0
                unew[beta > 1] = 1
                diff = l2_diff(u, beta)
                diff_unew = l2_diff(u, unew)
                difference_values.append(diff)
                pbar.update(1)
                it += 1
            else:
                print(f'Solution found with difference {diff}')
        return unew, difference_values


class GaussSeidelNuestro:
    def __init__(self) -> None:
        self.tolerance = 1e-10
        self.max_iteration = 3000

    #  Solución iterativa de:
    #   laplaciano de p(x,y) = b(x,y)
    def compute(self, p, b, dx):
        diff = 1000
        difference_values = []
        it = 0
        with tqdm(total=self.max_iteration, desc='Gauss Seidel iteration algorithm') as pbar:
            
            while (diff > self.tolerance):
                if it > self.max_iteration:
                    print(f'Solution did not converge, last diference is {diff}')
                    break
                pnew = p.copy()

                for i in range(1, p.shape[0] - 1):
                    for j in range(1, p.shape[1] - 1):
                        pnew[i, j] = 0.25 * (pnew[i - 1, j] + p[i + 1, j] + pnew[i, j - 1] + p[i, j+1] - b[i, j] * dx ** 2)
                
                diff = l2_diff(p, pnew)
                difference_values.append(diff)
                pbar.update(1)
                it += 1
            else:
                print(f'Solution found with difference {diff}')
        return pnew, difference_values
class GaussSeidel:
    def compute(self, p, b, dx):
        
        it = 0 # iteration counter
        diff = 1.0
        tol_hist_gs = []
        tolerance = 1e-10
        max_iter = 10000
        pbar = tqdm(total=max_iter)
        pbar.set_description("it / max_it")
        pnew = p.copy()
        
        nx = p.shape[0]
        ny = p.shape[1]
        while (diff > tolerance):
            if it > max_iter:
                print('\nSolution did not converged within the maximum'
                    ' number of iterations'
                    f'\nLast l2_diff was: {diff:.5e}')
                break

            np.copyto(p, pnew)

            # We only modify interior nodes. The boundary nodes remain equal to
            # zero and the Dirichlet boundary conditions are therefore automatically
            # enforced.
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    pnew[i, j] = (0.25 * (pnew[i-1, j]+p[i+1, j]+pnew[i, j-1]
                            + p[i, j+1]-b[i, j]*dx**2))

            diff = l2_diff(pnew, p)
            tol_hist_gs.append(diff)

            it += 1
            pbar.update(1)

        else:
            print(f'\nThe solution converged after {it} iterations')

        del(pbar)
        return pnew, tol_hist_gs
 
if __name__ == '__main__':
    # Grid parameters.
    nx = 101                  # number of points in the x direction
    ny = 101                  # number of points in the y direction
    xmin, xmax = 0.0, 1.0     # limits in the x direction
    ymin, ymax = -0.5, 0.5    # limits in the y direction
    lx = xmax - xmin          # domain length in the x direction
    ly = ymax - ymin          # domain length in the y direction
    dx = lx / (nx-1)          # grid spacing in the x direction
    dy = ly / (ny-1)          # grid spacing in the y direction
    # Create the gridline locations and the mesh grid;
    # see notebook 02_02_Runge_Kutta for more details
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    # We pass the argument `indexing='ij'` to np.meshgrid
    # as x and y should be associated respectively with the
    # rows and columns of X, Y.
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Compute the rhs. Note that we non-dimensionalize the coordinates
    # x and y with the size of the domain in their respective dire-
    # ctions.
    b = (np.sin(np.pi*X)*np.cos(np.pi*Y) + np.sin(5.0*np.pi*X)*np.cos(5.0*np.pi*Y))

    p0 = np.zeros((nx, ny))
    pnew, hist = GaussSeidelNuestro().compute(p0, b, dx)
    print()



