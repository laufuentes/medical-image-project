import numpy as np

from source.gauss_seidel import l2_diff
from source.image_force import ImageForce


class PDESolver:
    """Solve the PDE equation but without using bregman"""
    def __init__(self, force: ImageForce, lambda_value: float, epsilon_value: float) -> None:
        self._force = force
        self._lambda = lambda_value   # for our energy function
        self._epsilon = epsilon_value  # stop criterion

    def run(self, initial_level_set):
        self._initial_level_set = initial_level_set
        u = initial_level_set.copy()
        u_prev = np.ones_like(u)
        error = []
        while not l2_diff(u, u_prev) < self._epsilon:
            error.append(l2_diff(u, u_prev))
            print(f'----------------- Iteration error {error[-1]} ------------------------')
            # Calculate force
            r = self._force.get_force(u > 0)
            if np.all(u > 0):
                print('Level set is all positive')            
            # Save prev level set
            u_prev = u.copy()
            # Calculate new level set
            u = self.solve(u, -self._lambda * r, dt=1)
            if error[-1] > 10:
                raise ValueError
        print(f'Convergence with {error[-1]}')
        return u

    def solve(self, phi, r, dt):
        """
        This corresponds to equation (22) of the paper by Pascal Getreuer,
        which computes the next iteration of the level set based on a current
        level set, adapted for our energy function
        """
        eta = 1e-16
        mu = 1

        P = np.pad(phi, 1, mode='edge')

        phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
        phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
        phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0

        phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
        phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
        phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0

        
        C1 = 1. / np.sqrt(eta + phixp**2 + phiy0**2)
        C2 = 1. / np.sqrt(eta + phixn**2 + phiy0**2)
        C3 = 1. / np.sqrt(eta + phix0**2 + phiyp**2)
        C4 = 1. / np.sqrt(eta + phix0**2 + phiyn**2)
        K = (P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 +
         P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4)

        new_phi = phi + dt * (r+K)
        return new_phi / (1 + mu * dt * (C1+C2+C3+C4))
