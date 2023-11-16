from enum import Enum
import os
from matplotlib import pyplot as plt
import numpy as np
from .gauss_seidel import GaussSeidelGCS, l2_diff
from .image_force import ImageForce, Force1, Force2
import cv2 as cv

from .utils import normalization, normalization_automatic, remove_old_files


class NormalizationMode(Enum):
    Null = 0
    Clip = 1
    FirstImageParameters = 2


class SplitBregmanGCS():
    def __init__(
            self,
            force: ImageForce,
            lambda_value: float,
            nu_value: float,
            epsilon_value: float,
            gs_error: float,
            mode: NormalizationMode = NormalizationMode.Clip,
            debug: bool = False
            ) -> None:
        self._force = force
        self._lambda = lambda_value   # for our energy function
        self._nu = nu_value  # regularization for constraints
        self._epsilon = epsilon_value  # stop criterion
        self._gs_error = gs_error
        self._mode = mode
        self._debug = debug
        self._solver_next_u = GaussSeidelGCS(
            nu_value=self._nu,
            lambda_value=self._lambda,
            error=gs_error,
            debug=debug)

    def shrink(self, z):
        out = np.max(np.linalg.norm(z) - self._nu, 0) * (z / np.linalg.norm(z))
        out[np.isnan(out)] = 0
        return out

    def run(self, initial_level_set):
        self._initial_level_set = initial_level_set
        u = initial_level_set.copy()
        u_prev = np.ones_like(u)

        d = np.zeros((u.shape[0], u.shape[1], 2))
        b = np.zeros((u.shape[0], u.shape[1], 2))

        if self._debug:
            it = 0
            cv.imwrite(f'results/level_set_it_{it}.tif', u)
        error = []
        while not l2_diff(u, u_prev) < self._epsilon:
            error.append(l2_diff(u, u_prev))
            print(f'----------------- Iteration error {error[-1]} ------------------------')
            # Calculate force
            r = self._force.get_force(u > 0)

            # Save prev level set
            u_prev = u.copy()

            # Calculate new level set
            u, _ = self._solver_next_u.compute(u_prev.copy(), r, d, b)
            if self._debug:
                it += 1
                cv.imwrite(f'results/level_set__it_{it}.tif', u)

            u_no_normalized = u.copy()
            if self._mode == NormalizationMode.Clip:
                u[u > 1] = 1
                u[u < 0] = 0
            elif self._mode == NormalizationMode.FirstImageParameters:
                if np.all(u_prev == self._initial_level_set):
                    valid_min = np.min(u)
                    valid_max = np.max(u)
                u[u > valid_max] = valid_max
                u[u < valid_min] = valid_min
                u = normalization(u, valid_min, valid_max)
            elif self._mode == NormalizationMode.Null:
                pass

            if self._debug:
                cv.imwrite(f'results/level_set_it_normalized_{it}.tif', u)
            # image gradient
            grad_u = np.zeros((u.shape[0], u.shape[1], 2))
            grad_u[:, :, 0] = cv.Sobel(u, cv.CV_64F, dx=1, dy=0)
            grad_u[:, :, 1] = cv.Sobel(u, cv.CV_64F, dx=0, dy=1)

            # Update Bregman Parameters
            d = self.shrink(grad_u + b)
            b = b + grad_u - d
        print(f'Converged with an error {l2_diff(u, u_prev)}')
        if self._debug:
            plt.figure()
            plt.title("Split Bregman Error per iteration")
            plt.plot(error, '-o')
            plt.xlabel("Iterations")
            plt.ylabel("Level Set difference")
            plt.savefig('results/final_error.png')
        return u, u_no_normalized
  


if __name__ == '__main__':
    output_path = 'results/'
    alpha = 0.8
    remove_old_files(output_path)
    # test with a cell image
    initial_level_set = cv.imread('test_images/simplify_cells_distance_multiply_10.tif', cv.CV_8U)
    initial_level_set = normalization_automatic(initial_level_set)
    image = cv.imread('test_images/simplify_cells.tif', cv.CV_16U) 
    image = normalization_automatic(image)
    # initial_level_set = np.ones_like(image)
    cv.imwrite('results/input_image.tif', image)
    force1 = Force1(image, k0=1, k1=1)
    force2 = Force2(image)
    segmentator = SplitBregmanGCS(
        force1,
        lambda_value=1,
        nu_value=0.5,
        epsilon_value=0.031,
        gs_error=1e-3,
        debug=True)
    last_level_set, last_level_set_no_normalized = segmentator.run(
        initial_level_set)
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.contour(last_level_set == alpha)
    plt.savefig('results/last_countour.png')
    print()
