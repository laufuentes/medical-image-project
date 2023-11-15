import os
from matplotlib import pyplot as plt
import numpy as np
from gauss_seidel import GaussSeidelGCS, l2_diff
from image_force import ImageForce, Force1, Force2
import cv2 as cv

from utils import normalization, normalization_automatic, remove_old_files


class SplitBregmanGCS():
    def __init__(
            self,
            force: ImageForce,
            initial_level_set: np.array,
            lambda_value: float,
            nu_value: float,
            alpha_value: float,
            epsilon_value: float,
            gs_error: float,
            debug: bool = False) -> None:
        self._force = force
        self._initial_level_set = initial_level_set
        self._lambda = lambda_value   # for our energy function
        self._nu = nu_value  # regularization for constraints
        self._alpha = alpha_value  # for level set
        self._epsilon = epsilon_value  # stop criterion
        self._gs_error = gs_error
        self._debug = debug
        self._solver_next_u = GaussSeidelGCS(nu_value=self._nu, lambda_value=self._lambda, error=gs_error)

    def shrink(self, z):
        out = np.max(np.linalg.norm(z) - self._nu, 0) * (z / np.linalg.norm(z))
        out[np.isnan(out)] = 0
        return out

    def run(self):
        # we should change the function u as a level set function with
        # negative and positive values
        u = self._initial_level_set.copy()
        u_prev = np.ones_like(u)

        d = np.zeros((u.shape[0], u.shape[1], 2))  # np.random.rand(u.shape[0], u.shape[1], 2)
        b = np.zeros((u.shape[0], u.shape[1], 2))  # np.random.rand(u.shape[0], u.shape[1], 2)
        
        if self._debug:
            it = 0
            cv.imwrite(f'results/level_set_it_{it}.tif', u)
        error = []
        while not l2_diff(u, u_prev) < self._epsilon:
            error.append(l2_diff(u, u_prev))
            print(f'----------------- Iteration error {error} ------------------------')
            # Calculate force
            r = self._force.get_force(u >= self._alpha)

            # Save prev level set
            u_prev = u.copy()

            # Calculate new level set
            u, hist = self._solver_next_u.compute(u_prev.copy(), r, d, b)
            
            #normalization level set
            if np.all(u_prev==self._initial_level_set):
                valid_min = np.min(u)
                valid_max = np.max(u) 
                print(valid_min, valid_max)
            if self._debug:
                it += 1
                cv.imwrite(f'results/level_set__a_{alpha}_it_{it}.tif', u)
            u = normalization(u, valid_min, valid_max)
            
            if self._debug:
                cv.imwrite(f'results/level_set_a_{alpha}_it_normalized_{it}.tif', u)
            # image gradient
            grad_u = np.zeros((u.shape[0], u.shape[1], 2))
            grad_u[:, :, 0] = cv.Sobel(u, cv.CV_64F, dx=1, dy=0)
            grad_u[:, :, 1] = cv.Sobel(u, cv.CV_64F, dx=0, dy=1)

            # Update Bregman Parameters
            d = self.shrink(grad_u + b)
            b = b + grad_u - d

            
        print(f'Converged with an error {l2_diff(u, u_prev)}')
        plt.figure()
        plt.plot(error, '-o')
        plt.savefig('results/final_error.png')
        return u


def test_silly_example():
    # test the basic algorithm with silly example
    image = np.random.randint(0, 255, (10, 10))
    force1 = Force1(image, k0=1, k1=1)
    initial_mask = np.zeros((10, 10), bool)
    initial_mask[:5, :5] = True

    segmentator = SplitBregmanGCS(
        force1,
        initial_mask,
        lambda_value=1,
        mu_value=0.5,
        epsilon_value=0.29)
    last_level_set = segmentator.run()
    


if __name__ == '__main__':
    output_path = 'results/'
    alpha = 0.95
    #remove_old_files(output_path)
    # test with a cell image
    initial_level_set = cv.imread('test_images/simplify_cells_distance_multiply_10.tif', cv.CV_8U)
    initial_level_set = normalization_automatic(initial_level_set)
    image = cv.imread('test_images/simplify_cells.tif', cv.CV_16U) 
    image = normalization_automatic(image)
    # initial_level_set = np.ones_like(image)
    cv.imwrite('results/input_image.tif', image)
    force1 = Force1(image, k0=1, k1=1)
    segmentator = SplitBregmanGCS(
        force1,
        initial_level_set,
        lambda_value=1,
        nu_value=0.5,
        alpha_value=alpha,
        epsilon_value=0.031,
        gs_error=1e-5,
        debug=True)
    last_level_set = segmentator.run()
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.contour(last_level_set == alpha)
    plt.savefig('results/last_countour.png')
    print()
