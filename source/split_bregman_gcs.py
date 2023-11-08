import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from gauss_seidel import GaussSeidel, GaussSeidelGCS
from image_force import ImageForce, Force1, Force2
import cv2 as cv

class SplitBregmanGCS():
    def __init__(
            self,
            force:ImageForce,
            initial_level_set:np.array,
            lambda_value:float,
            nu_value:float,
            alpha_value:float,
            epsilon_value:float,
            debug: bool = False) -> None:
        self._force = force
        self._initial_level_set = initial_level_set
        self._lambda = lambda_value   # for our energy function
        self._nu = nu_value  # regularization for constraints
        self._alpha = alpha_value  # for level set
        self._epsilon = epsilon_value  # stop criterion
        self._debug = debug
        self._solver_next_u = GaussSeidelGCS(self._nu, self._lambda)

    def shrink(self, z):
        return np.max(np.linalg.norm(z) - self._nu, 0) * z / np.linalg.norm(z)

    def _generator(level_set_found):
        while level_set_found:
            yield

    def run(self):
        # we should change the function u as a level set function with
        # negative and positive values
        level_set_found = False
        u = self._initial_level_set
        d = np.random.rand(u.shape[0], u.shape[1], 2)  # i dont have really clear how to initialize
        b = np.random.rand(u.shape[0], u.shape[1], 2)
        if self._debug:
            it = 0
            cv.imwrite(f'results/level_set_it_{it}.png', u*255)
        while not level_set_found:
            r = self._force.get_force(u)
            next_u, hist = self._solver_next_u.compute(u, r, d, b)
            
            plt.plot(hist)
            plt.savefig('results/plot_gs.png')
            # image gradient
            grad_next_u = np.zeros((u.shape[0], u.shape[1], 2))
            grad_next_u[:, :, 0] = cv.Sobel(next_u, cv.CV_8U, dx=1, dy=0)
            grad_next_u[:, :, 1] = cv.Sobel(next_u, cv.CV_8U, dx=0, dy=1)
            
            d = self.shrink(grad_next_u + b)
            b = b + grad_next_u - d
            level_set_found = np.linalg.norm(next_u - u) < self._epsilon
            u = next_u
            if self._debug:
                cv.imwrite(f'results/level_set_it_{it}.png', u*255)
                it += 1
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
        lambda_value=10,
        mu_value=0.5,
        epsilon_value=0.01)
    last_level_set = segmentator.run()

def remove_old_files(output_path):
    for f in os.listdir(output_path):
        os.remove(os.path.join(output_path, f))
if __name__ == '__main__':
    output_path = 'results/'
    remove_old_files(output_path)
    # test with a cell image
    initial_level_set = cv.imread('test_images/simplify_cells_initial_mask_2.png', cv.CV_8U) * 255
    image = cv.imread('test_images/simplify_cells.tif', cv.CV_16U)
    image = cv.convertScaleAbs(image) / 255
    force1 = Force2(image)
    segmentator = SplitBregmanGCS(
        force1,
        initial_level_set,
        lambda_value=10,
        nu_value=5,  # constraint regulation
        alpha_value=.5,
        epsilon_value=0.01,
        debug=True)
    last_level_set = segmentator.run()
    print()