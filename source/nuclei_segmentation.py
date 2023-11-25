import numpy as np
import cv2 as cv

from source.pde_solver  import PDESolver

from .split_bregman_gcs import SplitBregmanGCS, NormalizationMode
from .image_force import Force1, Force2


class ThreeStepApproach:
    def __init__(
            self,
            image: np.array,
            lambda_value: float,
            nu_value: float,
            epsilon_value: float,
            error_gs: float,
            k0: float,
            k1: float,
            mode: NormalizationMode = NormalizationMode.FirstImageParameters,
            using_bregman: bool = True,
            debug: bool = False,
            ) -> None:
        self._image = image
        self._lambda = lambda_value
        self._nu = nu_value
        self._k0 = k0
        self._k1 = k1
        self._epsilon = epsilon_value
        self._error_gs = error_gs
        self._mode = mode
        self._debug = debug
        self._using_bregman = using_bregman
        self.__post_init__()

    def __post_init__(self):
        self._r1 = Force2(self._image)
        self._r2 = Force1(self._image, k0=self._k0, k1=self._k1)
        self._r3 = Force2(self._image)
        if self._using_bregman:
            self._GCS_step_1 = SplitBregmanGCS(
                self._r2,
                self._lambda,
                self._nu,
                self._epsilon,
                self._error_gs,
                self._mode,
                self._debug)
            self._GCS_step_2 = SplitBregmanGCS(
                self._r1,
                self._lambda,
                self._nu,
                self._epsilon,
                self._error_gs,
                self._mode,
                self._debug)

            self._GCS_step_3 = SplitBregmanGCS(
                self._r2,
                self._lambda,
                self._nu,
                self._epsilon,
                self._error_gs,
                self._mode,
                self._debug)
        else:
            self._GCS_step_1 = PDESolver(self._r2, self._lambda, self._epsilon)
            self._GCS_step_2 = PDESolver(self._r1, self._lambda, self._epsilon)
            self._GCS_step_3 = PDESolver(self._r2, self._lambda, self._epsilon)


    def step_1(self, initial_level_set: np.array) -> np.array:
        if self._using_bregman:
            level_set_function_1, _ = self._GCS_step_1.run(initial_level_set)
        else:
            level_set_function_1 = self._GCS_step_1.run(initial_level_set)
        return level_set_function_1

    def step_2(self, input_level_set: np.array) -> list:
        level_set_function_2 = []
        self.input_step_2 = []
        # For each level calculate the second step
        # FIXME: which alphas values they use (?)
        for i, alpha in enumerate(np.linspace(np.min(input_level_set), np.max(input_level_set), 5)):
            print(f'---------- Iteration {i} with alpha={alpha} --------------------')
            level_segmentation = input_level_set >= alpha
            if np.all(level_segmentation):
                continue
            # Apply the distance transform
            dist_level_segmentation = cv.distanceTransform(level_segmentation.astype('uint8')*255, cv.DIST_L2, 3)
            # Normalize the distance image for range = {0.0, 1.0}
            # so we can visualize and threshold it
            cv.normalize(dist_level_segmentation, dist_level_segmentation, 0, 1.0, cv.NORM_MINMAX)
            self.input_step_2.append(dist_level_segmentation)
            if self._using_bregman:
                level_segmentation_after_step_2, _= self._GCS_step_2.run(
                    dist_level_segmentation)
            else:
                level_segmentation_after_step_2 = self._GCS_step_2.run(
                    dist_level_segmentation)
            level_set_function_2.append(level_segmentation_after_step_2)
        return level_set_function_2

    def step_3(self, input_level_set: list) -> list:
        # Dilate each of the segmentation
        iterations = 1
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # FIXME: KERNEL SIZE (? and ITERATIONS

        level_set_function_3 = []
        self.input_levels_step_3 = []
        for i, level_segmentation in enumerate(input_level_set):
            print(f'---------- Iteration {i} -----------------------------')
            binary_level_mask = level_segmentation > np.median(level_segmentation)
            if np.all(binary_level_mask):
                continue
            if np.all(np.logical_not(binary_level_mask)):
                continue

            level_segmentation_dilated = cv.dilate(
                binary_level_mask.astype('uint8')*255,
                kernel,
                iterations=iterations)
            # Apply the distance transform
            dist_level_segmentation = cv.distanceTransform(level_segmentation_dilated, cv.DIST_L2, 3)
            # Normalize the distance image for range = {0.0, 1.0}
            # so we can visualize and threshold it
            cv.normalize(dist_level_segmentation, dist_level_segmentation, 0, 1.0, cv.NORM_MINMAX)
            self.input_levels_step_3.append(dist_level_segmentation)
            if self._using_bregman:
                level_segmentation_after_step_3, _ = self._GCS_step_3.run(
                    dist_level_segmentation)
            else:
                level_segmentation_after_step_3 = self._GCS_step_3.run(
                    dist_level_segmentation)
            level_set_function_3.append(level_segmentation_after_step_3)
        return level_set_function_3

    def run(self, initial_level_set: np.array) -> list:
        print('-------------------- Step 1 ------------------------------')
        level_set_1 = self.step_1(initial_level_set)
        print('-------------------- Step 2 ------------------------------')
        level_set_2 = self.step_2(level_set_1)
        print('-------------------- Step 3 ------------------------------')
        level_set_3 = self.step_3(level_set_2)
        print('------------ First Approach finished ---------------------')
        return level_set_3
