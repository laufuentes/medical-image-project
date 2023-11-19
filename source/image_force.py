import numpy as np
import cv2 as cv

class ImageForce:
    def __init__(self, image) -> None:
        self._image = image
        self._r = None

    def get_force(self, interior_mask):
        self._r = self.compute_force(interior_mask)
        return self._r

class Force1(ImageForce):
    def __init__(self, image, k0, k1) -> None:
        super().__init__(image)
        self._k0 = k0
        self._k1 = k1
        
    def compute_force(self, mask):
        self._mask_in = mask
        self._mask_out = ~mask
        self._mean_value_in = np.mean(self._image[self._mask_in])
        self._mean_value_out = np.mean(self._image[self._mask_out])
        print(f'in_{self._mean_value_in} out:{self._mean_value_out}')
        return self._k1 * (self._image - self._mean_value_in) ** 2 - self._k0 * (self._image - self._mean_value_out) ** 2


class Force2(ImageForce):
    def proba_in(self):
        return (1/np.sqrt(2*np.pi*self._var_value_in))*np.exp(-(self._image - self._mean_value_in)**2/(2*self._var_value_in))
    
    def proba_out(self):
        return (1/np.sqrt(2*np.pi*self._var_value_out))*np.exp(-(self._image - self._mean_value_out)**2/(2*self._var_value_out))

    def compute_force(self, mask):
        self._mask_in = mask
        self._mask_out = ~mask
        self._mean_value_in = np.mean(self._image[self._mask_in])
        self._var_value_in = np.std(self._image[self._mask_in])**2
        self._mean_value_out = np.mean(self._image[self._mask_out])
        self._var_value_out = np.std(self._image[self._mask_out])**2
        return np.log(self.proba_in()) - np.log(self.proba_out()) 


class Force3(ImageForce):
    def __init__(self, image, k0, k1, sigma) -> None:
        super().__init__(image)
        self._k0 = k0
        self._k1 = k1
        self._sigma = sigma
        self._kernel_dimension = 2 * sigma +1
        self._kernel_size = (self._kernel_dimension, self._kernel_dimension)

    def f_in(self) -> np.array:
        # In the paper do not specify how to treat border region
        image_in = self._image.copy()

        # Remove outside region
        image_in[self._mask_out] = 0

        # Gaussian Blur for the inside
        image_convoluted = cv.GaussianBlur(
            image_in,
            self._kernel_size,
            self._sigma)

        # Remove all results from region outside
        image_convoluted[self._mask_out] = 0
        return image_convoluted

    def f_out(self) -> np.array:
        # In the paper do not specify how to treat border region
        image_out = self._image.copy()

        # Remove inside region
        image_out[self._mask_in] = 0

        # Gaussian Blur for the outside
        image_convoluted = cv.GaussianBlur(
            image_out,
            self._kernel_size,
            self._sigma)

        # Remove all results from region inside
        image_convoluted[self._mask_in] = 0
        return image_convoluted
    
    def gaussian(self, x):
        c = 1 / np.sqrt(2*np.pi* self._sigma ** 2)
        return c * np.exp(-0.5 * np.linalg.norm(x) ** 2 / self._sigma ** 2)

    def compute_sum(self, mask, f, i, j):
        aux = 0
        rows, columns = np.where(mask)
        for m in rows:
            for n in columns:
                diff = self._image[i, j] - f[m, n]
                aux += self.gaussian(np.array([m - i, n - j])) * diff ** 2
        return aux

    def compute_force_2(self, mask):
        self._mask_in = mask
        self._mask_out = ~mask

        f_in = self.f_in()
        f_out = self.f_out()
        
        rows = self._image.shape[0]
        columns = self._image.shape[1]
        r3 = np.zeros_like(self._image)
        for i in np.arange(rows):
            for j in np.arange(columns):
                sum_in = self.compute_sum(self._mask_in, f_in, i, j)
                sum_out = self.compute_sum(self._mask_out, f_out, i, j)
                r3[i, j] = self._k1 * sum_in - self._k0 * sum_out
        return r3

    def compute_force(self, mask):
        self._mask_in = mask
        self._mask_out = ~mask

        f_in = self.f_in()
        f_out = self.f_out()

        # Separete in each term
        # See notebook for a detailed description
        t1 = self._k0 * self._k1 * self._image**2 * cv.GaussianBlur(
            np.ones_like(self._image),
            self._kernel_size,
            self._sigma)
        
        t2 = - 2 * self._k1 * self._image * cv.GaussianBlur(
            f_in.copy(),
            self._kernel_size,
            self._sigma)
        t2[self._mask_out] = 0

        t3 = 2 * self._k0 * self._image * cv.GaussianBlur(
            f_out.copy(),
            self._kernel_size,
            self._sigma)
        t3[self._mask_in] = 0

        t4 = self._k1 * cv.GaussianBlur(
            f_in.copy()**2,
            self._kernel_size,
            self._sigma)
        t4[self._mask_out] = 0

        t5 = - self._k0 * cv.GaussianBlur(
            f_out.copy()**2,
            self._kernel_size,
            self._sigma)
        t5[self._mask_in] = 0
        return t1 + t2 + t3 + t4 + t5


if __name__=='__main__':
    image=np.random.randint(0,255,(10,10))
    force = Force2(image)
    mask = np.zeros((10,10),bool) 
    mask[0:5,0:5]=True
    force.get_force(mask)
    print()