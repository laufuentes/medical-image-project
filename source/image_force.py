import numpy as np 

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
        self._mean_value_in = np.mean(self._image[mask])
        self._mean_value_out = np.mean(self._image[1-mask])
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
    def __init__(self, image, k0, k1) -> None:
        super().__init__(image)
        self._k0 = k0
        self._k1 = k1

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
    
if __name__=='__main__':
    image=np.random.randint(0,255,(10,10))
    force = Force2(image)
    mask = np.zeros((10,10),bool) 
    mask[0:5,0:5]=True
    force.get_force(mask)
    print()