import numpy as np 

class ImageForce:
    def __init__(self, image) -> None:
        self._image = image
        self._r = None

    def get_force(self, level_set_function):
        mask = level_set_function >= 0
        self._r = self.compute_force(mask)
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
        self._mask_in=mask
        self._mask_out = ~mask
        self._mean_value_in = np.mean(self._image[self._mask_in])
        self._var_value_in = np.std(self._image[self._mask_in])**2
        self._mean_value_out = np.mean(self._image[self._mask_out])
        self._var_value_out = np.std(self._image[self._mask_out])**2
        return np.log(self.proba_in()) - np.log(self.proba_out()) 
    
class Force3(ImageForce):
    def __init__(self, image, initial_mask, k0,k1) -> None:
        super().__init__(image, initial_mask)
        self._k0 = k0
        self._k1 = k1
        self._mean_value_initial = np.mean(image[self.initial_mask])
        self._sd_value_initial = np.mean(image[self.initial_mask] - self._mean_value_initial)
        self._mean_value_complement_intial = np.mean(image[self.complement_initial_mask])
        self._sd_value_complement_intial = np.mean(image[self.complement_initial_mask] - self._mean_value_complement_intial)

    def proba_initial(self): 
        return (1/np.sqrt(2*np.pi*self._sd_value_initial))*np.exp(-np.abs(self._image[self.initial_mask] - self._mean_value_initial)**2/(2*self._sd_value_initial))
    
    def proba_complement_initial(self): 
        return (1/np.sqrt(2*np.pi*self._sd_value_complement_intial))*np.exp(-np.abs(self._image[self.complement_initial_mask] - self._mean_value_complement_intial)**2/(2*self._sd_value_complement_intial))

    def compute_force(self):
        return np.log(self.proba_initial()) - np.log(self.proba_complement_initial()) 
    
if __name__=='__main__':
    image=np.random.randint(0,255,(10,10))
    force = Force2(image)
    mask = np.zeros((10,10),bool) 
    mask[0:5,0:5]=True
    force.get_force(mask)
    print()