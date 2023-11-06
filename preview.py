import nibabel as nib
from typing import Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt

class NiiPreviewer:

    img_data: nib.nifti1.Nifti1Image

    def __init__(self, filename) -> None:
        self.img_data = nib.load(filename)

    def get_shape(self) -> Tuple[int, ...]:
        return self.img_data.shape
    
    def get_array(self) -> np.ndarray:
        return self.img_data.get_fdata()
    
    def get_header(self) -> Dict:
        return self.img_data.header
    
    def show(self, depth:int, frame: int) -> None:
        plt.imshow(self.get_array()[:, :, depth, frame], cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    p = NiiPreviewer('./dataset/training/patient001/patient001_4d.nii.gz')
    height, width, depth, frame = p.get_shape()
    print(f'The image object height: {height}, width:{width}, depth:{depth}, frame:{frame}')
    print(p.get_array().shape)
    print(p.get_header()['pixdim']) # 分辨率数据
    p.show(3, 1)