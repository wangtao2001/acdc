import nibabel as nib
from typing import Tuple, Dict, List
import numpy as np
import skimage


class NIFITLoader:

    """
        对nibabel库的封装
    """

    img_data: nib.nifti1.Nifti1Image

    def __init__(self, filename) -> None:
        self.img_data = nib.load(filename)

    def get_shape(self) -> Tuple[int, ...]:
        return self.img_data.shape
    
    def get_array(self) -> np.ndarray:
        return self.img_data.get_fdata()
    
    def get_header(self) -> Dict:
        return self.img_data.header

    def get_pixdim(self) -> List[float]:
        return self.get_header()['pixdim'][1:4]
    
    def resample(self, spacing: List[int]) -> np.ndarray:
        shape = list(self.get_shape())
        pixdim = self.get_pixdim()
        new_shape = np.array(shape) * np.array(pixdim) / np.array(spacing)
        array_resampled = skimage.transform.resize(self.get_array(), new_shape, order=1, anti_aliasing=True) # 双线性插值，抗锯齿
        return array_resampled
