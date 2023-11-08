import os
import cv2
from loader import NIFITLoader
import numpy as np
from typing import Tuple, List

def covert(input_file, output_path):
    """
        水平方向上切片并处理
    """
    l = NIFITLoader(input_file)
    array = l.resample([1.37, 1.37, l.get_pixdim()[2]])
    for slice in range(array.shape[2]):
        data = array[:, :, slice]
        data = np.array(cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)) # 将值缩放到0-255
        output_file = os.path.join(output_path, input_file.split('/')[-1].rstrip('.nii.gz')+f'_{slice+1}.jpg')
        data = fill(data, (384, 384))
        cv2.imwrite(output_file, data)


def fill(data: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
        四周填充0值
    """
    pad_height = shape[0] - data.shape[0]
    pad_width = shape[1] - data.shape[1]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # 使用np.pad进行填充
    return np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)




if __name__ == '__main__':
    for root, _ , files in os.walk('acdc_challenge_20170617'):
        nii_files = [f for f in files if f.endswith('_frame01.nii.gz') or f.endswith('_frame01_gt.nii.gz') ]
        if len(nii_files) != 0:
            m = 'train' if root.split('/')[-2] == 'training' else 'test'
            nii_files.sort()
            # 分别处理数据和相应标签
            for i, v in enumerate(['data', 'label']):
                covert(os.path.join(root, nii_files[i]), f'dataset/{m}/{v}')
            
