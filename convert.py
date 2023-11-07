import os
import cv2
from loader import niiLoader

def covert_2d(input_file, output_path):
    """
        水平方向上切片
    """
    l = niiLoader(input_file)
    array = l.resample([1.37, 1.37, l.get_pixdim()[2]])
    for slice in range(array.shape[2]):
        slice_data = array[:, :, slice]
        # ... 图片裁剪
        output_file = os.path.join(output_path, input_file.split('/')[-1].rstrip('.nii.gz')+f'_{slice+1}.jpg')
        cv2.imwrite(output_file, slice_data)

if __name__ == '__main__':
    for root, _ , files in os.walk('./acdc_challenge_20170617'):
        if len(files) != 0:
            for f in files:
                if f.endswith('_frame01.nii.gz') or f.endswith('_frame01_gt.nii.gz'):
                    m1 = 'train' if root.split('/')[-2] == 'training' else 'test'
                    m2 = 'data' if f.endswith('_frame01.nii.gz') else 'label'
                    covert_2d(os.path.join(root, f), f'dataset/{m1}/{m2}')
