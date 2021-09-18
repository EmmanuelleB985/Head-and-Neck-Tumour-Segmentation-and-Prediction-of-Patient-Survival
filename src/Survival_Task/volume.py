import numpy as np
import pandas as pd
import SimpleITK as sitk
import pathlib 
import os
import math

def read_nifti(path):
    nifti = sitk.ReadImage(str(path))
    return nifti

def volume( mask_image ):
    """
    Input:
    image = sitk.Image, mask or binary image (1 values where tumour, 0 values otherwise)
    Output:
    vol = float, volume in mm3 
    """
    space = mask_image.GetSpacing()         # image spacing
    voxel = np.prod(space)                  # voxel volume
    img = sitk.GetArrayFromImage(mask_image)
    nonzero_voxel_count = np.count_nonzero(img)
    vol = voxel*nonzero_voxel_count

    return vol

def main():
    # Load masks:
    # Path to pre-processed resampled images
    path_to_masks = pathlib.Path('/well/papiez/users/gbo097/hecktor2021_train/hecktor_resampled/')

    patients = [p for p in os.listdir(path_to_masks) if os.path.isdir(path_to_masks)]
    masks_list = []
    patients_id = []
    
    for p in patients:
        patients_id.append(p)
        try:
            masks = read_nifti(path_to_masks / p / (p + '_gtvt.nii.gz'))
            masks_list.append(masks)
        except:
            print('Unable to read input image file.')

    import pandas as pd  

    # Calculate the volume
    i = 0
    vol = [] 
    loc = []
    for mask in masks_list:
    
        v = volume(mask)
        vol.append(v)
    

        
    dict = {'id': patients_id, 'volume': vol}  
        
    df = pd.DataFrame(dict) 
    
    # saving the volumes as csv dataframe 
    df.to_csv('/gpfs3/users/papiez/gbo097/HK-2021/src/volume.csv') 
    
if __name__ == "__main__":
    # execute only if run as a script
    main()