import numpy as np
import SimpleITK as sitk
import pathlib 
import os
import pandas as pd  
from radiomics import featureextractor
from itertools import product


def nifti_read(path):
    nifti_img = sitk.ReadImage(str(path))
    return nifti_img


#-----------------------------------------------------
# PET/CT feature extraction using pyradiomics package 
#-----------------------------------------------------
def main():

    # Path to pre-processed resampled images and segmentation maps
    path = pathlib.Path('/well/papiez/users/gbo097/hecktor2021_train/hecktor2021_test/hecktor_nii_resampled/')
    seg =  pathlib.Path('/well/papiez/users/gbo097/hecktor2021_train/results/predictions/')

    patients = [p for p in os.listdir(path) if os.path.isdir(path / p)]
    masks_list = []
    ct_list = []
    patients_id = []

    for p in patients:
        patients_id.append(p)
        try:
            masks = nifti_read(seg /(p + '.nii.gz'))
            ct_scans = nifti_read(path / p / (p + '_ct.nii.gz')) #/pet
            masks_list.append(masks)
            ct_list.append(ct_scans)
        except:
            print('Unable to read input image file.')
    

    # convert to simple itk format
    mask = sitk.GetArrayFromImage(masks)
    ct = sitk.GetArrayFromImage(ct_scans)
    m = sitk.GetImageFromArray(mask) 
    ct = sitk.GetImageFromArray(ct)

    # extractor radiomics features with feature extractor function
    extractor = featureextractor.RadiomicsFeatureExtractor()
    res = extractor.execute(ct, m)
    features_names = list(res.keys())
    features_list = dict.fromkeys(features_names)

    for feature in features_names:
        features_list[feature] = []

    for mask,ct in zip(masks_list,ct_list):

        # Feature extractor
        mask = sitk.GetArrayFromImage(mask)
        ct = sitk.GetArrayFromImage(ct)
        m = sitk.GetImageFromArray(mask) 
        ct = sitk.GetImageFromArray(ct)
        # extractor radiomics features with feature extractor function
        extractor = featureextractor.RadiomicsFeatureExtractor()
        res = extractor.execute(ct, m)

        for key, value in res.items():
                features_list[key].append(value)

    # Save extracted features from ct/pet
    features = pd.DataFrame(data=features_list)

    # Add patients id to dataframe
    features.insert(loc=0,column='patients_id',value=patients_id)

    # Save the dataframe 
    features.to_csv('/gpfs3/users/papiez/gbo097/HK-2021/src/extracted_features_ct_T.csv', sep=',',index=False) 

if __name__ == "__main__":
    # execute only if run as a script
    main()