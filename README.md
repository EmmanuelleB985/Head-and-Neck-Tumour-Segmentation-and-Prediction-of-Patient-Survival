# Head-and-Neck-Tumour-Segmentation-and-Prediction-of-Patient-Survival

<p align="center"><project-description></p>

Welcome to the Head and Neck Segmentation Tumour Segmentation and Prediction of Patient Survival!

This project aims to provide methods to automatically segment the primary gross target volume on
fluoro-deoxyglucose (FDG)-PET and Computed Tomography (CT) images and prediction of progression-free survival in H&N oropharyngeal cancer. We participated to the HEad and neCK TumOR Segmentation and Prediction of
Patient Outcome Challenge 2021 (HECKTOR 2021) which creates a platform for
comparing segmentation methods and predictions of patient survival.
For the segmentation task, we proposed a new network based on an encoder/decoder architecture with attention mechanisms and full inter- and intra-skip connections and attention mechanisms inspired from UNet3+ to take advantage of low-level and high-level semantics at full scales. Additionally, we used Conditional Random Fields (CRF) as a post-processing step to refine the predicted segmentation maps. 
For prediction of patient progression free survival task, we propose a Cox proportional hazard regression combining
clinical, radiomic, and deep learning features. 


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/EmmanuelleB985/Head-and-Neck-Tumour-Segmentation-and-Prediction-of-Patient-Survival.git
   ```
2. Install the packages
   ```sh
   pip install requirements.txt
   ```

## Usage

* For segmentation task
```sh
   cd src/Segmentation_Task
```
To train the model
```sh
   python main.py
```

* For Survival task
```sh
   cd src/Survival_Task
```

## License

Distributed under the MIT License. See `LICENSE` for more information.