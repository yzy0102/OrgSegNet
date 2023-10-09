# How to run demos

OrgSegNet provides a variety of running demos to reproduce the results in the article and provide more possibilities for expansion.

## 1. Run training demo
A training demo is shown in [Train_OrgSegNet_demo.ipynb](../../demo/Train_OrgSegNet_demo.ipynb). Before starting, please ensure that the operating environment is configured correctly, and ensure that you have an Nvidia GPU with more than 11GB GPU memory. Afterwards, you can follow the guidance in the notebook for the training process.


## 2. Run fine-tune demo
OrgSegNet (PlantOrganelle Hunter) is committed to providing a more robust segmentation method for the field of plant cell segmentation and providing assistance for downstream analysis. The original version of OrgSegNet was trained on more than 900 high-quality TEM images, and nearly 10,000 plant organelles were included. Hence, OrgSegNet can also be regarded as an important pre-training model, which can provide more possibilities for more forms of data sets. 

A fine-tuning demo is provided (see [Fine-tune_OrgSegNet_demo.ipynb](../../demo/Fine-tune_OrgSegNet_demo.ipynb)) to assist those in need to achieve fast fine-tuning of OrgSegNet, which can better realize the role of OrgSegNet in more detailed areas.

The important thing is that before fine-tune, a relatively complete data set needs to be constructed. One can use LabelMe to label the images and convert it into png format. The dataset should be constructed in the form required by OrgSegNet (same as the training dataset provided by OrgSegNet see https://cstr.cn/31253.11.sciencedb.01335).

The dataset used for fine-tune does not need to be particularly complex, e.g., for an 800-image TEM stack of cells, only 40 images with manually labeled labels are required. It only takes half an hour of training on a 3090GPU to complete the accurate 3D reconstruction of a plant cell.



## 3. Run inference demo
An inference demo was added to [inference_demo.ipynb](../../demo/inference_demo.ipynb). You can load your trained model _(checkpoint)_ or the pre-trained model officially provided by OrgSegNet. 

~~The pretrained checkpoint can be downloaded from the [google drive link](https://drive.google.com/file/d/12TYv8mEUWdVqjrfbrKZK_OtcV5pq6ejr/view?usp=drive_link), 
or run this code below:~~

<b>The pretrained checkpoint can be downloaded from the [zonodo link](https://doi.org/10.5281/zenodo.8419877).</b>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8419877.svg)](https://doi.org/10.5281/zenodo.8419877)



```
!mkdir ../checkpoints
!wget https://doi.org/10.5281/zenodo.8419877 -P ../checkpoints
```



Make sure that the environment is set up correctly. In the inference process, the performance requirements for the CPU and GPU are low. You can use a CPU for inference, and using a GPU for inference will speed up the process.



## 4. Run Generate Instance demo
A demo that generates organelle instances from segmentation results is provided [InstanceGenerate.ipynb](../../demo/InstanceGenerate.ipynb).

This demo represents the post-processing part of the article, whose role is to generate examples of various organelles from the semantic segmentation results of OrgSegNet to help the analysis of downstream tasks.


## 5. Run PlantHunter web demo
While providing the webpage PlantOrganelle Hunter, we also provide the function realization of the local version of PlantOrganelle Hunter in [PlantOrganelleHunterWebImplementation.ipynb](../../demo/PlantOrganelleHunterWebImplementation.ipynb). 

This demo implements the entire process of data input, OrgSegNet for segmentation, application of post-processing to generate instances of target organelles, and extraction of morphological features of target organelles.

You can get the segmentation result of the image, the example result of the organelle and the morphological characteristic parameters of the organelle (cross-sectional area, shape complexity, electron density) after running the demo.

What needs to be input is only the image and the scale of the image, and the parameters that need to be adjusted are only the threshold required for post-processing.


## 6. Other demos
We provide a demo to show the difference between ISA and SA [ISA and SA.ipynb](../../demo/ISA%20and%20SA.ipynb). 

We provide a demo to calculate the inference time [Calculate_inference_Time_Cost.ipynb](../../demo/Calculate_Inference_Time_Cost.ipynb). 