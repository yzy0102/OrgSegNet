# How to prepare the data for training or testing

If you want to train OrgSegNet on official datasets or custom datasets, you need to install the following instructions to build the dataset correctly.

## Set up an official dataset
The official dataset has been open sourced on ScienceDB and contains 19 plant varieties with over 900 high-quality images of plant organelles.
All the data can be downloaded in https://doi.org/10.11922/sciencedb.01335


<b>The dataset folder structure should look like this:</b>
```
    # -- CellData
    #   --- image
    #       ----img1.tif
    #       ----img2.tif
    #       ----imgxxx.tif
    #   --- label
    #       ----img1.png
    #       ----img2.png
    #       ----imgxxx.png
    #   --- splits
    #       ----train.txt 
    #       ----val.txt
    #       ----test.txt
```
Notes:
- The original image file is stored in the image folder, and the file suffix is tif.
- The label file is stored in the label folder, and the file suffix is png.
- The label file is stored in the label folder, and the file suffix is png.
- The splits folder stores the data set division information, and each txt file contains the name of the image.

All of these files can be downloaded from the above address (https://doi.org/10.11922/sciencedb.01335). In official dataset, 60% of the dataset was used as training set (541 images), 20% was used as validation set (180 images), and the remaining 20% was used for model testing (181 images).

</br>

## Set up personal fine-tuning datasets
In addition to training with official datasets, we also provide a build format for fine-tuning datasets.

Similarly, datasets should also be formatted as follows:
```
    # -- FinetuneDataset001
    #   --- image
    #       ----img1.tif
    #       ----img2.tif
    #       ----imgxxx.tif
    #   --- label
    #       ----img1.png
    #       ----img2.png
    #       ----imgxxx.png
    #   --- splits
    #       ----train.txt 
    #       ----val.txt
    #       ----test.txt
```

The original image suffix can be tif, jpg, jpeg, png, etc., but needs to be changed accordingly in the dataset file
[mmseg/datasets/plantcell.py](../mmseg/datasets/plantcell.py)

```
    def __init__(self,
        img_suffix='.tif', # .tif can be replaced with other images, such as .jpg
        seg_map_suffix='.png',
        **kwargs) -> None:
```