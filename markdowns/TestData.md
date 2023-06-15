# Test the model using the trained model on testing dataset

Use the following code on the command line (on the conda environment)
```
python tools/test.py - config file - checkpoint file
```

Test on GPU ,for example:
```
CUDA_VISIBLE_DEVICES=0 python tools/test.py demo/work_dirs/OrgSegNet/OrgSeg_PlantCell_768x512.py demo/work_dirs/OrgSegNet/best_mIoU_iter_160000.pth
```

Test on CPU: If the machine does not have a GPU, the process of training on the CPU is the same as single GPU training. If the machine has GPUs, but does not want to use them, we just need to turn off the GPUs training function in the following ways before training.

```
CUDA_VISIBLE_DEVICES=-1 python tools/test.py demo/work_dirs/OrgSegNet/OrgSeg_PlantCell_768x512.py demo/work_dirs/OrgSegNet/best_mIoU_iter_160000.pth
```