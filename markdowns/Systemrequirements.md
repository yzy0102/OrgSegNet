# System requirements

## Operating System
OrgSegNet has been tested on Linux (Ubuntu 18.04, 20.04, 22.04; RHEL) and Windows (Windows 10 and 11)! But it should be able to be used in any environment that can use mmcv and mmsegmentation (such as MacOS). It should work out of the box!

</br>

## Hardware requirements
We support GPU (recommended), CPU and Apple M1/M2 as devices.

</br>

## Hardware requirements for Training
We strongly recommend using the GPU for the training process.
For training an Nvidia GPU with at least 10 GB (popular non-datacenter options are the Nvidia GTX 1080ti, RTX 2080ti, RTX 3060/3080/3090 or RTX 4080/4090) is required. 
A strong CPU (with more than 6 cores) is also recommended for data preprocessing and data augmentation.


The minimum configuration we recommend is Intel i5-8400 CPU coupled with 16 GB RAM and Nvidia 1080ti GPU. 

## Hardware requirements for inference
We still recommend using the GPU for inference. 

In any case, the GPU specifications required for the inference can be lower, and any mainstream Nvidia GPU with more than 4GB memory can do the job (GTX 1070/1080/1080ti, RTX 2060/2060s/2070/2070s/2080/2080s/2080ti, RTX 3060/3060ti/3070/3070ti/3080/3080ti/3090/3090ti or RTX 4060ti/4080/4090).

The CPU can also do this within a reasonable time frame, and we also recommend CPUs with cores greater than 6 (from Intel i5-8400 to Intel i9-13900k, from Ryzen 5 2600 to Ryzen 9 7950x).


## Example hardware configurations
Example workstation configurations for training:

 - CPU: CPU can selected from Intel and AMD. Intel i5-10400 - Intel i9-13900k. Intel's 12th and 13th generation CPUs will be stronger; R5 3600 - R9 7950x. AMD's R9 7900X and R9 7950X will be stronger.
 - GPU: RTX 3090\3090ti\4090 with 24GB GPU memory
 - RAM: 32 GB or 64 GB
 - Storage: 512 GB storage; SSD (M.2 PCIe Gen 3 or better!) is better. 

---
Minimum configuration:
 - CPU: Intel i5-8400
 - GPU: RTX 1080ti
 - RAM: 16 GB
 - Storage: 512 GB SSD storage

---
Recommended configuration
 - CPU: Intel i5-10400
 - GPU: RTX 3090
 - RAM: 32 GB
 - Storage: 512 GB SSD storage
 