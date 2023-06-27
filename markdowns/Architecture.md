# The code architecture of OrgSegNet
## Inherited from MMSegMentation

[MMSegMentation](https://github.com/open-mmlab/mmsegmentation.git) is a powerful open source framework, integrating most of the methods from development to deployment

As a fork of MMsegMentation, OrgSegNet follows the usage of the official MMsegMentation library.

Our OrgSegNet is developed based on the MMsegmentation framework (v1.0.0), so for many problems, it can be found in [MMsegmentation guidelines](https://mmsegmentation.readthedocs.io/en/latest/). 

In principle, any relevant tutorials can be found in the official MMsegMentation github repository. But OrgSegNet still provides code-related tutorials in as much detail as possible.

## New features of OrgSegNet
Our OrgSegNet is dedicated to providing segmentation and reconstruction of important organelles in plant cells. We also provide a morphological feature extraction method for 2D organelle images.

Under the demo folder, we provide various demos to realize all the functions mentioned in our paper (waiting to be published).

- [How to train OrgSegNet](../demo/Train_OrgSegNet_demo.ipynb)
- [How to fine-tune OrgSegNet to any other dataset](../demo/Fine-tune_OrgSegNet_demo.ipynb)
- [How to inference on an image](../demo/inference_demo.ipynb)
- [How to generate instances of individual organelles based CAM](../demo/InstanceGenerate.ipynb)
- [How to extract morphological characteristics of plant organelles (same function as web version)](../demo/PlantOrganelleHunterWebImplementation.ipynb)

## Code architecture
- configs: The structure and parameters of the model are saved. Guide see [how to modify config](ArchGuide/HowToModifyConfig.md)
- demo: Demos of OrgSegNet
- markdowns: Some OrgSegNet usage tutorials
- mmseg: Some important components of MMsegmentation and OrgSegNet
- mmseg: Some important components of MMsegmentation and OrgSegNet
- tests: Some test project files
- tools: some tools for MMsegmentation and OrgSegNet
- utils: some important functions for extract morphological characteristics of plant organelles
- README.md: General Guidance Document.

## Code 

