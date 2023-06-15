# Prepare the enviroment for OrgSegNet
0. Download and install Anaconda from the  [official website](https://www.anaconda.com/download/).

1. Create a conda envirement using:
    ```
    conda create -n orgseg python==3.8.16
    ```

2. Activate the orgseg enviroment
    ```
    conda activate orgseg
    ```

3. Download the integrative framework (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed
    ```
    git clone https://github.com/yzy0102/OrgSegNet.git
    cd OrgSegNet
    ```

4. Choose one from 4.1 and 4.2 depending on your hardware configuration

-    4.1 Install pytorch=0.13.1 torchvision torchaudio based cuda=11.6 if   you have a GPU
        ```
        pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
        ```
-   4.2 Install pytorch=0.13.1 torchvision torchaudio based cpu if you do not have a GPU
    ```
        pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
    ```


5. Install MMCV using MIM.
    ```
    pip install -U openmim
    mim install mmengine
    mim install mmcv==2.0.0rc4
    ```

6. Install OrgSegNet
    ```
    git clone -b main https://github.com/open-mmlab/mmsegmentation.git
    cd OrgSegNet
    pip install -v -e .

    # '-v' means verbose, or more output
    # '-e' means installing a project in editable mode,
    # thus any local modifications made to the code will take effect without reinstallation.
    ```

7. Install the necessary dependencies
    ```
    pip install jupyter
    pip install future tensorboard
    pip install -r requirements.txt
    ```