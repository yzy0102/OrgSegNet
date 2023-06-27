# How to modify config for dataset, training shedule and model config

In configs, note that "_base_" is the config base for all models, containing:
- "datasets" config, 
- "models" config 
- "shedules" config.


## datasets config
We create the Plant Cell organelle dataset config for training OrgSegNet. 

Preprocessing information about the training process is stored here.

We can see "_train_pipeline_", "_test_pipeline_", "_val_pipeline_". These piplines contain enhanced flows of data, such as "_RandomResize_", "_RandomCrop_", "_RandomFlip_" , "_PhotoMetricDistortion_" and so on. More data augmentation operations can be added if needed.

We can also see "train_dataloader", "test_dataloader", "val_dataloader". These data loaders mainly undertake the function of reading data to memory. The only parameters that need to be modified are "_batch size_" and "_num workers_" to fit the computer hardware.


"_val_evaluator_" and "_test_evaluator_" specify which parameters the model calculates when performing tests and validation. Here we generally use the iou_metrics=['mIoU', 'mDice', 'mFscore'].


## models config
The config of the model structure and parameters is saved in each corresponding file.

Let's take OrgSegNet.py as an example

1. First, we set the norm_cfg for the model training, SyncBN for multi-gpus and BN for single GPU. Then a data preprocessor is applied to normalize the RGB information of the image.
    ```
    # model settings
    norm_cfg = dict(type='SyncBN', requires_grad=True)
    data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255)
    ```

2. We define the model config. "_EncoderDecoder_" structure is the most common and useful structure in segmentation tasks. A pretrained model for resnet50 can be downloaded from open-mmlab if one want to load the pretrained checkpoint.

    In OrgSegNet, _ResNetV1c_ (ResNet50) was used as encoder, _OrgSeg_Head_ was used as decoder, FCNHead was used as auxiliary decoder. We set the _num_classes_ as 5 for segmenting the background, chloroplast, mitochondrion, vacuole and nucleus.

    If one wants to migrate a model to another dataset, one often only need to modify _num_classes_ in decode_head and auxiliary_head.

    ```
    model = dict(
        type='EncoderDecoder',
        data_preprocessor=data_preprocessor,
        pretrained='open-mmlab://resnet50_v1c',
        backbone=dict(
            type='ResNetV1c',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        decode_head=dict(
            type='OrgSeg_Head',
            in_channels=(256, 512, 1024, 2048),
            input_transform = 'multiple_select',
            in_index=(0, 1, 2, 3),
            channels=512,
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            num_classes=5,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=1., 
                # We can set a loss weight for each category
                weight = (0.5 , 0.5, 1.3, 1.3, 1.4))),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=5,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=0.4, 
                # We can set a loss weight for each category
                weight = (0.5, 0.5, 1.3, 1.3, 1.4))),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='slide', crop_size=(512, 512) ,stride=(256, 256)))
        # test_cfg=dict(mode='whole'))
    ```

    In OrgSegNet, we choose the _slide_ mode for training and inference, one can choose either _whole_ mode or _slide_ mode.


## schedules config
Early deep learning models generally used epoch to control the training flow, but our OrgSegNet and MMsegmentation recommend using iteration to define the training flow. Iteration control will be more flexible than epoch control, it should be noted that the two can be converted to each other.

Let's see schedule_160k.py as an example.

```
# optimizer
## We set SGD optimizer for model training
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # Save checkpoint every 200 interval
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200),
    # Early Stop, monitor can be loss, mIoU, Fscore ...
    early_stop = dict(type='EarlyStoppingHook', monitor='mIoU', patience=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

```

