norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=16,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,
        in_index=4,
        channels=16,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=32,
        in_index=3,
        channels=16,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
dataset_type = 'PlantCellDataset'
data_root = '/home/CODE/mmsegmentation/CellData/'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(768, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(768, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 0.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 0.75,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.0,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.25,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.75,
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'horizontal'
                    }], [{
                        'type': 'LoadAnnotations'
                    }], [{
                        'type': 'PackSegInputs'
                    }]])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='PlantCellDataset',
        data_root='/home/CODE/mmsegmentation/CellData/',
        data_prefix=dict(img_path='image', seg_map_path='label'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(768, 512),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ],
        ann_file='splits/train.txt'))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PlantCellDataset',
        data_root='/home/CODE/mmsegmentation/CellData/',
        data_prefix=dict(img_path='image', seg_map_path='label'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(768, 512), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ],
        ann_file='splits/val.txt'))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PlantCellDataset',
        data_root='/home/CODE/mmsegmentation/CellData/',
        data_prefix=dict(img_path='image', seg_map_path='label'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(768, 512), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ],
        ann_file='splits/test.txt'))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=200,
        max_keep_ckpts=5,
        save_best=['mIoU'],
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
work_dir = './work_dirs/Unet_512max'
randomness = dict(seed=0)
