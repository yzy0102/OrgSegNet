_base_ = [
    '../_base_/models/pspnet_r50-d8.py', 
    '../_base_/datasets/PlantCell_768x512.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(depth=50),
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
