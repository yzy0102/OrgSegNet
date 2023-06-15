_base_ = [
    '../_base_/models/danet_r50-d8.py',
    '../_base_/datasets/PlantCell_768x512.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
