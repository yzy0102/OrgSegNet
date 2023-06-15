# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=1000,
        by_epoch=True)
]
# training schedule for 320k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs =1000, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    early_stop = dict(type='EarlyStoppingHook', monitor='loss', patience=10),
    visualization=dict(type='SegVisualizationHook'))
