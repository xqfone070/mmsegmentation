import os
import time


_base_ = [
    '../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../configs/_base_/datasets/pascal_voc12.py',
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_20k.py'
]

# common
model_name = 'deeplabv3plus_r50'

# train config
batch_size = 4
max_iters = 20000
val_interval = int(max_iters * 0.1)
lr_base = 0.01

# dataset config
dataset_type = 'PascalVOCDataset'
data_root = '/home/alex/data/TPS2000_body_seg_train_1001_20231229'

img_subdir = 'images'
ann_subdir = 'annotations'
set_subdir = 'sets'
train_ann_file = os.path.join(set_subdir, 'train.txt')
val_ann_file = os.path.join(set_subdir, 'val.txt')
test_ann_file = os.path.join(set_subdir, 'test.txt')

# model config
img_scale = (256, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=False,
    # 会将结果尺寸pad到该尺寸
    size=img_scale)
num_classes = 3


# train
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
dataset_name = os.path.basename(data_root)
run_name = '%s_%dx%d_%s' % (model_name, img_scale[0], img_scale[1], time_str)

load_from = None
resume = False
work_dir = os.path.join('work_dirs', dataset_name, run_name)

# model
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes))

# dataset
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=img_subdir, seg_map_path=ann_subdir),
        ann_file=train_ann_file,
        pipeline=train_pipeline)
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=img_subdir, seg_map_path=ann_subdir),
        ann_file=val_ann_file,
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=img_subdir, seg_map_path=ann_subdir),
        ann_file=test_ann_file,
        pipeline=test_pipeline)
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# optimizer
optimizer = dict(type='SGD', lr=lr_base, momentum=0.9, weight_decay=0.0005)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False)
]

# train schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)

wandb_init_kwargs = {'project': dataset_name,
                     'name': run_name}
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=wandb_init_kwargs)
])

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=val_interval,
        save_best='mIoU')
)