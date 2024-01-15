import os
import time

_base_ = [
    '../../configs/segmenter/segmenter_vit-b_mask_8xb1-160k_ade20k-512x512.py',
    # '../../configs/_base_/datasets/pascal_voc12.py',
    # '../../configs/_base_/default_runtime.py',
    # '../../configs/_base_/schedules/schedule_20k.py'
]

# common
model_name = 'segmenter-b'

# train config
batch_size = 8
max_iters = 160000
# val_interval = int(max_iters * 0.1)
val_interval = 2000
lr_base = 0.001

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
# 训练的尺度
img_scale = (256, 512)
# 注意crop的shape是(h, w)，不是(w, h)
crop_size = (512, 256)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # 不做通道颜色变换，因为灰度图
    bgr_to_rgb=False,
    # preprocess不做尺寸缩放或padding
    size=crop_size
)

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
    backbone=dict(
        img_size=crop_size,
    ),
    decode_head=dict(
        num_classes=num_classes,
    ),
    test_cfg=dict(mode='whole')
)

# dataset
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=8,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=9),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

dataset_type = 'PascalVOCDataset'
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

test_dataloader = dict(
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
optimizer = dict(type='SGD', lr=lr_base, momentum=0.9, weight_decay=0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

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
        save_best='mIoU', max_keep_ckpts=10)
)
