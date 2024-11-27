from mmengine.dataset.sampler import DefaultSampler
from mmengine.visualization.vis_backend import LocalVisBackend, TensorboardVisBackend

from mmdet3d.datasets.once_dataset import OnceDataset
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile)
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D
from mmdet3d.datasets.transforms.transforms_3d import (  # noqa
    ObjectNoise, ObjectRangeFilter, ObjectSample,
    PointShuffle, PointsRangeFilter, RandomFlip3D, PointsRotateZ90CW, GlobalRotScaleTrans)
from mmdet3d.evaluation import OnceMetric
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

# dataset settings
dataset_type = 'OnceDataset'
data_root = 'data/once/'
class_names = ('Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')
# copy from https://github.com/open-mmlab/OpenPCDet/blob/8cacccec11db6f59bf6934600c9a175dae254806/tools/cfgs/dataset_configs/once_dataset.yaml#L5
point_cloud_range = [-75.2, -75.2, -5.0, 75.2, 75.2, 3.0]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'once_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Truck=5, Bus=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Truck=15, Bus=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type=PointsRotateZ90CW),  # trans the points to mmdetection3d LiDAR coordinate system
    dict(type=LoadAnnotations3D, with_bbox_3d=True, with_label_3d=True),
    dict(type=ObjectSample, db_sampler=db_sampler),
    dict(
        type=ObjectNoise,
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type=RandomFlip3D, flip_ratio_bev_horizontal=0.5),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=ObjectRangeFilter, point_cloud_range=point_cloud_range),
    dict(type=PointShuffle),
    dict(
        type=Pack3DDetInputs, keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type=PointsRotateZ90CW),
    dict(
        type=MultiScaleFlipAug3D,
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type=GlobalRotScaleTrans,
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type=RandomFlip3D),
            dict(type=PointsRangeFilter, point_cloud_range=point_cloud_range)
        ]),
    dict(type=Pack3DDetInputs, keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type=PointsRotateZ90CW),
    dict(type=Pack3DDetInputs, keys=['points'])
]
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=OnceDataset,
        data_root=data_root,
        ann_file='once_infos_train.pkl',
        data_prefix=dict(pts=''),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=OnceDataset,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file='once_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=OnceDataset,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file='once_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type=OnceMetric,
    ann_file=data_root + 'once_infos_val.pkl',
    metric='mAP',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type=LocalVisBackend), dict(type=TensorboardVisBackend)]
visualizer = dict(
    type=Det3DLocalVisualizer, vis_backends=vis_backends, name='visualizer')
