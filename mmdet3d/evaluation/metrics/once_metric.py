# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes

from mmdet3d.evaluation import once_eval


@METRICS.register_module()
class OnceMetric(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'mAP',
                 pcd_limit_range: List[float] = [-75.2, -75.2, -5.0, 75.2, 75.2, 3.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 use_superclass: bool = False,
                 difficulty_mode: str = 'Overall&Distance',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'once metric'
        super(OnceMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args
        self.use_superclass = use_superclass
        self.difficulty_mode = difficulty_mode

        allowed_metrics = ['mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'mAP' "
                               f'but got {metric}.')

    def convert_annos_to_once_annos(self, data_infos: dict) -> List[dict]:
        """Convert loading annotations to once annotations.

        Args:
            data_infos (dict): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of Kitti annotations.
        """
        data_annos = data_infos['data_list']
        if not self.format_only:
            cat2label = data_infos['metainfo']['categories']
            label2cat = dict((v, k) for (k, v) in cat2label.items())
            assert 'instances' in data_annos[0]
            for i, annos in enumerate(data_annos):
                if len(annos['instances']) == 0:
                    once_annos = {
                        'name': np.array([]),
                        'boxes_3d': np.array([])
                    }
                else:
                    once_annos = {
                        'name': [],
                        'boxes_3d': []
                    }
                    for instance in annos['instances']:
                        label = instance['bbox_label']
                        once_annos['name'].append(label2cat[label])
                        once_annos['boxes_3d'].append(instance['bbox_3d'])
                    for name in once_annos:
                        once_annos[name] = np.array(once_annos[name])
                data_annos[i]['once_annos'] = once_annos
        return data_annos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # 使用下面的方式可以直接从data_batch中提取gt的label信息
        # ann_samples = data_batch['data_samples']
        # assert len(ann_samples) == len(
        #     data_samples), "The number of samples in ann_batch and data_batch should be the same."
        # for i in range(len(ann_samples)):
        #     anno = ann_samples[i]
        #     pred = data_samples[i]
        #     assert anno['sample_idx'] == pred['sample_idx']
        #     result = dict()
        #     pred_3d = anno['pred_instances_3d']
        #     for attr_name in pred_3d:
        #         pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
        #     result['pred_instances_3d'] = pred_3d
        #
        #     eval_ann_info = anno['eval_ann_info']
        #     result['eval_ann_info'] = eval_ann_info
        #     result['sample_idx'] = anno['sample_idx']
        #     self.results.append(result)
        # return None

        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            # pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            # for attr_name in pred_2d:
            #     pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            # result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        self.data_infos = self.convert_annos_to_once_annos(pkl_infos)

        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.submission_prefix)}')
            return metric_dict

        gt_annos = [
            self.data_infos[result['sample_idx']]['once_annos']
            for result in results
        ]

        for metric in self.metrics:
            ap_dict = self.once_evaluate(
                result_dict,
                gt_annos,
                metric=metric,
                logger=logger,
                classes=self.classes)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def once_evaluate(self,
                      results_dict: dict,
                      gt_annos: List[dict],
                      metric: Optional[str] = None,
                      classes: Optional[List[str]] = None,
                      logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in KITTI protocol.

        Args:
            results_dict (dict): Formatted results of the dataset.
            gt_annos (List[dict]): Contain gt information of each sample.
            metric (str, optional): Metrics to be evaluated. Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        ap_dict = dict()
        for name in results_dict:
            ap_result_str, ap_dict_ = once_eval(
                gt_annos, results_dict[name], classes,
                use_superclass=self.use_superclass,
                difficulty_mode=self.difficulty_mode, use_gpu=True)
            for ap_type, ap in ap_dict_.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')

            print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)
            break

        return ap_dict

    def format_results(
            self,
            results: List[dict],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None,
            classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to pkl file.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing the
            formatted result, tmp_dir is the temporal directory created for
            saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]
        for name in results[0]:
            if submission_prefix is not None:
                submission_prefix_ = osp.join(submission_prefix, name)
            else:
                submission_prefix_ = None
            if pklfile_prefix is not None:
                pklfile_prefix_ = osp.join(pklfile_prefix, name) + '.pkl'
            else:
                pklfile_prefix_ = None
            if 'pred_instances' in name and '3d' in name and name[0] != '_' and results[0][name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_once(net_outputs,
                                                     sample_idx_list, classes,
                                                     pklfile_prefix_,
                                                     submission_prefix_)
                result_dict[name] = result_list_
            # elif name == 'pred_instances' and name[0] != '_' and results[0][
            #     name]:
            #     net_outputs = [result[name] for result in results]
            #     result_list_ = self.bbox2result_kitti2d(
            #         net_outputs, sample_idx_list, classes, pklfile_prefix_,
            #         submission_prefix_)
            #     result_dict[name] = result_list_
        return result_dict, tmp_dir

    def bbox2result_once(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting 3D prediction to ONCE format')
        for idx, pred_dicts in enumerate(mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            info = self.data_infos[sample_idx]

            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'boxes_3d': [],
                'score': []
            }
            if len(box_dict['label_preds']) > 0:
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box_lidar, score, label in zip(box_preds_lidar, scores, label_preds):
                    anno['name'].append(class_names[int(label)])
                    anno['boxes_3d'].append(box_lidar)
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]),
                    'boxes_3d': np.array([]),
                    'score': np.array([]),
                }
            anno['name'] = anno['name'].reshape(-1)
            anno['boxes_3d'] = anno['boxes_3d'].reshape(-1, 7)
            anno['score'] = anno['score'].reshape(-1)

            anno['sample_idx'] = np.array([sample_idx] * len(anno['score']), dtype=np.int64)
            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict: dict, info: dict) -> dict:
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (Tensor): Scores of boxes.
                - labels_3d (Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

            - bbox (np.ndarray): 2D bounding boxes.
            - box3d_camera (np.ndarray): 3D bounding boxes in
              camera coordinate.
            - box3d_lidar (np.ndarray): 3D bounding boxes in
              LiDAR coordinate.
            - scores (np.ndarray): Scores of boxes.
            - label_preds (np.ndarray): Class label predictions.
            - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        # check box_preds_lidar
        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_lidar = box_preds
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds_lidar.center > limit_range[:3]) &
                              (box_preds_lidar.center < limit_range[3:]))
            valid_inds = valid_pcd_inds.all(-1)
        else:
            raise Exception("Invalid box type, only support LiDARInstance3DBoxes!")

        if valid_inds.sum() > 0:
            return dict(
                box3d_lidar=box_preds_lidar[valid_inds].numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)
