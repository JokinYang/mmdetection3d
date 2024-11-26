from collections import OrderedDict
from pathlib import Path

import mmcv
import mmengine
import numpy as np

from mmdet3d.structures.ops import box_np_ops
from .once_data_utils import get_once_image_info

once_categories = ('Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def create_once_info_file(data_path,
                          pkl_prefix='once',
                          with_plane=False,
                          save_path=None,
                          relative_path=True):
    """Create info file of ONCE dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'once'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    # test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    once_infos_train = get_once_image_info(data_path, image_ids=train_img_ids, relative_path=relative_path)
    # _calculate_num_points_in_gt(data_path, once_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Once info train file is saved to {filename}')
    mmengine.dump(once_infos_train, filename)

    once_infos_val = get_once_image_info(data_path, image_ids=val_img_ids, relative_path=relative_path)
    # _calculate_num_points_in_gt(data_path, once_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Once info val file is saved to {filename}')
    mmengine.dump(once_infos_val, filename)

    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Once info trainval file is saved to {filename}')
    mmengine.dump(once_infos_train + once_infos_val, filename)

    # once_infos_test = get_once_image_info(data_path, image_ids=test_img_ids, relative_path=relative_path)
    # filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    # print(f'Once info test file is saved to {filename}')
    # mmengine.dump(once_infos_test, filename)
