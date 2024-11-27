import numpy as np
import numba

from .iou_utils import rotate_iou_gpu_eval
from .rotate_iou_cpu_eval import rotate_iou_cpu_eval
from .eval_utils import compute_split_parts, overall_filter, distance_filter, overall_distance_filter

iou_threshold_dict = {
    'Car': 0.7,
    'Bus': 0.7,
    'Truck': 0.7,
    'Pedestrian': 0.3,
    'Cyclist': 0.5
}

superclass_iou_threshold_dict = {
    'Vehicle': 0.7,
    'Pedestrian': 0.3,
    'Cyclist': 0.5
}


def get_evaluation_results(gt_annos, pred_annos, classes,
                           use_superclass=True,
                           iou_thresholds=None,
                           num_pr_points=50,
                           difficulty_mode='Overall&Distance',
                           ap_with_heading=True,
                           num_parts=100,
                           use_gpu = True,
                           print_ok=False
                           ):
    if iou_thresholds is None:
        if use_superclass:
            iou_thresholds = superclass_iou_threshold_dict
        else:
            iou_thresholds = iou_threshold_dict

    assert len(gt_annos) == len(pred_annos), "the number of GT must match predictions"
    assert difficulty_mode in ['Overall&Distance', 'Overall', 'Distance'], "difficulty mode is not supported"
    if use_superclass:
        if ('Car' in classes) or ('Bus' in classes) or ('Truck' in classes):
            assert ('Car' in classes) and ('Bus' in classes) and (
                    'Truck' in classes), "Car/Bus/Truck must all exist for vehicle detection"
        classes = [cls_name for cls_name in classes if cls_name not in ['Car', 'Bus', 'Truck']]
        classes.insert(0, 'Vehicle')

    num_samples = len(gt_annos)
    split_parts = compute_split_parts(num_samples, num_parts)
    # commont out this line to use gpu version
    if use_gpu:
        ious = compute_iou3d(gt_annos, pred_annos, split_parts, with_heading=ap_with_heading)
    else:
        ious = compute_iou3d_cpu(gt_annos, pred_annos)

    num_classes = len(classes)
    if difficulty_mode == 'Distance':
        num_difficulties = 3
        difficulty_types = ['0-30m', '30-50m', '50m-inf']
    elif difficulty_mode == 'Overall':
        num_difficulties = 1
        difficulty_types = ['overall']
    elif difficulty_mode == 'Overall&Distance':
        num_difficulties = 4
        difficulty_types = ['overall', '0-30m', '30-50m', '50m-inf']
    else:
        raise NotImplementedError

    precision = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points + 1])

    for cls_idx, cur_class in enumerate(classes):
        iou_threshold = iou_thresholds[cur_class]
        for diff_idx in range(num_difficulties):
            ### filter data & determine score thresholds on p-r curve ###
            accum_all_scores, gt_flags, pred_flags = [], [], []
            num_valid_gt = 0
            for sample_idx in range(num_samples):
                gt_anno = gt_annos[sample_idx]
                pred_anno = pred_annos[sample_idx]
                pred_score = pred_anno['score']
                iou = ious[sample_idx]
                gt_flag, pred_flag = filter_data(gt_anno, pred_anno, difficulty_mode,
                                                 difficulty_level=diff_idx, class_name=cur_class,
                                                 use_superclass=use_superclass)
                gt_flags.append(gt_flag)
                pred_flags.append(pred_flag)
                num_valid_gt += sum(gt_flag == 0)
                accum_scores = accumulate_scores(iou, pred_score, gt_flag, pred_flag,
                                                 iou_threshold=iou_threshold)
                accum_all_scores.append(accum_scores)
            all_scores = np.concatenate(accum_all_scores, axis=0)
            thresholds = get_thresholds(all_scores, num_valid_gt, num_pr_points=num_pr_points)

            ### compute tp/fp/fn ###
            confusion_matrix = np.zeros([len(thresholds), 3])  # only record tp/fp/fn
            for sample_idx in range(num_samples):
                pred_score = pred_annos[sample_idx]['score']
                iou = ious[sample_idx]
                gt_flag, pred_flag = gt_flags[sample_idx], pred_flags[sample_idx]
                for th_idx, score_th in enumerate(thresholds):
                    tp, fp, fn = compute_statistics(iou, pred_score, gt_flag, pred_flag,
                                                    score_threshold=score_th, iou_threshold=iou_threshold)
                    confusion_matrix[th_idx, 0] += tp
                    confusion_matrix[th_idx, 1] += fp
                    confusion_matrix[th_idx, 2] += fn

            ### draw p-r curve ###
            for th_idx in range(len(thresholds)):
                recall[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                    (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 2])
                precision[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                       (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 1])

            for th_idx in range(len(thresholds)):
                precision[cls_idx, diff_idx, th_idx] = np.max(
                    precision[cls_idx, diff_idx, th_idx:], axis=-1)
                recall[cls_idx, diff_idx, th_idx] = np.max(
                    recall[cls_idx, diff_idx, th_idx:], axis=-1)

    AP = 0
    for i in range(1, precision.shape[-1]):
        AP += precision[..., i]
    AP = AP / num_pr_points * 100

    ret_dict = {}

    ret_str = "\n|AP@%-9s|" % (str(num_pr_points))
    for diff_type in difficulty_types:
        ret_str += '%-12s|' % diff_type
    ret_str += '\n'
    for cls_idx, cur_class in enumerate(classes):
        ret_str += "|%-12s|" % cur_class
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            key = 'AP_' + cur_class + '/' + diff_type
            ap_score = AP[cls_idx, diff_idx]
            ret_dict[key] = ap_score
            ret_str += "%-12.2f|" % ap_score
        ret_str += "\n"
    mAP = np.mean(AP, axis=0)
    ret_str += "|%-12s|" % 'mAP'
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        key = 'AP_mean' + '/' + diff_type
        ap_score = mAP[diff_idx]
        ret_dict[key] = ap_score
        ret_str += "%-12.2f|" % ap_score
    ret_str += "\n"

    if print_ok:
        print(ret_str)
    return ret_str, ret_dict


@numba.jit(nopython=True)
def get_thresholds(scores, num_gt, num_pr_points):
    eps = 1e-6
    scores.sort()
    scores = scores[::-1]
    recall_level = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (r_recall + l_recall < 2 * recall_level) and i < (len(scores) - 1):
            continue
        thresholds.append(score)
        recall_level += 1 / num_pr_points
        # avoid numerical errors
        # while r_recall + l_recall >= 2 * recall_level:
        while r_recall + l_recall + eps > 2 * recall_level:
            thresholds.append(score)
            recall_level += 1 / num_pr_points
    return thresholds


@numba.jit(nopython=True)
def accumulate_scores(iou, pred_scores, gt_flag, pred_flag, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    accum_scores = np.zeros(num_gt)
    accum_idx = 0
    for i in range(num_gt):
        if gt_flag[i] == -1:  # not the same class
            continue
        det_idx = -1
        detected_score = -1
        for j in range(num_pred):
            if pred_flag[j] == -1:  # not the same class
                continue
            if assigned[j]:
                continue
            iou_ij = iou[i, j]
            pred_score = pred_scores[j]
            if (iou_ij > iou_threshold) and (pred_score > detected_score):
                det_idx = j
                detected_score = pred_score

        if (detected_score == -1) and (gt_flag[i] == 0):  # false negative
            pass
        elif (detected_score != -1) and (gt_flag[i] == 1 or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected_score != -1:  # true positive
            accum_scores[accum_idx] = pred_scores[det_idx]
            accum_idx += 1
            assigned[det_idx] = True

    return accum_scores[:accum_idx]


@numba.jit(nopython=True)
def compute_statistics(iou, pred_scores, gt_flag, pred_flag, score_threshold, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    under_threshold = pred_scores < score_threshold

    tp, fp, fn = 0, 0, 0
    for i in range(num_gt):
        if gt_flag[i] == -1:  # different classes
            continue
        det_idx = -1
        detected = False
        best_matched_iou = 0
        gt_assigned_to_ignore = False

        for j in range(num_pred):
            if pred_flag[j] == -1:  # different classes
                continue
            if assigned[j]:  # already assigned to other GT
                continue
            if under_threshold[j]:  # compute only boxes above threshold
                continue
            iou_ij = iou[i, j]
            if (iou_ij > iou_threshold) and (iou_ij > best_matched_iou or gt_assigned_to_ignore) and pred_flag[j] == 0:
                best_matched_iou = iou_ij
                det_idx = j
                detected = True
                gt_assigned_to_ignore = False
            elif (iou_ij > iou_threshold) and (not detected) and pred_flag[j] == 1:
                det_idx = j
                detected = True
                gt_assigned_to_ignore = True

        if (not detected) and gt_flag[i] == 0:  # false negative
            fn += 1
        elif detected and (gt_flag[i] == 1 or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected:  # true positive
            tp += 1
            assigned[det_idx] = True

    for j in range(num_pred):
        if not (assigned[j] or pred_flag[j] == -1 or pred_flag[j] == 1 or under_threshold[j]):
            fp += 1

    return tp, fp, fn


def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level, class_name, use_superclass):
    """
    Filter data by class name and difficulty

    Args:
        gt_anno:
        pred_anno:
        difficulty_mode:
        difficulty_level:
        class_name:

    Returns:
        gt_flags/pred_flags:
            1 : same class but ignored with different difficulty levels
            0 : accepted
           -1 : rejected with different classes
    """
    num_gt = len(gt_anno['name'])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_or(gt_anno['name'] == 'Pedestrian', gt_anno['name'] == 'Cyclist')
        else:
            reject = gt_anno['name'] != class_name
    else:
        reject = gt_anno['name'] != class_name
    gt_flag[reject] = -1
    num_pred = len(pred_anno['name'])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_or(pred_anno['name'] == 'Pedestrian', pred_anno['name'] == 'Cyclist')
        else:
            reject = pred_anno['name'] != class_name
    else:
        reject = pred_anno['name'] != class_name
    pred_flag[reject] = -1

    if(num_gt==0 or num_pred==0):
        return gt_flag, pred_flag

    if difficulty_mode == 'Overall':
        ignore = overall_filter(gt_anno['boxes_3d'])
        gt_flag[ignore] = 1
        ignore = overall_filter(pred_anno['boxes_3d'])
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Distance':
        ignore = distance_filter(gt_anno['boxes_3d'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = distance_filter(pred_anno['boxes_3d'], difficulty_level)
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Overall&Distance':
        ignore = overall_distance_filter(gt_anno['boxes_3d'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = overall_distance_filter(pred_anno['boxes_3d'], difficulty_level)
        pred_flag[ignore] = 1
    else:
        raise NotImplementedError

    return gt_flag, pred_flag


def iou3d_kernel(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    # inter_h[intersection_2d <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    # eps = 1e-6
    # union_3d[union_3d<eps] = eps
    iou3d = intersection_3d / union_3d
    return iou3d


def iou3d_kernel_with_heading(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    # inter_h[intersection_2d <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    # eps = 1e-6
    # union_3d[union_3d<eps] = eps
    iou3d = intersection_3d / union_3d

    # rotation orientation filtering
    diff_rot = gt_boxes[:, [6]] - pred_boxes[:, [6]].T
    diff_rot = np.abs(diff_rot)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[diff_rot >= np.pi]  # constrain to [0-pi]
    iou3d[diff_rot > np.pi / 2] = 0  # unmatched if diff_rot > 90
    return iou3d


def rotate_iou_kernel_eval(gt_boxes, pred_boxes):
    iou3d_cpu = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
    return iou3d_cpu


def compute_iou3d(gt_annos, pred_annos, split_parts, with_heading):
    """
    Compute iou3d of all samples by parts

    Args:
        with_heading: filter with heading
        gt_annos: list of dicts for each sample
        pred_annos:
        split_parts: for part-based iou computation

    Returns:
        ious: list of iou arrays for each sample
    """
    gt_num_per_sample = np.stack([len(anno["name"]) for anno in gt_annos], 0)
    pred_num_per_sample = np.stack([len(anno["name"]) for anno in pred_annos], 0)
    ious = []
    sample_idx = 0
    for num_part_samples in split_parts:
        gt_annos_part = gt_annos[sample_idx:sample_idx + num_part_samples]
        pred_annos_part = pred_annos[sample_idx:sample_idx + num_part_samples]

        gt_boxes = np.concatenate([anno["boxes_3d"] for anno in gt_annos_part], 0)
        pred_boxes = np.concatenate([anno["boxes_3d"] for anno in pred_annos_part], 0)

        if with_heading:
            iou3d_part = iou3d_kernel_with_heading(gt_boxes, pred_boxes)
        else:
            iou3d_part = iou3d_kernel(gt_boxes, pred_boxes)

        gt_num_idx, pred_num_idx = 0, 0
        for idx in range(num_part_samples):
            gt_box_num = gt_num_per_sample[sample_idx + idx]
            pred_box_num = pred_num_per_sample[sample_idx + idx]
            ious.append(iou3d_part[gt_num_idx: gt_num_idx + gt_box_num, pred_num_idx: pred_num_idx + pred_box_num])
            gt_num_idx += gt_box_num
            pred_num_idx += pred_box_num
        sample_idx += num_part_samples
    return ious


def compute_iou3d_cpu(gt_annos, pred_annos):
    ious = []
    gt_num = len(gt_annos)
    for i in range(gt_num):
        gt_boxes = gt_annos[i]['boxes_3d']
        pred_boxes = pred_annos[i]['boxes_3d']

        iou3d_part = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
        ious.append(iou3d_part)
    return ious


if __name__ == '__main__':
    import numpy as np
    import copy
    pred_data = {
        "name": [
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Car",
            "Truck",
            "Cyclist",
            "Cyclist",
            "Cyclist",
            "Cyclist",
            "Cyclist",
            "Cyclist",
            "Cyclist",
            "Pedestrian"
        ],
        "boxes_3d": np.array([
            [
                -3.460779890336198, 11.889099506707907, -0.8206982783150591,
                4.555971205891034, 1.8030093908309937, 1.5509192471702893,
                5.2348885297696235
            ],
            [
                -0.9969175029538491, -38.69317913045427, -1.2679895304161408,
                4.708746910095215, 1.7456061840057373, 1.5380195379257202,
                0.14170256455475894
            ],
            [
                -11.875659942626953, -28.584253311157227, -1.1764512062072754,
                4.154930114746094, 1.8246525526046753, 1.4076584577560425,
                3.9471536000543317
            ],
            [
                26.029391688726562, 9.458823658682036, -0.878351778172369,
                4.8718061447143555, 2.0425333976745605, 1.8974950313568115,
                3.131668361025401
            ],
            [
                7.066287517547607, -41.02307891845703, -1.1766657829284668,
                4.748326301574707, 1.7213102579116821, 1.4444016218185425,
                6.081188654891694
            ],
            [
                45.00757598876953, 0.7134484648704529, -1.3726370267315207,
                4.079574108123779, 1.871203899383545, 1.6068871021270752,
                2.7457537968927106
            ],
            [
                33.84749221801758, 6.647745609283447, -1.163730502128601,
                4.244619846343994, 1.7117009162902832, 1.4961812496185303,
                2.753302129107066
            ],
            [
                -12.939452171325684, -38.86998748779297, -1.1824709177017212,
                4.087010383605957, 1.7874534130096436, 1.4045615196228027,
                0.34081760247284976
            ],
            [
                16.09961141847191, 2.2441862041183356, -1.0851360821293023,
                4.3121538162231445, 1.9871797561645508, 1.391538143157959,
                3.588504823046275
            ],
            [
                54.64401685621457, -2.9081035025770916, -1.5011897673062937,
                4.491567134857178, 1.8608672618865967, 1.6215827465057373,
                2.922738345461436
            ],
            [
                37.15433522149317, 9.52385295002776, -1.2339278905676163,
                4.498213291168213, 1.9067984819412231, 1.4956731796264648,
                2.734384687738963
            ],
            [
                0.8569065254568784, -14.71125619047831, -1.183545708656311,
                3.9722704887390137, 1.7630285024642944, 1.4633375406265259,
                4.290142091112681
            ],
            [
                -29.912790298461914, -46.6987419128418, -1.5271672010421753,
                4.4965620040893555, 1.7782410383224487, 1.6138628721237183,
                0.45587411721284
            ],
            [
                -21.40715789794922, -45.748043060302734, -1.2914018630981445,
                4.487395286560059, 1.7117500305175781, 1.5276386737823486,
                0.44109621842438784
            ],
            [
                -38.672725677490234, -47.02140808105469, -1.4273680448532104,
                4.5264363288879395, 1.7633424997329712, 1.5924698114395142,
                3.5285605510049542
            ],
            [
                34.55817212078739, 2.9413106086257557, -1.3670268467323625,
                4.448055267333984, 1.7787138223648071, 1.5423678159713745,
                2.717174323397227
            ],
            [
                -28.57036590576172, -49.468746185302734, -1.355146884918213,
                4.083809852600098, 1.6974924802780151, 1.5252057313919067,
                0.34226456482941714
            ],
            [
                -40.58879470825195, -55.847801208496094, -1.2195556163787842,
                4.496826171875, 1.7447901964187622, 1.5922259092330933,
                0.4245611747079572
            ],
            [
                42.1269688013447, 10.068271333382597, -1.0660970743981393,
                4.407517910003662, 1.7619510889053345, 1.6174633502960205,
                2.7486172590822537
            ],
            [
                17.742129626695544, -38.093257147063426, -0.3968806048707871,
                5.979849815368652, 2.516859292984009, 3.1814844608306885,
                6.148222899429047
            ],
            [
                17.100009612344763, -48.41649681150909, -1.2255488271075503,
                1.5471781492233276, 0.800000011920929, 1.3624770641326904,
                5.97439143656894
            ],
            [
                33.6546963846252, 15.037311442813689, -1.230517046341546,
                1.6226742267608643, 0.800000011920929, 1.603995680809021,
                2.813317807512828
            ],
            [
                25.753924132766628, 14.898764911988565, -1.0705579414767101,
                1.4544227123260498, 0.9581016302108765, 1.4188017845153809,
                3.121172221499034
            ],
            [
                -16.848543599483037, 19.006636140863463, -0.716509823080124,
                1.2435309886932373, 0.8035303950309753, 1.3958953619003296,
                5.20078942774936
            ],
            [
                -9.264792401310842, 44.088361671800826, 0.8896512093546477,
                1.2013179063796997, 0.9209445118904114, 1.4112517833709717,
                4.781368231765473
            ],
            [
                -36.885494232177734, -39.51515197753906, 0.020576000213623047,
                2.299999952316284, 0.800000011920929, 1.0937252044677734,
                1.4354871829324445
            ],
            [
                20.579928122556566, -61.34001813369708, -1.2065993237927943,
                2.316119909286499, 1.3219619989395142, 1.5640028715133667,
                4.821046328536713
            ],
            [
                23.305380231369465, 22.853182976351945, -1.199327267460939,
                0.30345029611085217, 0.27185997596153844, 0.8178032603639808,
                4.71238898038469
            ]
        ]).reshape(-1, 7),
        "score": np.array([
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9
        ])
    }
    info_data=copy.deepcopy(pred_data)
    del info_data['score']
    import pickle

    # info_data = pickle.load(
    #     open('once_infos_val.pkl', 'rb'))  # you can find this file in once_devkit/submission_format/
    # pred_data = pickle.load(open('../../../../../../../Downloads/result.pkl', 'rb'))  # your prediction file
    gt_data = list()
    # for item in info_data:
    #     if 'annos' in item:
    #         gt_data.append(item['annos'])
    gt_data.append(info_data)
    classes = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']
    result_str, result_dict = get_evaluation_results(gt_data, [pred_data],
                                                     ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist'], True)
    print(result_str)
    print(result_dict)
