import numpy as np
from copy import deepcopy
import math
import torch.nn.functional as F

def associate_detections_to_trackers_STSC(detections, trackers, model, iou_threshold, sim_threshold, edu_threshold, feat_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    detections:  N x 8 x 3
    trackers:    M x 8 x 3
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    dets_8corner = [convert_3dbox_to_8corner(
        det_tmp.bbox) for det_tmp in detections]
    if len(dets_8corner) > 0:
        dets_8corner = np.stack(dets_8corner, axis=0)
    else:
        dets_8corner = []

    trks_8corner = [convert_3dbox_to_8corner(
        trk_tmp.pose) for trk_tmp in trackers]
    if len(trks_8corner) > 0:
        trks_8corner = np.stack(trks_8corner, axis=0)
    if (len(trks_8corner) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0, 8, 3), dtype=int)

    iou_matrix = np.zeros(
        (len(dets_8corner), len(trks_8corner)), dtype=np.float32)

    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            # det: 8 x 3, trk: 8 x 3
            iou_matrix[d, t] = iou3d(det, trk)[0]

    rows, cols = iou_matrix.shape
    sim_ontshot_matrix = np.zeros_like(iou_matrix, dtype=iou_matrix.dtype)
    for i in range(rows):
        for j in range(cols):
            if cols >=2 :
                matrix = deepcopy(iou_matrix)

                row_without_current = np.delete(matrix[i, :], j)

                max_except_current = np.max(row_without_current)

                row_score = math.sqrt((iou_matrix[i,j]-1)**2 + (max_except_current-0)**2)
            else:
                row_score = math.sqrt((iou_matrix[i,j]-1)**2)
            
            if rows >= 2:
                matrix = deepcopy(iou_matrix)
                column_without_current = np.delete(matrix[:, j], i)

                max_except_current = np.max(column_without_current)
 
                col_score = math.sqrt((iou_matrix[i,j]-1)**2 + (max_except_current-0)**2)
            else:
                col_score = math.sqrt((iou_matrix[i,j]-1)**2)
            sim_ontshot = (row_score + col_score) / 2
            sim_ontshot_matrix[i, j] = sim_ontshot

    matches1 = []
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(
                1-iou_matrix, 1-iou_threshold)  
    else:
        matched_indices = np.empty(shape=(0, 2))

    if len(matched_indices) == 0:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(dets_8corner):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trks_8corner):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    for m in matched_indices:  
       
        if iou_matrix[m[0], m[1]] > iou_threshold and sim_ontshot_matrix[m[0], m[1]] < sim_threshold:
            matches1.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])

    if len(matches1) == 0:
        matches1 = np.empty((0, 2), dtype=int)
    else:
 
        matches1 = np.concatenate(matches1, axis=0)

    matches2 = []
    eucliDistance_matrix = np.full(
        (len(dets_8corner), len(trks_8corner)), 1000, dtype=np.float32)
    feats_matrix = np.zeros(
        (len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    for d, det in enumerate(detections):  # [1]
        for t, trk in enumerate(trackers):  # [2, 3]
            if d in unmatched_detections and t in unmatched_trackers:
                eucliDistance_matrix[d, t] = eucliDistance(
                    det.bbox[0:3], trk.pose[0:3])
                if eucliDistance_matrix[d, t] < edu_threshold:
                    feats_matrix[d, t] = feats_scores_img_point_CM(det, trk, model) # det: 8 x 3, trk: 8 x 3

    if not np.all(feats_matrix == 0):  

        mat = linear_assignment(-feats_matrix, -feat_threshold)  # array

        for m in mat:
            matches2.append(m.reshape(1, 2))

        if len(mat) > 0:
            # unmatched_detections = []#移除已经匹配的
            for d in unmatched_detections:
                if d in mat[:, 0]:
                    unmatched_detections.remove(d)

            # unmatched_trackers = []#移除
            for t in unmatched_trackers:
                if t in mat[:, 1]:
                    unmatched_trackers.remove(t)

        if len(matches2) == 0:
            matches2 = np.empty((0, 2), dtype=int)
        else:
            # [array([[0, 1]]), array([[2, 0]])]
            matches2 = np.concatenate(matches2, axis=0)

    elif len(matches2) == 0:
        matches2 = np.empty((0, 2), dtype=int)
    else:
        matches2 = np.concatenate(matches2, axis=0)

    if len(matches2) == 0:
        matches = matches1
    else:
        # print('match1:', matches1)
        # print('match2:', matches2)
        matches = np.vstack((matches1, matches2))
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

