def iou_batch(boxA, boxB):

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU, only working for object parallel to ground

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''

    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    _, inter_area = convex_hull_intersection(rect1, rect2)

    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def dist3d(detection, track):
    dist = math.sqrt((detection[0] - track[0]) ** 2 +
                     (detection[1] - track[1]) ** 2 + (detection[2] - track[2]) ** 2)
    return dist

def associate_with_DIoU(detections, trackers, aplha, iou_threshold):
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

    eucliDistance_square_matrix = np.full(
        (len(dets_8corner), len(trks_8corner)), 1000, dtype=np.float32)

    Diou_matrix = np.zeros(
        (len(dets_8corner), len(trks_8corner)), dtype=np.float32)

    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            # det: 8 x 3, trk: 8 x 3
            iou_matrix[d, t] = iou3d(det, trk)[0]
            eucliDistance_square_matrix[d, t] = eucliDistance_square(
                detections[d].bbox[0:3], trackers[t].pose[0:3])
            Diou_matrix[d, t] = iou_matrix[d, t] - \
                math.sqrt(eucliDistance_square_matrix[d, t]) / aplha

    matches = []
    if min(Diou_matrix.shape) > 0:
        matched_indices = linear_assignment_diou(
            1-Diou_matrix, 1-iou_threshold)

        if len(matched_indices) == 0:
            matched_indices = np.empty(shape=(0, 2))
    else:
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
        if Diou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)