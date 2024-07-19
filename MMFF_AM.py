def associate_tith_MMFF(detections, trackers, model, edu_threshold, feat_threshold):
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

    feats_matrix = np.zeros(
        (len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    
    eucliDistance_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)

    for d, det in enumerate(detections):  # [1]
        for t, trk in enumerate(trackers):  # [2, 3]

            feats_matrix[d, t] = feats_scores_img_point_CM(det, trk, model) # det: 8 x 3, trk: 8 x 3

    matches = []
    if min(feats_matrix.shape) > 0:
        matched_indices = linear_assignment(1-feats_matrix, 1-feat_threshold)
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
        if feats_matrix[m[0], m[1]] < feat_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)