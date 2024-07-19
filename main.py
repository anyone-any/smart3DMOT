# -*-coding:utf-8-*
# author: ---
from __future__ import print_function
import os
import shutil
import numpy as np
import time
import cv2
import torch
from copy import deepcopy
from os import listdir
from os.path import join
from file_operation.file import load_list_from_folder, mkdir_if_inexistence, fileparts
from detection.detection import Detection_2D, Detection_3D_only, Detection_3D_Fusion
from tracking.a3_tracker_DIoU import Tracker
from datasets.datafusion import datafusion2Dand3D
from datasets.coordinate_transformation import convert_3dbox_to_8corner, convert_x1y1x2y2_to_tlwh
from visualization.visualization_3d import show_image_with_boxes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from tracking.cost_function import Model, Model_v1, Model_without_geo, MLP_copy, MLP_copy_v1,Attention_model

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class DeepFusion(object):
    def __init__(self, max_age, min_hits):
        '''
        :param max_age:  The maximum frames in which an object disappears.
        :param min_hits: The minimum frames in which an object becomes a trajectory in succession.
        '''
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracker = Tracker(max_age, min_hits)
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.frame_count = 0

    def update(self, detection_3D_fusion, detection_2D_only, detection_3D_only, detection_3Dto2D_only,
               additional_info, calib_file, para1, para2, model):

        dets_3d_fusion = np.array(detection_3D_fusion['dets_3d_fusion'])
        dets_3d_fusion_info = np.array(detection_3D_fusion['dets_3d_fusion_info'])
        dets_3d_fusion_vis = np.array(detection_3D_fusion['vis_feats'])
        dets_3d_fusion_point = np.array(detection_3D_fusion['point_feats'])


        dets_3d_only = np.array(detection_3D_only['dets_3d_only'])
        dets_3d_only_info = np.array(detection_3D_only['dets_3d_only_info'])
        dets_3d_only_vis = np.array(detection_3D_only['vis_feats'])
        dets_3d_only_point = np.array(detection_3D_only['point_feats'])

        if len(dets_3d_fusion) == 0:
            dets_3d_fusion = dets_3d_fusion
        else:
            dets_3d_fusion = dets_3d_fusion[:, self.reorder]

        if len(dets_3d_only) == 0:
            dets_3d_only = dets_3d_only
        else:
            dets_3d_only = dets_3d_only[:, self.reorder]

        detection_3D_fusion = [Detection_3D_Fusion(
            det_fusion, dets_3d_fusion_info[i], dets_3d_fusion_vis[i], dets_3d_fusion_point[i]) for i, det_fusion in enumerate(dets_3d_fusion)]
        
        detection_3D_only = [Detection_3D_only(
            det_only, dets_3d_only_info[i], dets_3d_only_vis[i], dets_3d_only_point[i]) for i, det_only in enumerate(dets_3d_only)]
        
        detection_2D_only = [Detection_2D(
            det_fusion) for i, det_fusion in enumerate(detection_2D_only)]

        self.tracker.predict_2d()
        self.tracker.predict_3d()
        self.tracker.update(detection_3D_fusion, detection_3D_only,
                            detection_3Dto2D_only, detection_2D_only, calib_file, model, para1, para2, iou_threshold=0.5)

        self.frame_count += 1
        outputs = []
        for track in self.tracker.tracks_3d:
            if track.is_confirmed():
                bbox = np.array(track.pose[self.reorder_back])
                outputs.append(np.concatenate(
                    ([track.track_id_3d], bbox, track.additional_info[0:7])).reshape(1, -1))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    # Convert the coordinate format of the bbox box from center x, y, w, h to upper left x, upper left y, w, h
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), 0)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), 0)
        return x1, y1, x2, y2

    def _tlwh_to_x1y1x2y2(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return x1, y1, x2, y2


if __name__ == '__main__':

    para1_list = [25]
    para2_list = [-0.25] 

    hota_para1_para2_list_VAL1 = []
    hota_para1_para2_list_VAL2 = []

    for para1 in para1_list:
        for para2 in para2_list:

            mlp = Attention_model(256, 128, 1)  
            mlp_path = "./mlp_best.pt"
            mlp.load_state_dict(torch.load(mlp_path), strict=True)
            mlp.eval()
            mlp.cuda()
            data_root = 'datasets/kitti/train'
            detections_name_3D = '3D_CasA'
            detections_name_2D = '2D_yolov8'
            det_2d_score_filter = 0.75  

            calib_root = os.path.join(data_root, 'calib_train')
            dataset_dir = os.path.join(data_root, 'image_02')
            detections_root_3D = os.path.join(data_root, detections_name_3D)
            detections_root_2D = os.path.join(data_root, detections_name_2D)

            mkdir_if_inexistence('/result/')
            save_root = '/result'
            mkdir_if_inexistence(save_root)
            txt_path_0 = os.path.join(save_root, 'data')
            mkdir_if_inexistence(txt_path_0)
            image_path_0 = os.path.join(save_root, 'image')
            mkdir_if_inexistence(image_path_0)
            # Open file to save in list.
            det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
            calib_files = os.listdir(calib_root)

            detections_files_3D = sorted(os.listdir(detections_root_3D))

            detections_files_2D = sorted(os.listdir(detections_root_2D))

            image_files = sorted(os.listdir(dataset_dir))
            detection_file_list_3D, num_seq_3D = load_list_from_folder(
                detections_files_3D, detections_root_3D)
            detection_file_list_2D, num_seq_2D = load_list_from_folder(
                detections_files_2D, detections_root_2D)
            image_file_list, _ = load_list_from_folder(image_files, dataset_dir)

            # Tracker runtime, total frames and Serial number of the dataset
            total_time, total_frames, i = 0.0, 0, 0
            tracker = DeepFusion(max_age=para1, min_hits=3)  # Tracker initialization
            
            All_frames = 0
            Numbers_SG_all_seqs_fusion = []
            Numbers_SG_all_seqs_only = []
            Numbers_dioupro_fusion = []
            Numbers_dioupro_only = []
            val_seqs_name_list = ['0000','0002','0003','0004','0005','0007','0009','0011','0017','0020']
            # Iterate through each data set
            for seq_file_3D, image_filename in zip(detection_file_list_3D, image_files):
                print('--------------Start processing the {} dataset--------------'.format(image_filename))
                history_track_ifo = []
                total_image = 0  # Record the total frames in this dataset
                seq_file_2D = detection_file_list_2D[i]
                seq_name, datasets_name, _ = fileparts(seq_file_3D)

                txt_path = txt_path_0 + "/" + image_filename + '.txt'

                image_path = image_path_0 + '/' + image_filename
                mkdir_if_inexistence(image_path)

                calib_file = [
                    calib_file for calib_file in calib_files if calib_file == seq_name]
                calib_file_seq = os.path.join(calib_root, ''.join(calib_file))
                image_dir = os.path.join(dataset_dir, image_filename)

                image_filenames = sorted([join(image_dir, x)
                                        for x in listdir(image_dir) if is_image_file(x)])
                # load 3D detections, N x 15
                seq_dets_3D = np.loadtxt(seq_file_3D, delimiter=',')
                # load 2D detections, N x 6
                seq_dets_2D = np.loadtxt(seq_file_2D, delimiter=',')

                #进行2D过滤
                seq_dets_2D = seq_dets_2D[seq_dets_2D[:, -1] >= 0.15, :]

                min_frame, max_frame = int(
                    seq_dets_3D[:, 0].min()), len(image_filenames)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                
                All_frames += max_frame

                json_file = '/%s.json' % (datasets_name)

                with open(json_file, 'r', encoding='utf-8') as fp:

                    all_data = json.load(fp)

                mot_result_dict = {}
                for frame, img0_path in zip(range(min_frame, max_frame + 1), image_filenames):
                    #print(frame)
                    img_0 = cv2.imread(img0_path)
                    _, img0_name, _ = fileparts(img0_path)
                    
                    if str(frame) not in all_data:

                        detection_3D_fusion = {'dets_3d_fusion':[],'dets_3d_fusion_info':[],'vis_feats':[],'point_feats':[]}
                        detection_2D_only_tlwh = []
                        detection_3D_only = {'dets_3d_only':[],'dets_3d_only_info':[],'vis_feats':[],'point_feats':[]}
                        detection_3Dto2D_only = []
                        additional_info = []
                    else:
        
                        detection_3D_fusion = all_data[str(frame)]['detection_3D_fusion']
                        detection_2D_only_tlwh = all_data[str(frame)]['detection_2D_only_tlwh']
                        detection_3D_only = all_data[str(frame)]['detection_3D_only']
                        detection_3Dto2D_only = all_data[str(frame)]['detection_3Dto2D_only']
                        additional_info =  all_data[str(frame)]['addition_info']
                    
                    start_time = time.time()
                    trackers = tracker.update(detection_3D_fusion, detection_2D_only_tlwh, detection_3D_only, detection_3Dto2D_only,
                                            additional_info, calib_file_seq, para1, para2, mlp)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    # Outputs
                    total_frames += 1  # Total frames for all datasets
                    total_image += 1  # Total frames for a dataset
                    if total_image % 50 == 0:
                        print("Now start processing the {} image of the {} dataset".format(
                            total_image, image_filename))

                    if len(trackers) > 0:
                        for d in trackers:
                            bbox3d = d.flatten()
                            bbox3d_tmp = bbox3d[1:8]
                            id_tmp = int(bbox3d[0])
                            ori_tmp = bbox3d[8]
                            type_tmp = det_id2str[bbox3d[9]]
                            bbox2d_tmp_trk = bbox3d[10:14]
                            conf_tmp = bbox3d[14]
                            color = compute_color_for_id(id_tmp)
                            label = f'{id_tmp} {"car"}'
                            image_save_path = os.path.join(
                                image_path, '%06d.jpg' % (int(img0_name)))
    
                            with open(txt_path, 'a') as f:
                                str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, type_tmp, ori_tmp, bbox2d_tmp_trk[0],
                                                                                                        bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[
                                                                                                            3], bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3],
                                                                                                        bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp)
                                f.write(str_to_srite)
                
                            value = [frame, id_tmp, type_tmp, ori_tmp, bbox2d_tmp_trk[0],
                                    bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
                                    bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3],
                                    bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp]
                            if str(id_tmp) in mot_result_dict:
                                mot_result_dict[str(id_tmp)].append(value)
                            else:
                                mot_result_dict[str(id_tmp)] = [value]
                
                if image_filename in val_seqs_name_list:
                    Numbers_SG_all_seqs_fusion.append(np.sum(np.array(tracker.tracker.num_ifo_3dfusion), axis=0)) 
                    Numbers_SG_all_seqs_only.append(np.sum(np.array(tracker.tracker.num_ifo_3donly), axis=0)) 
                    Numbers_dioupro_fusion.extend(tracker.tracker.num_dioupro_3dfusion)
                    Numbers_dioupro_only.extend(tracker.tracker.num_dioupro_3donly)
                
                txt_path_new = './result/data' + "/" + image_filename + '.txt'
                if os.path.exists(txt_path_new):
                    os.remove(txt_path_new)
                with open(txt_path_new, 'a') as f:  
                    for ele in mot_result_dict:
                        if len(mot_result_dict[ele]) >= 3:
                            for det_trk in mot_result_dict[ele]:
                                str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (det_trk[0], det_trk[1], det_trk[2], 
                                                                                                        det_trk[3], det_trk[4], det_trk[5], 
                                                                                                        det_trk[6], det_trk[7], det_trk[8], 
                                                                                                        det_trk[9], det_trk[10], det_trk[11],
                                                                                                        det_trk[12], det_trk[13], det_trk[14], det_trk[15])
                                f.write(str_to_srite)
                i += 1
                print('--------------The time it takes to process all datasets are {}s --------------'.format(total_time))
            print('--------------FPS = {} --------------'.format(total_frames/total_time))
            print(total_frames, total_time)

           
