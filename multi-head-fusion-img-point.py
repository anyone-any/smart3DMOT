import copy
import math
import numpy as np
from numba import jit
from scipy.spatial import ConvexHull
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def batch_multiply(objs, dets):
    """

        :param objs: BxDxN
        :param dets: BxDxM
        :return:BxDxNxM
        """
    x = torch.einsum('bci,bcj->bcij', objs, dets)
    return x

def batch_minus(objs, dets):
    """

    :param objs: BxDxN
    :param dets: BxDxM
    :return: Bx2dxNxM
    """

    obj_mat = objs.unsqueeze(-1).repeat(1, 1, 1, dets.size(-1))  # BxDxNxM
    det_mat = dets.unsqueeze(-2).repeat(1, 1, objs.size(-1), 1)  # BxDxNxM
    related_pos = (obj_mat - det_mat) / 2  # BxDxNxM
    return related_pos

def batch_minus_abs(objs, dets):
    """

    :param objs: BxDxN
    :param dets: BxDxM
    :return: Bx2dxNxM
    """

    obj_mat = objs.unsqueeze(-1).repeat(1, 1, 1, dets.size(-1))  # BxDxNxM
    det_mat = dets.unsqueeze(-2).repeat(1, 1, objs.size(-1), 1)  # BxDxNxM
    related_pos = (obj_mat - det_mat) / 2  # BxDxNxM
    x = related_pos.abs()  # Bx2DxNxM
    return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, IMG, POINTS):
        FUSHION = torch.cat([IMG.squeeze(1), POINTS.squeeze(1)], 1).unsqueeze(2)  # (N+M)x642x1
        b, c, _  = FUSHION.size()
        weight = self.avg_pool(FUSHION).view(b, c)
        weight = self.fc(weight).view(b, c, 1)
        FUSHION = torch.mul(FUSHION, weight).squeeze(2)
        return FUSHION


class MMFeatFusion(torch.nn.Module): 
    def __init__(self, use_attention='multihead', lc_fused_type='cat'):
        super(MMFeatFusion, self).__init__()
        self.use_attention = use_attention
        self.lc_fused_type = lc_fused_type

        self.c2c_att = nn.MultiheadAttention(embed_dim=128, num_heads=2, kdim=128, vdim=128, batch_first=True)    
        self.l2l_att = nn.MultiheadAttention(embed_dim=128, num_heads=2, kdim=128, vdim=128, batch_first=True)

        self.senet = SELayer(256, reduction=16)

    def forward(self, img_feats, lidar_feats, edge_index): 

        x_img = img_feats

        x_lidar = lidar_feats

        if self.use_attention == 'multihead':
            x_j_img, x_i_img = x_img[edge_index[0],:,:].view(-1,1,128), x_img[edge_index[1],:,:].view(-1,1,128)
            x_j_lidar, x_i_lidar = x_lidar[edge_index[0],:,:].view(-1,1,128), x_lidar[edge_index[1],:,:].view(-1,1,128)
           
            x_j_img_att, _ = self.c2c_att(query=x_i_img, key=x_j_img, value=x_j_img, need_weights=False)
            x_i_img_att, _ = self.c2c_att(query=x_j_img, key=x_i_img, value=x_i_img, need_weights=False)

            x_j_lidar_att, _ = self.l2l_att(query=x_i_lidar, key=x_j_lidar, value=x_j_lidar, need_weights=False)
            x_i_lidar_att, _ = self.l2l_att(query=x_j_lidar, key=x_i_lidar, value=x_i_lidar, need_weights=False)


            x_j_img, x_i_img = x_j_img_att.squeeze(1), x_i_img_att.squeeze(1)
            x_j_lidar, x_i_lidar = x_j_lidar_att.squeeze(1), x_i_lidar_att.squeeze(1)

            x_sens_j, x_sens_i = torch.cat([x_j_lidar, x_j_img], dim=1), torch.cat([x_i_lidar, x_i_img], dim=1)

            x_sens_j = torch.transpose(x_sens_j.unsqueeze(0), 1, 2)  # BxDxN

            x_sens_i = torch.transpose(x_sens_i.unsqueeze(0), 1, 2)  # BxDxM

            if self.lc_fused_type == 'cat':
                att_fusion_feats = torch.cat([x_sens_i, x_sens_j], dim=1)  
            elif self.lc_fused_type == 'add':

                att_fusion_feats = x_sens_i.add(x_sens_j)
            elif self.lc_fused_type == 'multipy':
                att_fusion_feats = batch_multiply(x_sens_i, x_sens_j)  # BxDxNxM
            elif self.lc_fused_type == 'diff':
                att_fusion_feats = batch_minus(x_sens_i, x_sens_j)  # BxDxNxM
            elif self.lc_fused_type == 'diff-abs':
                att_fusion_feats = batch_minus_abs(x_sens_i, x_sens_j)  # BxDxNxM

        elif self.use_attention == 'multihead-self':
            x_j_img, x_i_img = x_img[edge_index[0],:,:].view(-1,1,128), x_img[edge_index[1],:,:].view(-1,1,128)
            x_j_lidar, x_i_lidar = x_lidar[edge_index[0],:,:].view(-1,1,128), x_lidar[edge_index[1],:,:].view(-1,1,128)
           
            x_j_img_att, _ = self.c2c_att(query=x_i_img, key=x_j_img, value=x_j_img, need_weights=False)
            x_i_img_att, _ = self.c2c_att(query=x_j_img, key=x_i_img, value=x_i_img, need_weights=False)

            x_j_lidar_att, _ = self.l2l_att(query=x_i_lidar, key=x_j_lidar, value=x_j_lidar, need_weights=False)
            x_i_lidar_att, _ = self.l2l_att(query=x_j_lidar, key=x_i_lidar, value=x_i_lidar, need_weights=False)


            x_j_img, x_i_img = x_j_img_att.squeeze(1), x_i_img_att.squeeze(1)
            x_j_lidar, x_i_lidar = x_j_lidar_att.squeeze(1), x_i_lidar_att.squeeze(1)

            x_sens_j, x_sens_i = torch.cat([x_j_lidar, x_j_img], dim=1), torch.cat([x_i_lidar, x_i_img], dim=1)

            x_sens_j = torch.transpose(x_sens_j.unsqueeze(0), 1, 2)  # BxDxN

            x_sens_i = torch.transpose(x_sens_i.unsqueeze(0), 1, 2)  # BxDxM


            if self.lc_fused_type == 'cat':
                att_fusion_feats = torch.cat([x_sens_i, x_sens_j], dim=1)  
            elif self.lc_fused_type == 'add':
                att_fusion_feats = torch.add([x_sens_i, x_sens_j], dim=1)
            elif self.lc_fused_type == 'diff':
                att_fusion_feats = batch_minus(x_sens_i, x_sens_j)  # BxDxNxM
            elif self.lc_fused_type == 'diff-abs':
                att_fusion_feats = batch_minus_abs(x_sens_i, x_sens_j)  # BxDxNxM
        
        elif self.use_attention == 'senet':
            x_j_img, x_i_img = x_img[edge_index[0],:,:].view(-1,1,128), x_img[edge_index[1],:,:].view(-1,1,128)
            x_j_lidar, x_i_lidar = x_lidar[edge_index[0],:,:].view(-1,1,128), x_lidar[edge_index[1],:,:].view(-1,1,128)

            x_j_img, x_i_img = self.senet.forward(x_j_img, x_i_img)
            x_j_lidar, x_i_lidar = self.senet.forward(x_j_lidar, x_i_lidar)

            x_sens_j, x_sens_i = torch.cat([x_j_lidar, x_j_img], dim=1), torch.cat([x_i_lidar, x_i_img], dim=1)

            # concat diff diff-abs add
            # 
            if self.lc_fused_type == 'cat':
                att_fusion_feats = torch.cat([x_sens_i, x_sens_j], dim=1)  
        
        elif self.use_attention == 'senet-self':
            x_j_img, x_i_img = x_img[edge_index[0],:,:].view(-1,1,128), x_img[edge_index[1],:,:].view(-1,1,128)
            x_j_lidar, x_i_lidar = x_lidar[edge_index[0],:,:].view(-1,1,128), x_lidar[edge_index[1],:,:].view(-1,1,128)

            x_j_img, x_i_img = self.senet.forward(x_j_img, x_i_img)
            x_j_lidar, x_i_lidar = self.senet.forward(x_j_lidar, x_i_lidar)

            x_sens_j, x_sens_i = torch.cat([x_j_lidar, x_j_img], dim=1), torch.cat([x_i_lidar, x_i_img], dim=1)

            # concat diff diff-abs addx_i_img
            if self.lc_fused_type == 'cat':
                att_fusion_feats = torch.cat([x_sens_i, x_sens_j], dim=1)  

        elif self.use_attention == 'Wo_att':
            x_j_lidar, x_i_lidar = x_lidar[edge_index[0]], x_lidar[edge_index[1]]
            x_j_img, x_i_img = x_img[edge_index[0]], x_img[edge_index[1]]
            att_fusion_feats = torch.cat([x_i_img, x_i_lidar, x_j_img, x_j_lidar], dim=1)


        return att_fusion_feats

class Model_A(nn.Module):
    def __init__(self, input_dim1, hidden1_dims1, hidden2_dims1):
        super(Model_A, self).__init__()
        self.binaryclassifier = Binaryclassifer(
            input_dim1, hidden1_dims1, hidden2_dims1)
        
    def forward(self, track_feats, det_feats):
        track_feats = track_feats.unsqueeze(0).permute(0, 2, 1)
        det_feats = det_feats.unsqueeze(0).permute(0, 2, 1)  # b c n
        fusion_feats = batch_minus_abs(track_feats, det_feats)
        b, dim, trk, det = fusion_feats.size()
        inputs = fusion_feats.permute(0, 2, 3, 1)
        inputs = inputs.view(-1, inputs.size(-1))
        output = self.binaryclassifier(inputs)
        output = output.reshape(trk, det)
        return output.unsqueeze(0)

class Model_B(nn.Module):
    def __init__(self, input_dim1, hidden1_dims1, hidden2_dims1):
        super(Model_B, self).__init__()
        self.binaryclassifier = Binaryclassifer(
            input_dim1, hidden1_dims1, hidden2_dims1)
        self.senet = SELayer(256, reduction=16)
       
    def forward(self, track_feats, det_feats):
        x_j_img, x_j_lidar = det_feats[:, :128].view(-1, 1, 128), det_feats[:, 128:].view(-1, 1, 128)
        x_i_img, x_i_lidar = track_feats[:, :128].view(-1, 1, 128), track_feats[:, 128:].view(-1, 1, 128)
        x_j_img, x_j_lidar = self.senet.forward(x_j_img, x_j_lidar)
        x_i_img, x_i_lidar = self.senet.forward(x_i_img, x_i_lidar)
        fusion_i = torch.cat([x_i_img, x_i_lidar], 1).unsqueeze(0).permute(0, 2, 1)
        fusion_j = torch.cat([x_j_img, x_j_lidar], 1).unsqueeze(0).permute(0, 2, 1)
        fusion_feats = batch_minus_abs(fusion_i, fusion_j)
        b, dim, trk, det = fusion_feats.size()
        inputs = fusion_feats.permute(0, 2, 3, 1)
        inputs = inputs.view(-1, inputs.size(-1))
        output = self.binaryclassifier(inputs)
        output = output.reshape(trk, det)
        return output.unsqueeze(0)
        
    def forward(self, track_feats, det_feats):
        track_feats = track_feats.unsqueeze(0).permute(0, 2, 1)
        det_feats = det_feats.unsqueeze(0).permute(0, 2, 1)
        fusion_feats = batch_minus_abs(track_feats, det_feats)
        b, dim, trk, det = fusion_feats.size()
        inputs = fusion_feats.permute(0, 2, 3, 1)
        inputs = inputs.view(-1, inputs.size(-1))
        output = self.binaryclassifier(inputs)
        output = output.reshape(trk, det)
        return output.unsqueeze(0)

class Attention_model(nn.Module):