## Imports
import argparse
import os
import sys
import cv2
import numpy as np
from .geo_utils import *
import torch
import matplotlib.pyplot as plt

## Geometric weight generation methods
## Consistency based:
##      - direct_joint_inconsistency: value-based mask 
##      - geo_consistency_to_inconsistency: {0,1} binary mask converted to Normal, Average, and Inverse mask
##      - projected_pixel_displacement: Value-based mask
##      - projected_depth_difference: Value-based mask
## Inconsistency based: 
##      - joint_inconsistency_mask: {0,1} binary mask converted to Normal and Average weights
##      - displacement_inconsistency_mask:{0,1} binary mask converted to Normal and Average weights 
##      - difference_inconsistency_mask: {0,1} binary mask converted to Normal and Average weights 

class GeometricWeights:
    def __init__(self, args):
        incosistency_base_masks = ["joint_inconsistency_mask", "displacement_inconsistency_mask", "difference_inconsistency_mask"]
        self.mask_type = args.mask_type
        self.geo_mask_sum_th = args.geo_mask_sum_thresh
        self.photo_mask_th = args.photo_mask_thresh
        self.avg_weight_gap = args.avg_weight_gap
        if self.mask_type == "direct_joint_inconsistency":
            self.lambda_1 = args.joint_lambda_1
            self.lambda_2 = args.joint_lambda_2
            self.dist_max_th = [float(e) for e in args.dist_max_thresh.split(",") if e]
            self.dist_min_th = [float(e) for e in args.dist_min_thresh.split(",") if e]
            self.depth_min_th = [float(e) for e in args.relative_depth_diff_min_thresh.split(",") if e]
            self.depth_max_th = [float(e) for e in args.relative_depth_diff_max_thresh.split(",") if e]
        elif self.mask_type in ["geo_consistency_to_inconsistency"] + incosistency_base_masks:
            self.dist_th = [float(e) for e in args.dist_thresh.split(",") if e] 
            self.depth_min_th = [float(e) for e in args.relative_depth_diff_min_thresh.split(",") if e]
            self.cons2incon_type = args.cons2incon_type
        elif self.mask_type == "projected_pixel_displacement":
            self.dist_max_th = [float(e) for e in args.dist_max_thresh.split(",") if e]
            self.dist_min_th = [float(e) for e in args.dist_min_thresh.split(",") if e]
        elif self.mask_type == "projected_depth_difference":
            self.depth_min_th = [float(e) for e in args.relative_depth_diff_min_thresh.split(",") if e]
            self.depth_max_th = [float(e) for e in args.relative_depth_diff_max_thresh.split(",") if e]
        else:
            raise ValueError ('mask type not provided')

    def pixel_displacement_error(self, x2d_reprojected, y2d_reprojected, x_ref, y_ref):
        """check ||p_reproj-p_1|| L2 norm"""
        return torch.sqrt((x2d_reprojected - x_ref.to(device='cuda')) ** 2 + (y2d_reprojected - y_ref.to(device='cuda')) ** 2)

    def depth_estimate_error(self, depth_reprojected, depth_ref, relative_=True):
        """check |d_reproj-d_1| / d_1"""
        depth_diff = torch.abs(depth_reprojected - depth_ref)
        if relative_:
            relative_depth_diff = depth_diff / depth_ref
        return relative_depth_diff if relative_ else depth_diff

    def get_geo_inconsistent_mask(self, mask_sum, total_views):
        """Generate pixel inconsistency across given views"""
        # to be used as poduct with other loss
        sum_th = self.geo_mask_sum_th
        condlist = [mask_sum < sum_th, mask_sum >= sum_th]
        choicelist = [total_views - mask_sum, 0.0] 
        return torch.from_numpy(np.select(condlist, choicelist, 1.0))
        
    def get_normal_inconsistent_mask(self, mask_sum, total_views):
        """Generate pixel inconsistency across given views"""
        # to be used as poduct with other loss
        sum_th = self.geo_mask_sum_th
        condlist = [mask_sum < sum_th, mask_sum >= sum_th]
        choicelist = [total_views - mask_sum, 1.0] 
        return torch.from_numpy(np.select(condlist, choicelist, 1.0)) 

    def consistency_inverse_mask(self, mask_sum):
        """Inverse each pixel value i.e. 1/x"""
        # to be used as product
        sum_th = self.geo_mask_sum_th
        condlist = [mask_sum <= 0.0, (mask_sum > 0.0) & (mask_sum < sum_th), mask_sum >= sum_th]
        choicelist = [1.0, 1.0/mask_sum, 0.0] 
        return torch.from_numpy(np.select(condlist, choicelist, 1.0))

    def get_pixel_displacement_mask(self, dist, stage_idx):
        """Generate mask due to pixel displacement error"""
        min_th = self.dist_min_th[stage_idx]
        max_th = self.dist_max_th[stage_idx]
        condlist = [dist <= min_th, (dist <= max_th) & (dist > min_th), dist > max_th]
        choicelist = [dist, np.sqrt(dist), 1.0]
        return torch.from_numpy(np.select(condlist, choicelist, 1.0))

    def get_proj_depth_mask(self, proj_depth, stage_idx):
        """generate depth difference mask"""
        min_th = self.depth_min_th[stage_idx]
        max_th = self.depth_max_th[stage_idx]
        condlist = [proj_depth < min_th, proj_depth >= max_th, (proj_depth > min_th) & (proj_depth < max_th)]
        choicelist = [1.0, 1.0, proj_depth]
        return torch.from_numpy(np.select(condlist, choicelist, 1.0))

    # project the reference point cloud into the source view, then project back
    def reproject_with_depth(self, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
        width, height = depth_ref.shape[1], depth_ref.shape[0]
        ## step1. project reference pixels to the source view
        # reference view x, y
        x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
        x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])

        # reference 3D space
        xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref),
                            torch.vstack((x_ref.to(device='cuda'), 
                                          y_ref.to(device='cuda'), 
                                          torch.ones_like(x_ref, device=torch.device('cuda')))) * depth_ref.reshape([-1]))
        # source 3D space
        xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.linalg.inv(extrinsics_ref)),
                            torch.vstack((xyz_ref.to(device='cuda'), 
                                          torch.ones_like(x_ref, device=torch.device('cuda')))))[:3]
        # source view x, y
        K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
        xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

        ## reproject the source view points with source view depth estimation
        # find the depth estimation of the source view
        x_src = xy_src[0].reshape([height, width]).cpu().detach().numpy()
        y_src = xy_src[1].reshape([height, width]).cpu().detach().numpy()
        sampled_depth_src = cv2.remap(np.squeeze(depth_src.cpu().detach().numpy()), 
                                      x_src, 
                                      y_src, 
                                      interpolation=cv2.INTER_LINEAR)
        sampled_depth_src = torch.from_numpy(sampled_depth_src)

        # source 3D space
        # NOTE that we should use sampled source-view depth_here to project back
        xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src),
                               torch.vstack((xy_src, torch.ones_like(x_ref, device=torch.device('cuda')))) * sampled_depth_src.reshape([-1]).to(device='cuda'))
        # reference 3D space
        xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)),
                                    torch.vstack((xyz_src.to(device='cuda'), 
                                                  torch.ones_like(x_ref, device=torch.device('cuda')))))[:3]
        # source view x, y, depth
        depth_reprojected = xyz_reprojected[2].reshape([height, width])
        K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
        xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
        x_reprojected = xy_reprojected[0].reshape([height, width])
        y_reprojected = xy_reprojected[1].reshape([height, width])
        
        ## put back to cuda
        x_src = torch.from_numpy(x_src)
        y_src = torch.from_numpy(y_src)

        return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

    def geometric_consistency_mask(self, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, stage_idx):
        width, height = depth_ref.shape[1], depth_ref.shape[0]
        x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
        
        depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = self.reproject_with_depth(depth_ref,
                                                                                                          intrinsics_ref,
                                                                                                          extrinsics_ref,
                                                                                                          depth_src,
                                                                                                          intrinsics_src,
                                                                                                          extrinsics_src)
        if self.mask_type == "geo_consistency_to_inconsistency":
            dist = self.pixel_displacement_error(x2d_reprojected, 
                                                 y2d_reprojected, 
                                                 x_ref, y_ref)
            relative_depth_diff = self.depth_estimate_error(depth_reprojected, 
                                                            depth_ref, 
                                                            relative_=True)
            ## Apply threshold and take logical AND
            dist = dist < self.dist_th[stage_idx]
            relative_depth_diff = relative_depth_diff < self.depth_min_th[stage_idx]
            mask = torch.logical_and(dist, relative_depth_diff).to(torch.float32)
            
        elif self.mask_type== "projected_pixel_displacement":
            dist = self.pixel_displacement_error(x2d_reprojected, y2d_reprojected, x_ref, y_ref)
            mask = self.get_pixel_displacement_mask(dist.cpu().detach().numpy(), stage_idx)
            
        elif self.mask_type == "projected_depth_difference":
            relative_depth_diff = self.depth_estimate_error(depth_reprojected, depth_ref, relative_=True)
            mask = relative_depth_diff
            
        elif self.mask_type == "direct_joint_inconsistency":
            dist = self.pixel_displacement_error(x2d_reprojected, 
                                                 y2d_reprojected, 
                                                 x_ref, y_ref)
            dist_mask = self.get_pixel_displacement_mask(dist.cpu().detach().numpy(), stage_idx)
            relative_depth_diff = self.depth_estimate_error(depth_reprojected, 
                                                            depth_ref, 
                                                            relative_=True)
            return (dist_mask, relative_depth_diff)
        return mask


    def geometric_inconsistency_mask(self, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, stage_idx):
        """Performs inconsistency check across different source views for three different 
        methods and returns geometric inconsistency mask sum"""
        width, height = depth_ref.shape[1], depth_ref.shape[0]
        x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
        
        depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = self.reproject_with_depth(depth_ref,
                                                                                                          intrinsics_ref,
                                                                                                          extrinsics_ref,
                                                                                                          depth_src,
                                                                                                          intrinsics_src,
                                                                                                          extrinsics_src)
        if self.mask_type == "joint_inconsistency_mask":
            ## apply threshold on dist and depth then take logical OR
            dist = self.pixel_displacement_error(x2d_reprojected, 
                                                 y2d_reprojected, 
                                                 x_ref, y_ref)
            relative_depth_diff = self.depth_estimate_error(depth_reprojected, 
                                                            depth_ref, 
                                                            relative_=True)
            ## Apply threshold and take logical OR
            dist = dist > self.dist_th[stage_idx]
            relative_depth_diff = relative_depth_diff > self.depth_min_th[stage_idx]
            mask = torch.logical_or(dist, relative_depth_diff).to(torch.float32)
            
        elif self.mask_type == "displacement_inconsistency_mask":
            ## apply threshold on dist and convert to 0/1 values
            dist = self.pixel_displacement_error(x2d_reprojected, y2d_reprojected, x_ref, y_ref)
            mask = dist > self.dist_th[stage_idx]
            mask = mask.to(torch.float32)
            
        elif self.mask_type == "difference_inconsistency_mask":
            ## apply threshold on depth and convert to 0/1 values
            relative_depth_diff = self.depth_estimate_error(depth_reprojected, depth_ref, relative_=True)
            relative_depth_diff = relative_depth_diff > self.depth_min_th[stage_idx]
            mask = relative_depth_diff.to(torch.float32)
            
        return mask


    def generate_geometric_weights(self, confidence, depth_est, p_mats, src_gt, stage_idx):
        ## loop variables 
        batch_size, _, _ = depth_est.shape
        total_src_views = len(src_gt)
        # print(f"total src views inside loss: {total_src_views}\npmats shape: {p_mats.shape}")
        ## for two loss with different mask types
        if self.cons2incon_type == "normal_and_average": 
            batch_geo_mask_avg, batch_geo_mask_normal = [], []
        else: 
            batch_geo_mask = []
        
        ## process each elements of a batch one-by-one
        for batch_idx in range(batch_size): 
            ref_depth_est = depth_est[batch_idx, :,:]
            photo_mask = confidence[batch_idx, :,:] > self.photo_mask_th
            photo_mask = photo_mask.to(torch.float32) # convert boolean tensor into int
            ref_extrinsics = p_mats[batch_idx, 0, 0, :4, :4] ## p_mats shape: [batch, nviews, 2, 4, 4]
            ref_intrinsics= p_mats[batch_idx, 0, 1, :3, :3]

            ## init geo consistency/inconsistency mask sum 
            if self.mask_type == "direct_joint_inconsistency":
                dist_sum =  torch.zeros(ref_depth_est.shape).to(device='cuda')
                depth_sum = torch.zeros(ref_depth_est.shape).to(device='cuda')
                
            else:
                mask_sum = torch.zeros(ref_depth_est.shape).to(device='cuda')
            
            ## Iterate over each source view to generate mask for it   
            for src_idx in range(total_src_views): 
                src_depth_est = src_gt[src_idx][batch_idx, :,:]
                src_extrinsics = p_mats[batch_idx, src_idx + 1, 0, :4, :4]
                src_intrinsics = p_mats[batch_idx, src_idx + 1, 1, :3, :3] 
                
                if self.mask_type == "direct_joint_inconsistency":
                    dist_in, depth_in = self.geometric_consistency_mask(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                         src_depth_est, src_intrinsics, src_extrinsics,
                                                                         stage_idx)
                    dist_sum += dist_in.to(device='cuda')
                    depth_sum += depth_in
                elif self.mask_type in ["joint_inconsistency_mask", "displacement_inconsistency_mask", "difference_inconsistency_mask"]: 
                    mask_in = self.geometric_inconsistency_mask(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                               src_depth_est, src_intrinsics, src_extrinsics, 
                                                               stage_idx)
                    mask_sum += mask_in.to(device='cuda') 
                else:
                    mask_in = self.geometric_consistency_mask(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                               src_depth_est, src_intrinsics, src_extrinsics, 
                                                               stage_idx)
                    mask_sum += mask_in.to(device='cuda')
            
            ## Convert geo consistency mask sum to final geo weights
            if self.mask_type == 'geo_consistency_to_inconsistency':
                if self.cons2incon_type == "average": ## Average inconsistency across all views 
                    geo_inconsistent_mask = self.get_geo_inconsistent_mask(mask_sum.cpu().detach().numpy(), 
                                                                           total_src_views)
                    avg_weight_controller = total_src_views if self.avg_weight_gap=="0.1" else total_src_views/2
                    geo_inconsistent_mask = 1.0 + mask_sum/avg_weight_controller
                elif self.cons2incon_type == "normal": ## normal inconsistency across all views 
                    geo_inconsistent_mask = self.get_normal_inconsistent_mask(mask_sum.cpu().detach().numpy(), 
                                                                           total_src_views)
                    geo_inconsistent_mask = 1.0 + geo_inconsistent_mask
                elif self.cons2incon_type == "inverse": ## Inverse of Consistency --> 1/x
                    geo_inconsistent_mask = 1.0 + self.consistency_inverse_mask(mask_sum.cpu().detach().numpy())
                else: 
                    raise ValueError ('geo consistency to inconsistency mask type not provided')        
            elif self.mask_type == "projected_pixel_displacement":
                ## generate average displacement across all views for each pixel
                geo_inconsistent_mask = mask_sum/total_src_views
            elif self.mask_type == "projected_depth_difference":
                ## generate average error across all views for each pixel
                geo_inconsistent_mask = self.get_proj_depth_mask(mask_sum.cpu().detach().numpy()/total_src_views, 
                                                                 stage_idx)
            elif self.mask_type == "direct_joint_inconsistency":
                dist_inconsistent_mask = dist_sum/total_src_views
                depth_inconsistent_mask = self.get_proj_depth_mask(depth_sum.cpu().detach().numpy()/ total_src_views, 
                                                                   stage_idx)
                dist_loss = self.lambda_1*dist_inconsistent_mask
                depth_loss = self.lambda_2*depth_inconsistent_mask.to(device='cuda')
                geo_inconsistent_mask = dist_loss + depth_loss
            elif self.mask_type in ["joint_inconsistency_mask", "displacement_inconsistency_mask", "difference_inconsistency_mask"]:
                if self.cons2incon_type == "average": ## Average inconsistency across all views 
                    avg_weight_controller = total_src_views if self.avg_weight_gap=="0.1" else total_src_views/2  
                    geo_inconsistent_mask = 1.0 + mask_sum/avg_weight_controller
                elif self.cons2incon_type == "normal": ## normal inconsistency across all views 
                    geo_inconsistent_mask = 1.0 + mask_sum
                elif self.cons2incon_type == "normal_and_average": 
                    ## normal for L1 and average for CE loss 
                    avg_weight_controller = total_src_views if self.avg_weight_gap=="0.1" else total_src_views/2 
                    geo_inconsistent_mask_avg = 1.0 + mask_sum/avg_weight_controller
                    geo_inconsistent_mask_normal = 1.0 + mask_sum 
            
            ## deal with two masks at once and only one mask at once. 
            if self.cons2incon_type == "normal_and_average": 
                batch_geo_mask_avg.append(geo_inconsistent_mask_avg.to(device='cuda'))
                batch_geo_mask_normal.append(geo_inconsistent_mask_normal.to(device='cuda')) 
            else: 
                ## collect generated geo mask in list for batch manipulation
                batch_geo_mask.append(geo_inconsistent_mask.to(device='cuda'))
        if self.cons2incon_type =="normal_and_average":
            output = (torch.stack(batch_geo_mask_normal, dim=0), torch.stack(batch_geo_mask_avg, dim=0)) 
        else: 
            output = (torch.stack(batch_geo_mask, dim=0),)
        return output

