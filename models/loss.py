import torch
import torch.nn as nn
import torch.nn.functional as F


gw_name = 0

def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()



def depth_wta(p, depth_values):
    '''Winner take all.'''
    wta_index_map = torch.argmax(p, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_values, 1, wta_index_map).squeeze(1)
    return wta_depth_map


def info_entropy_loss(prob_volume, prob_volume_pre, mask):
    # prob_colume should be processed after SoftMax
    B,D,H,W = prob_volume.shape
    LSM = nn.LogSoftmax(dim=1)
    valid_points = torch.sum(mask, dim=[1,2])+1e-6
    entropy = -1*(torch.sum(torch.mul(prob_volume, LSM(prob_volume_pre)), dim=1)).squeeze(1)
    entropy_masked = torch.sum(torch.mul(mask, entropy), dim=[1,2])
    return torch.mean(entropy_masked / valid_points)


###-------------------------------------------------------------------------------------------------------------------

def geo_entropy_loss(prob_volume, depth_gt, mask, depth_value, geo_weight, return_prob_map=False):
    # from AA
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W
    ## masked geo weights 
    geo_weight = torch.mul(mask_true, geo_weight) 
    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy_image = torch.mul(geo_weight, masked_cross_entropy_image) #geo weighted masked CE image
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


def geo_loss(for_loss_cal, for_geo_cal, geo_mask_obj, **kwargs):
    ## unpacking
    inputs, depth_gt_ms, mask_ms = for_loss_cal
    proj_mats, src_depths = for_geo_cal
    op = kwargs.get("operation", None)
    depth_loss_weights = kwargs.get("dlossw", None)
    
    mean_geo_mask = 0
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, p_mats, src_gt, stage_key) in [(inputs[k],proj_mats[k],src_depths[k],k) for k in inputs.keys() if "stage" in k]:
        stage_idx = int(stage_key.replace("stage", "")) - 1
        depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        confidence = stage_inputs["photometric_confidence"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 1.0
        valid_pixel_num = torch.sum(mask, dim=[1,2]) + 1e-6
        
        ## calculate geo weights 
        geo_weights = geo_mask_obj.generate_geometric_weights(confidence, depth_est, p_mats, src_gt, stage_idx)
        geo_weights = geo_weights[0] ## unpacking tuple

        geo_weights_sum = torch.sum(torch.mul(mask, geo_weights), dim=[1,2])
        if stage_idx == 2: 
            mean_geo_mask += torch.mean(geo_weights_sum / valid_pixel_num)
        ## Focal Loss calculation with geo weights
        w_entro_loss, depth_entropy = geo_entropy_loss(prob_volume, depth_gt, mask, depth_values, geo_weights)
        w_entro_loss = w_entro_loss * entropy_weight
        ## Depth loss calcuation (smooth L1)
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += w_entro_loss

        if depth_loss_weights is not None:
            total_loss += depth_loss_weights[stage_idx] * w_entro_loss
        else:
            total_loss += w_entro_loss

    return total_loss, depth_loss, total_entropy, depth_entropy, mean_geo_mask

#### -------------- geometric loss for Blended dataset 

def geo_loss_bld(for_loss_cal, for_geo_cal, geo_mask_obj, depth_interval, **kwargs):
    ## unpacking
    inputs, depth_gt_ms, mask_ms = for_loss_cal
    proj_mats, src_depths = for_geo_cal
    op = kwargs.get("operation", None)
    depth_loss_weights = kwargs.get("dlossw", None)
    
    mean_geo_mask = 0
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, p_mats, src_gt, stage_key) in [(inputs[k],proj_mats[k],src_depths[k],k) for k in inputs.keys() if "stage" in k]:
        stage_idx = int(stage_key.replace("stage", "")) - 1
        depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        confidence = stage_inputs["photometric_confidence"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 1.0
        valid_pixel_num = torch.sum(mask, dim=[1,2]) + 1e-6
        
        ## calculate geo weights 
        geo_weights = geo_mask_obj.generate_geometric_weights(confidence, depth_est, p_mats, src_gt, stage_idx)
        geo_weights = geo_weights[0] ## unpacking tuple
        # print(f"geo_weights:{geo_weights.shape}")

        geo_weights_sum = torch.sum(torch.mul(mask, geo_weights), dim=[1,2])
        if stage_idx == 2: 
            mean_geo_mask += torch.mean(geo_weights_sum / valid_pixel_num)
        ## Focal Loss calculation with geo weights
        w_entro_loss, depth_entropy = geo_entropy_loss(prob_volume, depth_gt, mask, depth_values, geo_weights)
        w_entro_loss = w_entro_loss * entropy_weight
        ## Depth loss calcuation (smooth L1)
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        # total_entropy += w_entro_loss

        if depth_loss_weights is not None:
            total_loss += depth_loss_weights[stage_idx] * w_entro_loss
        else:
            total_loss += w_entro_loss

    abs_err = (depth_gt_ms['stage3'] - inputs["stage3"]["depth"]).abs()
    abs_err_scaled = abs_err /(depth_interval).unsqueeze(1).unsqueeze(1)
    mask = mask_ms["stage3"]
    mask = mask > 0.5
    # print(f"loss script: {depth_interval}")
    ## EPE calculation on scaled error
    epe_scaled = abs_err_scaled[mask].mean()
    less1_scaled = (abs_err_scaled[mask] < 1.).to(depth_gt_ms['stage3'].dtype).mean()
    less3_scaled = (abs_err_scaled[mask] < 3.).to(depth_gt_ms['stage3'].dtype).mean()
    bld_metric_scaled = (epe_scaled, less1_scaled, less3_scaled)
    ## EPE calculation on absolute error
    epe = abs_err[mask].mean()
    less1 = (abs_err[mask] < 1.).to(depth_gt_ms['stage3'].dtype).mean()
    less3 = (abs_err[mask] < 3.).to(depth_gt_ms['stage3'].dtype).mean()
    bld_metric = (epe, less1, less3)

    return total_loss, depth_loss, mean_geo_mask, bld_metric_scaled, bld_metric

