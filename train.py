import argparse
import os
import sys
import time
import gc
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import torch.distributed as dist

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
local_rank = int(os.environ["LOCAL_RANK"])

## Geometric mask types
mask_types = ['geo_consistency_to_inconsistency', 'projected_pixel_displacement', 'projected_depth_difference' 'direct_joint_inconsistency']

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='official Implementation of GCMVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--online_augmentation', action='store_true', help="use online data augmentation on Ref image")
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')

parser.add_argument('--nviews', type=int, default=5, help='total number of views')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=local_rank)

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

## Geometric Mask Specific arguments
parser.add_argument('--mask_type', type=str, default=mask_types[0], help='mask generation method')
parser.add_argument('--operation', type=str, default="product", help='[product, sum]')
parser.add_argument('--joint_lambda_1', type=float, default=1.0, help='for joining dist and depth diff')
parser.add_argument('--joint_lambda_2', type=float, default=1.0, help='for joining dist and depth diff')
parser.add_argument('--dist_thresh', type=str, default="1,1,1", help='geo_const2inconsist dist thresh for each stage')
parser.add_argument('--dist_max_thresh', type=str, default="600,600,600", help='max pixel displacement thresh')
parser.add_argument('--cons2incon_type', type=str, default="average", help=' geo_cons2inconsis mask type [average, inverse]')
parser.add_argument('--avg_weight_gap', type=str, default="0.1", help='Avg weight gap b/w each inconsistency')
parser.add_argument('--dist_min_thresh', type=str, default="10,10,10", help='min pixel displacement thresh')
parser.add_argument('--relative_depth_diff_min_thresh', type=str, default="0.01,0.01,0.01", help='Min relative depth diff thresh for each stage')
parser.add_argument('--relative_depth_diff_max_thresh', type=str, default="0.6,0.6,0.6", help='Max relative depth diff thresh')
parser.add_argument('--geo_mask_sum_thresh', type=float, default=5, help='geo_mask_sum_thresh')
parser.add_argument('--photo_mask_thresh', type=float, default=0.9, help='photo_mask_thresh')


num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


# main function taken from TransMVS paper
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args, geo_obj):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        if is_distributed:
            TrainImgLoader.sampler.set_epoch(epoch_idx)
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args, geo_obj)
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    # save_images(logger, 'train', image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, entropy loss = {:.3f}, geo weights = {:3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss,
                           scalar_outputs['depth_loss'],
                           scalar_outputs['entropy_loss'],
                           scalar_outputs['geo_weights'],
                           time.time() - start_time))
                del scalar_outputs, image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args, geo_obj)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        # save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, entropy loss = {:.3f}, geo weights = {:3f}, time = {:3f}".format(epoch_idx, args.epochs,
                                    batch_idx,
                                    len(TestImgLoader), loss,
                                    scalar_outputs["depth_loss"],
                                    scalar_outputs['entropy_loss'],
                                    scalar_outputs['geo_weights'],
                                    time.time() - start_time))
                        avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, model_loss, TestImgLoader, args, geo_obj):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args, geo_obj)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args, geo_obj):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)

    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    src_depths = sample_cuda["src_depths"]
    
    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]
    try:
        ## extract training related items from sample and reduce to N-views value
        imgs_t = sample_cuda["imgs"][:, :args.nviews, :, :, :]
        proj_t = {k:v[:, :args.nviews, :, :, :] for (k,v) in sample_cuda["proj_matrices"].items()} 
        
        outputs = model(imgs_t, proj_t, sample_cuda["depth_values"])
        depth_est = outputs["depth"]
        
        for_loss_cal = (outputs, depth_gt_ms, mask_ms)
        for_geo_cal = (sample_cuda["proj_matrices"], src_depths)

        loss, depth_loss, entropy_loss, depth_entropy, geo_w = model_loss(for_loss_cal,
                                                                        for_geo_cal,
                                                                        geo_obj,
                                                                        dlossw=[float(e) for e in args.dlossw.split(",") if e],
                                                                        operation=args.operation)

        if np.isnan(loss.item()):
            raise NanError

        loss.backward()
        optimizer.step()

    except NanError:
        print(f'nan error occur!!')
        gc.collect()
        torch.cuda.empty_cache()

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "entropy_loss": entropy_loss,
                    #   "reproj_loss": reproj_loss,
                      "geo_weights": geo_w,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args, geo_obj):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    src_depths = sample_cuda["src_depths"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    
    for_loss_cal = (outputs, depth_gt_ms, mask_ms)
    for_geo_cal = (sample_cuda["proj_matrices"], src_depths)

    loss, depth_loss, entropy_loss, depth_entropy, geo_w = model_loss(for_loss_cal, 
                                                                                for_geo_cal, 
                                                                                geo_obj,
                                                                                dlossw=[float(e) for e in args.dlossw.split(",") if e],
                                                                                operation=args.operation)

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "entropy_loss": entropy_loss,
                    #   "reproj_loss": reproj_loss,
                      "geo_weights": geo_w,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                      "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [2.0, 4.0]),
                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [4.0, 8.0]),
                      "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [8.0, 14.0]),
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
                    }

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_entropy - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    # device = torch.device(args.device)
    device = torch.device(args.local_rank)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = GCMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method)
    model.to(device)
    ##LOSS: geo_loss
    model_loss = geo_loss 
    geo_obj = GeometricWeights(args)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr, 
                            betas=(0.9, 0.999), 
                            weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 
                                args.nviews, args.numdepth, args.interval_scale,
                                args.geo_mask_sum_thresh)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 
                                args.nviews, args.numdepth, args.interval_scale,
                                args.geo_mask_sum_thresh)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=2,drop_last=True, pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=2, drop_last=False, pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args, geo_obj)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args, geo_obj)
    else:
        raise NotImplementedError
