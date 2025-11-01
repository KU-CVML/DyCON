import os
import sys
import shutil
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from networks.net_factory_3d import net_factory_3d
from utils import ramps, metrics, losses, dycon_losses, test_3d_patch, monitor
from dataloaders.pancreas import Pancreas, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler 

# Argument parsing
parser = argparse.ArgumentParser(description="Training DyCON on PancreasCT Dataset")

parser.add_argument('--root_dir', type=str, default="../data/Pancreas", help='Path to PancreasCT dataset')
parser.add_argument('--patch_size', type=str, default=[112, 112, 96], help='Input image patch size')

parser.add_argument('--exp', type=str, default='PancreasCT', help='Experiment name')
parser.add_argument('--gpu_id', type=str, default=0, help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic training (0 or 1)')

parser.add_argument('--model', type=str, choices=['unet_3D', 'vnet'], default='unet_3D', help='Model architecture')
parser.add_argument('--in_ch', type=int, default=1, help='Number of input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
parser.add_argument('--feature_scaler', type=int, default=2, help='Feature scaling factor for contrastive loss')

parser.add_argument('--max_iterations', type=int, default=20000, help='Maximum number of training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='Total batch size per GPU')
parser.add_argument('--labeled_bs', type=int, default=4, help='Labeled batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')

parser.add_argument('--labelnum', type=int, default=12, help='Number of labeled samples per class')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for teacher model')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_type', type=str, default="mse", help='Consistency loss type')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='Ramp-up duration for consistency weight')

# === DyCon-specific Parameters === #
parser.add_argument('--gamma', type=float, default=2.0, help='Focusing parameter for hard positives/negatives in FeCL (γ)')
parser.add_argument('--beta_min', type=float, default=0.5, help='Minimum value for entropy weighting (β)')
parser.add_argument('--beta_max', type=float, default=5.0, help='Maximum value for entropy weighting (β)')
parser.add_argument('--s_beta', type=float, default=None, help='If provided, use this static beta for UnCLoss instead of adaptive beta.')
parser.add_argument('--temp', type=float, default=0.6, help='Temperature for contrastive softmax scaling (optimal: 0.6)')
parser.add_argument('--l_weight', type=float, default=1.0, help='Weight for supervised loss')
parser.add_argument('--u_weight', type=float, default=0.5, help='Weight for unsupervised loss')
parser.add_argument('--use_focal', type=int, default=1, help='Whether to use focal weighting (1 for True, 0 for False)')
parser.add_argument('--use_teacher_loss', type=int, default=1, help='Use teacher-based auxiliary loss (1 for True, 0 for False)')

args = parser.parse_args()

if args.s_beta is not None:
    beta_str = f"_beta{args.s_beta}"
else:
    beta_str = f"_beta{args.beta_min}-{args.beta_max}"

focal_str = "Focal" if bool(args.use_focal) else "NoFocal"
gamma_str = f"_gamma{args.gamma}" if bool(args.use_focal) else ""
teacher_str = "Teacher" if bool(args.use_teacher_loss) else "NoTeacher"

snapshot_path = (
    f"../models/{args.exp}/{args.model.upper()}_{args.labelnum}labels_"
    f"{args.consistency_type}{gamma_str}_{focal_str}_{teacher_str}_temp{args.temp}"
    f"{beta_str}_max_iterations{args.max_iterations}"
)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) # Only GPU `args.gpu_id` is visible

batch_size = args.batch_size 
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = True 
    cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = args.num_classes = 2
patch_size = args.patch_size = (112, 112, 96) 

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

from matplotlib import pyplot as plt

def plot_samples(image, mask, epoch):
    """Plot sample slices of the image/preds and mask"""
    # image: (C, H, W, D), mask: (H, W, D)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image[1][:, :, image.shape[-1]//2], cmap='gray') # access the class at index 1
    ax[1].imshow(mask[:, :, mask.shape[-1]//2], cmap='viridis')
    plt.savefig(f'../misc/train_preds/PancreasCT_sample_slice_{str(epoch)}.png')
    plt.close()


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        net = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes, scaler=args.feature_scaler)
        model = net.cuda() # Now 'cuda:0' refers to physical GPU 3
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # Model definition
    model = create_model()
    ema_model = create_model(ema=True)
    logging.info("Total params of model: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    # Read dataset
    db_train = Pancreas(base_dir=args.root_dir,
                        split='train', 
                        transform=T.Compose([
                        RandomCrop(patch_size),
                        RandomRotFlip(),
                        ToTensor(),
                    ]))
                
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, db_train.__len__()))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        
    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} Itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    uncl_criterion = dycon_losses.UnCLoss()
    fecl_criterion = dycon_losses.FeCLoss(device=f"cuda:0", temperature=args.temp, gamma=args.gamma, use_focal=bool(args.use_focal), rampup_epochs=1500)
    
    for epoch_num in iterator:

        if args.s_beta is not None:
            beta = args.s_beta
        else:
            beta = dycon_losses.adaptive_beta(epoch=epoch_num, total_epochs=max_epoch, max_beta=args.beta_max, min_beta=args.beta_min)

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise

            _, stud_logits, stud_features = model(volume_batch) 
            with torch.no_grad():
                _, ema_logits, ema_features = ema_model(ema_inputs)
           
            # Apply softmax for probability outputs
            stud_probs = F.softmax(stud_logits, dim=1)
            ema_probs = F.softmax(ema_logits, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            # Calculate the supervised loss
            loss_seg = F.cross_entropy(stud_logits[:labeled_bs], label_batch[:labeled_bs]) # 0.9020
            loss_seg_dice = losses.dice_loss(stud_probs[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1) # 0.9880
            
            B, C, _, _, _ = stud_features.shape
            stud_embedding = stud_features.view(B, C, -1) 
            stud_embedding = torch.transpose(stud_embedding, 1, 2) 
            stud_embedding = F.normalize(stud_embedding, dim=-1)  

            ema_embedding = ema_features.view(B, C, -1) 
            ema_embedding = torch.transpose(ema_embedding, 1, 2) 
            ema_embedding = F.normalize(ema_embedding, dim=-1) 
           
            # Mask contrastive
            # mask_con = F.avg_pool3d(label_batch.float(), kernel_size=args.feature_scaler*4, stride=args.feature_scaler*4) 
            mask_con = F.interpolate(label_batch.unsqueeze(1).float(), scale_factor=1/(args.feature_scaler * 4), mode='trilinear', align_corners=False).squeeze(1)  # torch.Size([8, 12, 12, 12])
            mask_con = (mask_con > 0.5).float()
            mask_con = mask_con.reshape(B, -1)
            mask_con = mask_con.unsqueeze(1) 

            # Plot sample images
            if iter_num % 200 == 0:
                # mask = mask_con[0].cpu().detach().numpy().reshape(14, 14, 10)
                # plot_samples(stud_features[0].cpu().detach().numpy(), mask, iter_num)
                path2save = os.path.join(snapshot_path, 'PancreasCT_similarity')
                os.makedirs(path2save, exist_ok=True)
                monitor.monitor_similarity_distributions(stud_embedding, mask_con, epoch=iter_num, path_prefix=path2save)

            # # Incorporate uncertainty mask into the contrastive loss
            # p_gs = dycon_losses.gambling_softmax(stud_logits) 
            # entropy = -torch.sum(p_gs * torch.log(p_gs + 1e-6), dim=1, keepdim=True) 
            # entropy = F.interpolate(entropy, scale_factor=1/(args.feature_scaler * 4), mode='trilinear', align_corners=False).squeeze(1)
            # gambling_uncertainty = entropy.view(B, -1) 

            teacher_feat = ema_embedding if args.use_teacher_loss else None
            f_loss = fecl_criterion(feat=stud_embedding[labeled_bs:],
                                    mask=mask_con[labeled_bs:], 
                                    teacher_feat=teacher_feat[labeled_bs:], # None,
                                    gambling_uncertainty=None, # gambling_uncertainty,
                                    epoch=epoch_num)
            u_loss = uncl_criterion(stud_logits, ema_logits, beta)
            consistency_loss = consistency_criterion(stud_probs[labeled_bs:], ema_probs[labeled_bs:]).mean()
            
            # Gather losses
            loss = args.l_weight * (loss_seg + loss_seg_dice) + consistency_weight * consistency_loss + args.u_weight * (f_loss + u_loss)

            # Check for NaN or Inf values
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf found in loss at iteration {iter_num}")
                continue

            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
             
            writer.add_scalar('info/loss', loss, iter_num)
            writer.add_scalar('info/f_loss', f_loss, iter_num)
            writer.add_scalar('info/u_loss', u_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_seg, iter_num)
            writer.add_scalar('info/loss_dice', loss_seg_dice, iter_num) 
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            
            del noise, stud_embedding, ema_logits, ema_features, ema_probs, mask_con

            # Batched Dice and HD95 metrics
            with torch.no_grad():
                outputs_bin = (stud_probs[:, 1, :, :, :] > 0.5).float()
                dice_score = metrics.compute_dice(outputs_bin, label_batch)
                H, W, D = stud_logits.shape[-3:]
                max_dist = np.linalg.norm([H, W, D])
                hausdorff_score = metrics.compute_hd95(outputs_bin, label_batch, max_dist)

            writer.add_scalar('train/Dice', dice_score.mean().item(), iter_num)
            writer.add_scalar('train/HD95', np.mean(hausdorff_score).item(), iter_num)
            
            logging.info(
                'Iteration %d : Loss : %03f, Loss_CE: %03f, Loss_Dice: %03f, UnCLoss: %03f, FeCLoss: %03f, mean_dice: %03f, mean_hd95: %03f' %
                (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item(), u_loss.item(), f_loss.item(), dice_score.mean().item(), np.mean(hausdorff_score).item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_3d_patch.var_all_case_Pancreas(model, args.root_dir, num_classes=args.num_classes, patch_size=patch_size, stride_xy=64, stride_z=64)
                if avg_metric > best_performance:
                    best_performance = round(avg_metric, 4)

                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format( iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/Dice', avg_metric, iter_num)
                writer.add_scalar('info/Best_dice', best_performance, iter_num)
                logging.info('Iteration %d : Dice: %03f Best_dice: %03f' % (iter_num, avg_metric, best_performance))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
            
    writer.close()
    print("Training Finished!")

