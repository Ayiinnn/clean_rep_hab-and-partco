import argparse
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root

from models.model import (
    DINOHead, 
    Patch_Projection,
    DistillLoss, 
    ContrastiveLearningViewGenerator, 
    get_params_groups
)
from models.partco_loss import PatchSupConLoss
from clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,is_code=False):#, smoothing=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist=-dist/(dist.sum(dim=1)+1e-10)
        else:
            dist = -torch.cdist(anchor_feature, contrast_feature)

        anchor_dot_contrast = torch.div(dist, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def info_nce_logits(features, confusion_factor, args, is_code=False):
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    if is_code:
        dist=torch.cdist(features, features,p=2)
        similarity_matrix =-dist/(dist.sum(dim=1)+1e-10)
    else:
        similarity_matrix=-torch.cdist(features, features)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs= confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs= confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / args.temperature
    return logits, labels, log_confs

def extract_labeled_protos(model, projector, train_loader, args):
    model.eval()
    projector.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids=np.array([])
    mask_cls=np.array([])
    metrics=dict()

    for batch_idx, (images, label, _, _, uq_idx, mask_lab_) in enumerate(tqdm(train_loader)):

        images = images[0].cuda()
        label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        feats, _ = projector(feats)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        ids=np.append(ids, uq_idx.cpu().numpy())
        mask = np.append(mask, mask_lab_.cpu().bool().numpy())
        mask_cls = np.append(mask_cls,np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
    mask = mask.astype(bool)
    mask_cls = mask_cls.astype(bool)
    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = targets[mask]  # Get labelled targets
    u_targets = targets[~mask]  # Get unlabelled targets
    n_samples =len(targets)

    if args.unbalanced: cluster_size=None
    else: cluster_size=math.ceil(n_samples /(args.num_labeled_classes + args.num_unlabeled_classes))
    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeanssem.labels_
    mask_cls=mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls,
                                                    eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    prototype_higher=[]
    prototypes = kmeanssem.cluster_centers_
    prototype_higher.append(prototypes.clone())
    n_labeled=args.num_labeled_classes
    n_novel= args.num_unlabeled_classes
    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes,:]
    preds_higher=[]

    preds_higher.append(all_preds.clone())
    print('Hierarchy clustering')
    mask_known=(all_preds<args.num_labeled_classes).cpu().numpy()
    l_feats = all_feats[mask_known]  # Get labelled set
    u_feats = all_feats[~mask_known]
    l_feats, u_feats= (torch.from_numpy(x).to(device) for  x in (l_feats, u_feats))

    while n_labeled>1:
        n_labeled=max(int(n_labeled/2),1)
        n_novel=max(int(n_novel/2),1)

        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0, n_init=10).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).to(device)
        level_l_targets=preds_labels[all_preds[mask_known]]
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil( n_samples / (n_labeled+n_novel))
        kmeans_higher =SemiSupKMeans(k=n_labeled+n_novel, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats, l_feats, level_l_targets)
        preds_level = kmeans_higher.labels_
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.to(device).clone())

    return ids,all_preds, prototype_higher,preds_higher, metrics

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity,smoothing = 0.5):
        target_smooth = F.one_hot(target,input.size(1)).float()*(1-smoothing) +smoothing*similarity#F.one_hot(similarity,input.size(1)).float()#s1/input.size(0)#coef# / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)


def train(model, projector, patch_projector, train_loader, test_loader, unlabelled_train_loader, merge_train_loader, args):
    params_groups = get_params_groups(model)
    params_groups += get_params_groups(projector)
    params_groups += get_params_groups(patch_projector)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    # # inductive
    best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0
    unsupervised_smoothing = args.unsupervised_smoothing
    prototype_extraction_interval=args.prototype_extraction_interval
    Distance=args.distance
    strategy=args.strategy
    cluster_momentum=args.cluster_momentum

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        with torch.no_grad():
            if epoch%prototype_extraction_interval==0:
                uq_index, all_preds, cluster_protos_list, preds_ind_list,  metrics= extract_labeled_protos(model, projector,  merge_train_loader, args=args)
                for i in range(len(preds_ind_list)):
                    preds_ind_list[i] = preds_ind_list[i].to(device).long()
                    cluster_protos_list[i] = cluster_protos_list[i].to(device)
                    if Distance == 'cosine':
                        cluster_protos_list[i]=cluster_protos_list[i]/torch.norm(cluster_protos_list[i],dim=1).unsqueeze(1)

                cluster_distances_list=[]
                cluster_radius_list=[]
                for i in range(len(preds_ind_list)):
                    if Distance=='euclidean':
                        cluster_distances = torch.cdist(cluster_protos_list[i], cluster_protos_list[i])
                    else:
                        cluster_distances = torch.matmul(cluster_protos_list[i], cluster_protos_list[i].T)

                    cluster_distances_list.append(cluster_distances.clone())
                    cluster_radius = \
                    (cluster_distances + torch.eye(cluster_distances.shape[0]).to(device) * cluster_distances.max()).min(dim=1)[0] / 2
                    cluster_radius_list.append(cluster_radius.clone())

        model.train()
        projector.train()
        patch_projector.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, patch_label_1, patch_label_2, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            patch_label_1 = patch_label_1.cuda(non_blocking=True)
            patch_label_2 = patch_label_2.cuda(non_blocking=True)
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            # with torch.cuda.amp.autocast(fp16_scaler is not None):
            with torch.amp.autocast('cuda', enabled=fp16_scaler is not None):
                features = model(images)
                # pass features to projector
                student_proj, student_out = projector(features)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                features_1, features_2 = student_proj.detach().chunk(2)
                all_features = torch.cat([features_1, features_2], dim=0)

                # represent learning, unsup
                with torch.no_grad():
                    confusion_factor=0
                    if Distance == 'euclidean':
                        pair_dist = torch.cdist(all_features, all_features)
                    else:
                        normalized_feats = all_features/torch.norm(all_features.unsqueeze(1))
                        pair_dist = torch.matmul(normalized_feats, normalized_feats.T)

                    n_labeled = args.num_labeled_classes
                    n_unlabeled = args.num_unlabeled_classes

                    for i in range(len(preds_ind_list)):
                        cluster_labels=(preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]).clone()
                        cluster_indexer = F.one_hot(cluster_labels.long(), n_labeled +n_unlabeled ).float().T
                        n_labeled = max(int(n_labeled / 2), 1)
                        n_unlabeled = max(int(n_unlabeled / 2), 1)

                        cluster_indexer = torch.cat([cluster_indexer,cluster_indexer],dim=1)
                        n_samples = torch.sum(cluster_indexer, dim=1).unsqueeze(1)
                        n_samples[n_samples == 0] = 1

                        if Distance=='euclidean':
                            distance = torch.cdist(all_features, cluster_protos_list[i].float())
                        else:
                            normalized_feats = all_features / torch.norm(all_features.unsqueeze(1))
                            distance = torch.matmul(normalized_feats, cluster_protos_list[i].float().T)

                        cluster_radius_list[i]= (cluster_indexer*distance.T).sum(dim=1)/n_samples.squeeze()\
                                                *(1-cluster_momentum)+cluster_radius_list[i]* cluster_momentum

                        cluster_labels = torch.cat([cluster_labels,cluster_labels])
                        if Distance=='euclidean':
                            if strategy=='zero_one':
                                confusion_factor+=(pair_dist>2*cluster_radius_list[i][cluster_labels]).float()/2 ** i
                            elif strategy=='pair_dist':
                                confusion_factor += pair_dist / 2 ** i
                            elif strategy=='pair_cluster':
                                confusion_factor += distance[:,cluster_labels] / 2 ** i
                            else:
                                pass

                        else:
                            if strategy=='zero_one':
                                confusion_factor += (pair_dist< distance[:,cluster_labels]/2).float()/ 2 ** i
                            elif strategy == 'pair_dist':
                                confusion_factor += -pair_dist / 2 ** i
                            elif strategy=='pair_cluster':
                                confusion_factor += -distance[:,cluster_labels] / 2 ** i

                confusion_factor = (confusion_factor - confusion_factor.min()) / (confusion_factor.max() - confusion_factor.min() + 0.0000001)
                confusion_factor = confusion_factor / confusion_factor.sum(dim=1)
                torch.cuda.empty_cache()

                torch.cuda.empty_cache()
                contrastive_logits, contrastive_labels, similarity = info_nce_logits(features=student_proj, confusion_factor=confusion_factor, args=args)
                contrastive_loss = LabelSmoothingLoss()(contrastive_logits, contrastive_labels, similarity, unsupervised_smoothing)

                # representation learning, sup
                sup_student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                sup_student_proj = torch.nn.functional.normalize(sup_student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(sup_student_proj, labels=sup_con_labels)

                # representation learning, semi-sup
                f1n, f2n = student_proj.chunk(2)
                semisup_con_feats = torch.cat([f1n.unsqueeze(1), f2n.unsqueeze(1)], dim=1)
                semisup_con_feats = torch.nn.functional.normalize(semisup_con_feats, dim=-1)
                dimension=semisup_con_feats.shape[-1]
                for i in range(len(preds_ind_list)):
                    sup_con_loss += SupConLoss()(
                        semisup_con_feats[:, :, :int(dimension/2**(i+1))],
                        labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]
                    ) / 2**(i+1)

                # patch representation learning
                patch_features = model.forward_features(images.chunk(2)[0])['x_norm_patchtokens'] # [B. 256, 768]
                patch_out = patch_projector(patch_features) # [B, 256, 128]
                patch_out = torch.nn.functional.normalize(patch_out, dim=-1)
                patch_sup_con_loss = PatchSupConLoss()(patch_out[mask_lab], patch_label_1[mask_lab], sup_con_labels)

                pstr = ''
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'patch_sup_con_loss: {patch_sup_con_loss.item():.4f} '

                # Loss calculation
                loss = 0
                loss += (1 - args.sup_weight) * contrastive_loss + (args.sup_weight * sup_con_loss * 0.5)
                loss += (args.sup_weight * patch_sup_con_loss)

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        with torch.no_grad():
            all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)


        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if old_acc > best_test_acc_lab:
            
            args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc:.4f}...')
            args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            
            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            
            # inductive
            best_test_acc_lab = old_acc
            # transductive            
            best_train_acc_lab = old_acc
            best_train_acc_ubl = new_acc
            best_train_acc_all = all_acc
        
        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


def test_kmeans(model, test_loader, epoch, save_name, args, projection_head=None, Use_GPU=False):

    model.eval()
    all_feats = []

    targets = np.array([])
    mask = np.array([])

    # First extract all features
    for batch_idx, (images, label, _, _, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask.astype(bool)
    all_feats = np.concatenate(all_feats)
    # -----------------------
    # EVALUATE
    # -----------------------
    kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
        all_feats
    )
    preds = kmeanss.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)


    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--seed', default=1, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    
    # selex hyperparameters
    parser.add_argument('--unsupervised_smoothing', type=float, default=1)
    parser.add_argument('--prototype_extraction_interval', default=1, type=int)
    parser.add_argument('--distance', type=str, default='euclidean', help='options: euclidean, cosine')
    parser.add_argument('--strategy', type=str, default='zero_one')
    parser.add_argument('--cluster_momentum', type=float, default=1)
    parser.add_argument('--unbalanced', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0) # for selex's infonce


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    print(f'Using seed {args.seed} for model training')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    # backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    backbone.to(device)
    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, contrastive_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=[train_transform, contrastive_transform], 
        n_views=args.n_views
    )
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    merge_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projector.to(device)

    # ----------------------
    # PATCH PROJECTION HEAD
    # ----------------------
    patch_projector = nn.Sequential(
        nn.LayerNorm(768),
        nn.Dropout(0.1),
        Patch_Projection(),
    ).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    train(backbone, projector, patch_projector, train_loader, None, test_loader_unlabelled, merge_train_loader, args)
