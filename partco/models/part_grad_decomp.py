"""[hab] GT-quadrant gradient decomposition of the unsup part loss (pure
diagnostic; ground truth enters ONLY statistics and gradient meters, never the
optimizer).

Answers the last open question of the A+B failure analysis: through WHICH
channel do wrong pseudo-labels poison the backbone --

    pull_T   true-positive attraction   (pseudo same & GT same)
    pull_F   FALSE-POSITIVE attraction  (pseudo same & GT diff)   <- gluing
             parts of different classes together
    push_FN  FALSE-NEGATIVE repulsion   (pseudo diff & GT same)   <- pushing
             parts of the same class apart
    push_R   remaining repulsion        (denominator flow through positives
             and true negatives)

Exact additive decomposition of the MAIN SupCon term of PartCo's
PatchUnConLoss (per-group re-normalize, no-diagonal denominator, confidence x
class-balance anchor weights, (T/bT) factor, /valid_parts -- all replicated
verbatim; the inert hard-negative term and the label-free consistency term are
outside the decomposition by design):

    L_i = -mean_{j in P_i} S_ij + logsumexp_{j != i} S_ij
    numerator  -> split by GT over j in P_i                    (exactly additive)
    denominator-> sum_j p_ij.detach() * S_ij with p = softmax  (d/dS_ij of the
                  logsumexp is p_ij, so quadrant-masked p.detach()*S recovers
                  the exact per-quadrant gradient component)

Completeness: grad(pull_T + pull_F + push_FN + push_R) == grad(L_main) up to
float error; ``total`` returns the recomposed main term so the caller can
verify the residual.
"""
import torch
import torch.nn.functional as F


def decompose_unsup_channels(patch_out, patch_label, class_logits, mask_lab,
                             gt_labels, epoch, temperature=0.07,
                             base_temperature=0.07, confidence_threshold=0.7,
                             dynamic_threshold=True, total_epochs=100,
                             class_balance_weight=0.3):
    """Returns (channels, stats): channels are differentiable scalars
    {pull_T, pull_F, push_FN, push_R, total}; stats are plain numbers."""
    B, device = patch_out.size(0), patch_out.device
    zero = patch_out.sum() * 0.0
    out = {k: zero.clone() for k in ('pull_T', 'pull_F', 'push_FN', 'push_R', 'total')}
    stats = {'n_TP': 0, 'n_FP': 0, 'n_FN': 0, 'n_groups': 0, 'n_anchor': 0}
    if (~mask_lab).sum() == 0:
        return out, stats

    threshold = confidence_threshold
    if dynamic_threshold and epoch is not None:
        threshold = 0.8 - 0.5 * min(1.0, epoch / total_epochs)

    plf = patch_label.reshape(B, -1)
    with torch.no_grad():
        probs = F.softmax(class_logits, dim=1)
        confidence, pseudo = torch.max(probs, dim=1)
        cmask = torch.zeros_like(mask_lab, dtype=torch.bool)
        cmask[~mask_lab] = confidence[~mask_lab] > threshold
        sel = torch.nonzero(cmask, as_tuple=False).flatten()
        if sel.numel() < 2:
            return out, stats
        # class-balance weights over the gated pseudo-labels (verbatim recipe)
        cw = None
        upl = pseudo[sel]
        if upl.numel() > 0:
            counts = torch.bincount(upl)
            cw = torch.ones(counts.size(0), device=device)
            cw = cw / counts.float().clamp_min(1)
            cw = cw / cw.sum() * len(counts)

    # group by part id (background 0 excluded), pooled means per (sample, part)
    part_ids = torch.unique(plf[sel])
    part_ids = part_ids[part_ids > 0]
    for pid in part_ids.tolist():
        feats, labs, gts, confs = [], [], [], []
        for i in sel.tolist():
            m = plf[i] == pid
            if m.sum() > 0:
                feats.append(patch_out[i][m].mean(dim=0))
                labs.append(pseudo[i]); gts.append(gt_labels[i]); confs.append(confidence[i])
        if len(feats) < 2:
            continue
        Ft = F.normalize(torch.stack(feats), dim=1)
        y = torch.stack(labs); g = torch.stack(gts); w0 = torch.stack(confs)
        uy, cnt = torch.unique(y, return_counts=True)
        if (cnt > 1).sum() == 0:
            continue
        with torch.no_grad():
            if cw is not None:
                sw = 1.0 + class_balance_weight * (cw[y] - 1.0)
                w = w0 * sw
            else:
                w = w0.clone()
            w = w / w.sum() * len(w)

        S = Ft @ Ft.t() / temperature
        N = S.size(0)
        eye = torch.eye(N, device=device)
        pos = (y.unsqueeze(0) == y.unsqueeze(1)).float() - eye
        pos = pos.clamp_min(0)
        gt_same = (g.unsqueeze(0) == g.unsqueeze(1)).float() - eye
        gt_same = gt_same.clamp_min(0)
        S = S - S.max(dim=1, keepdim=True).values.detach()
        p = torch.exp(S) * (1 - eye)
        p = p / (p.sum(dim=1, keepdim=True) + 1e-8)          # softmax over j != i
        p = p.detach()

        va = pos.sum(1) > 0
        if va.sum() == 0:
            continue
        wv = w[va]; wn = (wv / wv.sum())                     # anchor weights, sum 1
        inv_pos = 1.0 / pos.sum(1).clamp(min=1)

        def _agg(M):                                          # sum_i wn_i * sum_j M_ij
            return (wn * M[va].sum(dim=1)).sum()

        scale = (temperature / base_temperature)
        pull_T = scale * _agg(-(pos * gt_same) * inv_pos.unsqueeze(1) * S)
        pull_F = scale * _agg(-(pos * (1 - eye - gt_same).clamp_min(0)) * inv_pos.unsqueeze(1) * S)
        neg = (1 - eye) - pos                                 # pseudo-different
        push_FN = scale * _agg(p * (neg * gt_same) * S)
        push_R = scale * _agg(p * ((1 - eye) - neg * gt_same) * S)

        out['pull_T'] = out['pull_T'] + pull_T
        out['pull_F'] = out['pull_F'] + pull_F
        out['push_FN'] = out['push_FN'] + push_FN
        out['push_R'] = out['push_R'] + push_R
        stats['n_groups'] += 1
        stats['n_anchor'] += int(va.sum().item())
        with torch.no_grad():
            stats['n_TP'] += int((pos * gt_same).sum().item())
            stats['n_FP'] += int((pos * (1 - eye - gt_same).clamp_min(0)).sum().item())
            stats['n_FN'] += int((neg * gt_same).sum().item())

    if stats['n_groups'] > 0:
        for k in ('pull_T', 'pull_F', 'push_FN', 'push_R'):
            out[k] = out[k] / stats['n_groups']
    out['total'] = out['pull_T'] + out['pull_F'] + out['push_FN'] + out['push_R']
    return out, stats
