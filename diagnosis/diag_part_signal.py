#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diag_part_signal.py -- diagnose the pseudo-label signal passed from the main
branch to the part branch.

Location: <repos>/diagnosis/diag_part_signal.py (next to hab/ and partco/)

============================= CONTROLLED VARIABLES =============================
1. The script does not modify any file in hab/ or partco/. It reads each
   training entry point, uses AST rewriting to rename the final `train(...)`
   call to `__diag_train__(...)`, and then executes the script. Therefore, the
   seed, backbone, dataset, sampler, loader, heads, and optimizer are built
   entirely by the original code. This script then invokes the original train().
2. PatchSupConLoss.forward and PatchUnConLoss.forward are replaced with
   functions that return zero in both implementations. Thus the part-branch
   weight is zero: the branch is observed but produces no gradient. Because
   patch_projector receives no gradient, SGD skips it. The main-branch
   trajectory is the corresponding baseline without the part branch.
3. A forward hook on the classifier captures student_out at every training
   iteration and reconstructs the original quantity:
       class_logits = (student_out / 0.1).chunk(2)[0]
     - partco: student_out is the weight-normalized cosine-logit output of
       DINOHead (in [-1, 1]).
     - hab: student_out is the HypLinear output, i.e. points in the Poincare
       ball (norm <= 1/sqrt(c)).
   Both are divided by 0.1. This scale difference is the direct source of the
   sharpness difference measured by the script. A one-time runtime check
   verifies that the hooked tensor matches the tensor received by
   PatchUnConLoss.
4. Pair construction exactly matches PatchUnConLoss: samples within the same
   part ID form a positive pair when their pseudo-labels match. The primary
   statistics are collected before the confidence gate over all samples and
   pairs. Post-gate statistics are also reported to show what the gate removes
   and admits.

==================================== USAGE ====================================
  # Run each implementation separately. Use identical --seed and --stop_epoch.
  python diagnosis/diag_part_signal.py --impl partco --stop_epoch 31 -- \
      --dataset_name cub --seed 0 --weight_decay 5e-5 --memax_weight 2 --exp_name DIAG

  python diagnosis/diag_part_signal.py --impl hab --stop_epoch 31 -- \
      --dataset_name cub --seed 0 --weight_decay 5e-5 --memax_weight 1.0 \
      --model_name v2 --c 0.1 --cr 1.2 --hyper_temp_scale 0.4 --deterministic \
      --exp_name DIAG

  # Compare the two runs side by side.
  python diagnosis/diag_part_signal.py --compare \
      diagnosis/out/partco_DIAG/metrics.jsonl diagnosis/out/hab_DIAG/metrics.jsonl

Arguments after `--` are forwarded unchanged to the selected training script.
Do not pass --epochs: retaining the original default of 200 preserves the real
learning-rate and teacher-temperature schedules. Use --stop_epoch to limit the
training budget. PatchUnConLoss hard-codes total_epochs=100 for its dynamic
threshold, independently of args.epochs. This script reproduces the same rule:
thr = 0.8 - 0.5 * min(1, epoch/100).

=================================== METRICS ===================================
Cohorts:
  pre       all unlabeled samples before the gate (primary result)
  gate      samples actually passed to the loss by the original implementation
  lab       labeled samples (reference)
Label-level metrics:
  conf      mean pseudo-label confidence max(p), plus p10/p50/p90
  H/effK    softmax entropy and effective class count exp(H); lower is sharper
  margin    top-1 minus top-2 probability
  acc_H     Hungarian-matched accuracy over all accumulated (pseudo, GT) pairs
            in an epoch, split into all/old/new. New-class pseudo-labels are
            arbitrary cluster IDs and must be matched before comparison, as in
            the v2 evaluation protocol.
  dir_old   direct argmax == GT accuracy; meaningful only for old classes whose
            indices are aligned by supervised cross-entropy
  logit_*   student_out scale before division by 0.1: |z|max/std/L2/range
Pair-level metrics:
  pairs     all sample pairs within the same part ID
  pos%      fraction of pairs predicted as positive
  prec      fraction of predicted-positive pairs with matching GT labels
  base      random baseline: fraction of all pairs with matching GT labels
  lift      prec / base
  rec       fraction of GT-matched pairs predicted as positive; measures
            pseudo-label fragmentation
  pconf     positive-pair confidence: means of (ci+cj)/2 and min(ci,cj)

Note: hab's train() calls .train() only on student and patch_projector; it never
returns hyperbolic_projector or classifier to training mode. They therefore
remain in eval mode after the first test(). HypLinear and ToPoincare contain no
batch normalization or dropout, so this has no numerical effect. Consequently,
the script identifies training forwards with a loader-set pending flag instead
of module.training.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import unicodedata
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

ENTRY = {'hab': 'train/train_hab_v2.py', 'partco': 'train_partco_simgcd.py'}
LOGIT_TEMP = 0.1        # Original code: class_logits = student_out / 0.1
THR_TOTAL_EPOCHS = 100  # PatchUnConLoss.__init__ default; neither side overrides it


class _Stop(Exception):
    """Exit the original train() cleanly at stop_epoch."""


# ==========================================================================
# 1. Load the original entry point; rewrite only train() -> __diag_train__().
# ==========================================================================
class _RenameTrainCall(ast.NodeTransformer):
    hits = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'train':
            node.func = ast.Name(id='__diag_train__', ctx=ast.Load())
            _RenameTrainCall.hits += 1
        return node


def load_setup(repo_root, entry_rel, forwarded_argv):
    """Execute __main__ except its final train call; return (ns, args, kwargs)."""
    path = os.path.join(repo_root, entry_rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    _RenameTrainCall.hits = 0
    tree = _RenameTrainCall().visit(ast.parse(open(path, encoding='utf-8').read(), path))
    ast.fix_missing_locations(tree)
    if not _RenameTrainCall.hits:
        raise RuntimeError(f'No train(...) call found in {path}')

    captured = {}

    def __diag_train__(*a, **kw):
        captured['a'], captured['kw'] = a, kw

    cwd, syspath, argv = os.getcwd(), list(sys.path), sys.argv
    os.chdir(repo_root)                       # Both config.py files use relative paths.
    sys.path.insert(0, repo_root)
    sys.argv = [path] + list(forwarded_argv)
    try:
        ns = {'__name__': '__main__', '__file__': path, '__diag_train__': __diag_train__}
        exec(compile(tree, path, 'exec'), ns)
    finally:
        sys.argv, sys.path[:] = argv, syspath
        os.chdir(cwd)

    if 'a' not in captured:
        raise RuntimeError('The entry point did not reach train(...) (was --eval_only set?)')
    return ns, list(captured['a']), dict(captured['kw'])


# ==========================================================================
# 2. Probe
# ==========================================================================
class ProbeLoader:
    """Wrap train_loader to count epochs, retain batches, and mark pending hooks."""

    def __init__(self, loader, state, stop_epoch, on_epoch_end):
        self.loader, self.state = loader, state
        self.stop_epoch, self.on_epoch_end = stop_epoch, on_epoch_end

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.state['epoch'] += 1
        if self.state['epoch'] >= self.stop_epoch:
            raise _Stop
        for batch in self.loader:
            self.state['batch'] = batch
            self.state['pending'] = True      # This iteration has not been probed yet.
            yield batch
        self.state['pending'] = False
        self.on_epoch_end(self.state['epoch'])   # Finalize a normally completed epoch.


def find_train_loader(a, kw):
    """Find the sole drop_last=True DataLoader; return its container and key."""
    hits = [(a, i) for i, o in enumerate(a) if isinstance(o, DataLoader) and o.drop_last]
    hits += [(kw, k) for k, o in kw.items() if isinstance(o, DataLoader) and o.drop_last]
    if len(hits) != 1:
        raise RuntimeError(f'Could not identify train_loader: found {len(hits)} candidates')
    return hits[0]


def pick_logit_module(ns, override=None):
    """Return the module producing student_out and how to extract its output."""
    if override:
        return ns[override], 'tensor'
    if 'hyperbolic_classifier' in ns:      # hab: HypLinear -> points in the Poincare ball
        return ns['hyperbolic_classifier'], 'tensor'
    if 'projector' in ns:                  # partco: DINOHead -> (proj, logits)
        return ns['projector'], 'tuple1'
    raise RuntimeError('Classifier not found; specify its namespace name with --logits_module')


def dyn_threshold(epoch):
    """Reproduce the PatchUnConLoss dynamic threshold."""
    return 0.8 - 0.5 * min(1.0, epoch / THR_TOTAL_EPOCHS)


class Acc:
    """Accumulator for one epoch."""

    def __init__(self):
        self.c = defaultdict(float)   # Counts and sums.
        self.v = defaultdict(list)    # Vectors retained for percentiles.

    def add(self, k, x):
        self.c[k] += float(x)

    def vec(self, k, x):
        self.v[k].append(np.asarray(x, dtype=np.float32))

    def cat(self, k):
        return np.concatenate(self.v[k]) if self.v[k] else np.zeros(0, np.float32)


@torch.no_grad()
def probe(student_out, state, acc, cfg):
    """Accumulate label and pair statistics once per training iteration."""
    images, class_labels, patch_label, uq_idxs, mask_lab = state['batch']
    dev = student_out.device
    B = class_labels.shape[0]

    # ---- Match the original implementation exactly. ----
    logits_raw = student_out.detach().float().chunk(2)[0]     # View 0, before temperature.
    assert logits_raw.shape[0] == B, (logits_raw.shape, B)
    cls_logits = logits_raw / LOGIT_TEMP                      # Actual part-branch input.
    state['cls_logits'] = cls_logits
    probs = cls_logits.softmax(dim=1)
    conf, pseudo = probs.max(dim=1)

    gt = class_labels.to(dev)
    lab = mask_lab[:, 0].to(dev).bool()
    thr = dyn_threshold(state['epoch'])

    cohorts = {
        'pre':  ~lab,                      # All unlabeled samples before the gate.
        'gate': (~lab) & (conf > thr),     # Samples actually passed to the loss.
        'lab':  lab,                       # Labeled-sample reference.
    }

    # ---- Native student_out scale: the source of sharpness differences. ----
    acc.add('n_iter', 1)
    acc.add('thr_sum', thr)
    acc.add('logit_l2', logits_raw.norm(dim=1).mean().item())
    acc.add('logit_absmax', logits_raw.abs().max(dim=1).values.mean().item())
    acc.add('logit_std', logits_raw.std(dim=1).mean().item())
    acc.add('logit_range', (logits_raw.max(1).values - logits_raw.min(1).values).mean().item())

    top2 = probs.topk(2, dim=1).values
    ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=1)

    for name, m in cohorts.items():
        n = int(m.sum())
        acc.add(f'{name}/n', n)
        if n:
            acc.vec(f'{name}/conf', conf[m].cpu().numpy())
            acc.add(f'{name}/ent', ent[m].sum().item())
            acc.add(f'{name}/margin', (top2[m, 0] - top2[m, 1]).sum().item())
            acc.vec(f'{name}/pseudo', pseudo[m].cpu().numpy())
            acc.vec(f'{name}/gt', gt[m].cpu().numpy())

    # ---- Match PatchUnConLoss pair grouping, without applying the gate. ----
    pl = patch_label.reshape(B, -1).to(dev)
    parts = torch.unique(pl)
    parts = parts[parts > 0]                      # Match the original: exclude background 0.

    for name in ('pre', 'gate'):
        m = cohorts[name]
        if int(m.sum()) < 2:
            continue
        for p in parts.tolist():
            idx = ((pl == p).any(dim=1) & m).nonzero(as_tuple=True)[0]
            k = idx.numel()
            acc.add(f'{name}/inst', k)            # Number of (sample, part) instances.
            if k < 2:
                continue
            ps, g, c = pseudo[idx], gt[idx], conf[idx]
            tri = torch.triu(torch.ones(k, k, dtype=torch.bool, device=dev), 1)
            eqp = (ps[:, None] == ps[None, :]) & tri
            eqg = (g[:, None] == g[None, :]) & tri
            acc.add(f'{name}/pairs', tri.sum().item())
            acc.add(f'{name}/pos', eqp.sum().item())
            acc.add(f'{name}/pos_ok', (eqp & eqg).sum().item())
            acc.add(f'{name}/gt_same', eqg.sum().item())
            if bool(eqp.any()):
                acc.add(f'{name}/pconf_avg', (0.5 * (c[:, None] + c[None, :]))[eqp].sum().item())
                acc.add(f'{name}/pconf_min', torch.minimum(c[:, None], c[None, :])[eqp].sum().item())
            if cfg.per_part and name == 'pre':
                acc.add(f'part{p}/inst', k)
                acc.add(f'part{p}/pos', eqp.sum().item())
                acc.add(f'part{p}/pos_ok', (eqp & eqg).sum().item())


# ==========================================================================
# 3. Aggregation
# ==========================================================================
def hungarian_acc(y_true, y_pred, n_old):
    """Apply one global Hungarian match, then split old/new as in v2."""
    nan = float('nan')
    if y_true.size == 0:
        return nan, nan, nan
    D = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    np.add.at(w, (y_pred.astype(int), y_true.astype(int)), 1)
    r, c = linear_sum_assignment(w.max() - w)
    mp = dict(zip(r.tolist(), c.tolist()))
    hit = np.array([mp.get(int(p), -1) for p in y_pred]) == y_true
    old = y_true < n_old
    f = lambda m: float(hit[m].mean()) if m.sum() else nan
    return float(hit.mean()), f(old), f(~old)


def summarize(acc, epoch, cfg, n_old):
    """Convert one epoch's accumulator into one metrics record."""
    nan, c, out = float('nan'), acc.c, {'epoch': epoch}
    ni = max(c['n_iter'], 1)
    out['thr'] = c['thr_sum'] / ni
    for k in ('logit_l2', 'logit_absmax', 'logit_std', 'logit_range'):
        out[k] = c[k] / ni

    for name in ('pre', 'gate', 'lab'):
        n, d = c[f'{name}/n'], {'n': int(c[f'{name}/n'])}
        if n:
            cf = acc.cat(f'{name}/conf')
            d['conf'] = float(cf.mean())
            d['conf_p10'], d['conf_p50'], d['conf_p90'] = \
                [float(x) for x in np.percentile(cf, [10, 50, 90])]
            d['ent'] = c[f'{name}/ent'] / n
            d['effK'] = float(np.exp(d['ent']))
            d['margin'] = c[f'{name}/margin'] / n
            ps = acc.cat(f'{name}/pseudo').astype(int)
            gt = acc.cat(f'{name}/gt').astype(int)
            d['n_uniq_pseudo'] = int(np.unique(ps).size)
            d['top1_share'] = float(np.bincount(ps).max() / ps.size)
            d['acc_dir'] = float((ps == gt).mean())
            d['acc_dir_old'] = float((ps == gt)[gt < n_old].mean()) if (gt < n_old).any() else nan
            if name != 'lab':
                d['acc_H'], d['acc_H_old'], d['acc_H_new'] = hungarian_acc(gt, ps, n_old)
        if name != 'lab':
            pairs, pos, same = c[f'{name}/pairs'], c[f'{name}/pos'], c[f'{name}/gt_same']
            d['inst'], d['pairs'], d['pos'] = int(c[f'{name}/inst']), int(pairs), int(pos)
            d['pos_rate'] = pos / pairs if pairs else nan
            d['prec'] = c[f'{name}/pos_ok'] / pos if pos else nan
            d['base'] = same / pairs if pairs else nan
            d['lift'] = d['prec'] / d['base'] if d['base'] else nan
            d['rec'] = c[f'{name}/pos_ok'] / same if same else nan
            d['pconf_avg'] = c[f'{name}/pconf_avg'] / pos if pos else nan
            d['pconf_min'] = c[f'{name}/pconf_min'] / pos if pos else nan
        out[name] = d

    out['gate_pass'] = c['gate/n'] / c['pre/n'] if c['pre/n'] else nan
    if cfg.per_part:
        out['per_part'] = {
            k.split('/')[0]: {'inst': int(c[k.split("/")[0] + '/inst']),
                              'pos': int(c[k]),
                              'prec': c[k.split("/")[0] + '/pos_ok'] / c[k]}
            for k in list(c) if k.startswith('part') and k.endswith('/pos') and c[k]}
    return out


def fmt_row(r):
    p, g = r['pre'], r['gate']
    f = lambda x, n=3: ('%.*f' % (n, x)) if isinstance(x, float) and x == x else ' nan'
    return (
        "[E{ep:03d}] pre  n={n:<5d} conf {cf}({c10}/{c90}) H={H:.2f} effK={eK:>6.1f} "
        "acc_H {aa}/{ao}/{an} dir_old {ad}\n"
        "        pair inst={ins:<6d} pairs={pr:<9d} pos={po:>5.2f}% prec {pc} "
        "base {bs} lift {lf:>6.1f} rec {rc} pconf {cv}/{cm}\n"
        "        gate thr={thr} pass={gp:>5.1f}% prec {gpc} pos={gpo:>5.2f}% | "
        "logit |z|max {lm:.2f} std {ls:.2f} L2 {ll:.2f} range {lr:.2f}"
    ).format(
        ep=r['epoch'], n=p['n'], cf=f(p.get('conf', float('nan'))),
        c10=f(p.get('conf_p10', float('nan')), 2), c90=f(p.get('conf_p90', float('nan')), 2),
        H=p.get('ent', float('nan')), eK=p.get('effK', float('nan')),
        aa=f(p.get('acc_H', float('nan'))), ao=f(p.get('acc_H_old', float('nan'))),
        an=f(p.get('acc_H_new', float('nan'))), ad=f(p.get('acc_dir_old', float('nan'))),
        ins=p['inst'], pr=p['pairs'], po=100 * p['pos_rate'], pc=f(p['prec']),
        bs=f(p['base'], 4), lf=p['lift'], rc=f(p['rec']),
        cv=f(p['pconf_avg']), cm=f(p['pconf_min']),
        thr=f(r['thr'], 2), gp=100 * r['gate_pass'], gpc=f(g['prec']),
        gpo=100 * g['pos_rate'] if g['pairs'] else float('nan'),
        lm=r['logit_absmax'], ls=r['logit_std'], ll=r['logit_l2'], lr=r['logit_range'],
    )


# ==========================================================================
# 4. Main workflow
# ==========================================================================
def run(cfg, forwarded):
    here = os.path.dirname(os.path.abspath(__file__))
    root = cfg.repo_root or os.path.dirname(here)
    repo = os.path.join(root, cfg.repo_dir or cfg.impl)
    entry = cfg.entry or ENTRY[cfg.impl]

    print(f'== [{cfg.impl}] repo={repo}  entry={entry}', flush=True)
    ns, cargs, ckwargs = load_setup(repo, entry, forwarded)
    args = ns['args']
    n_old = args.num_labeled_classes
    state = {'epoch': -1, 'batch': None, 'pending': False}
    print(f'== Setup complete: num_labeled={n_old} num_unlabeled={args.num_unlabeled_classes} '
          f'seed={args.seed} epochs={args.epochs} warmup={args.warmup_teacher_temp_epochs}')

    # ---------- Set the part-branch weight to zero: observe without gradients. ----------
    def zero(self, *a, **kw):
        dev = next((t.device for t in a if torch.is_tensor(t)), torch.device('cpu'))
        return torch.zeros((), device=dev, requires_grad=True)

    def zero_unsup(self, *a, **kw):
        # One-time check: hooked tensor == class_logits received by PatchUnConLoss.
        if not state.get('verified'):
            state['verified'] = True
            ref, got = state.get('cls_logits'), (a[2] if len(a) > 2 else kw.get('class_logits'))
            ok = (torch.is_tensor(got) and torch.is_tensor(ref) and ref.shape == got.shape
                  and torch.allclose(ref, got.float(), atol=1e-4))
            print(f'== Consistency check: hooked class_logits and actual part-branch input '
                  f'{"match (OK)" if ok else "do not match; check --logits_module"}', flush=True)
        return zero(self, *a, **kw)

    for cls_name, fn in (('PatchSupConLoss', zero), ('PatchUnConLoss', zero_unsup)):
        if cls_name in ns:
            ns[cls_name].forward = fn
        else:
            print(f'!! Warning: {cls_name} is absent from the namespace; the part branch may not be zeroed')

    # ---------- Optionally disable evaluation/checkpoints for clean, fast diagnosis. ----------
    if not cfg.keep_eval and 'test' in ns:
        ns['test'] = lambda *a, **kw: (0.0, 0.0, 0.0)
    if not cfg.keep_ckpt:
        torch.save = lambda *a, **kw: None

    # ---------- Output ----------
    acc = {'a': Acc()}
    out_dir = os.path.join(here, 'out', f'{cfg.impl}_{cfg.tag}')
    os.makedirs(out_dir, exist_ok=True)
    fout = open(os.path.join(out_dir, 'metrics.jsonl'), 'w')
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump({'impl': cfg.impl, 'repo': repo, 'entry': entry, 'argv': forwarded,
                   'seed': args.seed, 'epochs': args.epochs, 'stop_epoch': cfg.stop_epoch,
                   'n_old': n_old, 'n_new': args.num_unlabeled_classes,
                   'warmup_teacher_temp_epochs': args.warmup_teacher_temp_epochs,
                   'part_weight': 0.0}, fh, indent=2, default=str)

    def on_epoch_end(ep):
        row = summarize(acc['a'], ep, cfg, n_old)
        fout.write(json.dumps(row) + '\n')
        fout.flush()
        print(fmt_row(row), flush=True)
        acc['a'] = Acc()

    # ---------- Register the hook ----------
    mod, how = pick_logit_module(ns, cfg.logits_module)
    print(f'== Hook target: {type(mod).__name__} (output mode: {how})')

    def hook(module, inp, out):
        if not state['pending']:               # Training forwards only, once per iteration.
            return
        state['pending'] = False
        probe(out[1] if how == 'tuple1' else out, state, acc['a'], cfg)

    h = mod.register_forward_hook(hook)

    # ---------- Replace the loader with the probe wrapper and call train(). ----------
    box, key = find_train_loader(cargs, ckwargs)
    box[key] = ProbeLoader(box[key], state, cfg.stop_epoch, on_epoch_end)

    t0 = time.time()
    try:
        ns['train'](*cargs, **ckwargs)
    except _Stop:
        print(f'== Reached --stop_epoch {cfg.stop_epoch}; stopping')
    finally:
        h.remove()
        fout.close()
    print(f'== Elapsed {time.time() - t0:.0f}s -> {out_dir}/metrics.jsonl')


# ==========================================================================
# 5. Comparison mode
# ==========================================================================
def _pad(s, w, right=True):
    """Pad according to display width; wide characters occupy two columns."""
    n = w - sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in s)
    return (s + ' ' * n) if right else (' ' * n + s)


def compare(paths):
    runs = []
    for p in paths:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        runs.append((os.path.basename(os.path.dirname(p)), {r['epoch']: r for r in rows}))

    keys = [
        ('sharp conf(pre)',  lambda r: r['pre'].get('conf')),
        ('sharp effK(pre)',  lambda r: r['pre'].get('effK')),
        ('sharp margin',     lambda r: r['pre'].get('margin')),
        ('logit |z|max',    lambda r: r['logit_absmax']),
        ('logit std',       lambda r: r['logit_std']),
        ('label acc_H all',  lambda r: r['pre'].get('acc_H')),
        ('label acc_H old',  lambda r: r['pre'].get('acc_H_old')),
        ('label acc_H new',  lambda r: r['pre'].get('acc_H_new')),
        ('label uniq clusters', lambda r: r['pre'].get('n_uniq_pseudo')),
        ('pair prec',       lambda r: r['pre']['prec']),
        ('pair base',       lambda r: r['pre']['base']),
        ('pair lift',       lambda r: r['pre']['lift']),
        ('pair pos%',       lambda r: 100 * r['pre']['pos_rate']),
        ('pair conf',       lambda r: r['pre']['pconf_avg']),
        ('gate pass%',      lambda r: 100 * r['gate_pass']),
        ('post-gate prec',  lambda r: r['gate']['prec']),
        ('post-gate conf',  lambda r: r['gate']['pconf_avg']),
    ]
    eps = sorted(set.intersection(*[set(d) for _, d in runs]))
    if not eps:
        print('The result files have no common epoch')
        return
    show = sorted({eps[0], eps[len(eps) // 2], eps[-1]})

    w = max(max(len(n) for n, _ in runs) + 2, 11)
    for ep in show:
        print(f'\n===== epoch {ep} ' + '=' * 44)
        print(_pad('', 18) + ''.join(_pad(n, w, False) for n, _ in runs))
        for label, fn in keys:
            cells = []
            for _, d in runs:
                try:
                    v = fn(d[ep])
                except (KeyError, TypeError):
                    v = None
                ok = isinstance(v, (int, float)) and v == v
                cells.append(_pad(('%.4g' % v) if ok else '-', w, False))
            print(_pad(label, 18) + ''.join(cells))


def main():
    ap = argparse.ArgumentParser(
        description='Diagnose pseudo-label and pair signals passed to the part branch (weight zero)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--impl', choices=['hab', 'partco'], help='implementation to run')
    ap.add_argument('--stop_epoch', type=int, default=31,
                    help='stop before this epoch (default: 31, just after the 30-epoch warmup)')
    ap.add_argument('--repo_root', default=None, help='parent of hab/ and partco/; default: script parent')
    ap.add_argument('--repo_dir', default=None, help='repository directory; default: same as --impl')
    ap.add_argument('--entry', default=None, help='override the relative training entry-point path')
    ap.add_argument('--logits_module', default=None, help='override the classifier variable name')
    ap.add_argument('--tag', default='DIAG', help='output-directory suffix')
    ap.add_argument('--per_part', action='store_true', help='also report pair metrics for each part ID')
    ap.add_argument('--keep_eval', action='store_true', help='retain the original test() evaluation')
    ap.add_argument('--keep_ckpt', action='store_true', help='retain torch.save checkpoint writes')
    ap.add_argument('--compare', nargs='+', metavar='metrics.jsonl', help='comparison mode')
    cfg, forwarded = ap.parse_known_args()
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]

    if cfg.compare:
        return compare(cfg.compare)
    if not cfg.impl:
        ap.error('specify --impl {hab,partco} or --compare')
    run(cfg, forwarded)


if __name__ == '__main__':
    main()
