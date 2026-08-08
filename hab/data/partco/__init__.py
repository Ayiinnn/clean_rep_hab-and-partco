"""PartCo data layer, ported verbatim from the (stably running) partco repo.

Self-contained subpackage so the mainline HypCD ``data/`` package stays
untouched: every dataset here additionally loads the PartCo part-level
correspondence label (a small PNG whose pixel values are part ids on the
ViT patch grid; 0 = background) and returns 4-tuples
``(img, target, patch_label, uq_idx)``.

Port deltas vs. the partco repo (nothing else was changed):
  1. imports rewired ``data.*`` -> ``data.partco.*`` (config imports keep the
     partco label roots, which are appended additively to config.py);
  2. a ``random_hflip`` switch (default True = original behavior) on the
     datasets that flip img+label inside ``__getitem__``.  The trainer turns
     it off ONLY on the two evaluation datasets: the original PartCo code
     random-flips test images too, which would break the determinism / best-
     model-selection contract of the *_det_ab baselines.  Training-time
     behavior is byte-identical to the partco repo.

Scope: the fine-grained datasets used with part labels in this project
(cub / scars / aircraft / pets).  The remaining partco datasets can be
ported the same way if ever needed.
"""
