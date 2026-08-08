# -----------------
# DATASET ROOTS
# -----------------
cub_root = '/data/datasets/cub'
car_root = '/data/datasets/cars'
aircraft_root = '/data/datasets/fgvc-aircraft-2013b'
cifar_10_root = '/data/datasets/cifar10'
cifar_100_root = '/data/datasets/cifar100'
imagenet_root = '/data/datasets/imagenet'
herbarium_dataroot = '/data/datasets/herbarium_19'
pets_root = '/data/datasets/oxford_pets'

# -----------------
# PARTCO LABELS ROOTS ([hypartco] additive; used only by data/partco/*)
# Directory layout mirrors the partco repo config, e.g. <root>/<img>_label.png
# -----------------
cub_partco_root = '/data/projects/partco/datasets/partco_labels/cub/cub'
car_partco_root = '/data/projects/partco/datasets/partco_labels/cars/cars' 
aircraft_partco_root = '/data/projects/partco/datasets/partco_labels/aircraft/aircraft'
pets_partco_root = '/data/projects/partco/datasets/partco_labels/pets/'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/data/pretrained/dino_vitb16_pretrain.pth'
dinov2_pretrain_path = '/data/pretrained/dinov2_vitb14_reg4_pretrain.pth'
feature_extract_dir = 'osr_novel_categories/extracted_features_public_impl'

# -----------------
# OTHER PATHS
# -----------------
exp_root = '/data/projects/HypCD/dev_outputs'  # All logs and checkpoints will be saved here
