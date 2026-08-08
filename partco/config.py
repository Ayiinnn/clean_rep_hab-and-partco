# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = 'generic_datasets'
cifar_100_root = 'generic_datasets'
cub_root = '/data/datasets/cub'
aircraft_root = 'datasets/fgvc-aircraft-2013b'
car_root = 'datasets/cars'
herbarium_dataroot = 'datasets/herbarium_19'
imagenet_root = 'datasets/ImageNet'
pets_root = 'datasets/pets/'

# -----------------
# PARTCO LABELS ROOTS
# -----------------
cifar_10_partco_root = 'datasets/partco_labels/cifar10'
cifar_100_partco_root = 'datasets/partco_labels/cifar100'
cub_partco_root = '/data/projects/partco/datasets/partco_labels/cub/cub'
aircraft_partco_root = 'datasets/partco_labels/aircraft/aircraft'
car_partco_root = 'datasets/partco_labels/cars/cars'
herbarium_partco_dataroot = 'datasets/partco_labels/herbarium'
imagenet_partco_root = 'datasets/partco_labels/imagenet100'
pets_partco_root = 'datasets/partco_labels/pets/'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = 'dev_outputs' # All logs and checkpoints will be saved here
feature_extract_dir = 'dev_outputs/feature_extraction' # All logs and checkpoints will be saved here
# fix-qxz: alias for data/herbarium_19.py
herbarium_partco_root = herbarium_partco_dataroot
