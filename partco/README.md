# PartCo

## Quick start

- **Install dependencies:**  
```bash
pip install -r requirements.txt
```

- **Place PARTCO labels:**  
Download the PartCo label files and put them under the repository `datasets/partco_labels/`, with one subfolder per dataset, for example:
```
datasets/partco_labels/
  cifar10/
  cifar100/
  cub/
  aircraft/
  cars/
  herbarium/
  imagenet100/
  pets/
```

- **Configure paths:**  
If your datasets are in custom locations, update the dataset root variables in config.py.

- **Run experiments:**  
See the scripts directory for dataset-specific run scripts (files named `run_*.sh`). Example:
```bash
bash scripts/run_cifar10_partco_simgcd.sh
```

- **Outputs:**  
Logs and checkpoints are written to `dev_outputs` (see `exp_root` in config.py).