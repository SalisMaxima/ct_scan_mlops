#This script downloads the latest version of the Brain MRI Images for Brain Tumor Detection dataset from Kaggle.
import shutil
from pathlib import Path

import kagglehub

# Define the target directory
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# Download latest version
cache_path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")

# Copy to data/raw folder
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
target_path = RAW_DATA_DIR / "chest-ctscan-images"
if target_path.exists():
    shutil.rmtree(target_path)
shutil.copytree(cache_path, target_path)

print("Dataset downloaded to:", target_path)