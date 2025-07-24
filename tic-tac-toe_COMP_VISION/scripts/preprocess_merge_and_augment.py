import os
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image

# CONFIGURABLE PATHS
OLD_DATASET_DIR = '/kaggle/input/rock-paper-scissors-dataset/'
NEW_DATASET_ROOT = '/kaggle/input/d/glushko/rock-paper-scissors-dataset/'
NEW_TRAIN_DIR = os.path.join(NEW_DATASET_ROOT, 'train')
WORKING_ROOT = '/kaggle/working/rock-paper-scissors-dataset/'
WORKING_TRAIN_DIR = os.path.join(WORKING_ROOT, 'train')
CLASS_NAMES = ['rock', 'paper', 'scissors']

# 1. Merge old and new train datasets into working train/ folder
def merge_to_working_train():
    for cls in CLASS_NAMES:
        working_cls_dir = os.path.join(WORKING_TRAIN_DIR, cls)
        os.makedirs(working_cls_dir, exist_ok=True)
        # Copy from new dataset's train
        new_cls_dir = os.path.join(NEW_TRAIN_DIR, cls)
        if os.path.exists(new_cls_dir):
            for img_path in glob(os.path.join(new_cls_dir, '*')):
                fname = os.path.basename(img_path)
                dest_path = os.path.join(working_cls_dir, f"new_{fname}")
                shutil.copy2(img_path, dest_path)
        # Copy from old dataset
        old_cls_dir = os.path.join(OLD_DATASET_DIR, cls)
        if os.path.exists(old_cls_dir):
            for img_path in glob(os.path.join(old_cls_dir, '*')):
                fname = os.path.basename(img_path)
                dest_path = os.path.join(working_cls_dir, f"old_{fname}")
                shutil.copy2(img_path, dest_path)
    print("Merging old and new train datasets into working train/ complete.")

# 2. Data augmentation for all images in working train/ folder
def augment_image(img):
    from torchvision import transforms
    aug_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1)),
    ]
    augmented = []
    for t in aug_transforms:
        augmented.append(t(img))
    return augmented

def augment_working_train_dataset():
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(WORKING_TRAIN_DIR, cls)
        img_paths = glob(os.path.join(cls_dir, '*'))
        for img_path in tqdm(img_paths, desc=f"Augmenting {cls}"):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Could not open {img_path}: {e}")
                continue
            for i, aug_img in enumerate(augment_image(img)):
                aug_fname = os.path.splitext(os.path.basename(img_path))[0] + f'_aug{i}.jpg'
                aug_path = os.path.join(cls_dir, aug_fname)
                aug_img.save(aug_path)
    print("Augmentation of working train/ complete.")

if __name__ == '__main__':
    print("Merging old and new train datasets into working train/ folder...")
    merge_to_working_train()
    print("Performing data augmentation on working train/ folder...")
    augment_working_train_dataset()
    print("Preprocessing complete.") 