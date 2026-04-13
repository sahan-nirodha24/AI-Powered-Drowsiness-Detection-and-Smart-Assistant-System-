import os
import shutil
import random
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
OUTPUT_DIR = Path("Processed_Dataset")
SPLIT_RATIO = (0.7, 0.15, 0.15)  # Train, Val, Test

# Extensions to look for
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']

def find_source_directory():
    """Locate the Imageset directory."""
    possible_paths = [
        Path("Imageset"),
        Path("Drowsiness Detection/Imageset"),
        Path("../Imageset")
    ]
    
    for p in possible_paths:
        if p.exists() and p.is_dir():
            return p
    
    # Deep search if standard paths fail
    found = list(Path(".").rglob("Imageset"))
    if found:
        return found[0]
        
    return None

def setup_directories(categories):
    """Create train/val/test directories for each category."""
    if OUTPUT_DIR.exists():
        logging.info(f"Refreshing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        for category in categories:
            (OUTPUT_DIR / split / category).mkdir(parents=True, exist_ok=True)

def get_image_files(directory):
    """Recursively find image files."""
    files = []
    for ext in EXTENSIONS:
        files.extend(list(directory.rglob(ext)))
    return files

def split_dataset():
    source_dir = find_source_directory()
    
    if not source_dir:
        logging.error("Could not locate 'Imageset' folder.")
        return

    logging.info(f"Using Source Directory: {source_dir.absolute()}")
    
    # Target Categories to merge into
    # We explicitly look for folders named 'drowsy' and 'non-drowsy' anywhere in the tree
    target_classes = ['drowsy', 'non-drowsy']
    
    # Reset Output Directory
    setup_directories(target_classes)

    total_images = 0
    
    for category in target_classes:
        logging.info(f"Collecting images for class '{category}'...")
        
        # Find all folders named exactly 'category' (e.g., all 'drowsy' folders)
        # We search recursively from the source root
        category_folders = [p for p in source_dir.rglob(category) if p.is_dir()]
        
        all_files = []
        for folder in category_folders:
            # Get all images inside this folder (including its subfolders like 'with glasses')
            files = get_image_files(folder)
            logging.info(f"  Found {len(files)} images in {folder.relative_to(source_dir)}")
            all_files.extend(files)
            
        # Check if we missed files directly in source_dir/category if the structure is flat
        if not category_folders:
             direct_path = source_dir / category
             if direct_path.exists():
                 files = get_image_files(direct_path)
                 all_files.extend(files)

        total_cat = len(all_files)
        logging.info(f"Total images for '{category}': {total_cat}")
        
        if total_cat == 0:
            logging.warning(f"No images found for class '{category}'. Skipping.")
            continue

        # Shuffle and Split
        random.shuffle(all_files)
        
        train_end = int(total_cat * SPLIT_RATIO[0])
        val_end = train_end + int(total_cat * SPLIT_RATIO[1])

        splits = {
            'train': all_files[:train_end],
            'val': all_files[train_end:val_end],
            'test': all_files[val_end:]
        }

        for split_name, split_files in splits.items():
            for f in split_files:
                # Custom Naming Logic: Color_Glasses_OriginalName
                # Expected Source Path: Source / Color / State / Glasses / Image
                try:
                    rel_path = f.relative_to(source_dir)
                    parts = rel_path.parts
                    
                    if len(parts) >= 4:
                        # parts[0] = 'Black & White' or 'Colored'
                        # parts[1] = 'drowsy' or 'non-drowsy'
                        # parts[2] = 'with glasses' or 'without glasses'
                        # parts[3] = filename
                        
                        color_prefix = parts[0].replace(" ", "") # 'Black & White' -> 'Black&White'
                        glasses_suffix = parts[2].replace(" ", "_") # 'with glasses' -> 'with_glasses'
                        
                        new_name = f"{color_prefix}_{glasses_suffix}_{parts[-1]}"
                    else:
                        # Fallback for unexpected structure
                        new_name = f"{f.parent.parent.name}_{f.parent.name}_{f.name}".replace(" ", "_")
                except Exception as e:
                    logging.warning(f"Could not construct custom name for {f}: {e}")
                    new_name = f"{f.parent.name}_{f.name}".replace(" ", "_")

                dest = OUTPUT_DIR / split_name / category / new_name
                shutil.copy2(f, dest)
        
        logging.info(f"  Split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        total_images += total_cat

    logging.info(f"✅ Splitting Complete. Total {total_images} images organized in '{OUTPUT_DIR.absolute()}'.")

if __name__ == "__main__":
    split_dataset()
