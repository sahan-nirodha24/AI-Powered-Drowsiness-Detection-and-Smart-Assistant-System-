import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = r"c:\Users\SAHAN\Desktop\SVM\Imageset"
TARGET_BASE_DIR = r"c:\Users\SAHAN\Desktop\SVM\dataset_split"
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # Train, Val, Test

def create_structure():
    if os.path.exists(TARGET_BASE_DIR):
        print(f"Removing existing {TARGET_BASE_DIR}...")
        shutil.rmtree(TARGET_BASE_DIR)
    
    for split in ['train', 'val', 'test']:
        for category in ['drowsy', 'non_drowsy']:
            os.makedirs(os.path.join(TARGET_BASE_DIR, split, category), exist_ok=True)

def get_image_files(base_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files_list = []
    
    # Structure: Type (BW/Color) -> Category (Drowsy/Non) -> Glasses (With/Without)
    # Actually based on exploration:
    # Imageset -> Black & White -> drowsy -> with glasses
    
    # We will traverse recursively
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                full_path = Path(root) / file
                
                # Extract metadata from path
                # Path parts relative to Imageset:
                # e.g. "Black & White\drowsy\with glasses\img.jpg"
                rel_parts = full_path.relative_to(base_path).parts
                
                if len(rel_parts) < 3:
                    continue
                
                img_type = rel_parts[0] # "Black & White" or "Colored"
                category_raw = rel_parts[1] # "drowsy" or "non-drowsy"
                glasses = rel_parts[2] # "with glasses" or "without glasses"
                
                # Normalize for SVM classes
                if "non" in category_raw.lower():
                    final_category = "non_drowsy"
                else:
                    final_category = "drowsy"
                    
                # Simplify metadata for filename
                type_code = "bw" if "black" in img_type.lower() else "col"
                glasses_code = "glass" if "with" in glasses.lower() and "out" not in glasses.lower() else "no_glass"
                
                files_list.append({
                    'path': full_path,
                    'category': final_category,
                    'prefix': f"{type_code}_{glasses_code}_"
                })
    return files_list

def split_and_copy(files_list):
    # Group by category to ensure stratified split if needed, 
    # but simple shuffle per category is usually sufficient.
    
    category_map = {'drowsy': [], 'non_drowsy': []}
    for f in files_list:
        category_map[f['category']].append(f)
        
    for category, items in category_map.items():
        random.shuffle(items)
        total = len(items)
        train_count = int(total * SPLIT_RATIOS[0])
        val_count = int(total * SPLIT_RATIOS[1])
        # remaining goes to test
        
        train_items = items[:train_count]
        val_items = items[train_count:train_count+val_count]
        test_items = items[train_count+val_count:]
        
        print(f"Category {category}: Total {total} -> Train {len(train_items)}, Val {len(val_items)}, Test {len(test_items)}")
        
        splits = [
            ('train', train_items),
            ('val', val_items),
            ('test', test_items)
        ]
        
        for split_name, split_items in splits:
            for item in split_items:
                src = item['path']
                # Create unique filename: prefix + original name
                # Handle potential duplicate names from different folders by adding hash if needed
                # For now prefix should distinguish enough providing original names aren't identical across BW/Color
                
                new_name = item['prefix'] + src.name
                dest = os.path.join(TARGET_BASE_DIR, split_name, category, new_name)
                
                # Handle potential collision by checking existence
                if os.path.exists(dest):
                    base, ext = os.path.splitext(new_name)
                    new_name = f"{base}_{random.randint(1000,9999)}{ext}"
                    dest = os.path.join(TARGET_BASE_DIR, split_name, category, new_name)
                
                shutil.copy2(src, dest)

if __name__ == "__main__":
    print("Starting dataset split...")
    create_structure()
    all_files = get_image_files(SOURCE_DIR)
    print(f"Found {len(all_files)} total images.")
    split_and_copy(all_files)
    print("Dataset split complete!")
