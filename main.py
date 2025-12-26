import os
import shutil
import random
import math

def split_dataset(source_dir, base_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Splits a directory of classed images into train, validation, and test sets.

    :param source_dir: Path to the original dataset with class sub-folders.
    :param base_dir: Path to the new directory where 'train', 'validation',
                     and 'test' will be created.
    :param split_ratio: A tuple (train, validation, test) ratio.
    """

    # --- 1. Basic Setup and Error Checking ---
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        print(f"Please create the folder: {base_dir}")
        return

    if sum(split_ratio) != 1.0:
        print(f"Error: Split ratios must sum to 1.0. Got: {sum(split_ratio)}")
        return

    train_ratio, val_ratio, test_ratio = split_ratio

    # Define the train, validation, and test directory paths
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # Create the directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Directories created at: {base_dir}")

    # --- 2. Iterate Through Each Class ---
    try:
        class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        if not class_names:
            print(f"Error: No class sub-folders found in {source_dir}")
            return
        print(f"Found {len(class_names)} classes. Starting split...")
    except Exception as e:
        print(f"Error reading source directory: {e}")
        return

    for class_name in class_names:
        print(f"Processing class: {class_name}")

        # Create class sub-folders in train, val, and test
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # --- 3. Get All Files and Split Them ---
        class_source_dir = os.path.join(source_dir, class_name)
        try:
            # Get a list of all image files for the current class
            files = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]
            random.shuffle(files) # Shuffle the list for a random split
        except Exception as e:
            print(f"  Could not read files from {class_source_dir}: {e}")
            continue

        if not files:
            print(f"  No files found for class {class_name}.")
            continue

        # Calculate the split indices
        total_files = len(files)
        train_end = math.floor(total_files * train_ratio)
        val_end = math.floor(total_files * (train_ratio + val_ratio))

        # Slice the list of files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # --- 4. Copy Files to New Directories ---

        # Copy training files
        for f in train_files:
            shutil.copy(os.path.join(class_source_dir, f), os.path.join(train_class_dir, f))

        # Copy validation files
        for f in val_files:
            shutil.copy(os.path.join(class_source_dir, f), os.path.join(val_class_dir, f))

        # Copy test files
        for f in test_files:
            shutil.copy(os.path.join(class_source_dir, f), os.path.join(test_class_dir, f))

    print(f"\n--- Dataset splitting complete! ---")
    print(f"Your new dataset is ready at: {base_dir}")


# ==================================================================
#                       HOW TO USE THIS SCRIPT
# ==================================================================

# 1. SET YOUR PATHS (I have already done this for you)
# Path to your *original* PlantVillage 'color' folder
SOURCE_DIR = r'D:\Projects\Agri Detect\PlantVillage-Dataset-master\PlantVillage-Dataset-master\raw\color'

# Path to the *new, empty* folder you will create
BASE_DIR = r'D:\Projects\Agri Detect\PlantVillage-Dataset-master\plant_dataset_split'

# 2. SET YOUR SPLIT RATIO
# (Train, Validation, Test) - Must add up to 1.0
SPLIT_RATIO = (0.7, 0.15, 0.15) # 70% train, 15% validation, 15% test

# 3. RUN THE FUNCTION
split_dataset(SOURCE_DIR, BASE_DIR, SPLIT_RATIO)