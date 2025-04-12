import os
import shutil

def flatten_dataset_structure(root_dir):
    """
    Moves all images from nested subfolders up into the 'adm' or 'real' folders.
    Assumes the directory structure looks like:
    
    root_dir/
        train/
            adm/
                subfolder1/
                subfolder2/
            real/
                subfolder1/
                subfolder2/
        val/
        test/
    """

    for split in ["train", "val", "test"]:
        for cls in ["adm", "real"]:
            base_dir = os.path.join(root_dir, split, cls)
            if not os.path.isdir(base_dir):
                print(f"[!] Skipping non-existent directory: {base_dir}")
                continue

            # Traverse subfolders under each class
            for subfolder in os.listdir(base_dir):
                sub_path = os.path.join(base_dir, subfolder)
                if not os.path.isdir(sub_path):
                    continue  # Skip if it's not a folder

                # Move each image inside the subfolder to the class folder
                for filename in os.listdir(sub_path):
                    src_path = os.path.join(sub_path, filename)
                    dst_filename = f"{subfolder}_{filename}"
                    dst_path = os.path.join(base_dir, dst_filename)
                    
                    # Avoid overwriting existing files
                    if not os.path.exists(dst_path):
                        shutil.move(src_path, dst_path)

                # Remove the now-empty subfolder
                os.rmdir(sub_path)
            print(f"[✓] Flattened: {split}/{cls}")

if __name__ == "__main__":
    # Replace this path with the root directory of your ImageNet-style dataset
    dataset_root = r"D:\VS Projects\ECE-580\Proj\datasets\imagenet"
    flatten_dataset_structure(dataset_root)
