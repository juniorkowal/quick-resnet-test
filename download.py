import os
import shutil
import subprocess

# config
REPO_URL = "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes.git"
TARGET_DIR = "imagenet"
VAL_DIR = os.path.join(TARGET_DIR, "val")

def download_dataset():
    '''Downloads dataset with subset of 5000 images from IMAGENET/ILVSR2012/val'''
    print("Cloning repository...")
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    subprocess.run(["git", "clone", REPO_URL, TARGET_DIR], check=True)
    print("Deleting non-JPG files...")
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(('.jpg', '.jpeg')):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    os.makedirs(VAL_DIR, exist_ok=True)
    print("Moving JPG files to val directory...")
    for root, _, files in os.walk(TARGET_DIR):
        if root == VAL_DIR:
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                src = os.path.join(root, file)
                dest = os.path.join(VAL_DIR, file)
                counter = 1
                while os.path.exists(dest):
                    name, ext = os.path.splitext(file)
                    dest = os.path.join(VAL_DIR, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(src, dest)
                print(f"Moved: {src} -> {dest}")

    print("Cleaning up empty directories...")
    for root, dirs, _ in os.walk(TARGET_DIR, topdown=False):
        if root == VAL_DIR:
            continue
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
            except OSError:
                pass

    print(f"All images are now in {VAL_DIR}.")
    url_json = r'https://raw.githubusercontent.com/Jasonlee1995/ImageNet-1K/refs/heads/main/ImageNet_class_index.json'
    url_txt = r'https://raw.githubusercontent.com/Jasonlee1995/ImageNet-1K/refs/heads/main/ImageNet_val_label.txt'
    subprocess.run(["wget", "--no-check-certificate", url_json])
    subprocess.run(["wget", "--no-check-certificate", url_txt])

    shutil.move('ImageNet_class_index.json', f'{TARGET_DIR}/ImageNet_class_index.json')
    shutil.move('ImageNet_val_label.txt', f'{TARGET_DIR}/ImageNet_val_label.txt')
    print(f"Labels and class indexes downloaded to {TARGET_DIR}.")


if __name__ == "__main__":
    download_dataset()


