import zipfile
import os

# 设置源 zip 路径
zip_base_path = r"D:\Programming\datasets"
val_zip = os.path.join(zip_base_path, "val2017.zip")
ann_zip = os.path.join(zip_base_path, "annotations_trainval2017.zip")

# 设置目标解压路径
val_target_dir = "./data/images"
ann_target_dir = "./data/annotations"

# 解压 val2017.zip
os.makedirs(val_target_dir, exist_ok=True)
with zipfile.ZipFile(val_zip, 'r') as zip_ref:
    zip_ref.extractall(val_target_dir)
print("✅ 解压 val2017 完成")

# 解压 annotations
os.makedirs(ann_target_dir, exist_ok=True)
with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
    zip_ref.extractall(ann_target_dir)
print("✅ 解压 annotations 完成")
