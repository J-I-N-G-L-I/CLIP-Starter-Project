import json
import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CocoCaptionsDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, max_images=500):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())[: max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        captions = [ann["caption"] for ann in self.coco.loadAnns(ann_ids)]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, captions, image_path
#
#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
#         ann_ids = self.coco.getAnnIds(imgIds=image_id)
#         captions = [ann["caption"] for ann in self.coco.loadAnns(ann_ids)]
#         image_info = self.coco.loadImgs(image_id)[0]
#         image_path = os.path.join(self.image_dir, image_info["file_name"])
#         image = Image.open(image_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, captions, image_path
