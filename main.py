import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import CocoCaptionsDataset
from encoder import CLIPModel
from matcher import match_caption

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel(device=device)

    dataset = CocoCaptionsDataset(
        image_dir="data/images/val2017",
        annotation_file="data/annotations/annotations/captions_val2017.json",
        transform=clip_model.preprocess,
        max_images=100  # 先用前100张图像
    )
    dataloader = DataLoader(dataset, batch_size=1)

    # for image_tensor, captions, path in tqdm(dataloader):
    #     image_tensor = image_tensor.to(device)
    #     image_feature = clip_model.encode_image(image_tensor)
    #
    #     text_features = clip_model.encode_text(captions)
    #     matched = match_caption(image_feature, text_features, captions[0], top_k=1)
    #
    #     print(f"Image: {os.path.basename(path[0])}")
    #     print(f"Best Caption: {matched[0]}")
    #     print("-" * 40)

    for image, captions, image_path in tqdm(dataloader):
        image = image.to(device)
        image_feature = clip_model.encode_image(image)

        # captions 是 list[str]
        text_features = clip_model.encode_text(captions)

        # 传完整 captions list 给 matcher
        matched = match_caption(image_feature, text_features, captions, top_k=1)

        print(f"Image: {os.path.basename(image_path[0])}")
        print(f"Best Caption: {matched[0]}")
        print("-" * 40)

if __name__ == "__main__":
    main()
