import os
import clip
import torch
from PIL import Image
from tqdm import tqdm

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模型
model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("ViT-B/32", device=device)
# 图像路径
image_folder = "images/"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
# image_folder = "images/"
# image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
# 候选描述句子
candidate_captions = [
    "A dog running in the grass",
    "A city skyline with tall buildings",
    "A delicious pizza with cheese",
    "A person riding a bicycle",
    "A cat sitting on the sofa",
    "Mountains under a blue sky",
    "A busy street with cars",
    "People walking in a park",
    "A plate of fresh fruit",
    "A beach during sunset",
    "A brown bear",
    "Several apples",
    "Several bananas",
    "A kid is playing black sands",
    "A women is smoking",
]

# 先将所有文本编码
with torch.no_grad():
    text_tokens = clip.tokenize(candidate_captions).to(device)
    text_features = model.encode_text(text_tokens)

# 为每张图片找到最相似的文本
for image_path in tqdm(image_paths):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        similarity = (image_features @ text_features.T).squeeze(0)  # 点积
        best_idx = similarity.argmax().item()
        best_caption = candidate_captions[best_idx]

    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Caption: {best_caption}")
    print("=" * 40)