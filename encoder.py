import clip
import torch

class CLIPModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode_image(self, image):
        with torch.no_grad():
            return self.model.encode_image(image)

    # def encode_text(self, texts):
    #     with torch.no_grad():
    #         tokens = clip.tokenize(texts).to(self.device)
    #         return self.model.encode_text(tokens)

    # def encode_text(self, texts):
    #     # 兼容传入 tuple 或 str 的情况
    #     if isinstance(texts, str):
    #         texts = [texts]
    #     elif isinstance(texts, tuple):
    #         texts = list(texts)
    #
    #     with torch.no_grad():
    #         tokens = clip.tokenize(texts).to(self.device)
    #         return self.model.encode_text(tokens)

    def encode_text(self, texts):
        # 单条字符串变为 list[str]
        if isinstance(texts, str):
            texts = [texts]
        # tuple 变为 list
        elif isinstance(texts, tuple):
            texts = list(texts)

        # 如果是 list of tuple 或 list of list，提取每个元素中第一个子元素
        if isinstance(texts, list):
            if all(isinstance(t, (tuple, list)) for t in texts):
                texts = [t[0] for t in texts]

        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            return self.model.encode_text(tokens)