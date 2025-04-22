import clip
import torch

class CLIPModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_image(self, image):
        with torch.no_grad():
            return self.model.encode_image(image) # 为什么已经preprocess

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
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, tuple):
            texts = list(texts)

        if isinstance(texts, list):
            if all(isinstance(t, (tuple, list)) for t in texts):
                texts = [t[0] for t in texts]

        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            return self.model.encode_text(tokens)