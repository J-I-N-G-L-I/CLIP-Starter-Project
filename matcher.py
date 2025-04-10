import torch

def match_caption(image_feature, text_features, captions, top_k=3):
    similarities = (image_feature @ text_features.T).squeeze(0)
    topk_idx = similarities.topk(top_k).indices.cpu().numpy()
    return [captions[i] for i in topk_idx]