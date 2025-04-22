# # import clip
# # import torch
# # # print(clip.available_models())
# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print (device)
# #
# # from torch.utils.tensorboard import SummaryWriter
# # writer = SummaryWriter("logs")
# # # writer.add_image()
# # for i in range(10):
# #     writer.add_scalar("y=x", i, i)
# #
# # writer.close()
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
path1 = r"D:\Programming\Jetbrains\PyCharm\Workspace\CLIP_caption\images\bananas.jpg"
img = Image.open(path1)
writer = SummaryWriter(log_dir="logs")


tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_tensor(tag="tensor_images", tensor=tensor_img)
writer.close()

# print(tensor_img)


