from PIL import Image
from torchvision import transforms




img = Image.open("./images/bananas.jpg")

transto_tensor = transforms.ToTensor()
image_tensor = transto_tensor(img)

print(img)
print("*" * 50)
print (image_tensor)

# normalize
print(image_tensor[0][0][0])
trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
img_norm = trans_norm(image_tensor)
print(img_norm[0][0][0])

trans_resize = transforms.Resize((224, 224))
img_resized = trans_resize(img)
print(img_resized)
img_resized = transto_tensor(img_resized)
print(img_resized)


data_augument = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_