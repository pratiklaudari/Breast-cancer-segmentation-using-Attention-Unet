import torchvision.datasets as datasets
from torchvision import transforms
import torch
import cv2
from glob import glob as gb
#find out average and stanadar deviation of whole dataset
mean=0.0
std=0.0
pixels=0
data_dir = 'C:\major\datasetfornormalization\*.png'
dataset=gb(data_dir)
for image in dataset:
    img=cv2.imread(image)
    Tensor_img=transforms.Compose([transforms.ToTensor()])
    image_tensor=Tensor_img(img)
    t_mean=torch.mean(image_tensor)
    t_std=torch.std(image_tensor)
    mean=(mean+t_mean)
    std=(std+t_std)
mean=mean/len(dataset)
std=std/len(dataset)
print(str(mean)+str(std))
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
data_dir ='C:\major\datasetfornormalization'
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)