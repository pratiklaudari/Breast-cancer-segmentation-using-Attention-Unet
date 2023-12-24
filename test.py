import encoder_block
from glob import glob as gb

#dataset_dir=input("path to dataset")
dataset=[]
dataset_dir=r"C:\major\Breast-cancer-segmentation-using-Attention-Unet\images\benign\*.png"
dataset=gb(dataset_dir)

#num_filters=int(input("number of filters"))
img=[]
mask=[]
for i in dataset:
    x=i.find('mask')
    if x==-1:
        img.append(i)
    else:
        mask.append(i)
print(img)
print(mask)

#encoder_block.encoder(dataset_dir=dataset_dir,num_filters=num_filters)
