import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import os
import numpy as np
from PIL import Image
import random
# import models
#import numpy as np


THREAD_NUM = 16
BATCH_SIZE = 256
PATH = r"D:\Image\Image"
CLASS_NUM = 5
EPOCHS = 200
DEVICE = torch.device("cuda")
Learn_rate = 1e-3
Weight_decay = 0

train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                     torchvision.transforms.RandomCrop(224),
                                                     torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize([0.485, 0.456, -.406],
                                                                                      [0.229, 0.224, 0.225])
                                                     ])


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        super(Mydataset, self).__init__()
        self.folder = np.array([x.path for x in os.scandir(root)])
        print(self.folder)
        self.image_files = []
        self.transforms = transforms
        for i in range(self.folder.size-1):
            print(i,self.folder[i])
            x = list([(x.path,i) for x in os.scandir(self.folder[i])])

            # print(x)
            # random.shuffle(x)
            # random.shuffle(x)
            print(len(x))
            for j in range(np.max([len(x)//10,350])  ):
                self.image_files.append(x[j])
            # self.image_files.insert(-1,*x[:1000])
        # print(self.image_files)
        print(len(self.image_files))

    def __getitem__(self, index):
        x,y = self.image_files[index]
        img = Image.open(x)
        img = self.transforms(img)
        return img, int(y)

    def __len__(self):
        return int(len(self.image_files))

class Testdata(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        super(Testdata, self).__init__()
        self.folder = np.array([x.path for x in os.scandir(root)])
        self.image_files = []
        self.transforms = transforms
        for i in range(self.folder.size-1):
            x = list([(x.path,i) for x in os.scandir(self.folder[i])])
            random.shuffle(x)
            random.shuffle(x)
            for j in range(100):
                self.image_files.append(x[j])
            # self.image_files.insert(-1,*x[:1000])
        # print(self.image_files)
        # print(len(self.image_files))

    def __getitem__(self, index):
        x,y = self.image_files[index]
        img = Image.open(x)
        img = self.transforms(img)
        return img, int(y)

    def __len__(self):
        return int(len(self.image_files))


if __name__ == '__main__':
    mydataset = Mydataset(PATH, train_augmentation)
    model = torchvision.models.densenet169(pretrained=True).to(DEVICE)
    model.eval()
    # print(model)
    # raise 0
    for param in model.parameters():
        param.requires_grad = False
    # model.classifier = Mynet22().to(DEVICE)
    # mymodel = Mynet22().to(DEVICE)
    # model.classifier = nn.Sequential(
    #         nn.Linear(1664, 500),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm1d(500),
    #         nn.Linear(500, 100),
    #         # nn.Dropout(p=0.5),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(100, CLASS_NUM)
    #     ).to(DEVICE)
    model.classifier = torch.load('./save_model2/30no_other_small.pt') # 改为torch.load('./saved_model.pt')即可使用
    # mymodel = Mynet2().to(DEVICE)
    # mymodel = torch.load('./save_model2/241two_without_dropout.pt')
    # mymodel.eval()
    # mymodel.train()
    train_loader = data.DataLoader(dataset=mydataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=THREAD_NUM)

    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=Learn_rate,
                           weight_decay=Weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    ####### test
    t=[[0 for i in range(6)] for j in range(6)]
    tt=[0 for i in range(6)]
    v = []
    for batch_idx,(image,label) in enumerate(train_loader):
        image, label = image.to(DEVICE), label.to(DEVICE)
        # print('this is forward')
        #optimizer.zero_grad()
        temp = model(image)
        output = temp
        # output = mymodel(temp)
        loss = loss_fn(output, label)
        for ind,j in enumerate(output):
                t[label[ind].item()][torch.argmax(output[ind].cpu()).item()]+=1
                tt[label[ind].item()]+=1
        v.append(loss.item())

    for i in range(CLASS_NUM):
        for j in range(CLASS_NUM):
            t[i][j]=t[i][j]/tt[i]
    print(t)
    print(np.mean(v))
    ### train
    # outfile = 'no_other.txt'
    # outfile1 = open(outfile,'w')
    # for epoch in range(0, EPOCHS + 1):
    #     print("epoch:", epoch)
    #     cos=[]
    #     for batch_idx, (image, label) in enumerate(train_loader):
    #         image, label = image.to(DEVICE), label.to(DEVICE)
    #         # print('this is forward')
    #         optimizer.zero_grad()
    #         temp = model(image)
    #         output = temp #mymodel(temp)
    #         loss = loss_fn(output, label)
    #         cos.append(loss.item())
    #         # print('back')
    #         loss.backward()
    #         optimizer.step()
    #     # outfile1.write(str(epoch)+' '+str(np.mean(cos))+'\n')
    #     # outfile1.write(str(epoch)+' '+str(np.mean(cos))+'\n')
    #     print('train ',np.mean(cos))
    #     testdataset = Testdata(PATH,train_augmentation)
    #     test_loader = data.DataLoader(dataset=testdataset,
    #                                    batch_size=BATCH_SIZE,
    #                                    shuffle=True,
    #                                    num_workers=THREAD_NUM)
    #     cos=[]
    #     for batch_idx, (image, label) in enumerate(test_loader):
    #         image, label = image.to(DEVICE), label.to(DEVICE)
    #         temp = model(image)
    #         output = temp  # mymodel(temp)
    #         loss = loss_fn(output, label)
    #         cos.append(loss.item())
    #     print('test ',np.mean(cos))
    #     if epoch % 10 ==0 :
    #         torch.save(model.classifier, './save_model2/'+str(epoch)+'no_other_small.pt')
    #         #pass
