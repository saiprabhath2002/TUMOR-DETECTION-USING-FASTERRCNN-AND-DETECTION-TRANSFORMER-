!pip install torch torchvision
!pip install transformers
!pip install pycocotools

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from torchvision.transforms.functional import to_tensor, resize
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
import time
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import json
import sys
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from collections import OrderedDict, Counter
from torchvision.ops import boxes as box_ops
import os
import cv2
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from transformers import DetrFeatureExtractor, AutoImageProcessor, DetrForObjectDetection, DetrImageProcessor, DetrConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class CustomDataset(Dataset):
    
    def __init__(self, root_dir, transform=None,annotation_path=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotation_path=annotation_path
        self.json_data=None
        self.read_annotated_data()
        self.data={}
        self.img_name_id={}
        self.make_dict_name_id()
        self.annotation_data=None
        self.get_annotaion_data()

    def make_dict_name_id(self):
        for i in range(len(self.json_data["images"])):
            temp_data=self.json_data["images"][i]
            self.img_name_id[temp_data["id"]]=temp_data["file_name"]

    def get_annotaion_data(self):
        visited={}
        n=0
        for i in range(len(self.json_data["annotations"])):
            temp_data=self.json_data["annotations"][i]
            if temp_data["image_id"] not in visited:
                self.data[n]={}
                self.data[n]["bbox"]=[]
                self.data[n]["cls"]=[]
                self.data[n]["image_id"]=temp_data["image_id"]
                self.data[n]["image_path"]=os.path.join(self.root_dir,self.img_name_id[temp_data["image_id"]])
                x1,y1,w,h=temp_data["bbox"]
                self.data[n]["bbox"].append([x1,y1,x1+w,y1+h])
                self.data[n]["cls"].append(1)
                visited[temp_data["image_id"]]=n
                n+=1
    #                 print(i,self.data[i])
            else:
                x1,y1,w,h=temp_data["bbox"]
                self.data[visited[temp_data["image_id"]]]["bbox"].append([x1,y1,x1+w,y1+h])
                self.data[visited[temp_data["image_id"]]]["cls"].append(1)

    def read_annotated_data(self):
        with open(self.annotation_path,'r') as f:
            self.json_data=json.load(f)

    def display_img(self,idx):
        img_path = self.data[idx]['image_path']
        orig_img=cv2.imread(img_path)
        for i in range(len(self.data[idx]["bbox"])):
            x1,y1,x2,y2=self.data[idx]["bbox"][i]
    #             print(x1,y1,w,h)
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)
    #             print(x1,y1,x2,y2)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (255,0,0), 1)
        plt.imshow(orig_img)
        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    #         print(idx)
        img_path = self.data[idx]['image_path']
        image = Image.open(img_path)
        image = self.transform(image)
        target = {}
        target["boxes"] = torch.tensor(self.data[idx]["bbox"]).to(device)
        target["labels"] = torch.tensor(self.data[idx]['cls']).to(device)
        return (torch.tensor(image).to(device),target)


def get_model_A(num_classes=2, nms_thresh=0.5, score_thresh=0.05, detections_per_img=100, pretrained=True):

    model = fasterrcnn_resnet50_fpn(weights=None,num_classes=2)
    num_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_in_features, num_classes)
    model.roi_heads.nms_thresh = nms_thresh
    return model

def load_data_A(i,data_set,indices,batch_size):
    img_list=[]
    targets_list=[]
    for i in range(i,i+batch_size):
        data=data_set[indices[i]]
        img_list.append(data[0])
        targets_list.append(data[1])
    return img_list,targets_list


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder,annotation_path, train=True):
        super(CocoDetection, self).__init__(img_folder, annotation_path)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        encoding = processor(images=img, annotations={'image_id': self.ids[idx], 'annotations': target}, return_tensors="pt")
        return encoding["pixel_values"].squeeze() , encoding["labels"][0]

def Train_model_A(train_annotation,val_annotation,train_img_path,val_img_path):
    batch_size=4
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])

    train_dataset = CustomDataset(root_dir=train_img_path , transform=transform,annotation_path=train_annotation)
    val_dataset = CustomDataset(root_dir=val_img_path , transform=transform,annotation_path=val_annotation)
    train_num_batches=len(train_dataset)//batch_size
    valid_num_batches=len(val_dataset)//batch_size

    model=get_model_A().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    valid_loss=[]
    train_loss=[]
    model.train()
    st=time.time()
    for epoch in range(30):
        batch_loss=[]
        indices = torch.randperm(len(train_dataset)).tolist()
        for i in range(train_num_batches):
            images,targets=load_data_A(i,train_dataset,indices,batch_size)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            batch_loss.append(losses.item())
            del images
            del targets
        train_loss.append(np.mean(batch_loss))
        batch_loss=[]
        indices = torch.randperm(len(val_dataset)).tolist()
        for i in range(valid_num_batches):
            images,targets=load_data_A(i,val_dataset,indices,batch_size)
            with torch.no_grad():
                l = model(images, targets)
                losses = sum(loss for loss in l.values())
                batch_loss.append(losses.item())
                del images
                del targets
        valid_loss.append(np.mean(batch_loss))
        scheduler.step()
        print(f"epoch : {epoch} train loss {train_loss[-1]}   validation_loss : {valid_loss[-1]} time : {(time.time()-st)/60} min")
    torch.save(model,"model_A.pth")
    plt.plot(train_loss,label="train loss")
    plt.plot(valid_loss,label="valid loss")
    plt.legend()
    plt.show()

def Train_model_B(train_annotation,val_annotation,train_img_path,val_img_path):

    train_dataset = CocoDetection(img_folder=train_img_path,annotation_path=train_annotation)
    val_dataset = CocoDetection(img_folder=val_img_path,annotation_path=val_annotation, train=False)
    
    def collate_fn(batch):
        pixel_values, labels = zip(*batch)
        encoding = processor.pad(list(pixel_values), return_tensors="pt")
        return {'pixel_values': encoding['pixel_values'], 'pixel_mask': encoding['pixel_mask'], 'labels': list(labels)}

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=8)

    model=DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                                revision="no_timm",
                                                                num_labels=1,
                                                                ignore_mismatched_sizes=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    train_loss=[]
    val_loss=[]
    num_epochs=30
    st=time.time()
    for epoch in range(num_epochs):
        l=[]
        model.train()
        for batch in (train_dataloader):
            images,pixel_mask, targets = batch["pixel_values"],batch["pixel_mask"],batch["labels"]
            images = images.to(device)
            pixel_mask=pixel_mask.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(pixel_values=images, pixel_mask=pixel_mask, labels=targets)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())
        train_loss.append(np.mean(l))
        
        model.eval()
        with torch.no_grad():
            l=[]
            for batch in (val_dataloader):
                images,pixel_mask, targets = batch["pixel_values"],batch["pixel_mask"],batch["labels"]
                images = images.to(device)
                pixel_mask=pixel_mask.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(pixel_values=images, pixel_mask=pixel_mask, labels=targets)
                loss = outputs.loss
                l.append(loss.item())
            val_loss.append(np.mean(l))
        scheduler.step()
        print(f"Epoch {epoch}, Loss: {train_loss[-1]}  val_loss : {val_loss[-1]} time : {(time.time()-st)/60} min")
    torch.save(model,"model_B.pth")
    plt.plot(train_loss, label = 'Train Loss')
    plt.plot(val_loss, label = 'val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()    
    plt.show()


def main():
    model_type=int(sys.argv[1])
    main_folder=sys.argv[2]
    train_annotation=os.path.join(main_folder,"annotations/instances_train2017.json")
    val_annotation=os.path.join(main_folder,"annotations/instances_val2017.json")
    train_img_path=os.path.join(main_folder,"train2017")
    val_img_path=os.path.join(main_folder,"val2017")
    if(model_type==1):
        Train_model_A(train_annotation,val_annotation,train_img_path,val_img_path)
    if(model_type==2):
        Train_model_B(train_annotation,val_annotation,train_img_path,val_img_path)


if __name__ == "__main__":
    main()