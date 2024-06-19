# !pip install torch torchvision
# !pip install transformers
# !pip install pycocotools

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
import shutil
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from transformers import DetrFeatureExtractor, AutoImageProcessor, DetrForObjectDetection, DetrImageProcessor, DetrConfig, DeformableDetrForObjectDetection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.list_imgs=os.listdir(root_dir)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        img_path = os.path.join(self.root_dir,self.list_imgs[idx])
        orig_img=cv2.imread(img_path)
        image = Image.open(img_path)
        encoding = processor(images=image, annotations=None, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() 
#         image = self.transform(image)
        mask=torch.ones((pixel_values.shape[1],pixel_values.shape[2]))
        return (torch.tensor(pixel_values).unsqueeze(0),mask.unsqueeze(0),orig_img ,self.list_imgs[idx])



def get_model_A(num_classes=2, nms_thresh=0.5, score_thresh=0.05, detections_per_img=100, pretrained=True):

    model = fasterrcnn_resnet50_fpn(weights=None,num_classes=2)
    num_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_in_features, num_classes)
    model.roi_heads.nms_thresh = nms_thresh
    return model


def inv_transform_test(box, scores, img_name, oh, ow,folder_path=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_name = img_name[:-4] + "_preds.txt"
    fin_list = []
    conf_list = [l for l in scores]
    for i, item in enumerate(box):
        x_c, y_c, w, h = item
        x_c, y_c, w, h = x_c / ow, y_c / oh, w / ow, h / oh
        fin_list.append([x_c.item(), y_c.item(), w.item(), h.item(), conf_list[i].item()])
    
    img_path = os.path.join(folder_path, file_name)
    with open(img_path, 'w') as file:
        for item in fin_list:
            formatted_item = ' '.join(['{:.18e}'.format(num) for num in item])
            file.write(formatted_item + '\n')


def gen_predictions_modelA(model,test_img_dir,folder_name="predictions_modelA"):
    model = model.to(device)
    images = os.listdir(test_img_dir)

    for k, img in enumerate(images):
        file = os.path.join(test_img_dir, img)
        image = cv2.imread(file)
        h, w = image.shape[0], image.shape[1]
        orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(orig_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            pred = model([image_tensor])
            inv_transform_test(pred[0]['boxes'], pred[0]["scores"], img, h, w,folder_path=folder_name)
    shutil.make_archive(folder_name, 'zip', folder_name)



def inv_transform_test_b(box, scores, img_name, oh, ow,folder_path=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_name = img_name[:-4] + "_preds.txt"
    fin_list = []
    conf_list = [l for l in scores]
    for i, item in enumerate(box):
        x_c, y_c, w, h = item
        x_c, y_c, w, h = x_c / ow, y_c / oh, w / ow, h / oh
        fin_list.append([x_c.item(), y_c.item(), w.item(), h.item(), conf_list[i]])
    
    img_path = os.path.join(folder_path, file_name)
    with open(img_path, 'w') as file:
        for item in fin_list:
            formatted_item = ' '.join(['{:.18e}'.format(num) for num in item])
            file.write(formatted_item + '\n')


def gen_predictions_modelB(model,test_img_dir,infer_data_loader,folder_name="predictions_modelB"):
    
    for k, (pixel_values, mask, image, image_name) in enumerate(infer_data_loader):
        pixel_values = pixel_values.to(device)
        pixel_mask = mask.to(device)
        h, w = image.shape[0], image.shape[1]
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            logits=outputs["logits"][0]
            prob=torch.softmax(logits, dim=1)
            cl=torch.argmax(prob, dim=1)
            box=[]
            prob_score=[]
            out_tensor = outputs["pred_boxes"][0]
    #         print(out_tensor)
            for i in range(300):
                if(cl[i]==0 and prob[i][0]>0.5):
                    box.append(out_tensor[i])
                    prob_score.append(prob[i][0].item())
            if len(prob_score)==0:
                b = []
                s = []
            else:
                max_pos = np.argmax(prob_score)
                b = [box[max_pos]]
                s = [prob_score[max_pos]]
            inv_transform_test_b(b,s, image_name, h, w,folder_name )
    shutil.make_archive(folder_name, 'zip', folder_name)


def infer_model_A(main_folder,model_path):
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    model=get_model_A()
    trained = torch.load(model_path)
    model.load_state_dict(trained.state_dict())
    model=model.to(device)
    model.eval()
    gen_predictions_modelA(model,main_folder)

def infer_model_B(test_images_dir,model_path):
    model= DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    loaded_model = torch.load(model_path)
    state_dict = loaded_model.state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    infer_data_loader = CustomDataset2(root_dir = test_images_dir)
    gen_predictions_modelB(model,test_images_dir,infer_data_loader,folder_name="predictions_modelB")

def main():
    model_type=int(sys.argv[1])
    main_folder=sys.argv[2]
    modelpath=sys.argv[3]
    
    if(model_type==1):
        infer_model_A(main_folder,modelpath)
    if(model_type==2):
        infer_model_B(main_folder,modelpath)


if __name__ == "__main__":
    main()