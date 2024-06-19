!pip install torch torchvision
!pip install transformers
!pip install pycocotools

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
import numpy as np
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import json
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from collections import OrderedDict, Counter
from torchvision.ops import boxes as box_ops
import os
import cv2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights , fasterrcnn_resnet50_fpn
# from gradcam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import sys
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),  
])

coco_names=["bg","tumor"]
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_model_A(num_classes=2, nms_thresh=0.5, score_thresh=0.05, detections_per_img=100, pretrained=True):

    model = fasterrcnn_resnet50_fpn(weights=None,num_classes=2)
    num_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_in_features, num_classes)
    model.roi_heads.nms_thresh = nms_thresh
    return model


def g2(img,model,cam,np_image=None,threshold=0):
    image_float_np=np.transpose(img[0].cpu().numpy(),(1,2,0))
    outputs = model(img)
    pred_classes,pred_labels,pred_scores,pred_bboxes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()],outputs[0]['labels'].cpu().numpy(),outputs[0]['scores'].detach().cpu().numpy(),outputs[0]['boxes'].detach().cpu().numpy()
    indices = [i for i, score in enumerate(pred_scores) if score >= threshold]
    boxes = pred_bboxes[indices].astype(np.int32)
    classes = [pred_classes[i] for i in indices]
    labels = pred_labels[indices]
    boxes = np.int32(boxes)
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    stacked_tensor = torch.stack(img, dim=0)
    grayscale_cam = cam(stacked_tensor, targets=targets)[0, :]
#     print("cam")
#     plt.imshow(grayscale_cam)
#     plt.show()
    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    for i, box in enumerate(boxes):
        color = (0,0,255)
        cv2.rectangle(
            cam_image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(cam_image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
#     plt.imshow(cam_image)
#     plt.show()
    return cam_image


def infer_model_A(main_folder,model_path):
    saveto="gradcam_model1"
    if os.path.exists(saveto):
        shutil.rmtree(saveto)
    os.mkdir(saveto)
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    model=get_model_A()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained = torch.load(model_path)
    model.load_state_dict(trained.state_dict())
    model=model.to(device)
    model.eval()
    all_imgs=os.listdir(main_folder)
    
    target_layers = [model.backbone]
    cam = EigenCAM(model,
                target_layers, 
                reshape_transform=fasterrcnn_reshape_transform)
    for img in tqdm(all_imgs):
#         print(img)
        image = Image.open(os.path.join(main_folder,img))
        images=transform(image).to(device)
        ip=[]
        ip.append(images)
        pred=model(ip)
        cam_img=g2(ip,model,cam)
        cv2.imwrite(os.path.join(saveto,img),cam_img)
    shutil.make_archive(saveto, 'zip', saveto)

def infer_model_B(test_images_dir,model_path):
    pass


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
