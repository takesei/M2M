#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import random
import os
import sys
import time


# In[2]:


# Define save/load func
def load_checkpoint(model_dir, epoch, model):
    load_path = os.path.join(model_dir, f"shufflenet_epoch_{epoch}.pkl")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {load_path}")


# In[3]:


# Settings
# random seeds
np.random.seed(723)
random.seed(723)
torch.manual_seed(723)
torch.cuda.manual_seed(723)

# Networks
batchsize = 32
epochs = 10
epoch_start = 1

# GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("CUDA detected")
    cudnn.benchmark = True
    cudnn.deterministic = True
    
# PATH
checkout_dir = "./checkout"
if os.path.exists(checkout_dir) is False:
    os.mkdir(checkout_dir)
    print("create ./checkout")


# In[4]:


# Model
n_classes = 3
device = torch.device("cuda" if use_gpu else "cpu")
model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", pretrained=True)
# model = models.resnet18(pretrained=True).to(device)
# model = models.shufflenet_v2_x0_5(pretrained=False).to(device)
n_filters = model.fc.in_features
model.fc = nn.Linear(n_filters, n_classes)

if use_gpu:
    model.cuda()
    print("mount model in CUDA")

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
use_scheduler= False
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Loss
criterion = nn.CrossEntropyLoss()


# In[9]:


# If you want to restart learning, set model_load as True
model_load = True
epoch_start = 10

if model_load:
    load_checkpoint(checkout_dir, epoch_start, model)


# In[10]:

# Process our image
def process_image(image_path):
    img = Image.open(image_path)

    width, height = img.size

    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))

    width, height = img.size

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))

    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img/255

    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225

    img = img[np.newaxis,:]
    image = torch.from_numpy(img)
    image = image.float()
    return image

# Using our model to predict the label
def predict(image, model):
    output = model.forward(image)
    output = torch.exp(output)
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

def show_image(image):
    image = image.numpy()
    image[0] = image[0] * 0.226 + 0.445
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))

image_class = {0: "GlassBottle", 1: "Can", 2: "PET"}

import RPi.GPIO as GPIO
import time

# Pin Definitions

def move(pos):
    pos = 11
    neg = 13
    rot = 15
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup((pos, neg, rot), GPIO.OUT)
    GPIO.output(pos, GPIO.HIGH)
    time.sleep(0.22*pos)
    GPIO.output(pos, GPIO.LOW)
    GPIO.cleanup()

import cv2
import os

DEVICE_NUM = 0
DIR_PATH = "images"

def process(image_dir):
    image = process_image(f"{image_dir}")
    top_prob, top_class = predict(image.to(device), model)
    print(f"The model is {top_prob*100}% certain that the image has a predicted class of {image_class[top_class]}")
    return top_prob, top_class

def save_frame_camera_key(device_num, dir_path, name, ext="jpg", delay=1, window_name="frame"):
    cap = cv2.VideoCapture(device_num)

    assert cap.isOpened(), f"Video Device dev/video{device_num} Not Found"

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, name)

    n = 0

    while True:
        ret, frame = cap.read()
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("c"):
            cv2.imwrite(f"{base_path}_{n}.{ext}", frame)
            prob, top_class = process(f"{base_path}_{n}.{ext}")
            move(top_class)
            n+=1
        elif key==ord("q"):
            break
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    save_frame_camera_key(DEVICE_NUM, DIR_PATH, "cap")
