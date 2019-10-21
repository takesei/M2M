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

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import random
import os
import time


# In[2]:


# Define save/load func
def save_checkpoint(path, epoch, model):
    save_path = os.path.join(path, f"mobilenet_epoch_{epoch}.pkl")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model_dir, epoch, model):
    load_path = os.path.join(model_dir, f"mobilenet_epoch_{epoch}.pkl")
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
model = models.mobilenet_v2(pretrained=True).to(device)
n_filters = model.classifier[1].in_features
model.classifier[1] = nn.Linear(n_filters, n_classes)

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


# In[5]:


# Dataset
data_transform = transforms.Compose([
    transforms.Resize([224, 224]), transforms.RandomHorizontalFlip()
    , transforms.ToTensor()
    , transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

full_dataset = datasets.ImageFolder(root="trashes", transform=data_transform)

dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

print(f"Number of Dataset: {dataset_size}")
print(f"Number of Train Dataset: {len(train_dataset)}")
print(f"Number of Test Dataset: {len(test_dataset)}")
print(f"epoch: {epochs}")
print(f"batchsize: {batchsize}")


# In[6]:


# Train
def train(model, train_loader, epoch):
    model.train()
    print(f"\nEpoch: {epoch}")
    train_loss = 0
    correct = 0
    total = 0
    
    if use_scheduler:
        scheduler.step()
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    print(f"Train Loss:{train_loss/(batch_idx+1)} | Acc:{100.*correct/total} ({correct}/{total})")
    return train_loss, 100.*correct/total


# In[7]:


# Test
def test(model, test_loader, epoch):
    model.eval()
    running_loss = 0
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (image, label) in enumerate(test_loader):
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        loss = criterion(outputs, label)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    print(f"Test Loss:{test_loss/batch_idx+1} | Acc:{100.*correct/total} ({correct}/{total})")
    return test_loss, 100.*correct/total


# In[8]:


# Evaluation
def evaluation(model_dir, epoch, model, test_loader):
    print("\nEvaluation")
    load_checkpoint(model_dir, epoch, model)
    model.eval()
    y_test = []
    y_pred = []
    for batch_idx, (image, label) in enumerate(test_loader):
        image, label = image.to(device), label.to(device)
        outpus = model(image)
        _, predictions = outpus.max(1)
        y_test.append(label.data.cpu().numpy())
        y_pred.append(predictions.data.cpu().numpy())
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}, confusion matrix: \n{confusion_mat}")


# In[9]:


# If you want to restart learning, set model_load as True
model_load = False
epoch_start = 0

if model_load:
    load_checkpoint(checkout_dir, epoch_start, model)


# In[10]:


# Main
if __name__ == "__main__":
    summary(model, (3, 244, 244))

    train_loss_log = []
    test_loss_log = []
    train_acc_log = []
    test_acc_log = []

    for epoch in range(epoch_start, epochs + 1):
        train_loss, train_acc = train(model, train_loader, epoch)
        test_loss, test_acc = test(model, test_loader, epoch)
        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)
        train_acc_log.append(train_acc)
        test_acc_log.append(test_acc)
        save_checkpoint(checkout_dir, epoch, model)
    evaluation(checkout_dir, epochs, model, test_loader)

