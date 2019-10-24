import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


# load func
def load_checkpoint(model_dir, name, epoch, model):
    load_path = os.path.join(model_dir, f"{name}_epoch_{epoch}.pkl")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint)
    logger.debug(f"Checkpoint loaded from {load_path}")


# Process image
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

    img = img[np.newaxis, :]
    image = torch.from_numpy(img)
    image = image.float()
    return image


def show_image(image):
    image = image.numpy()
    image[0] = image[0] * 0.226 + 0.445
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))


class InfModel():

    def __init__(self, model_dir, model_name,  epoch_start, model, image_class={0: "GlassBottle", 1: "Can", 2: "PET"}):
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.model = model.to(self.device)
        self.image_class = image_class
        if self.use_gpu:
            logger.debug("CUDA detected")
            cudnn.benchmark = True
            cudnn.deterministic = True
            self.model.cuda()
            logger.debug("mount model on CUDA")
        load_checkpoint(model_dir, model_name, epoch_start, self.model)

    def predict(self, image_path):
        image = process_image(image_path)
        output = self.model.forward(image.to(self.device))
        output = torch.exp(output)
        probs, classes = output.topk(1, dim=1)
        top_prob, top_class = probs.item(), classes.item()
        logger.debug(f"The model is {top_prob*100}% certain that the image has a predicted class of {self.image_class[top_class]}")
        return probs, classes


if __name__ == "__main__":
    device = torch.device("cuda")
    model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", pretrained=True)
    # model = models.resnet18(pretrained=True).to(device)
    # model = models.shufflenet_v2_x0_5(pretrained=False).to(device)
    n_filters = model.fc.in_features
    model.fc = nn.Linear(n_filters, 3)

    md = InfModel("../checkout", "shufflenet", 10, model)
    md.predict("../images/cap_0.jpg")
