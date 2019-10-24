import torch
import cv2
from torchvision import models
import asyncio

from evaluation.camera import Camera
from evaluation.inference import InfModel
from evaluation.output_io import Conveyer


async def cam_action(image, frame, md, conv, cam):
    prob, pos = md.predict(image)
    conv.convey(pos.item()).drop_out(sec=0.7)
    cam.write_string(md.image_class[pos.item()])

if __name__ == "__main__":
    cam = Camera(0, "./images", "cap", fps=24)
    conv = Conveyer(11, 13, 15, 19)

    model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", pretrained=True)
    # model = models.resnet18(pretrained=True).to(device)
    # model = models.shufflenet_v2_x0_5(pretrained=False).to(device)
    n_filters = model.fc.in_features
    model.fc = torch.nn.Linear(n_filters, 3)

    md = InfModel("./checkout", "shufflenet", 10, model)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(cam.save_frame_with(lambda x, frame: cam_action(x, frame, md, conv, cam)))
    del conv
