import cv2
import os
from logging import getLogger, StreamHandler, DEBUG
import asyncio

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

DEVICE_NUM = 0
DIR_PATH = "images"


class Camera():

    def __init__(self, device_num, dir_path, name, ext="jpg", delay=1, window_name="frame", fps=10):
        os.makedirs(dir_path, exist_ok=True)
        self.cap = cv2.VideoCapture(device_num)
        self.base_path = os.path.join(dir_path, name)
        self.n = 0
        assert self.cap.isOpened(), f"Video Device dev/video{device_num} Not Found"

        self.name = name
        self.ext = ext
        self.delay = delay
        self.window_name = window_name
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.string = ""

    async def save_frame_with(self, action=lambda x, frame: print("No Action registered"), shutter="c", quit="q"):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while True:
            ret, frame = self.cap.read()
            cv2.putText(frame, self.string, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(self.delay) & 0xFF
            if key == ord(shutter):
                file_name = f"{self.base_path}_{self.n}.{self.ext}"
                cv2.imwrite(file_name, frame)
                logger.debug(f"Image saved as {file_name}")
                self.n += 1
                await action(file_name, frame)
            elif key == ord(quit):
                logger.debug("Camera Quit")
                break
        cv2.destroyWindow(self.window_name)

    def write_string(self, string):
        self.string = string


if __name__ == "__main__":
    cam = Camera(DEVICE_NUM, DIR_PATH, "cap")
    cam.save_frame_with()
