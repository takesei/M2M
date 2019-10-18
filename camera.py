import cv2
import os

DEVICE_NUM = 0
DIR_PATH = "images"

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
            n+=1
        elif key==ord("q"):
            break
    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    save_frame_camera_key(DEVICE_NUM, DIR_PATH, "cap")
