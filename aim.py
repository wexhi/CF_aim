from torch.cuda import device
from ultralytics import YOLO
import pygetwindow
import pyautogui as pt
import numpy as np
import cv2 as cv
import torch
from PIL import ImageGrab
import keyboard
from mouseinfo import screenshot
import win32api
import win32con
import time
import ctypes

# 初始化参数
cofig_thread = 0.75  # 置信度阈值
aim_speed = 0.8  # 瞄准速度


# 获取窗口句柄
window_title = "穿越火线"
window = pygetwindow.getWindowsWithTitle(window_title)[0]

classNames = ["body", "head"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("./models/best _cf.pt")
model = model.to(device)

# 按键开关状态变量
mouse_move_enabled = False
locked_target = None  # 用于锁定目标的变量


# 切换开关状态的函数
def toggle_mouse_move():
    global mouse_move_enabled
    mouse_move_enabled = not mouse_move_enabled
    print(f"Mouse move enabled: {mouse_move_enabled}")


# 注册按键监听器，按下'z'键时切换状态
keyboard.add_hotkey("z", toggle_mouse_move)

# 定义鼠标事件
MOUSEEVENTF_MOVE = 0x0001


# 函数来移动鼠标
def move_mouse(x, y):
    x, y = int(x), int(y)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, x, y, 0, 0)


def move_mouse_relative(x_target, y_target, x_current, y_current, speed=1):
    delta_x = int((x_target - x_current) * speed)
    delta_y = int((y_target - y_current) * speed)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, delta_x, delta_y, 0, 0)


while True:
    if window:
        win_x, win_y, win_w, win_h = (
            window.left,
            window.top,
            window.width,
            window.height,
        )
        screenshot = ImageGrab.grab(bbox=(win_x, win_y, win_x + win_w, win_y + win_h))
        img_src = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        win_size_x, win_size_y = img_src.shape[1], img_src.shape[0]
        img_det = cv.resize(img_src, (640, 640))
        results = model.predict(img_det, conf=cofig_thread, imgsz=640, verbose=False)

        current_mouse_x, current_mouse_y = pt.position()
        min_distance = float("inf")
        target_x, target_y = None, None
        found_locked_target = False  # 用于检测锁定的目标是否存在

        for result in results:
            boxes = result.boxes.xywhn.cpu().numpy()  # 获取每个框的坐标
            classes = result.boxes.cls.cpu().numpy()  # 获取每个框的类别索引
            for box, cls in zip(boxes, classes):
                x_center, y_center = box[0] * win_w, box[1] * win_h
                distance = np.sqrt(
                    (x_center - current_mouse_x) ** 2
                    + (y_center - current_mouse_y) ** 2
                )

                if locked_target is not None:
                    locked_target_x, locked_target_y = locked_target
                    if (
                        np.sqrt(
                            (locked_target_x - x_center) ** 2
                            + (locked_target_y - y_center) ** 2
                        )
                        < 50
                    ):  # 阈值可以根据实际情况调整
                        target_x, target_y = locked_target_x, locked_target_y
                        found_locked_target = True
                        break

                if distance < min_distance:
                    min_distance = distance
                    target_x = x_center + win_x - box[2] * win_w / 4.9
                    target_y = y_center + win_y - box[3] * win_h / 1.82
                    locked_target = (target_x, target_y)
                    target_label = classNames[int(cls)]

        if not found_locked_target:
            locked_target = None

        if mouse_move_enabled and target_x is not None and target_y is not None:
            if target_label == "body":
                move_mouse_relative(
                    target_x,
                    target_y,
                    current_mouse_x,
                    current_mouse_y,
                    speed=aim_speed,
                )

        # 绘制矩形框
        for result in results:
            boxes = result.boxes.xywhn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = (
                    (box[0] - box[2] / 2) * win_w,
                    (box[1] - box[3] / 2) * win_h,
                    (box[0] + box[2] / 2) * win_w,
                    (box[1] + box[3] / 2) * win_h,
                )
                if cls == 0:  # 如果是人体
                    cv.rectangle(
                        img_src, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                    )
                elif cls == 1:  # 如果是头部
                    cv.rectangle(
                        img_src, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )

        status_text = f"Mouse move enabled: {mouse_move_enabled}"
        cv.putText(
            img_src,
            f"Moving mouse to position: ({target_x}, {target_y})",
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv.putText(
            img_src,
            f"Current mouse position: {win32api.GetCursorPos()}",
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv.putText(
            img_src, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
        )
        cv.imshow("window", img_src)
        if cv.waitKey(1) == ord("q"):
            break
