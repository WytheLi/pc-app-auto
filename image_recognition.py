"""
这里使用图片识别，提供另外一种自动化方案思路
"""

import time

import cv2
import numpy as np
import pyautogui
import win32con
import win32gui
from PIL import ImageGrab

from pywinauto import Application


def top_level_application(window_title):
    """将指定标题的窗口置为前台"""
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd:
        # 如果窗口最小化则恢复
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        # 将窗口置前
        win32gui.SetForegroundWindow(hwnd)
        return True
    return False


def find_template_on_screen(template_path, threshold=0.8):
    """
    在屏幕上查找模板图像
    :param template_path: 模板图片路径
    :param threshold: 匹配阈值 (0-1)
    :return: 匹配区域的中心坐标 (x, y)，未找到返回None
    """
    # 读取模板图像
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"模板图片未找到: {template_path}")

    # 获取屏幕截图
    screenshot = ImageGrab.grab()
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # 模板匹配
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 检查匹配结果
    if max_val < threshold:
        return None

    # 计算中心坐标
    h, w = template.shape[:2]
    top_left = max_loc
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2

    return center_x, center_y


def auto_click_button(template_path, threshold=0.8, click_delay=0.5):
    """
    自动查找并点击屏幕上的按钮
    :param template_path: 模板图片路径
    :param threshold: 匹配阈值
    :param click_delay: 点击前延迟(秒)
    """
    location = find_template_on_screen(template_path, threshold)

    if location:
        print(f"找到按钮，坐标: {location}")
        time.sleep(click_delay)  # 点击前等待
        pyautogui.click(location[0], location[1])
        print("点击完成")
        return True
    else:
        print("未找到按钮")
        return False


DOUYIN_APP_PATH = r"D:\Program Files (x86)\webcast_mate\10.2.2.174153822\直播伴侣.exe"

app = Application(backend="uia").start(DOUYIN_APP_PATH)
time.sleep(5)  # 等待应用启动加载

# top_level_application("直播伴侣")

# 设置参数
TEMPLATE_IMAGE = "resources/launch-window.png"  # 替换为你的模板图片路径
CONFIDENCE_THRESHOLD = 0.8  # 匹配置信度阈值

# 执行自动点击
auto_click_button(TEMPLATE_IMAGE, CONFIDENCE_THRESHOLD)
