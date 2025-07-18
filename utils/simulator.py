import pyautogui
import random
import time


class MouseDragSimulator:
    """
    鼠标点击拖动轨迹模拟
    """

    def __init__(self, duration=1.0, steps=50):
        self.duration = duration
        self.steps = steps

    def bezier_curve(self, p0, p1, p2, p3):
        """生成贝塞尔曲线路径"""
        points = []
        for i in range(self.steps + 1):
            t = i / self.steps
            x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
            y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
            points.append((x, y))
        return points

    def generate_path(self, start_pos, end_pos):
        """默认使用贝塞尔曲线生成路径"""
        x1, y1 = start_pos
        x4, y4 = end_pos
        offset = (x4 - x1) * 0.3
        p0 = (x1, y1)
        p1 = (x1 + offset, y1 + random.randint(-50, 50))
        p2 = (x4 - offset, y4 + random.randint(-50, 50))
        p3 = (x4, y4)
        return self.bezier_curve(p0, p1, p2, p3)

    def drag(self, start_pos, end_pos):
        path = self.generate_path(start_pos, end_pos)
        pyautogui.moveTo(*start_pos)
        pyautogui.mouseDown()
        interval = self.duration / self.steps
        for x, y in path[1:]:
            pyautogui.moveTo(x, y)
            time.sleep(interval + random.uniform(0, interval / 3))
        pyautogui.mouseUp()


if __name__ == "__main__":
    time.sleep(2)  # 等待两秒，用于切换到目标窗口/激活目标窗口
    simulator = MouseDragSimulator(duration=1.5, steps=60)
    simulator.drag((500, 500), (800, 500))
