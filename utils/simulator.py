import pyautogui
import random
import time


class MouseDragSimulator:
    """
    鼠标点击拖动轨迹模拟（贝塞尔曲线速率）
    """

    def __init__(self, duration=1.0, steps=50):
        self.duration = duration
        self.steps = steps
        # 贝塞尔曲线控制点（用于速率变化）
        self.control_points = [
            (0.0, 0.0),
            (random.uniform(0.2, 0.4), random.uniform(0.3, 0.7)),
            (random.uniform(0.6, 0.8), random.uniform(0.3, 0.7)),
            (1.0, 1.0)
        ]

    def bezier_easing(self, t):
        """计算贝塞尔曲线在t时刻的进度值（0-1之间）"""
        # 解压控制点
        p0, p1, p2, p3 = self.control_points

        # 三次贝塞尔曲线公式
        mt = 1 - t
        x = (mt ** 3 * p0[0] +
             3 * mt ** 2 * t * p1[0] +
             3 * mt * t ** 2 * p2[0] +
             t ** 3 * p3[0])

        y = (mt ** 3 * p0[1] +
             3 * mt ** 2 * t * p1[1] +
             3 * mt * t ** 2 * p2[1] +
             t ** 3 * p3[1])

        return y  # 返回y值作为进度百分比

    def generate_path(self, start_pos, end_pos):
        """生成直线路径，但移动速率按贝塞尔曲线变化"""
        points = []
        sx, sy = start_pos
        ex, ey = end_pos

        # 计算总位移
        dx = ex - sx
        dy = ey - sy

        # 生成路径点（直线）
        for i in range(self.steps + 1):
            t = i / self.steps
            # 获取贝塞尔曲线决定的进度
            progress = self.bezier_easing(t)
            # 计算当前位置（直线移动）
            x = sx + dx * progress
            y = sy + dy * progress
            points.append((x, y))

        return points

    def drag(self, start_pos, end_pos):
        path = self.generate_path(start_pos, end_pos)
        pyautogui.moveTo(*start_pos)
        pyautogui.mouseDown()

        interval = self.duration / self.steps
        for i in range(1, len(path)):
            # 添加随机延迟使移动更自然
            delay = interval * random.uniform(0.8, 1.2)
            pyautogui.moveTo(*path[i])
            time.sleep(delay)

        pyautogui.mouseUp()


if __name__ == "__main__":
    time.sleep(2)  # 等待两秒，用于切换到目标窗口/激活目标窗口
    simulator = MouseDragSimulator(duration=1.5, steps=60)
    simulator.drag((500, 500), (800, 500))
