import argparse

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from utils.simulator import MouseDragSimulator
from utils.window_manager import WindowManager


class SliderVerification:
    """
    滑块认证

    1. 获取滑块图、背景图，web端可以解析网页源码获取图片URL（Selenium定位并裁剪）；PC/App/小程序通过抓包获取
    2. opencv识别计算滑块和阴影位中心点坐标
    3. pyautogui模拟用户点击拖动鼠标，完成认证
    """

    def __init__(self, debug=False):
        """
        初始化滑块验证识别器

        参数:
            debug: 是否启用调试模式（显示处理过程）
        """
        self.debug = debug

    def load_image(self, image_path):
        """
        加载图像文件，支持多种格式

        参数:
            image_path: 图像文件路径

        返回:
            image: OpenCV图像对象
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"文件不存在: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            # 尝试其他读取方式
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError(f"无法解码图像文件: {image_path}")

        print(f"成功加载图像: {os.path.basename(image_path)}, 尺寸: {image.shape[1]}x{image.shape[0]}")

        # 处理透明通道
        if image.shape[2] == 4:
            # 分离透明通道
            alpha = image[:, :, 3]
            _, mask = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)
            # 转换为BGR三通道
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image, mask
        else:
            return image, None

    def detect_gap_with_slider(self, background, slider_image, slider_mask=None):
        """
        使用滑块图像检测背景图中的阴影坑位中心点

        参数:
            background: 背景图像 (BGR格式)
            slider_image: 滑块图像 (BGR格式)
            slider_mask: 滑块掩码 (可选)

        返回:
            center: 阴影坑位中心点坐标 (x, y)
            confidence: 检测置信度 (0-1)
        """
        try:
            # 方法1: 模板匹配
            print("尝试模板匹配方法...")
            bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            slider_gray = cv2.cvtColor(slider_image, cv2.COLOR_BGR2GRAY)

            # 如果滑块掩码可用，使用掩码进行匹配
            if slider_mask is not None:
                print("使用滑块掩码进行模板匹配")
                result = cv2.matchTemplate(bg_gray, slider_gray, cv2.TM_CCORR_NORMED, mask=slider_mask)
            else:
                result = cv2.matchTemplate(bg_gray, slider_gray, cv2.TM_CCOEFF_NORMED)

            # 找到最佳匹配位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 如果使用TM_CCORR_NORMED，最大值为最佳匹配
            if max_val > 0.5:  # 设置置信度阈值
                top_left = max_loc
                w, h = slider_gray.shape[::-1]
                center = (top_left[0] + w // 2, top_left[1] + h // 2)
                print(f"模板匹配成功: 置信度 {max_val:.2f}, 中心点 {center}")
                return center, max_val

            # 方法2: 特征匹配
            print("模板匹配置信度低，尝试特征匹配方法...")
            # 创建ORB特征检测器
            orb = cv2.ORB_create(nfeatures=500)

            # 检测关键点和计算描述符
            kp1, des1 = orb.detectAndCompute(slider_gray, None)
            kp2, des2 = orb.detectAndCompute(bg_gray, None)

            if des1 is None or des2 is None:
                raise ValueError("无法计算图像特征描述符")

            # 创建BFMatcher对象
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # 匹配描述符
            matches = bf.match(des1, des2)

            # 按距离排序
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 10:
                raise ValueError("匹配的特征点太少")

            # 计算匹配点的位置
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # 使用RANSAC计算单应性矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                raise ValueError("无法计算单应性矩阵")

            # 计算滑块在背景图中的位置
            h, w = slider_gray.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # 计算中心点
            center_x = int(np.mean(dst[:, :, 0]))
            center_y = int(np.mean(dst[:, :, 1]))
            center = (center_x, center_y)

            # 计算置信度 (匹配点数量和质量)
            inlier_ratio = np.sum(mask) / len(mask)
            confidence = min(1.0, inlier_ratio * len(matches) / 100)

            print(f"特征匹配成功: 置信度 {confidence:.2f}, 中心点 {center}")
            return center, confidence

        except Exception as e:
            print(f"结合滑块检测坑位失败: {str(e)}")
            # 方法3: 边缘检测与轮廓分析 (备选方案)
            print("尝试边缘检测方法...")
            try:
                # 创建边缘图像
                bg_edges = cv2.Canny(bg_gray, 50, 150)
                slider_edges = cv2.Canny(slider_gray, 50, 150)

                # 使用滑块边缘作为模板
                result = cv2.matchTemplate(bg_edges, slider_edges, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > 0.3:
                    top_left = max_loc
                    w, h = slider_edges.shape[::-1]
                    center = (top_left[0] + w // 2, top_left[1] + h // 2)
                    print(f"边缘匹配成功: 置信度 {max_val:.2f}, 中心点 {center}")
                    return center, max_val

                # 最终备选: 使用简单的轮廓分析
                contours, _ = cv2.findContours(bg_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 按面积排序轮廓
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    # 取最大轮廓
                    x, y, w, h = cv2.boundingRect(contours[0])
                    center = (x + w // 2, y + h // 2)
                    confidence = 0.4
                    print(f"轮廓分析得到坑位中心: {center} (置信度: {confidence:.2f})")
                    return center, confidence

                raise ValueError("所有方法均无法检测坑位")
            except Exception as e2:
                print(f"备选方法失败: {str(e2)}")
                return None, 0.0

    def detect_slider_center(self, slider_image):
        """
        检测滑块图中的滑块中心点

        参数:
            slider_image: 滑块图像 (BGR格式)

        返回:
            center: 滑块中心点坐标 (x, y)
            confidence: 检测置信度 (0-1)
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(slider_image, cv2.COLOR_BGR2GRAY)

            # 二值化处理
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                # 尝试直接使用中心点
                h, w = slider_image.shape[:2]
                center = (w // 2, h // 2)
                confidence = 0.5
                print(f"未检测到轮廓，使用图像中心: {center}")
            else:
                # 找到最大轮廓
                slider_contour = max(contours, key=cv2.contourArea)

                # 计算轮廓的矩
                M = cv2.moments(slider_contour)
                if M['m00'] == 0:
                    # 使用边界矩形中心
                    x, y, w, h = cv2.boundingRect(slider_contour)
                    center = (x + w // 2, y + h // 2)
                    confidence = 0.7
                else:
                    # 计算质心
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    center = (cx, cy)

                    # 计算置信度
                    img_area = slider_image.shape[0] * slider_image.shape[1]
                    contour_area = cv2.contourArea(slider_contour)
                    confidence = min(1.0, contour_area / img_area)

                print(f"检测到滑块中心: {center}, 置信度: {confidence:.2f}")

            return center, confidence

        except Exception as e:
            print(f"滑块中心检测失败: {str(e)}")
            return None, 0.0

    def calculate_slide_distance(self, gap_center, slider_center):
        """
        计算滑动距离

        参数:
            gap_center: 阴影坑位中心点 (x, y)
            slider_center: 滑块中心点 (x, y)

        返回:
            distance: 需要滑动的水平距离 (像素)
        """
        if gap_center is None or slider_center is None:
            raise ValueError("无法计算距离 - 中心点坐标无效")

        # 计算水平距离
        distance = gap_center[0] - slider_center[0]

        print(f"计算滑动距离: {distance} 像素")
        return distance

    def plot_trajectory(self, background, slider_image, gap_center, slider_center, distance, slider_y=None):
        """
        移动轨迹可视化绘制
        :param background: 背景图片
        :param slider_image: 滑块图片
        :param gap_center: 识别的缺口中心
        :param slider_center: 滑块中心
        :param distance: 滑动距离
        :param slider_y: 滑块在背景图上的垂直位置 (None表示自动居中)
        :return:
        """
        # 创建可视化图像副本
        vis = background.copy()

        # 获取图像尺寸
        slider_height, slider_width = slider_image.shape[:2]
        bg_height, bg_width = background.shape[:2]

        # 计算滑块位置 (水平位置固定在左侧)
        if slider_y is None:
            # 默认垂直居中
            slider_y = (bg_height - slider_height) // 2
        else:
            # 确保滑块不会超出背景图边界
            if slider_y < 0:
                slider_y = 0
            elif slider_y + slider_height > bg_height:
                slider_y = bg_height - slider_height

        slider_pos = (0, slider_y)

        # 绘制滑块位置标记
        # cv2.rectangle(vis, (0, slider_y), (10, slider_y + slider_height), (0, 255, 0), 2)
        # cv2.putText(vis, f"滑块位置: y={slider_y}",
        #             (15, slider_y + 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 将滑块叠加到背景图上（处理透明通道）
        if slider_image.shape[2] == 4:  # 如果有透明通道
            # 分离滑块图像和透明通道
            slider_rgb = slider_image[:, :, :3]
            alpha_channel = slider_image[:, :, 3] / 255.0

            # 提取背景图的相应区域
            x1, y1 = slider_pos
            x2, y2 = x1 + slider_width, y1 + slider_height
            roi = vis[y1:y2, x1:x2]

            # 创建逆透明通道
            alpha_inv = 1.0 - alpha_channel

            # 为每个通道应用透明混合
            for c in range(0, 3):
                roi[:, :, c] = (alpha_channel * slider_rgb[:, :, c] +
                                alpha_inv * roi[:, :, c])

            # 更新背景图的相应区域
            vis[y1:y2, x1:x2] = roi
        else:
            # 没有透明通道，直接覆盖滑块
            x1, y1 = slider_pos
            x2, y2 = x1 + slider_width, y1 + slider_height
            vis[y1:y2, x1:x2] = slider_image

        # 绘制滑块中心点（在背景图坐标系中的位置）
        slider_vis_center = (slider_center[0] + slider_pos[0],
                             slider_center[1] + slider_pos[1])

        # 绘制滑块中心点
        cv2.circle(vis, slider_vis_center, 8, (0, 255, 0), -1)
        cv2.circle(vis, slider_vis_center, 10, (0, 100, 0), 2)
        # cv2.putText(vis, "Slider center",
        #             (slider_vis_center[0] + 15, slider_vis_center[1]),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制坑位中心点
        if gap_center:
            cv2.circle(vis, gap_center, 10, (0, 0, 255), -1)
            cv2.circle(vis, gap_center, 12, (0, 0, 100), 2)
            # cv2.putText(vis, "Gap center",
            #             (gap_center[0] + 15, gap_center[1]),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 绘制滑动路径
            cv2.line(vis, slider_vis_center, gap_center,
                     (255, 0, 0), 3, cv2.LINE_AA)

            # 绘制距离标注
            mid_x = (slider_vis_center[0] + gap_center[0]) // 2
            mid_y = (slider_vis_center[1] + gap_center[1]) // 2

            # 绘制距离标注框
            text = f"Sliding distance: {abs(distance):.1f}px"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 2)[0]
            text_x = mid_x - text_size[0] // 2
            text_y = mid_y - 10

            # 绘制半透明背景
            overlay = vis.copy()
            cv2.rectangle(overlay,
                          (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10),
                          (50, 50, 50), -1)

            # 应用半透明效果
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

            # 绘制文本
            cv2.putText(vis, text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 绘制方向指示箭头
            if distance > 0:
                cv2.arrowedLine(vis,
                                (text_x + text_size[0] + 15, mid_y),
                                (text_x + text_size[0] + 45, mid_y),
                                (255, 200, 0), 2, tipLength=0.3)
            else:
                cv2.arrowedLine(vis,
                                (text_x - 15, mid_y),
                                (text_x - 45, mid_y),
                                (255, 200, 0), 2, tipLength=0.3)

        # # 添加标题和说明
        # cv2.putText(vis, "滑块验证分析结果",
        #             (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        # cv2.putText(vis, "滑块验证分析结果",
        #             (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # 添加位置信息
        position_info = f"Slider position: y={slider_y} (height: {slider_height}px)"
        cv2.putText(vis, position_info,
                    (bg_width - 400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 2)

        # 添加图例
        legend_y = bg_height - 30
        cv2.circle(vis, (20, legend_y), 8, (0, 255, 0), -1)
        cv2.putText(vis, "Slider center", (40, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        cv2.circle(vis, (120, legend_y), 8, (0, 0, 255), -1)
        cv2.putText(vis, "Gap center", (140, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        cv2.line(vis, (220, legend_y), (250, legend_y), (255, 0, 0), 2)
        cv2.putText(vis, "Sliding distance", (260, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # 显示结果
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Slider verification analysis result")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        return vis  # 返回可视化图像

    def process_verification(self, background_path, slider_path, slider_y):
        """
        完整的滑块验证处理流程

        参数:
            background_path: 背景图路径
            slider_path: 滑块图路径

        返回:
            result: 包含所有结果的字典
        """
        result = {
            'gap_center': None,
            'slider_center': None,
            'distance': None
        }

        try:
            # 1. 加载图像
            background, _ = self.load_image(background_path)
            slider_img, slider_mask = self.load_image(slider_path)

            # 2. 使用滑块检测阴影坑位中心
            gap_center, gap_confidence = self.detect_gap_with_slider(background, slider_img, slider_mask)
            result['gap_center'] = gap_center
            result['gap_confidence'] = gap_confidence

            # 3. 检测滑块中心
            slider_center, slider_confidence = self.detect_slider_center(slider_img)
            result['slider_center'] = slider_center
            result['slider_confidence'] = slider_confidence

            # 4. 计算滑动距离
            if gap_center and slider_center:
                distance = self.calculate_slide_distance(gap_center, slider_center)
                result['distance'] = distance

                # 可视化结果
                if self.debug:
                    self.plot_trajectory(background, slider_img, gap_center, slider_center, distance, slider_y * 2)

        except Exception as e:
            print(f"处理失败: {str(e)}")

        return result


def main():
    parser = argparse.ArgumentParser(description='抖音滑块认证工具')
    parser.add_argument('--background', required=True, help='背景图片')
    parser.add_argument('--slider', required=True, help='滑块图片')
    parser.add_argument('--slider_y', help='滑块y轴坐标')    # 原图和弹窗显示图片缩放比例为2倍

    args = parser.parse_args()

    recognizer = SliderVerification(debug=True)

    background_path = args.background
    slider_path = args.slider
    slider_y = args.slider_y and int(args.slider_y)

    # 执行滑块验证处理
    result = recognizer.process_verification(background_path, slider_path, slider_y)

    # 打印结果
    print("\n处理结果:")
    print(f"阴影坑位中心: {result.get('gap_center', 'N/A')}")
    print(f"滑块中心: {result.get('slider_center', 'N/A')}")
    print(f"滑动距离: {result.get('distance', 'N/A')} 像素")


    wm = WindowManager(title="图片显示器")
    window_info = wm.get_window_info()

    # 获取滑块拖动按钮的位置（原图和弹窗显示图片缩放比例为2倍）
    slider_center_x, slider_center_y = result.get('slider_center', (0, 0))
    gap_center_x, gap_center_y = result.get('gap_center', (0, 0))

    # slider_pos_y = (slider_center_y // 2) + slider_y + window_info.y + window_info.title_height
    slider_pos_y = (344 // 2) + window_info.y + window_info.title_height
    slider_pos_x = (slider_center_x // 2) + window_info.x

    # gap_pos_y = (gap_center_y // 2) + window_info.y + window_info.title_height
    gap_pos_y = (344 // 2) + window_info.y + window_info.title_height
    gap_pos_x = (gap_center_x // 2) + window_info.x

    # 显示窗口
    wm.activate()

    # pyautogui模拟用户点击拖动鼠标，完成认证
    simulator = MouseDragSimulator(duration=1.5, steps=60)
    print(f"点击拖动起始点位置：{(slider_pos_x, slider_pos_y), (gap_pos_x, gap_pos_y)}")
    simulator.drag((slider_pos_x, slider_pos_y), (gap_pos_x, gap_pos_y))


if __name__ == "__main__":
    """
    抖音pc端直播伴侣滑块验证功能实现
    1. 基于抓包获取到的滑块、背景图片；
    2. opencv进行轮廓比对计算，分别获取滑块和缺口在背景图中坐标、滑动的横向距离；
    3. 结合win32gui模块截图指定滑块窗口，计算在windows窗口中的坐标；
    4. 使用pyautogui自动化工具模拟用户点击拖动滑块
    
    python main.py --background resources/background/bg0.png --slider resources/slider/cut0.png --slider_y 172
    """
    main()
