import argparse

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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

    def visualize_moving_track(self, background, slider_image, gap_center, slider_center, distance):
        """
        移动轨迹可视化
        :param background:
        :param slider_image:
        :param gap_center:
        :param slider_center:
        :param distance:
        :return:
        """
        # 创建可视化图像
        vis = background.copy()

        # 绘制坑位中心
        if gap_center:
            cv2.circle(vis, gap_center, 10, (0, 0, 255), -1)
            cv2.putText(vis, "Gap Center", (gap_center[0] + 15, gap_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制滑块位置和中心
        if slider_center:
            # 假设滑块位于左侧边缘
            slider_height = slider_image.shape[0]
            y_offset = (background.shape[0] - slider_height) // 2
            slider_pos = (0, y_offset)

            # 绘制滑块轮廓
            cv2.rectangle(vis,
                          slider_pos,
                          (slider_pos[0] + slider_image.shape[1], slider_pos[1] + slider_height),
                          (0, 255, 0), 2)

            # 绘制滑块中心点
            slider_vis_center = (slider_center[0] + slider_pos[0],
                                 slider_center[1] + slider_pos[1])
            cv2.circle(vis, slider_vis_center, 5, (0, 255, 0), -1)
            cv2.putText(vis, "Slider Center", (slider_vis_center[0] + 15, slider_vis_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 绘制距离线
            if gap_center:
                cv2.line(vis, slider_vis_center, gap_center,
                         (255, 0, 0), 2, cv2.LINE_AA)
                mid_point = ((slider_vis_center[0] + gap_center[0]) // 2,
                             (slider_vis_center[1] + gap_center[1]) // 2)
                cv2.putText(vis, f"Distance: {distance}px",
                            (mid_point[0], mid_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 显示结果
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        plt.title('Original background image'), plt.axis('off')
        plt.subplot(122), plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title('Detected center and moving track'), plt.axis('off')
        plt.tight_layout()
        plt.show()

    def process_verification(self, background_path, slider_path):
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
                    self.visualize_moving_track(background, slider_img, gap_center, slider_center, distance)

        except Exception as e:
            print(f"处理失败: {str(e)}")

        return result


def main():
    parser = argparse.ArgumentParser(description='抖音滑块认证工具')
    parser.add_argument('--background', required=True, help='背景图片')
    parser.add_argument('--slider', required=True, help='滑块图片')
    parser.add_argument('--slider_y', required=True, help='滑块y轴坐标')

    args = parser.parse_args()

    recognizer = SliderVerification(debug=True)

    background_path = args.background
    slider_path = args.slider
    slider_y = int(args.slider_y)

    # 执行滑块验证处理
    result = recognizer.process_verification(background_path, slider_path)

    # 打印结果
    print("\n处理结果:")
    print(f"阴影坑位中心: {result.get('gap_center', 'N/A')}")
    print(f"滑块中心: {result.get('slider_center', 'N/A')}")
    print(f"滑动距离: {result.get('distance', 'N/A')} 像素")

    # 获取滑块拖动按钮的位置

    # pyautogui模拟用户点击拖动鼠标，完成认证

    """
接下来需要完成：
1. 获取滑块拖动按钮的位置（我的思路是截图滑块验证的窗口，拖动按钮在窗口位置是固定的，我可以获取拖动按钮的坐标）
2. pyautogui模拟用户点击（点击的坐标要考虑拖动按钮在整个窗口的坐标）拖动鼠标，完成认证

请结合以上需求，用代码实现

visualize_results()绘制结果中，把slider_image作为图层叠加进去一同绘制应该怎么处理
    """


if __name__ == "__main__":
    """
    python main.py --background resources/background/bg0.png --slider resources/slider/cut0.png --slider_y 170
    """
    main()
