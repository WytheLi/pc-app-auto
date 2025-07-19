import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageViewer(QMainWindow):
    """
    图片浏览器

    模拟滑块验证弹窗
    """
    def __init__(self, width=800, height=600):
        super().__init__()

        # 设置窗口尺寸
        self.setWindowTitle("图片显示器")
        self.setGeometry(100, 100, width, height)

        # 创建中央部件
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setCentralWidget(self.image_label)

        # 创建菜单栏
        self.create_menu()

        # 初始状态
        self.current_image = None

    def create_menu(self):
        # 文件菜单
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")

        # 打开动作
        open_action = QAction("打开图片", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        # 退出动作
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def open_image(self):
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        # 加载图片
        self.current_image = QImage(file_path)

        if not self.current_image.isNull():
            # 缩放图片以适应窗口
            scaled_pixmap = QPixmap.fromImage(self.current_image).scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("无法加载图片")

    def resizeEvent(self, event):
        # 窗口大小改变时重新缩放图片
        if self.current_image and not self.current_image.isNull():
            scaled_pixmap = QPixmap.fromImage(self.current_image).scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)


if __name__ == "__main__":
    # 设置窗口尺寸 (宽x高)
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 700

    app = QApplication(sys.argv)
    viewer = ImageViewer(WINDOW_WIDTH, WINDOW_HEIGHT)
    viewer.show()
    sys.exit(app.exec_())
