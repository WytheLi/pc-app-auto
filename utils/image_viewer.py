import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QFileDialog, QSizePolicy, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageViewer(QMainWindow):
    """
    图片浏览器

    打包成exe程序，模拟滑块验证弹窗
    """
    def __init__(self, width=800, height=600):
        super().__init__()

        # 设置窗口尺寸
        self.setWindowTitle("图片显示器")
        # 创建中央部件和布局 - 无内边距
        central_widget = QWidget()
        central_widget.setContentsMargins(0, 0, 0, 20)  # 无内边距
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 20)  # 无内边距
        layout.setSpacing(0)  # 无间距

        # 创建固定尺寸的图片显示区域
        self.image_label = QLabel()
        self.image_label.setFixedSize(276, 172)  # 固定显示区域尺寸
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")

        # 添加到布局
        layout.addWidget(self.image_label)

        # 创建菜单栏
        self.create_menu()

        # 调整窗口大小以刚好容纳图片区域（加上菜单栏高度）
        menu_height = self.menuBar().sizeHint().height()
        self.setFixedSize(276, 172 + menu_height + 20)

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
        image = QImage(file_path)

        if not image.isNull():
            # 缩放图片以适应固定尺寸的显示区域
            pixmap = QPixmap.fromImage(image)

            # 保持宽高比缩放
            scaled_pixmap = pixmap.scaled(
                276, 172,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)
        else:
            # 清空图片并显示错误信息
            self.image_label.clear()
            self.image_label.setText("无法加载图片")
            self.image_label.setStyleSheet("background-color: #f0f0f0; color: red; font: bold;")


if __name__ == "__main__":
    # 确保高DPI屏幕适配
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
