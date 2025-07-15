import time

from pywinauto import Application, mouse

DOUYIN_APP_PATH = r"D:\Program Files (x86)\webcast_mate\10.2.2.174153822\直播伴侣.exe"

try:
    app = Application(backend="uia").connect(title_re="直播伴侣")
except Exception:
    app = Application(backend="uia").start(DOUYIN_APP_PATH)
    time.sleep(5)  # 等待应用启动加载

dlg = app.window(class_name="Chrome_WidgetWin_1", found_index=0)

dlg.child_window(title="窗口", control_type="Image").click_input()

document = dlg.child_window(title="请选择", control_type="Edit")
# print(document.window_text())
document.click_input()

# 下拉框实际上是一个Document控件（而非传统的List或ComboBox），项只在滚动时动态生成
# 获取文本框的矩形区域
rect = document.rectangle()
# 计算中心点坐标
center_x = (rect.left + rect.right) // 2
center_y = (rect.top + rect.bottom) // 2 + 50
mouse.scroll(coords=(center_x, center_y), wheel_dist=-3)  # 向下滚动
