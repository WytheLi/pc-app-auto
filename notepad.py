import time
from pywinauto import Application

app = Application(backend="win32").start("notepad.exe")

# 获取关于窗口信息
app.Notepad.type_keys("%HA", pause=1)   # Alt+H+A
dlg = app.window(title_re="关于")
dlg["确认Button"].click()

# 输入文本
app.Notepad.type_keys("自动化测试成功！{ENTER}第二行")

# 保存文件
app.Notepad.menu_select("文件(F)->另存为(A)...")
time.sleep(2)
save_dlg = app.window(title="另存为")

# 精确输入文件名（使用中文冒号）
save_dlg.window(title_re="文件名").type_keys("test.txt")

# 点击保存按钮
save_dlg.window(title_re="保存", control_type="Button").click()

# 处理覆盖确认
if app.window(title="确认另存为").exists(timeout=3):
    app.window(title="确认另存为").child_window(title="是(Y)").click()
