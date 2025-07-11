from pywinauto.application import Application
from pywinauto.findwindows import find_elements

# 获取Windows当前所有开启窗口
elems = find_elements(backend="uia")
print(elems)

# 连接指定应用
app = Application(backend="uia").connect(title_re="Progress Telerik Fiddler Web Debugger")

# 获取当前主窗口
dlg = app.window(title_re="Progress Telerik Fiddler Web Debugger")

# 获取当前窗口所有控件
# dlg.print_control_identifiers()

# 点击指定的元素
# el = dlg.child_window(best_match="Capturing")
# el.click_input()

# 操作菜单栏
# 菜单栏下面有子菜单不能直接操作，需要使用子菜单的定位方式完成操作
# dlg.menu_select("File -> Export Sessions -> ALL Sessions...")
dlg.menu_select("File -> Export Sessions")

# 点击子菜单
menu_item = dlg.child_window(best_match="ALL Sessions...")
menu_item.click_input()

# 操作弹窗
# 操作组合框，并且点击下拉框中的选项
combo_box = dlg.child_window(best_match="ComboBox")
combo_box.chick_input()
select_item = dlg.child_window(best_match="WCAT Script")
select_item.click_input()

# 点击操作Button
button = dlg.child_window(best_match="Next")

# 文件对话框输入文件名
dlg.child_window(title_re="文件名:", control_type="Edit").set_text("export_sessions")

# 点击保存
dlg.child_window(title_re="保存").click_input()
