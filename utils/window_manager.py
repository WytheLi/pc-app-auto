import ctypes
from typing import Dict, Any

import win32api
import win32gui
import win32con
import win32ui
import pygetwindow as gw
from PIL import Image


class WindowInfo:
    """封装窗口信息的类"""

    def __init__(self, handle, title, class_name, x, y, width, height,
                 title_height, title_width, is_active, state):
        self.handle = handle
        self.title = title
        self.class_name = class_name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title_height = title_height
        self.title_width = title_width
        self.is_active = is_active
        self.state = state

    def __str__(self):
        """自定义输出格式"""
        return f"WindowInfo(handle={self.handle}, title='{self.title}')"

    def __repr__(self):
        return self.__str__()

    def to_dict(self, exclude_fields=[]) -> Dict[str, Any]:
        data = vars(self)

        res = {}
        for k, v in data.items():
            if k in exclude_fields:
                continue
            if k.startswith('_') or k.startswith('__'):
                continue

            res[k] = v
        return res


class WindowManager:
    """
    用于窗口操作的实用类：包括捕获、截图、调整大小、移动、显示/隐藏以及信息获取等功能。
    """

    def __init__(self, title=None):
        if title:
            self.hw = self.get_handle_window_by_title(title)
        else:
            self.hw = self.get_active_window()

    def get_active_window(self):
        """Return handle of the current foreground window."""
        return win32gui.GetForegroundWindow()

    def get_handle_window_by_title(self, title):
        """Return the handle of the most-recent window matching the title."""
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            raise ValueError(f"No window found with title: {title}")
        return windows[-1]._hWnd

    # def screenshot(self):
    #     """
    #     从窗口关联的显示设备上下文（GetWindowDC）读像素
    #
    #     窗口必须实际可见（未最小化、在屏幕上未被完全遮挡），否则截图为黑幕
    #     :return:
    #     """
    #     # 获取窗口矩形区域
    #     left, top, right, bot = win32gui.GetWindowRect(self.hw)
    #     width = right - left
    #     height = bot - top
    #
    #     # 获取设备上下文
    #     hwndDC = win32gui.GetWindowDC(self.hw)
    #     mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    #     saveDC = mfcDC.CreateCompatibleDC()
    #
    #     # create bitmap
    #     saveBitMap = win32ui.CreateBitmap()
    #     saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    #     saveDC.SelectObject(saveBitMap)
    #
    #     # bit blt into saveDC
    #     saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
    #
    #     # save bitmap to memory
    #     bmpinfo = saveBitMap.GetInfo()
    #     bmpstr = saveBitMap.GetBitmapBits(True)
    #     im = Image.frombuffer(
    #         'RGB',
    #         (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
    #         bmpstr, 'raw', 'BGRX', 0, 1
    #     )
    #
    #     # clean up
    #     win32gui.DeleteObject(saveBitMap.GetHandle())
    #     saveDC.DeleteDC()
    #     mfcDC.DeleteDC()
    #     win32gui.ReleaseDC(self.hw, hwndDC)
    #
    #     return im

    def screenshot(self):
        """
        系统调用 PrintWindow(hw, hdc, PW_RENDERFULLCONTENT)，强制窗口自己把内部内容渲染到指定 DC

        即便最小化或被遮挡，也会让窗口把内容绘制出来
        :return:
        """
        # 获取窗口矩形区域
        left, top, right, bottom = win32gui.GetWindowRect(self.hw)
        width, height = right - left, bottom - top

        # 创建设备上下文
        hdc_window = win32gui.GetWindowDC(self.hw)
        mfc_window_dc = win32ui.CreateDCFromHandle(hdc_window)
        mem_dc = mfc_window_dc.CreateCompatibleDC()

        # 创建位图对象
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_window_dc, width, height)
        mem_dc.SelectObject(bmp)

        # 使用 PrintWindow 强制窗口重绘到内存 DC
        PW_RENDERFULLCONTENT = 0x00000002
        ctypes.windll.user32.PrintWindow(self.hw, mem_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)

        # 从位图获取像素数据
        bmp_info = bmp.GetInfo()
        bmp_str = bmp.GetBitmapBits(True)
        image = Image.frombuffer(
            'RGB',
            (bmp_info['bmWidth'], bmp_info['bmHeight']),
            bmp_str, 'raw', 'BGRX', 0, 1
        )

        # 清理
        win32gui.DeleteObject(bmp.GetHandle())
        mem_dc.DeleteDC()
        mfc_window_dc.DeleteDC()
        win32gui.ReleaseDC(self.hw, hdc_window)

        return image

    def move(self, x, y):
        """Move window to (x, y)."""
        rect = win32gui.GetWindowRect(self.hw)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        win32gui.MoveWindow(self.hw, x, y, width, height, True)

    def resize_window_size(self, width, height):
        """Resize window to width x height at current position."""
        rect = win32gui.GetWindowRect(self.hw)
        x, y = rect[0], rect[1]
        win32gui.MoveWindow(self.hw, x, y, width, height, True)

    def show(self, flags=win32con.SW_SHOW):
        """Show or restore the window. Default flags=SW_SHOW."""
        win32gui.ShowWindow(self.hw, flags)

    def hide(self):
        """Hide the window."""
        win32gui.ShowWindow(self.hw, win32con.SW_HIDE)

    def get_window_info(self) -> WindowInfo:
        """Return dict with position, size, and active state."""
        # 窗口坐标、宽高
        rect = win32gui.GetWindowRect(self.hw)
        x, y, right, bottom = rect
        width = right - x
        height = bottom - y

        # 标题栏高度：边框+标题栏
        # SM_CYCAPTION 标题栏高度，SM_CYFRAME（或 SM_CYSIZEFRAME）是顶边和底边框高度
        frame_height = win32api.GetSystemMetrics(win32con.SM_CYFRAME)
        caption_height = win32api.GetSystemMetrics(win32con.SM_CYCAPTION)
        title_height = caption_height + frame_height

        # 标题栏宽度：去掉左右边框后的窗口宽度
        # SM_CXFRAME（或 SM_CXSIZEFRAME）是左+右边框宽度
        frame_width = win32api.GetSystemMetrics(win32con.SM_CXFRAME)
        title_width = width - 2 * frame_width

        # 是否激活
        is_active = (self.hw == win32gui.GetForegroundWindow())

        # 窗口状态
        placement = win32gui.GetWindowPlacement(self.hw)
        if placement[1] == win32con.SW_SHOWMINIMIZED:
            state = 'minimized'
        elif placement[1] == win32con.SW_SHOWMAXIMIZED:
            state = 'maximized'
        else:
            state = 'normal'

        return WindowInfo(
            handle=self.hw,
            title=win32gui.GetWindowText(self.hw),
            class_name=win32gui.GetClassName(self.hw),
            x=x,
            y=y,
            width=width,
            height=height,
            title_height=title_height,
            title_width=title_width,
            is_active=is_active,
            state=state
        )

    def is_window_active(self):
        """Return True if the specified window is the foreground window."""
        return self.hw == win32gui.GetForegroundWindow()

    def activate(self):
        """Bring the window to the foreground and activate it."""
        # 如果窗口被最小化，则先恢复
        win32gui.ShowWindow(self.hw, win32con.SW_RESTORE)
        # 设置到前台
        win32gui.SetForegroundWindow(self.hw)


if __name__ == "__main__":
    wm = WindowManager(title="图片显示器")
    im = wm.screenshot()
    im.save('image_viewer.png')
    info = wm.get_window_info()
    print(info.to_dict())
