# matplotlib_config.py
import matplotlib
import os
import platform

def configure_matplotlib():
    """配置matplotlib以支持中文字体"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系统字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # 苹方
            "/System/Library/Fonts/STHeiti Light.ttc",  # 华文黑体
        ]
    else:  # Linux/WSL
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto字体
            "/mnt/c/Windows/Fonts/msyh.ttc",  # WSL中访问Windows字体
        ]
    
    # 查找可用的字体文件
    available_font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            available_font = font_path
            break
    
    if available_font:
        try:
            # 添加字体
            matplotlib.font_manager.fontManager.addfont(available_font)
            font_name = matplotlib.font_manager.FontProperties(fname=available_font).get_name()
            
            # 设置matplotlib参数
            matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            print(f"✅ 已配置中文字体: {font_name} ({available_font})")
        except Exception as e:
            print(f"⚠️ 字体配置失败: {e}")
            # 设置后备字体
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
    else:
        print("⚠️ 未找到中文字体文件，使用默认字体")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    
    return True