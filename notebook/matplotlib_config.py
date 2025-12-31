# matplotlib_config.py
import matplotlib.pyplot as plt
import matplotlib
import platform
import os

def configure_matplotlib():
    """配置Matplotlib以支持中文显示"""
    
    # 根据不同操作系统设置字体
    system = platform.system()
    
    if system == "Windows":
        # Windows系统常见中文字体路径
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # 苹方
            "/System/Library/Fonts/STHeiti Light.ttc",  # 黑体
            "/System/Library/Fonts/Hiragino Sans GB.ttc",  # 冬青黑体
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Droid字体
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto字体
        ]
    
    # 尝试添加字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                matplotlib.font_manager.fontManager.addfont(font_path)
                font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用字体: {font_name}")
                return True
            except Exception as e:
                print(f"加载字体失败 {font_path}: {e}")
    
    # 如果找不到中文字体，使用默认支持中文的字体
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("使用备用字体: DejaVu Sans")
        return True
    except:
        return False