# spark_windows_patch.py
"""
此脚本用于在Windows上修复PySpark导入时遇到的
'socketserver.UnixStreamServer' 缺失错误。
必须在任何PySpark模块导入之前运行。
"""
import sys
import socketserver
import os

# 检查并修复 UnixStreamServer 缺失的问题
if not hasattr(socketserver, 'UnixStreamServer'):
    print("检测到Windows环境，正在修补 socketserver 模块...")
    # 创建一个虚拟的 UnixStreamServer 类
    class UnixStreamServer:
        pass
    # 将其“注入”到 socketserver 模块中
    socketserver.UnixStreamServer = UnixStreamServer
    print("修补完成。")

# ====== Windows环境修复 ======
if sys.platform.startswith('win'):
    # 修复1: 设置必要的环境变量
    hadoop_home = "C:\\hadoop"  # 根据你的实际路径调整
    if os.path.exists(hadoop_home):
        os.environ['HADOOP_HOME'] = hadoop_home
        # 添加Hadoop bin目录到PATH
        hadoop_bin = os.path.join(hadoop_home, "bin")
        if os.path.exists(hadoop_bin):
            os.environ['PATH'] = f"{hadoop_bin};{os.environ['PATH']}"
    else:
        print(f"⚠️  HADOOP_HOME路径不存在: {hadoop_home}")
    
    # 修复2: 临时解决socketserver问题
    import socketserver
    if not hasattr(socketserver, 'UnixStreamServer'):
        class UnixStreamServer:
            pass
        socketserver.UnixStreamServer = UnixStreamServer
    
    # 修复3: 设置Python worker环境
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # 修复4: 避免端口冲突
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

print("✅ Spark Windows补丁已应用")