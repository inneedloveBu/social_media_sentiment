# run_analysis.py
"""
一键运行社交媒体情感分析
自动选择最佳运行方案
"""

import os
import sys
import subprocess

def check_spark_availability():
    """检查Spark是否可用"""
    try:
        import pyspark
        print("✅ PySpark已安装")
        return True
    except ImportError:
        print("❌ PySpark未安装")
        return False
    except Exception as e:
        print(f"⚠️  PySpark导入出错: {e}")
        return False

def run_spark_version():
    """运行Spark版本"""
    print("\n🚀 尝试运行Spark版本...")
    
    # 应用Windows补丁
    try:
        import socketserver
        if not hasattr(socketserver, 'UnixStreamServer'):
            class UnixStreamServer:
                pass
            socketserver.UnixStreamServer = UnixStreamServer
    except:
        pass
    
    # 运行简化版Spark分析
    try:
        exec(open("spark_sentiment_final.py").read())
        return True
    except Exception as e:
        print(f"❌ Spark版本运行失败: {e}")
        return False

def run_pandas_version():
    """运行Pandas版本"""
    print("\n📊 运行Pandas/Scikit-learn版本...")
    try:
        import pandas as pd
        import sklearn
        print("✅ Pandas和Scikit-learn可用")
        
        # 这里可以调用您的pandas版本分析脚本
        # 例如：exec(open("pandas_sentiment_analysis.py").read())
        
        print("\n📋 创建演示数据和分析...")
        
        # 简单演示
        data = pd.DataFrame({
            'text': [
                '这部电影很棒！',
                '太糟糕了，不推荐',
                '一般般，还可以',
                '强烈推荐！'
            ],
            'label': [1, 0, 0, 1]
        })
        
        print("示例数据:")
        print(data)
        
        # 简单的文本分析
        data['text_length'] = data['text'].str.len()
        data['sentiment'] = data['label'].map({1: '正面', 0: '负面'})
        
        print("\n分析结果:")
        print(data[['text', 'text_length', 'sentiment']])
        
        # 保存结果
        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)
        
        data.to_csv(os.path.join(output_dir, "pandas_analysis.csv"), 
                   index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 结果已保存到: {output_dir}/pandas_analysis.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Pandas版本运行失败: {e}")
        return False

def run_streamlit_app():
    """运行Streamlit应用"""
    print("\n🌐 启动Streamlit Web应用...")
    try:
        # 检查Streamlit是否安装
        import streamlit
        print("✅ Streamlit已安装")
        
        print("\n📢 启动命令:")
        print("  streamlit run streamlit_app.py")
        print("\n或者运行英文版:")
        print("  streamlit run streamlit_app_en.py")
        
        # 询问是否立即启动
        response = input("\n是否立即启动Streamlit应用？(y/n): ")
        if response.lower() == 'y':
            print("正在启动Streamlit...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        
        return True
    except ImportError:
        print("❌ Streamlit未安装")
        print("  安装命令: pip install streamlit")
        return False

def main():
    """主函数"""
    print("="*60)
    print("社交媒体情感分析 - 一键运行脚本")
    print("="*60)
    
    print("\n🔍 检查环境...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version.split()[0]}")
    
    # 检查操作系统
    print(f"操作系统: {sys.platform}")
    
    # 菜单选择
    print("\n📋 请选择运行模式:")
    print("  1. 尝试Spark版本 (Windows可能有问题)")
    print("  2. 运行Pandas版本 (推荐，稳定)")
    print("  3. 启动Streamlit Web应用")
    print("  4. 全部运行")
    print("  0. 退出")
    
    choice = input("\n请选择 (0-4): ")
    
    if choice == '1':
        if check_spark_availability():
            run_spark_version()
        else:
            print("❌ Spark不可用，请安装或使用其他选项")
    elif choice == '2':
        run_pandas_version()
    elif choice == '3':
        run_streamlit_app()
    elif choice == '4':
        print("\n🚀 运行所有版本...")
        print("\n" + "="*40)
        print("1. 尝试Spark版本")
        print("="*40)
        if check_spark_availability():
            run_spark_version()
        
        print("\n" + "="*40)
        print("2. 运行Pandas版本")
        print("="*40)
        run_pandas_version()
        
        print("\n" + "="*40)
        print("3. 启动Streamlit应用")
        print("="*40)
        run_streamlit_app()
    elif choice == '0':
        print("👋 退出程序")
        return
    else:
        print("❌ 无效选择")
    
    print("\n" + "="*60)
    print("程序执行完成！")
    print("="*60)

if __name__ == "__main__":
    main()