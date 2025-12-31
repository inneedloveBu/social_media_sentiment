import pandas as pd
import os

# 请修改此处为你的实际文件名和路径
DATA_FILE_PATH = './data/weibo_senti_100k.csv'  # 示例，请替换成你实际的文件名

def load_data(file_path):
    """通用数据加载函数，自动尝试常见编码格式。"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']  # 中文数据集常见的编码
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 '{encoding}' 编码加载文件: {file_path}")
            return df
        except (UnicodeDecodeError, LookupError):
            continue
        except FileNotFoundError:
            print(f"错误：未在路径找到文件 -> {file_path}")
            print("请检查：1. 文件是否已下载；2. DATA_FILE_PATH 变量中的文件名是否正确。")
            return None
    print("错误：尝试了所有常见编码均无法读取文件，请检查文件格式。")
    return None

# 主程序
if __name__ == '__main__':
    # 1. 加载数据
    df = load_data(DATA_FILE_PATH)
    
    if df is not None:
        # 2. 查看数据基本信息
        print("\n=== 数据加载成功 ===")
        print(f"数据形状: {df.shape} (行数, 列数)")
        print(f"数据前3行预览:\n{df.head(3)}")
        print(f"\n列名: {df.columns.tolist()}")
        
        # 3. 检查典型的情感分析数据集列名（根据你的数据集调整查看的列名）
        # 常见文本列名
        possible_text_columns = ['review', 'text', 'content', 'comment', '微博中文内容', 'sentence']
        # 常见标签列名
        possible_label_columns = ['label', 'sentiment', 'category', '情感倾向', 'label_ft']
        
        text_col = None
        label_col = None
        
        for col in possible_text_columns:
            if col in df.columns:
                text_col = col
                break
        for col in possible_label_columns:
            if col in df.columns:
                label_col = col
                break
                
        if text_col:
            print(f"\n文本列名为: '{text_col}'")
            print(f"文本示例: {df[text_col].iloc[0][:100]}...")  # 打印第一段文本的前100字符
        if label_col:
            print(f"标签列名为: '{label_col}'")
            print(f"标签分布:\n{df[label_col].value_counts()}")
            
        # 4. 保存一份清洗前的副本（可选）
        df.to_csv('./data/raw_data_backup.csv', index=False, encoding='utf-8-sig')
        print(f"\n数据备份已保存至: ./data/raw_data_backup.csv")
        print("\n>>> 数据准备阶段完成！接下来可以进行数据清洗和特征工程。 <<<")