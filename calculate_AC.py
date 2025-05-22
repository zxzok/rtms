import numpy as np
import pandas as pd
from nctpy.utils import matrix_normalization
from nctpy.metrics import ave_control

def calculate_average_controllability(file_path, roi_count=90):
    """
    使用nctpy计算网络中每个节点的平均可控性
    
    参数:
    file_path (str): 连接矩阵文件的路径
    roi_count (int): 需要分析的ROI数量，默认为90
    
    返回:
    numpy.ndarray: 包含每个ROI的平均可控性值的数组
    """
    # 读取连接矩阵
    A = np.loadtxt(file_path)
    
    # 只取前roi_count行和列
    # A = A[:roi_count, :roi_count]
    
    # 检查矩阵的有效性
    if A.shape[0] != A.shape[1]:
        raise ValueError("连接矩阵必须是方阵。")
    
    # 处理无效值
    # 1. 先将对角线上的inf替换为1（自相关）
    np.fill_diagonal(A, 1.0)
    # 2. 处理其他无效值
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 矩阵归一化
    system = 'discrete'  # 使用离散时间系统，因为是相关性矩阵
    A_norm = matrix_normalization(A=A, c=1, system=system)
    
    # 计算平均可控性
    average_controllability_values = ave_control(A_norm=A_norm, system=system)
    
    return average_controllability_values

def save_results_to_csv(controllability_values, output_file):
    """
    将平均可控性结果保存到CSV文件
    
    参数:
    controllability_values (numpy.ndarray): 平均可控性值数组
    output_file (str): 输出CSV文件的路径
    """
    # 创建数据框，ROI_Index从0开始
    df = pd.DataFrame({
        'ROI_Index': range(0, len(controllability_values)),
        'Average_Controllability': controllability_values
    })
    
    # 保存到CSV文件
    df.to_csv(output_file, index=False)


def process_single_file(input_file, output_dir):
    """
    处理单个连接矩阵文件并保存结果
    
    参数:
    input_file (str): 输入文件路径
    output_dir (str): 输出目录路径
    """
    # 从文件名中提取被试ID
    subject_id = os.path.splitext(os.path.basename(input_file))[0]
    
    # 计算平均可控性
    print(f"正在处理被试 {subject_id} 的数据...")
    controllability_values = calculate_average_controllability(input_file)
    
    # 创建输出文件路径
    output_file = os.path.join(output_dir, f"{subject_id}_ave_control.csv")
    
    # 保存结果
    save_results_to_csv(controllability_values, output_file)
    print(f"已成功处理被试 {subject_id} 的数据并保存到 {output_file}")

def main():
    """
    主函数：处理所有连接矩阵文件
    """
    # 设置输入和输出目录
    input_dir = "/home/user/zhangyan/dominating_set/data/power264/pre"
    output_dir = "/home/user/zhangyan/dominating_set/计算平均可控性/平均可控性结果/power264/rtms_pre"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有txt文件
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    # 处理每个文件
    for file_name in input_files:
        input_file = os.path.join(input_dir, file_name)
        process_single_file(input_file, output_dir)
        print("-----处理----",file_name)

if __name__ == "__main__":
    main()
