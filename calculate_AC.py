"""Utilities for computing average controllability of connectivity matrices."""

from pathlib import Path

import numpy as np
import pandas as pd
from nctpy.metrics import ave_control
from nctpy.utils import matrix_normalization

def calculate_average_controllability(file_path: str) -> np.ndarray:
    """Return the average controllability values of ``file_path``.

    The matrix is normalised using :func:`nctpy.utils.matrix_normalization` and
    invalid values are replaced with zeros.
    """

    A = np.loadtxt(file_path)
    
    # 检查矩阵是否为方阵
    if A.shape[0] != A.shape[1]:
        raise ValueError("连接矩阵必须是方阵")
    
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

def save_results_to_csv(values: np.ndarray, output_file: Path) -> None:
    """Save controllability ``values`` to ``output_file``."""

    df = pd.DataFrame({
        "ROI_Index": range(len(values)),
        "Average_Controllability": values,
    })
    df.to_csv(output_file, index=False)


def process_single_file(input_file: Path, output_dir: Path) -> None:
    """Compute controllability for ``input_file`` and save it into ``output_dir``."""

    subject_id = input_file.stem
    print(f"Processing {subject_id} ...")
    values = calculate_average_controllability(str(input_file))
    output_file = output_dir / f"{subject_id}_ave_control.csv"
    save_results_to_csv(values, output_file)
    print(f"Saved results to {output_file}")

def main() -> None:
    """Process all connectivity matrices in ``input_dir``."""

    input_dir = Path(
        "/home/user/zhangyan/dominating_set/data/power264/pre"
    )
    output_dir = Path(
        "/home/user/zhangyan/dominating_set/计算平均可控性/平均可控性结果/power264/rtms_pre"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt in input_dir.glob("*.txt"):
        process_single_file(txt, output_dir)
        print("-- processed", txt.name)

if __name__ == "__main__":
    main()
