
import os
import numpy as np
import networkx as nx

def generate_percolation_net(matrix):
    import numpy as np
    import networkx as nx

    # 计算所有边的绝对值并排序
    temp_list = np.abs(matrix).flatten(order="C")
    sort_list = np.sort(temp_list)
    
    # 初始化图，节点数量与矩阵行数相同
    nxG = nx.DiGraph()
    for i in range(matrix.shape[0]):
        nxG.add_node(i)
    
    # 处理矩阵，将负值转为正值
    matrix = np.abs(matrix)
    
    # 初始化返回值
    return_nxG = None
    return_matrix = None
    critical_threshold = None
    
    # 用于记录当前筛选使用的阈值索引
    sort_index = 0
    
    # 进入迭代，每次用当前阈值筛选边
    while sort_index < len(sort_list):
        threshold = sort_list[sort_index]
        nxG.clear()
        for i in range(matrix.shape[0]):
            nxG.add_node(i)
        # 复制矩阵用于更新
        temp_matrix = matrix.copy()
        
        # 根据阈值筛选边
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > threshold:
                    nxG.add_edge(i, j, weight=matrix[i, j])
                else:
                    temp_matrix[i, j] = 0
        
        # 检查最大弱连通分支的节点数
        largest_cc = max(nx.weakly_connected_components(nxG), key=len).__len__()
        
        if largest_cc == matrix.shape[0]:
            # 网络全连通，记录状态和临界阈值，然后继续增加阈值
            critical_threshold = threshold
            return_nxG = nxG.copy()
            return_matrix = temp_matrix.copy()
            sort_index += 1
        else:
            # 当当前阈值破坏全连通性时，退出循环
            break

    return return_nxG, return_matrix, critical_threshold


def optimized_generate_percolation_net(matrix):
    import numpy as np
    import networkx as nx

    n = matrix.shape[0]
    abs_matrix = np.abs(matrix)
    
    # 生成边列表并按权重升序排序
    edges = []
    for i in range(n):
        for j in range(i+1):  # 无向图只需处理一半矩阵
            if abs_matrix[i, j] > 0:
                edges.append((abs_matrix[i, j], i, j))
    edges.sort()  # 按权重从小到大排序

    # 初始化并查集（跟踪连通分量大小）
    parent = list(range(n))
    size = [1] * n
    max_size = n  # 当前最大连通分量大小

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # 路径压缩
            u = parent[u]
        return u

    def union(u, v):
        nonlocal max_size
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return
        if size[root_u] < size[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        size[root_u] += size[root_v]
        if size[root_u] > max_size:
            max_size = size[root_u]

    # 二分查找临界阈值
    low, high = 0, len(edges)
    critical_threshold = 0.0
    best_matrix = np.zeros_like(abs_matrix)

    while low <= high:
        mid = (low + high) // 2
        
        # 重置并查集
        parent = list(range(n))
        size = [1] * n
        max_size = 1
        
        # 添加权重大于当前阈值的边
        current_threshold = edges[mid][0] if mid < len(edges) else float('inf')
        threshold = current_threshold
        
        # 临时矩阵和边集合
        temp_matrix = np.zeros_like(abs_matrix)
        active_edges = []
        
        # 添加所有权重大于阈值的边
        for w, i, j in edges[mid:]:
            if i != j:  # 避免自环
                active_edges.append((i, j, w))
                union(i, j)
                temp_matrix[i, j] = w
                temp_matrix[j, i] = w  # 无向图对称
        
        if max_size == n:  # 仍然全连通
            critical_threshold = current_threshold
            best_matrix = temp_matrix.copy()
            low = mid + 1  # 尝试更小的mid（更大的阈值）
        else:
            high = mid - 1  # 需要更小的阈值

    # 构建最终网络
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_list = [(i, j, {'weight': w}) for i, j, w in active_edges]
    G.add_edges_from(edge_list)

    return G, best_matrix, critical_threshold


def matrix_proprecess(matrix):
    # 去掉对角线
    for i in range(0, matrix.shape[0]):
        matrix[i,i] = 0
    return matrix

def matrix_abs(matrix):
    # 取绝对值
    return np.abs(matrix)

def matrix_group_average(data_path):
    num = 0
    all_matrix = []
    for patient in list(os.walk(data_path))[0][2]:
        patient_file_path = os.path.join(data_path, patient)
        print("Loading:", patient_file_path)  # 打印文件路径

        patient_data = np.loadtxt(patient_file_path)
        matrix_data = np.array(patient_data)

        matrix = matrix_proprecess(matrix_data)
        num += 1
        if num == 1:
            all_matrix = matrix
        else:
            all_matrix = all_matrix + matrix
    
    average_matrix = all_matrix / num
    
    # 先算组平均，再取绝对值
    for i in range(0, average_matrix.shape[0]):
        for j in range(0, average_matrix.shape[1]):
            if average_matrix[i,j] < 0:
                average_matrix[i, j] = abs(average_matrix[i,j])
    return average_matrix

def main():
    data_path = "/home/user/zhangyan/dominating_set/data/power264/pre"
    
    # 计算组平均矩阵
    print("计算组平均FC矩阵...")
    average_matrix = matrix_group_average(data_path)
    
    # 计算图渗流网络并保存
    print("-------图渗流--------")
    nxG, matrix,critical_threshold = generate_percolation_net(average_matrix)
    print("data_path:", data_path)
    print("图渗流结束后的阈值：", critical_threshold)
    
    # # 保存结果
    # result_dir = "/home/user/zhangyan/dominating_set/Individual_computing/图渗流找阈值/result"
    # os.makedirs(result_dir, exist_ok=True)
    
    # # 保存阈值结果
    # with open(os.path.join(result_dir, "power264-nanjing_HC-critical_threshold.txt"), "w") as f:
    #     f.write(f"Critical Threshold: {critical_threshold}\n")
    
    # 保存平均矩阵
    # np.savetxt(os.path.join(result_dir, "average_matrix.txt"), average_matrix)
    
    return critical_threshold

if __name__ == "__main__":
    threshold = main()
    print(f"计算完成，建网阈值为: {threshold}")