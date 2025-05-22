import numpy as np
import os
import networkx as nx
import datetime
import itertools
import csv
import random
import ast
from AAL116_brain_area import brain_area_90
from glasser360_brain_area import brain_area_360
from power264_brain_area import brain_area_264

def remove_prefix_suffix(s):
    """
    移除字符串s中的'ROICorrelation_FisherZ_'前缀和'.txt'后缀。
    
    参数:
    s (str): 需要处理的原始字符串。
    
    返回:
    str: 处理后的字符串。
    """
    # 去除前缀
    without_prefix = s.replace("ROICorrelation_FisherZ_", "")
    # 去除后缀
    final_string = without_prefix[:-4] if without_prefix.endswith(".txt") else without_prefix
    return final_string

def save_to_txt(min_sets, times, data_type, root_dir, template_name, subject_name):
    # 构建路径：root_dir/template_name/data_type/subject_name/greedy_result
    output_dir = os.path.join(root_dir, template_name, data_type, subject_name, "greedy_result")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"dominating_sets_times_{times}.txt")
    with open(filename, 'w') as f:
        for i, ds in enumerate(min_sets):
            f.write(f"Set {i+1}: {ds}\n")
    return filename

def matrix_proprecess(matrix):
    # #  去掉对角线
    for i in range(0, matrix.shape[0]):
        matrix[i,i] = 0
    return matrix

def matrix_abs(matrix):
    # 权重取绝对值
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if matrix[i,j] < 0:
                matrix[i, j] = abs(matrix[i,j])
    return matrix

def matrix_group_average(data_path):
    num = 0
    all_matrix = []
    for patient in list(os.walk(data_path))[0][2]:

        patient_file_path = os.path.join(data_path, patient)
        print("Loading:", patient_file_path)  # 打印文件路径

        patient_data = np.loadtxt(patient_file_path)
        matrix_data = np.array(patient_data)

        # print("matrix_data:",matrix_data.shape)
        # print("matrix_data:",matrix_data)

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


def generate_percolation_net(matrix):
    # 筛选阈值,最大弱连通分支被破坏时，终止筛选
    temp_list = matrix.flatten(order="C")
    # 只看绝对强度
    temp_list = abs(temp_list)
    sort_list = np.sort(temp_list)
    sort_index = -1
    largest_cc = matrix.shape[0]
    nxG = nx.DiGraph()
    return_nxG = nxG.copy()
    # 权重取绝对值
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if matrix[i,j] < 0:
                matrix[i, j] = abs(matrix[i,j])
    return_matrix = matrix.copy()
    temp_matrix = matrix.copy()
    while largest_cc == matrix.shape[0]:
        return_nxG = nxG.copy()
        return_matrix = temp_matrix.copy()
        nxG.clear()
        for i in range(0,matrix.shape[0]):
            nxG.add_node(i)
        sort_index = sort_index + 1
        temp_data = matrix
        threshold = sort_list[sort_index]
        # print("threshold: "+str(threshold))
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                if abs(temp_data[i, j]) > threshold:
                    nxG.add_edge(i, j, weight=abs(temp_data[i, j]))
                else:
                    temp_matrix[i,j] = 0
        largest_cc = max(nx.weakly_connected_components(nxG),key=len).__len__()
        # print("largest_cc: " + str(largest_cc))
        # print()
    return return_nxG,return_matrix

def all_min_dominating_set(nxG):
    min_dominating_set_result = []
    strength_of_all_min_dominating_set = []
    # nxG中的节点非重复且不考虑顺序的排列组合,暴力求解最小支配集
    node_list = nx.nodes(nxG)
    node_num = nxG.number_of_nodes()
    # 初始化最小支配集大小为所有节点
    min_dominating_size = node_num
    print("start searching....")
    for i in range(1,node_num+1):
        print("searching for size " + str(i) +" ...")
        print(str(datetime.datetime.now()))
        set_list = list(itertools.combinations(node_list, i))
        for temp_set in set_list:
            if nx.is_dominating_set(nxG,temp_set):
                min_dominating_size = i
                min_dominating_set_result.append(temp_set)
                # 计算最小支配集权重
                temp_total_strength = 0
                for dominating_node in temp_set:
                    temp_strength = nxG.degree(dominating_node,weight='weight')
                    temp_total_strength = temp_total_strength + temp_strength
                strength_of_all_min_dominating_set.append(temp_total_strength)
        # 如果已经找到最小支配集，那么就不再继续寻找最小支配集
        if i >= min_dominating_size:
            break
        print(str(datetime.datetime.now()))
    return min_dominating_set_result,strength_of_all_min_dominating_set

# 贪心算法寻找最小支配集（输入参数：NetworkX 图对象 nxG、times 表示进行搜索的次数）
def greedy_minimum_dominating_set(nxG, times):
    min_dominating_set = []

    for time in range(times):
        nxG_copy = nxG.copy()
        dominating_set = []

        # 当 nxG_copy 中还有节点时，执行以下循环： 
        # 1. 随机选择一个节点 node 加入到 dominating_set。 
        # 2. 创建并清空一个列表 remove_list，将 node 添加进去。 
        # 3. 遍历 node 的所有邻居，并将它们添加到 remove_list 中。
        # 4. 将 remove_list 中的所有节点从 nxG_copy 中移除。
        # 5. 重复步骤 1-4，直到 nxG_copy 中没有节点为止。

        while nxG_copy.nodes():
            node = random.choice(list(nxG_copy.nodes()))
            dominating_set.append(node)
            remove_list = []
            remove_list.clear()
            remove_list.append(node)
            for neighbor in nxG_copy.neighbors(node):
                remove_list.append(neighbor)
                # # 这里remove list要set化，因为原来的节点中可能存在环，使得原来的list中有重复被删的节点
                # remove_list_set = set(remove_list)

            for node in remove_list:
                if node in nxG_copy.nodes:
                    nxG_copy.remove_node(node)
        
        # 如果当前 dominating_set 大小等于已知最小支配集大小但不在已知集合中，则将其添加到结果列表。 
        # 如果当前 dominating_set 更小，则清空已知最小支配集列表，并将当前集合添加进去。
        dominating_set = set(dominating_set)
        if len(min_dominating_set) == 0:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) == len(dominating_set) and dominating_set not in min_dominating_set:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) > len(dominating_set):
            min_dominating_set.clear()
            min_dominating_set.append(dominating_set)
        # print("times: " + str(time + 1))
        # print("times: " + str(time + 1) +" MDSet size: "+ str(len(min_dominating_set[0]))+ " MDSet number: "+ str(len(min_dominating_set)) +"  MDSet: " + str(min_dominating_set))

    return min_dominating_set

def dominating_frequency(all_dom_set,nxG):
    num_dom_set = len(all_dom_set)
    node_num = nxG.number_of_nodes()
    # init
    as_dom_node_count = {}
    for node_index in range(0,node_num):
        as_dom_node_count[node_index] = 0
    # count
    for min_dom_set in all_dom_set:
        for dom_node in min_dom_set:
            as_dom_node_count[dom_node] = as_dom_node_count[dom_node] + 1

    for node_index in as_dom_node_count:
        as_dom_node_count[node_index] = as_dom_node_count[node_index] / num_dom_set
    print(as_dom_node_count)
    return as_dom_node_count

# 整合贪心算法寻找的所有最小支配集
def consolidate_and_save_results(data_type, root_dir, template_name, subject_name):
    all_sets = []
    result_dir = os.path.join(root_dir, template_name, data_type, subject_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # 结果整合 删除重复的最小支配集
    for file_name in os.listdir(os.path.join(result_dir, "greedy_result")):
        if file_name.startswith("dominating_sets_times_") and file_name.endswith(".txt"):
            with open(os.path.join(result_dir, "greedy_result", file_name), 'r') as f:
                for line in f.readlines():
                    all_sets.append(set(eval(line.split(": ")[1].strip())))
    unique_sets = set(map(tuple, all_sets))

    # 筛选出节点数等于最小节点数的最小支配集
    min_set_size = min(len(s) for s in unique_sets)
    smallest_sets = [set(s) for s in unique_sets if len(s) == min_set_size]
    
    # 构建目标路径
    target_dir = os.path.join(result_dir, 'consolidate_result')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    output_file = os.path.join(target_dir, f"{subject_name}_consolidate_result.txt")
    
    # 保存整合后的结果
    with open(output_file, 'w') as f:
        for index, set_items in enumerate(smallest_sets, start=1):
            f.write(f"Set {index}: {set(set_items)}\n")
    return output_file

def open_txt(path):
    all_dom_set = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("Set "):
                # 使用ast.literal_eval来安全地评估字符串中的集合
                dom_set = ast.literal_eval(line.split(": ")[1])
                all_dom_set.append(dom_set)
        return all_dom_set

def save_csv(df, data_type, root_dir, template_name, subject_name):
    # 构建路径：root_dir/template_name/data_type/subject_name/frequency
    output_dir = os.path.join(root_dir, template_name, data_type, subject_name, "frequency")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dominating_frequency_{subject_name}.csv")
    
    # 将字典数据写入CSV文件
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'Frequency'])  # 写入表头
        for node, frequency in df.items():
            writer.writerow([node, frequency])
    
    return output_path

def count_nodes(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip() 
        if first_line.startswith('Set'):
            set_content = first_line.split(':')[1].strip()  
            nodes = set_content[1:-1] 
            node_list = [int(n) for n in nodes.split(', ')]  
            print(f"集合中的节点数量: {len(node_list)}")

def replace_to_brainarea(input_csv_path, data_type, root_dir, template_name, subject_name):
    # 根据模板名称选择对应的字典
    if template_name == 'aal116':
        brain_area_dict = brain_area_90
    elif template_name == 'glasser360':
        brain_area_dict = brain_area_360
    elif template_name == 'power264':
        brain_area_dict = brain_area_264
    else:
        raise ValueError("未知的脑模板名称")
    
    # 构建输出路径：root_dir/template_name/data_type/subject_name/brain_area
    output_dir = os.path.join(root_dir, template_name, data_type, subject_name, "brain_area")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"dominating_frequency_{subject_name}_brain_area.csv")
    
    # 读取原始CSV文件
    with open(input_csv_path, 'r') as infile:
        reader = csv.DictReader(infile)
        rows = [row for row in reader]
    
    # 替换节点编号为脑区名称
    for row in rows:
        node = int(row['Node'])
        row['Node'] = brain_area_dict.get(node, "未知区域")
    
    # 保存替换后的结果
    with open(output_csv_path, 'w', newline='') as outfile:
        fieldnames = ['Node', 'Frequency']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print("替换完成,新的CSV文件已保存.")
    return output_csv_path

def generate_network_with_fixed_threshold(matrix, threshold):
    """
    使用固定阈值生成网络
    
    参数:
    matrix: 输入的连接矩阵
    threshold: 固定的阈值值
    
    返回:
    G: networkx图对象
    thresholded_matrix: 应用阈值后的矩阵
    threshold: 使用的阈值
    """
    import numpy as np
    import networkx as nx
    
    n = matrix.shape[0]
    abs_matrix = np.abs(matrix)  # 使用绝对值
    
    # 创建阈值化的矩阵
    thresholded_matrix = np.zeros_like(abs_matrix)
    
    # 创建图对象
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # 添加边（权重大于阈值的连接）
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):  # 无向图只需处理上三角矩阵
            if abs_matrix[i, j] > threshold:
                edge_list.append((i, j, {'weight': abs_matrix[i, j]}))
                thresholded_matrix[i, j] = abs_matrix[i, j]
                thresholded_matrix[j, i] = abs_matrix[i, j]  # 保持对称性
    
    G.add_edges_from(edge_list)
    
    return G, thresholded_matrix, threshold


if __name__ == '__main__':
    data_path ="/home/user/zhangyan/dominating_set/data/power264/post"
    root_dir = "/home/user/zhangyan/dominating_set/Individual_computing/result/power264"
    data_type = 'nj_mdd_rtms_post'
    template_name = 'power264'
    fixed_threshold = 0.3230072808988765

    for filename in os.listdir(data_path):
        subject_path = os.path.join(data_path, filename)
        subject_name = remove_prefix_suffix(filename)
        # print("subject_name:",subject_name)
        # print("subject_path:",subject_path)
        patient_data = np.array(np.loadtxt(subject_path))
        patient_data = matrix_abs(matrix_proprecess(patient_data))

        # 根据模板名称判断是否需要截取矩阵
        if template_name.lower() == 'aal116':
            print("检测到AAL116模板，只保留前90个脑区")
            patient_data = patient_data[:90, :90]
        
        # 使用固定阈值建网
        print(f"-------使用固定阈值{fixed_threshold}建网--------")
        nxG, matrix, used_threshold = generate_network_with_fixed_threshold(patient_data, fixed_threshold)
        print(f"网络构建完成，使用的阈值: {used_threshold}")

        # 最小支配集分析   - 贪心 xxxxxx 次
        print("贪心算法寻找最小支配集:")
        for times in range(1000, 10001, 50000):
            # 运算得到结果
            all_dom_set = greedy_minimum_dominating_set(nxG, times)
            save_to_txt(all_dom_set, times, data_type, root_dir, template_name, subject_name)
        print("----贪心算法寻找结束----")

        # 整合所有支配集结果 
        print("-------整合贪心算法所有结果------")
        consolidate_result_path = consolidate_and_save_results(data_type, root_dir, template_name, subject_name)
        print("consolidate_result_path: ",consolidate_result_path)

        # 节点支配频率
        print("--------节点支配频率---------- ")
        dom_set = open_txt(consolidate_result_path)
        dom_frequency_result = dominating_frequency(dom_set,nxG)

        # 保存频率结果
        dom_frequency_result_path = save_csv(dom_frequency_result, data_type, root_dir, template_name, subject_name)

        # 分析节点支配频率——从局部脑区分析差异变化
        replace_to_brainarea(dom_frequency_result_path, data_type, root_dir, template_name, subject_name)
