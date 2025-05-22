"""Group level RTMS network analysis utilities."""

import datetime
import itertools
import os
import random
import csv

import numpy as np
import networkx as nx
import pandas as pd

from AAL116_brain_area import (
    brain_area_90,
    brain_area_90_en,
    brain_area_yeo_mapping,
)
from rtms.common import (
    matrix_preprocess,
    group_average,
    generate_network_with_threshold,
    greedy_minimum_dominating_set,
    dominating_frequency,
    read_sets,
)
import time
from glasser360_brain_area import brain_area_360
from power264_brain_area import brain_area_264,brain_area_subgraph_mapping

def save_to_txt(min_sets, times, data_type,root_dir):
    output_dir = os.path.join(root_dir, data_type, "greedy_result")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{root_dir}/{data_type}/greedy_result/dominating_sets_times_{times}.txt"
    with open(filename, 'w') as f:
        for i, ds in enumerate(min_sets):
            f.write(f"Set {i+1}: {ds}\n")


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


# 计算模块间控制强度
def module_controllability(nxG,all_dom_set,louvain_communities):
    # louvain_communities = louvain.best_partition(nxG)
    number_of_communities = max(louvain_communities.values())+1
    print("module number: "+str(number_of_communities))
    # init
    module = {}
    for index in range(0,number_of_communities):
        module[index] = []
    # finding module
    for node in louvain_communities:
        community_index = louvain_communities[node]
        module[community_index].append(node)

    for i in range(0,number_of_communities):
        print("module "+str(i)+" has "+ str(module[i])+" nodes")

    # 初始化结果
    average_module_controllability_result = {}
    for module_source in module:
        for module_target in module:
            average_module_controllability_result[str(module_source) + "_" + str(module_target)] = 0


    # all_dom_set,strength_of_all_dom_set = all_min_dominating_set(nxG)
    # print(all_dom_set)
    for min_dom_set in all_dom_set:
        dominated_area = {}
        for dom_node in min_dom_set:
            temp_neighbor = set()
            temp_neighbor.clear()
            for neighbor in nxG.neighbors(dom_node):
                temp_neighbor.add(neighbor)
            #支配域还有节点自身
            temp_neighbor.add(dom_node)
            dominated_area[dom_node] = temp_neighbor

        # 计算社团支配域
        modules_control_area = {}
        for module_index in module:
            node_in_module = module[module_index]
            single_module_control_area = set()
            for node in node_in_module:
                if node in min_dom_set:
                    # 添加module支配域
                    temp_dom_set = set()
                    for temp_node in dominated_area[node]:
                        single_module_control_area.add(temp_node)
            modules_control_area[module_index] = single_module_control_area
        # print(modules_control_area)

        # 计算社团间支配能力
        temp_module_controllability_result = {}
        temp_module_controllability_result.clear()
        for module_source in module:
            for module_target in module:
                # 社团控制域
                control_area = modules_control_area[module_source]
                # 被控社团节点集
                target_module_area = module[module_target]
                # 两者交集大小
                target_module_area_set = set(target_module_area)
                inter = control_area.intersection(target_module_area_set)
                temp_module_controllability_result[str(module_source)+"_"+str(module_target)] = len(inter) / len(target_module_area)
                average_module_controllability_result[str(module_source) + "_" + str(module_target)] = average_module_controllability_result[str(module_source) + "_" + str(module_target)] + (len(inter) / len(target_module_area))
        # print("dom_set: "+str(min_dom_set) + "   module_controllability: "+ str(temp_module_controllability_result))
    for total_module_controllability in average_module_controllability_result:
        average_module_controllability_result[total_module_controllability] = average_module_controllability_result[total_module_controllability] / len(all_dom_set)
    print("average_module_controllability: "+ str(average_module_controllability_result))
    return average_module_controllability_result


# 整合贪心算法寻找的所有最小支配集
def merge_dominating_results(data_type,root_dir):
    all_sets = []
    input_dir = os.path.join(root_dir, data_type, "greedy_result")
    # 结果整合
    for file_name in os.listdir(input_dir):
        if file_name.startswith("dominating_sets_times_") and file_name.endswith(".txt"):
            with open(os.path.join(input_dir, file_name), 'r') as f:
                for line in f.readlines():
                    all_sets.append(set(eval(line.split(": ")[1].strip())))
    # 删除重复的最小支配集
    unique_sets = set(map(tuple, all_sets))
    # 筛选出节点数等于最小节点数的最小支配集
    min_set_size = min(len(s) for s in unique_sets)
    smallest_sets = [set(s) for s in unique_sets if len(s) == min_set_size]

    output_file = f"{root_dir}/{data_type}/consolidate_result.txt"
    # 保存整合后的HC结果
    with open(output_file, 'w') as f:
        for index, set_items in enumerate(smallest_sets, start=1):
            f.write(f"Set {index}: {set(set_items)}\n")
    return output_file


# FTY 保存所有结果
def save_network_analysis_to_csv(specific_result_path, data_file, nxG, network_analysis_result, louvain_communities, as_dom_node_count, average_module_controllability_result, all_dom_set, algorithm):
    # 确保 specific_result_path 文件夹存在
    os.makedirs(specific_result_path, exist_ok=True)

    # 创建节点属性的 DataFrame
    node_data = {
        "item": list(nxG.nodes),
        "degree_centrality": [network_analysis_result["degree"][node] for node in nxG.nodes],
        "average_strength": [network_analysis_result["average_strength"][node] for node in nxG.nodes],
        "clustering": [network_analysis_result["clustering"][node] for node in nxG.nodes],
        "closeness": [network_analysis_result["closeness"][node] for node in nxG.nodes],
        "betweenness": [network_analysis_result["betweenness"][node] for node in nxG.nodes],
        "kcore": [network_analysis_result["kcore"][node] for node in nxG.nodes],
        "module": [louvain_communities[node] for node in nxG.nodes],
        "CF": [as_dom_node_count[node] for node in nxG.nodes]
    }
    node_df = pd.DataFrame(node_data)

    # 保存节点属性的 DataFrame 到 CSV
    node_df.to_csv(os.path.join(specific_result_path, f"{data_file}_nodes.csv"), index=False)

    # 创建边属性的 DataFrame
    edge_data = {
        "source": [edge[0] for edge in nxG.edges],
        "target": [edge[1] for edge in nxG.edges],
        "weight": [nxG.get_edge_data(edge[0], edge[1])['weight'] for edge in nxG.edges]
    }
    edge_df = pd.DataFrame(edge_data)

    # 保存边属性的 DataFrame 到 CSV
    edge_df.to_csv(os.path.join(specific_result_path, f"{data_file}_edges.csv"), index=False)

    # 创建模块可控性结果的 DataFrame
    module_controllability_data = {
        "module_2_module(direct)": list(average_module_controllability_result.keys()),
        "AMCS": list(average_module_controllability_result.values())
    }
    module_controllability_df = pd.DataFrame(module_controllability_data)

    # 保存模块可控性结果的 DataFrame 到 CSV
    module_controllability_df.to_csv(os.path.join(specific_result_path, f"{data_file}_module_controllability.csv"), index=False)

    # 保存支配频率到 CSV
    try:
        cf_output_file = os.path.join(specific_result_path, f"{data_file}_CF.csv")
        cf_df = pd.DataFrame(list(as_dom_node_count.items()), columns=['Node', 'Frequency'])
        cf_df.to_csv(cf_output_file, index=False, encoding='utf-8')
        print(f"支配频率已保存到文件: {cf_output_file}")
    except IOError as e:
        print(f"无法写入文件 {cf_output_file}，错误：{e}")

    # 保存最小支配集及其权重
    try:
        with open(os.path.join(specific_result_path, f"{data_file}_dominating_set.csv"), 'w', encoding='utf-8') as file_object:
            file_object.write(f"Minimum dominating set ({algorithm}) size: {len(all_dom_set[0])}\n")
            for mds in all_dom_set:
                file_object.write(f"{mds}\n")
        print(f"最小支配集已保存到 {specific_result_path}")
    except IOError as e:
        print(f"无法写入文件 {specific_result_path}，错误：{e}")

    print(f"数据已保存到 {specific_result_path}，并生成了 CSV 文件。")



def save_csv(df,data_type,root_dir):
    # dominating_frequency_HC.csv
    output_file = f"{root_dir}/{data_type}/dominating_frequency_{data_type}.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'Frequency'])
        for node, frequency in df.items():
            writer.writerow([node, frequency])
    return output_file

def count_nodes(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip() 
        if first_line.startswith('Set'):
            set_content = first_line.split(':')[1].strip()  
            nodes = set_content[1:-1] 
            node_list = [int(n) for n in nodes.split(', ')]  
            print(f"集合中的节点数量: {len(node_list)}")

def replace_to_brainarea(input_csv_path,data_type,root_dir,template_name):

    # 根据模板名称选择对应的字典
    if template_name == 'AAL116':
        brain_area_dict = brain_area_90
    elif template_name == 'glasser360':
        brain_area_dict = brain_area_360
    elif template_name == 'power264':
        brain_area_dict = brain_area_264
    else:
        raise ValueError("未知的脑模板名称")
    
    # f"{root_dir}/{data_type}/dominating_frequency_{data_type}.csv"
    output_csv_path = f"{root_dir}/{data_type}/dominating_frequency_{data_type}(brain_area).csv"

    # 读取原始CSV文件
    with open(input_csv_path, 'r') as infile:
        reader = csv.DictReader(infile)
        rows = [row for row in reader]
    for row in rows:
        node = int(row['Node'])  # 节点编号转换为整数
        row['Node'] = brain_area_dict.get(node, "未知区域")  # 替换为脑区域名称
    
    # 将替换后的数据写入新的CSV文件
    with open(output_csv_path, 'w', newline='') as outfile:
        fieldnames = ['Node', 'Frequency']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("替换完成,新的CSV文件已保存.")


def create_communities(brain_area_264):
    """
    根据power网络模板创建communities字典
    返回格式与原来的louvain_communities相同: {node_index: community_id}
    """
    communities = {}
    for node_idx in range(264):                    
        # 获取该节点对应的英文名称
        area_name = brain_area_264[node_idx]
        # 从brain_area_yeo_mapping中获取对应的网络编号
        community_id = brain_area_subgraph_mapping[area_name]
        communities[node_idx] = community_id
    return communities

if __name__ == '__main__':
    data_path ="/home/user/zhangyan/dominating_set/data/power264/pre/all"
    data_type = 'rtms_pre_all'
    template_name = 'power264'
    fixed_threshold = 0.3230072808988765
    # ----------------结果路径-----------------
    root_dir="/home/user/zhangyan/dominating_set/module_result/确定阈值后结果/power264"

    start_time = time.time()
    
    # 1.计算组平均网络  
    average_matrix = group_average(data_path)
    print("组平均网络:",average_matrix)
    # （如果是AAL则去掉后二十个脑区，只保留前90个脑区）
    # average_matrix_90 = average_matrix[:90, :90]


    # 2.固定阈值
    nxG, matrix, used_threshold = generate_network_with_threshold(
        average_matrix, fixed_threshold
    )
    print("matrix:",matrix)   
    print("used_threshold:",used_threshold)   

    ## 3.模块控制 (使用power网络模板替代Louvain社区检测)
    communities = create_communities(brain_area_264)
    number_of_communities = 14     
    print("communities:", communities)
    print("number_of_communities:", number_of_communities)

    # 4.最小支配集分析 - 贪心 xxxxxx 次,并整合多次贪心算法的结果 
    print("贪心算法寻找最小支配集:")
    for times in range(1000, 50001, 1000):
            # 运算得到结果
            all_dom_set = greedy_minimum_dominating_set(nxG, times)            
            save_to_txt(all_dom_set, times, data_type,root_dir)
            # print("all_dom_set_HC:",all_dom_set_HC)    
    print("----贪心算法寻找结束----")
    print("-----整合多次贪心算法的结果------")
    merge_result_path = merge_dominating_results(data_type,root_dir)
    print("merge_result_path: ",merge_result_path)
    
    # # 打印节点数\边数 ，打印网络支配集规模  （集合中的节点数量）
    # print("节点数:", nxG.number_of_nodes(),"网络边数:", nxG.number_of_edges())
    # count_nodes(consolidate_result_path)


    # # 5.节点支配频率
    all_dom_set = read_sets(merge_result_path)

    ## module controllability 这里必须传入louvain community的结果，不能在函数中算，社团编号可能会错乱
    
    print("----------计算average_module_controllability_result--------------")
    average_module_controllability_result = module_controllability(nxG,all_dom_set,communities)
    print("average_module_controllability_result: ",average_module_controllability_result)


    print("----------计算节点支配频率--------------")
    dom_frequency_result = dominating_frequency(all_dom_set,nxG)
    dom_frequency_result_path = save_csv(dom_frequency_result,data_type,root_dir)

    # 分析节点支配频率——从局部脑区分析差异变化
    replace_to_brainarea(dom_frequency_result_path,data_type,root_dir,template_name)
    

    # 保存 average_module_controllability_result 为 CSV data_type
    module_controllability_path = os.path.join(root_dir, data_type)
    module_controllability_path = os.path.join(module_controllability_path, "average_module_controllability_result.csv")
    with open(module_controllability_path, 'w', newline='') as csvfile:
        fieldnames = ['Module', 'Community', 'Controllability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in average_module_controllability_result.items():
            module, community = key.split('_')                                 
            writer.writerow({'Module': module, 'Community': community, 'Controllability': value})

    # 保存 louvain_communities 为 CSV louvain_communities.csv
    louvain_communities_path = os.path.join(root_dir, data_type)
    louvain_communities_path = os.path.join(louvain_communities_path, "louvain_communities.csv")

    with open(louvain_communities_path, 'w', newline='') as csvfile:
        fieldnames = ['Node', 'Community']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for node, community in communities.items():
            writer.writerow({'Node': node, 'Community': community})

    # 计算程序运行时间
    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.4f} 秒")
