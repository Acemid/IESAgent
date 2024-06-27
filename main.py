# 小论文-LLMopt区域供热系统LLM辅助调控优化
# -- coding: utf-8 --
# -------------------------------
# @Author : Ning Zhang
# @Email : zhang_n@zju.edu.cn
# -------------------------------
# @File : main.py
# @Time : 2024/6/25 下午4:22
# -------------------------------
# 主程序-End2End LLMopt Agent
# LLM赋予opt，而不是opt赋予LLM
# Data-->Agent-->Output

import pandas as pd
import json
from load_pred import pred_build

# 数据输入
model_path = '/root/Shanghai_AI_Laboratory/internlm2-math-7b'  # 模型选择

mech_info = pd.read_csv('data/info_data.csv', index_col=0)
mech_json = mech_info.to_json(orient='index')  # 负荷机理数据情况

data_load1 = pd.read_csv('data/load1.csv')
data_load2 = pd.read_csv('data/load2.csv')
data_load3 = pd.read_csv('data/load3.csv')
his_data = {
    'load1': {"历史数据量": len(data_load1), 'data_source': 'data/load1.csv'},
    'load2': {"历史数据量": len(data_load2), 'data_source': 'data/load2.csv'},
    'load3': {"历史数据量": len(data_load3), 'data_source': 'data/load3.csv'}
}  # 负荷历史数据情况
his_json = json.dumps(his_data)

# 负荷预测
pred_models = pred_build(model_path, his_json, mech_json)  # 返回根据各个load对应的负荷预测模型的路径列表

# 系统模型


# 优化调控


# 输出
