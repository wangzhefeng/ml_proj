# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052222
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_json_config(config_filename):
    """
    读取项目配置参数
    """
    # json 配置文件所在路径
    cfg_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(cfg_dir, config_filename)
    # 读取项目配置 json文件
    with open(cfg_path, "r", encoding = "utf-8") as infile:
        cfg_params = json.load(infile)

    # 构建模型保存文件夹
    if not os.path.exists(cfg_params['model']['save_dir']): 
        os.makedirs(cfg_params['model']['save_dir'])

    return cfg_params




# 测试代码 main 函数
def main():
    configs = load_json_config("config_sp500_1.json")
    logger.info(configs)

if __name__ == "__main__":
    main()
