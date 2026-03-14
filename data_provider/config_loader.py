# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091416
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import yaml
from typing import Dict

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_yaml(file_name):
    with open(file_name, 'r', encoding = "utf-8") as infile:
        return yaml.load(
            infile, 
            Loader = yaml.FullLoader
        )


def get_params(yaml_path: str) -> Dict:
    """
    读取项目配置参数

    Returns:
        Dict: 项目配置参数
    """
    # 配置文件读取
    cfg_dir = os.path.dirname(__file__)
    # 项目配置 yaml 文件
    cfg_params = load_yaml(os.path.join(cfg_dir, yaml_path))

    return cfg_params




# 测试代码 main 函数
def main(): 
    sys_cfg_path = "./config/config.yaml"
    cfg_params = get_params(sys_cfg_path)
    print(cfg_params)

if __name__ == "__main__":
    main()
