# -*- coding: utf-8 -*-

# ***************************************************
# * File        : onnx_resnet50.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-01
# * Version     : 0.1.090120
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

import torch
import torchvision

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


model = torchvision.models.resnet50(
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)

op_set = 13
dummy_data = torch.randn((1, 3, 224, 224))
dummdy_data_128 = torch.randn((128, 3, 224, 224))

# 固定 batch = 1
torch.onnx.export(
    model,
    (dummy_data),
    "resnet50_bs_1.onnx",
    opset_version = op_set,
    input_names = ["input"],
    output_names = ["output"],
)
# 固定 batch = 128
torch.onnx.export(
    model,
    (dummdy_data_128),
    "resnet50_bs_128.onnx",
    opset_version = op_set,
    input_names = ["input"],
    output_names = ["output"],
)
# 动态 batch
torch.onnx.export(
    model,
    (dummy_data),
    "resnet50_bs_dynamic.onnx",
    opset_version = op_set,
    input_names = ["input"],
    output_names = ["output"],
    dynamic_axes = {
        "input": {0: "batch_axes"},
        "output": {0: "batch_axes"},
    },
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
