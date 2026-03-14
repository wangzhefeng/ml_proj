# -*- coding: utf-8 -*-

# ***************************************************
# * File        : onnx_lr.py
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

import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model, 
    make_node,
    make_graph,
    make_tensor_value_info,
)

print(f"onnx_version = {onnx.version}, opset={onnx.defs.onnx_opset_version()}")
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# tensor value info
# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

# node
node1 = make_node("MatMul", ["X", "A"], ["XA"])
node2 = make_node("Add", ["XA", "B"], ["Y"])

# graph
graph = make_graph(
    [node1, node2],  # nodes
    "lr",  # name
    [X, A, B],  # inputs
    [Y]  # output
)

# model
onnx_model = make_model(graph)

with open("results/pretrained_models/linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
