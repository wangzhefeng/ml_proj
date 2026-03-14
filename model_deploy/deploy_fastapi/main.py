# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
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
from typing import List

import joblib
import uvicorn
from fastapi import FastAPI

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


app = FastAPI(
    title = "Iris Prediction Model API",
    description = "A simple API that use LogisticRegression model to predict the Iris species",
    version = "0.1",
) 


def model_load():
    """
    加载训练好的分类模型

    :return: _description_
    :rtype: _type_
    """
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            "models/IrisClassifier.pkl"
        ), "rb"
    ) as f:
        model = joblib.load(f)
    
    return model


def data_parse(params_str):
    """
    请求参数处理

    :param params_str: _description_
    :type params_str: _type_
    :return: _description_
    :rtype: _type_
    """
    params_array = params_str.split(",")
    params_array = list(map(float, params_array))

    return params_array


@app.get("/predict-result")
def predict_iris(request):
    """
    create prediction endpoint

    :param request: _description_
    :type request: _type_
    :return: _description_
    :rtype: _type_
    """
    # get model 
    model = model_load()
    # perform prediction
    request = data_parse(request)
    prediction = model.predict([request])
    output = int(prediction[0])
    probas = model.predict_proba([request])
    output_probability = "{:.2f}".format(float(probas[:, output]))

    # output dictionary
    species = {
        0: "Setosa",
        1: "Versicolour",
        2: "Virginica",
    }

    # show results
    result = {
        "prediction": species[output],
        "Probability": output_probability,
    }

    return result



# 测试代码 main 函数
def main():
    uvicorn.run(app, host = "127.0.0.1", port = 8001)
    # request = "7.233,4.652,7.39,0.324"
    # result = predict_iris(request)
    # print(result)

if __name__ == "__main__":
    main()
