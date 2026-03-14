# -*- coding: utf-8 -*-


# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************


# python libraries
import os
import io
import json
import flask
from flask import Flask, jsonify, request
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


print(flask.__version__)
print(torchvision.__version__)


# flask app
app = Flask(__name__)
# Trained on 1000 classes from ImageNet
model = models.densenet121(pretrained = True)
# Turns off autograd
model.eval()


# image class index
img_class_map = None
mapping_file_path = "/Users/zfwang/projects/cvproj/deeplearning/src/src_pytorch/deploy/deploy_flask/imagenet_class_index.json"
if os.path.isfile(mapping_file_path):
    with open(mapping_file_path) as f:
        img_class_map = json.load(f)


def transform_image(infile):
    """
    Transform input into the form model expects

    Args:
        infile ([type]): [description]

    Returns:
        [type]: [description]
    """
    # open the image file
    image = Image.open(infile)
    # image = Image.open(io.BytesIO(infile))
    
    # use multiple TorchVision transforms to transform 
    # PIL image to appropriately-shaped PyTorch tensor
    input_transforms = [
        # resize
        transforms.Resize(255),
        # center crop
        transforms.CenterCrop(224),
        # to tensor
        transforms.ToTensor(),
        # Standard normalization for ImageNet model input
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    my_transforms = transforms.Compose(input_transforms) 
    timg = my_transforms(image)
    
    # PyTorch models expect batched input; create a batch of 1
    timg.unsqueeze_(0)
    
    return timg


def get_prediction(input_tensor):
    """
    get a prediction

    Args:
        input_tensor ([type]): [description]

    Returns:
        [type]: [description]
    """
    # get likelihoods for all ImageNet classes
    outputs = model.forward(input_tensor)
    # extract the most likely class
    __, y_hat = outputs.max(1)
    prediction = y_hat.item()

    return prediction


def render_prediction(prediction_idx):
    """
    make the prediction human-readable

    Args:
        prediction_idx ([type]): [description]

    Returns:
        [type]: [description]
    """
    stridx = str(prediction_idx)
    class_name = "Unkonwn"
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]
        
    return prediction_idx, class_name


@app.route("/", methods = ["GET"])
def root():
    return jsonify({
        "msg": "Try POSTing to the /predict endpoint with an RGB image attachment."
    })


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({
                'class_id': class_id, 
                'class_name': class_name
            })




# 测试代码 main 函数
def main():
    app.run()


if __name__ == "__main__":
    main()
