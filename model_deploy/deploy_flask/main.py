import requests


resp = requests.post(
    "http://localhost:5000/predict", 
    files = {"file": open("/Users/zfwang/projects/cvproj/deeplearning/src/src_pytorch/deploy/deploy_flask/cat.jpg", "rb")}
)
print(resp.json())
