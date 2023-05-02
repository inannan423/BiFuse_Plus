from flask_cors import CORS
import numpy as np
import argparse
from imageio import imread
import matplotlib.pyplot as plt
import cv2
from flask import Flask, request, jsonify
import json
import torch
import BiFusev2
import requests
from io import BytesIO

app = Flask(__name__)

# 允许跨域访问
CORS(app, resources=r'/*')  # 注册 CORS, "/*" 允许访问所有api

network_args = {
    'save_path': './save',
    'dnet_args': {
        'layers': 34,
        'CE_equi_h': [8, 16, 32, 64, 128, 256, 512]
    },
    'pnet_args': {
        'layers': 18,
        'nb_tgts': 2
    }
}


def process_image(img_url):
    img = imread(img_url, pilmode='RGB').astype(np.float32) / 255.0
    # 将图片重设大小为 1024 * 512 imageio
    img = cv2.resize(img, (1024, 512))
    [h, w, _] = img.shape
    assert h == 512 and w == 1024
    batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...]

    # 加载模型
    if request.form['mode'] == 'supervised':
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    elif request.form['mode'] == 'selfsupervised':
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)
    # load ./pretrain/supervised_pretrain.pkl or ./pretrain/selfsupervised_pretrain.pkl 根据 mode 加载不同的模型
    param = torch.load(
        './pretrain/{}_pretrain.pkl'.format(request.form['mode']), map_location='cpu')
    model.load_state_dict(param, strict=False)
    model = model.cpu()
    model.eval()

    # 预测深度图
    with torch.no_grad():
        depth = model.dnet(batch)[0]
    if request.form['mode'] == 'selfsupervised':
        depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

    depth = depth[0, 0, ...].cpu().numpy().clip(0, 10)

    # 将处理后的深度图保存到本地 1.jpg
    plt.imsave('1.jpg', depth, cmap='gray')

    # 上传图片到 https://oss.jetlab.live/api/upload/
    url = 'https://oss.jetlab.live/api/upload/'
    files = {'files': open('1.jpg', 'rb')}
    response = requests.post(url, files=files)
    res = json.loads(response.text)
    # 将 res 保存到本地 res.json
    with open('res.json', 'w') as f:
        json.dump(res, f)
    return res[0]['url']


@app.route('/predict', methods=['POST'])
def predict():
    print(request.form)
    # 如果传入的参数不够，返回错误信息
    # if 'img' not in request.form:
    #     return jsonify({'error': 'missing img'}), 400
    if 'mode' not in request.form:
        return jsonify({'error': 'missing mode'}), 400
    # 如果 img not in request.files，返回错误信息
    if 'imgs' not in request.form:
        return jsonify({'error': 'missing imgs'}), 400

    urls = request.form.getlist('imgs')
    # 将 urls 转为一个数组，数组中的每个元素是一个图片的 url
    urls = urls[0].split(',')
    res_urls = []
    for url in urls:
        res_url = process_image(url)
        res_urls.append(res_url)

    # 将 res_urls 每个元素添加前缀 https://oss.jetlab.live，并返回

    for i in range(len(res_urls)):
        res_urls[i] = 'https://oss.jetlab.live' + res_urls[i]

    # 返回深度图,以图片形式返回
    return jsonify({'urls': res_urls})


if __name__ == '__main__':
    app.run()
