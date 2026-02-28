#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THUCNews新闻分类 - Flask API服务
提供RESTful API接口用于新闻分类预测
"""

import os
import json
import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

# 配置
MODEL_PATH = './thucnews_final_model'
RESULTS_PATH = './thucnews_final_model/results.json'
CATEGORIES = ['财经', '股票', '教育', '科技', '社会', '时尚', '时政', '体育', '游戏', '娱乐']

app = Flask(__name__, template_folder='templates', static_folder='static')

# 全局变量
tokenizer = None
model = None
device = None


def load_model():
    """加载BERT模型"""
    global tokenizer, model, device
    
    print("加载模型中...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    print(f"模型加载成功! 设备: {device}")
    print(f"可用类别: {CATEGORIES}")


def load_results():
    """加载训练结果"""
    try:
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def predict(text: str, top_k: int = 3):
    """预测新闻类别"""
    if not text or len(text.strip()) == 0:
        return []
    
    # 分词
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    # 获取top_k结果
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(CATEGORIES)))
    
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            'category': CATEGORIES[idx.item()],
            'confidence': round(prob.item() * 100, 2)
        })
    
    return results


@app.route('/')
def index():
    """主页"""
    results = load_results()
    return render_template('index.html', results=results)


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """预测API"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': '请提供text字段'}), 400
    
    text = data['text']
    top_k = data.get('top_k', 3)
    
    try:
        results = predict(text, top_k=top_k)
        return jsonify({
            'success': True,
            'data': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """获取所有类别"""
    return jsonify({
        'success': True,
        'categories': CATEGORIES
    })


if __name__ == '__main__':
    load_model()
    print("\n" + "=" * 50)
    print("Flask服务启动成功!")
    print("访问地址: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
