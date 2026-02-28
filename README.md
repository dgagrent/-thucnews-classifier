# THUCNews 新闻分类系统

基于 BERT 的中文新闻分类模型，支持10个新闻类别的自动分类。

## 项目简介

本项目使用 `bert-base-chinese` 预训练模型，在 THUCNews 数据集上进行微调训练，实现中文新闻的自动分类。

## 模型性能

| 指标 | 数值 |
|------|------|
| 测试准确率 | 85.00% |
| F1 Score | 0.83 |
| 训练数据量 | 76.7万条 |

## 新闻类别

支持以下10个新闻类别的分类：

- 财经
- 股票
- 教育
- 科技
- 社会
- 时尚
- 时政
- 体育
- 游戏
- 娱乐

## 环境要求

- Python 3.8+
- PyTorch
- transformers
- CUDA (GPU推荐)

## 数据集下载

本项目使用 THUCNews 数据集，需要单独下载：

**下载地址**：https://github.com/THUCNews

**本地路径配置**：
训练脚本 `bert_text_classifier.py` 中 `data_path` 默认配置为：
```python
data_path = r'D:\\浏览器下载\\数据集\\THUCNews'
```
请根据实际情况修改为本地数据集路径。

## 项目结构

## 项目结构

```
pytest/
├── bert_text_classifier.py  # 训练脚本
├── app.py                  # Flask API服务
├── test_model.py           # 测试脚本
├── templates/
│   └── index.html         # 前端页面
└── thucnews_final_model/  # 训练好的模型
```

## 快速开始

### 1. 训练模型

```bash
python bert_text_classifier.py
```

### 2. 启动API服务

```bash
python app.py
```

然后访问 http://localhost:5000

### 3. 测试模型

```bash
python test_model.py
```

## 技术特点

- **模型**: BERT-base-chinese (1.02亿参数)
- **优化**: FP16混合精度训练
- **显存优化**: 梯度累积、4GB显存适配
- **训练技巧**: 标签平滑、余弦学习率、早停机制

## 使用示例

通过前端界面：
1. 输入新闻文本
2. 点击"开始分类"
3. 查看分类结果

通过API：

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A股市场今日大涨，沪指突破3500点"}'
```

## 许可证

MIT License
