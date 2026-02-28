#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THUCNews新闻分类 - BERT训练脚本 (GPU优化版)

基于bert-base-chinese的中文多分类模型，针对4GB显存环境优化，
提高GPU利用率，包含完整的训练流水线、评估指标和模型保存功能。

作者: AI Assistant
日期: 2024
"""

import os
import sys
import json
import time
import random
import gc
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# ==================== 配置 ====================

@dataclass
class Config:
    """
    训练配置类 - 集中管理所有超参数和路径
    
    针对4GB显存环境优化，默认参数适合受限显存场景。
    """
    # 数据路径
    data_path: str = r'D:\浏览器下载\数据集\THUCNews'
    output_dir: str = './thucnews_final_model'
    best_model_dir: str = './best_model'
    
    # 模型配置
    model_name: str = 'bert-base-chinese'
    num_labels: int = 10
    
    # 训练参数 - 针对4GB显存优化
    max_length: int = 128  # 序列最大长度
    batch_size: int = 8    # 批次大小
    gradient_accumulation_steps: int = 4  # 梯度累积步数，有效batch_size = 8 * 4 = 32
    learning_rate: float = 3e-5
    epochs: int = 5
    warmup_ratio: float = 0.1
    
    # 优化器配置
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 高级优化
    use_fp16: bool = True  # 混合精度训练，大幅提高GPU利用率
    use_gradient_checkpointing: bool = False  # 梯度检查点，节省显存
    label_smoothing: float = 0.1  # 标签平滑
    use_cosine_scheduler: bool = True  # 余弦学习率调度
    
    # 早停配置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # 性能优化配置
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # 数据集划分比例
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 随机种子
    seed: int = 42
    
    # 分类标签
    categories: List[str] = field(default_factory=lambda: [
        '财经', '股票', '教育', '科技', '社会',
        '时尚', '时政', '体育', '游戏', '娱乐'
    ])
    
    # 设备
    device: torch.device = field(init=False)
    
    def __post_init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 验证num_labels与categories一致
        if self.num_labels != len(self.categories):
            self.num_labels = len(self.categories)


# ==================== 工具函数 ====================

def set_seed(seed: int = 42) -> None:
    """
    设置随机种子，确保实验可复现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config(config: Config) -> None:
    """打印训练配置信息"""
    print("=" * 70)
    print("THUCNews新闻分类 - BERT训练脚本 (GPU优化版)")
    print("=" * 70)
    
    print("\n【训练配置】")
    print(f"  模型: {config.model_name}")
    print(f"  类别数: {config.num_labels}")
    print(f"  最大序列长度: {config.max_length}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"  有效批次大小: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  早停耐心: {config.early_stopping_patience}")
    print(f"  标签平滑: {config.label_smoothing}")
    print(f"  混合精度(FP16): {config.use_fp16}")
    print(f"  学习率调度: {'余弦退火' if config.use_cosine_scheduler else '线性'}")
    
    print(f"\n【设备信息】")
    print(f"  使用设备: {config.device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU总显存: {total_mem:.2f} GB")
        # 估算当前显存使用
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"  已分配显存: {allocated:.2f} GB")
    
    print("=" * 70)


def print_gpu_memory(prefix: str = "") -> None:
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"{prefix}GPU显存 - 已分配: {allocated:.2f} GB, 预留: {reserved:.2f} GB")


def clean_gpu_memory() -> None:
    """清理GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== 数据加载模块 ====================

class THUCNewsDataLoader:
    """
    THUCNews数据集加载器
    
    负责从指定目录加载THUCNews数据集，支持自动处理编码问题。
    """
    
    def __init__(self, config: Config):
        """
        初始化数据加载器
        
        Args:
            config: 训练配置对象
        """
        self.config = config
        self.label_to_id = {cat: idx for idx, cat in enumerate(config.categories)}
        self.id_to_label = {idx: cat for cat, idx in self.label_to_id.items()}
    
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
        """
        加载THUCNews数据集
        
        Returns:
            DataFrame: 包含text和label列的数据框
            label_to_id: 标签到ID的映射
            id_to_label: ID到标签的映射
        
        Raises:
            FileNotFoundError: 当数据路径不存在时
        """
        data_path = self.config.data_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        
        print(f"\n开始加载数据集: {data_path}")
        
        all_texts: List[str] = []
        all_labels: List[int] = []
        category_counts: Dict[str, int] = {cat: 0 for cat in self.config.categories}
        
        # 遍历每个类别目录
        for category in tqdm(self.config.categories, desc="加载类别"):
            category_path = os.path.join(data_path, category)
            
            if not os.path.exists(category_path):
                print(f"  警告: 类别目录不存在 - {category}")
                continue
            
            category_id = self.label_to_id[category]
            
            try:
                # 获取该类别下的所有txt文件
                files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
            except OSError as e:
                print(f"  警告: 无法读取目录 {category} - {e}")
                continue
            
            # 读取每个文件
            for file_name in files:
                file_path = os.path.join(category_path, file_name)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                    
                    # 简单的文本质量过滤
                    if self._is_valid_text(text):
                        all_texts.append(text)
                        all_labels.append(category_id)
                        category_counts[category] += 1
                        
                except (OSError, IOError, UnicodeDecodeError):
                    continue
        
        if len(all_texts) == 0:
            raise ValueError("未能加载任何有效数据，请检查数据路径和文件格式")
        
        # 创建DataFrame
        df = pd.DataFrame({
            'text': all_texts,
            'label': all_labels
        })
        
        print(f"\n数据加载完成! 共加载 {len(df):,} 个样本")
        
        # 打印数据分布
        print("\n【数据分布】")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {category}: {count:,}")
        
        return df, self.label_to_id, self.id_to_label
    
    def _is_valid_text(self, text: str) -> bool:
        """
        检查文本是否有效
        
        Args:
            text: 待检查的文本
            
        Returns:
            bool: 文本是否有效
        """
        # 基本过滤条件
        if not text:
            return False
        if len(text) < 20:  # 文本太短
            return False
        if len(text) > 5000:  # 文本太长
            return False
        return True


# ==================== 数据集类 ====================

class THUCDataset(Dataset):
    """
    THUCNews数据集类
    
    用于PyTorch DataLoader的Dataset实现。
    """
    
    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ):
        """
        初始化数据集
        
        Args:
            texts: 原始文本数组
            labels: 标签数组
            input_ids: BERT输入IDs
            attention_mask: BERT注意力掩码
        """
        self.texts = texts
        self.labels = labels
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def tokenize_data(
    texts: np.ndarray,
    tokenizer: BertTokenizer,
    max_length: int,
    batch_size: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对文本进行分词
    
    Args:
        texts: 文本数组
        tokenizer: BERT分词器
        max_length: 最大序列长度
        batch_size: 处理批次大小
        
    Returns:
        input_ids和attention_mask的numpy数组
    """
    print(f"开始分词 (max_length={max_length}, batch_size={batch_size})...")
    
    input_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="分词进度"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx].tolist()
        
        # 分词
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        
        input_ids_list.extend(encoded['input_ids'].tolist())
        attention_mask_list.extend(encoded['attention_mask'].tolist())
    
    print(f"分词完成! 处理了 {len(input_ids_list):,} 个样本")
    
    return np.array(input_ids_list), np.array(attention_mask_list)


# ==================== 损失函数 ====================

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    
    通过对标签进行平滑处理，减少模型对标签的过度自信，提高泛化能力。
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        初始化标签平滑损失
        
        Args:
            num_classes: 类别数量
            smoothing: 平滑因子 (0-1之间)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            pred: 模型预测 logits
            target: 真实标签
            
        Returns:
            损失值
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ==================== 训练器 ====================

class Trainer:
    """
    BERT模型训练器
    
    封装模型训练、验证、评估的完整逻辑。
    """
    
    def __init__(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        config: Config,
        label_to_id: Dict[str, int],
        id_to_label: Dict[int, str]
    ):
        """
        初始化训练器
        
        Args:
            model: BERT分类模型
            tokenizer: BERT分词器
            config: 训练配置
            label_to_id: 标签映射
            id_to_label: ID映射
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.use_fp16 else None
        
        # 损失函数
        self.criterion = LabelSmoothingLoss(
            config.num_labels,
            config.label_smoothing
        )
        
        # 训练历史
        self.history: List[Dict[str, float]] = []
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        
        # 移至设备
        self.model.to(config.device)
    
    def _create_optimizer(self) -> optim.AdamW:
        """创建优化器"""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            },
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        total_steps = self.config.gradient_accumulation_steps * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.use_cosine_scheduler:
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            包含训练指标的字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc='训练', leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            # 混合精度前向传播
            if self.config.use_fp16:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = self.criterion(outputs.logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # 混合精度反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.criterion(outputs.logits, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            # 统计
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            with torch.no_grad():
                predictions = torch.argmax(outputs.logits, dim=1)
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_samples += len(labels)
            
            # 进度条显示
            progress_bar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{total_correct / total_samples:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            包含评估指标的字典
        """
        self.model.eval()
        
        all_predictions: List[int] = []
        all_labels: List[int] = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='评估', leave=False):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                if self.config.use_fp16:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        # 每个类别的指标
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_predictions,
            'labels': all_labels,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1_per_class.tolist()
        }
    
    def get_classification_report(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> str:
        """
        生成分类报告
        
        Args:
            predictions: 预测结果
            labels: 真实标签
            
        Returns:
            格式化的分类报告字符串
        """
        target_names = [self.id_to_label[i] for i in range(self.config.num_labels)]
        
        return classification_report(
            labels,
            predictions,
            target_names=target_names,
            digits=4,
            zero_division=0
        )
    
    def save_best_model(self) -> None:
        """保存最佳模型"""
        os.makedirs(self.config.best_model_dir, exist_ok=True)
        
        self.model.save_pretrained(self.config.best_model_dir)
        self.tokenizer.save_pretrained(self.config.best_model_dir)
        
        print(f"  -> 已保存最佳模型到: {self.config.best_model_dir}")
    
    def should_early_stop(self, val_f1: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_f1: 验证集F1分数
            
        Returns:
            是否应该早停
        """
        improvement = val_f1 - self.best_val_f1
        
        if improvement > self.config.early_stopping_threshold:
            self.best_val_f1 = val_f1
            self.best_val_acc = accuracy_score(
                self.history[-1]['val_labels'],
                self.history[-1]['val_predictions']
            )
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience


# ==================== 主函数 ====================

def create_dataloaders(
    train_dataset: THUCDataset,
    val_dataset: THUCDataset,
    test_dataset: THUCDataset,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        config: 配置对象
        
    Returns:
        训练、验证、测试数据加载器
    """
    # 训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=True
    )
    
    # 验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    # 测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def main():
    """主函数 - 完整的训练流水线"""
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 禁用警告
    warnings.filterwarnings('ignore')
    
    # 设置随机种子
    set_seed(42)
    
    # 创建配置
    config = Config()
    
    # 打印配置
    print_config(config)
    
    total_start_time = time.time()
    
    try:
        # ==================== 1. 数据加载 ====================
        print("\n" + "=" * 70)
        print("【步骤1】加载数据集")
        print("=" * 70)
        
        data_loader = THUCNewsDataLoader(config)
        df, label_to_id, id_to_label = data_loader.load_data()
        
        if len(df) == 0:
            raise ValueError("数据加载失败，没有有效样本")
        
        print_gpu_memory("数据加载后: ")
        
        # ==================== 2. 数据划分 ====================
        print("\n" + "=" * 70)
        print("【步骤2】划分数据集")
        print("=" * 70)
        
        # 划分训练集和测试集
        train_df, temp_df = train_test_split(
            df,
            test_size=config.val_ratio + config.test_ratio,
            random_state=config.seed,
            stratify=df['label']
        )
        
        # 划分验证集和测试集
        val_df, test_df = train_test_split(
            temp_df,
            test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
            random_state=config.seed,
            stratify=temp_df['label']
        )
        
        print(f"数据集划分:")
        print(f"  训练集: {len(train_df):,} 样本 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  验证集: {len(val_df):,} 样本 ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  测试集: {len(test_df):,} 样本 ({len(test_df)/len(df)*100:.1f}%)")
        
        # 释放内存
        del df, temp_df
        gc.collect()
        
        # ==================== 3. 分词 ====================
        print("\n" + "=" * 70)
        print("【步骤3】分词处理")
        print("=" * 70)
        
        print("加载BERT分词器...")
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        
        # 分词处理
        train_texts = train_df['text'].values
        train_labels = train_df['label'].values
        val_texts = val_df['text'].values
        val_labels = val_df['label'].values
        test_texts = test_df['text'].values
        test_labels = test_df['label'].values
        
        train_input_ids, train_attention_mask = tokenize_data(
            train_texts, tokenizer, config.max_length
        )
        
        val_input_ids, val_attention_mask = tokenize_data(
            val_texts, tokenizer, config.max_length
        )
        
        test_input_ids, test_attention_mask = tokenize_data(
            test_texts, tokenizer, config.max_length
        )
        
        # 释放内存
        del train_texts, val_texts, test_texts
        gc.collect()
        
        # ==================== 4. 创建数据集 ====================
        print("\n" + "=" * 70)
        print("【步骤4】创建数据集")
        print("=" * 70)
        
        train_dataset = THUCDataset(
            train_df['text'].values,
            train_labels,
            train_input_ids,
            train_attention_mask
        )
        
        val_dataset = THUCDataset(
            val_df['text'].values,
            val_labels,
            val_input_ids,
            val_attention_mask
        )
        
        test_dataset = THUCDataset(
            test_df['text'].values,
            test_labels,
            test_input_ids,
            test_attention_mask
        )
        
        # 释放分词结果内存
        del train_input_ids, train_attention_mask
        del val_input_ids, val_attention_mask
        del test_input_ids, test_attention_mask
        del train_df, val_df, test_df
        gc.collect()
        
        clean_gpu_memory()
        
        print(f"数据集创建完成!")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        # ==================== 5. 创建数据加载器 ====================
        print("\n" + "=" * 70)
        print("【步骤5】创建数据加载器")
        print("=" * 70)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, config
        )
        
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        
        # ==================== 6. 加载模型 ====================
        print("\n" + "=" * 70)
        print("【步骤6】加载BERT模型")
        print("=" * 70)
        
        model = BertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        
        # 梯度检查点（可选，节省显存）
        if config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("已启用梯度检查点")
        
        print(f"模型加载成功!")
        print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        print_gpu_memory("模型加载后: ")
        
        # ==================== 7. 训练 ====================
        print("\n" + "=" * 70)
        print("【步骤7】开始训练")
        print("=" * 70)
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            label_to_id=label_to_id,
            id_to_label=id_to_label
        )
        
        train_start_time = time.time()
        
        for epoch in range(config.epochs):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config.epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_metrics = trainer.train_epoch(train_loader)
            
            # 验证
            val_metrics = trainer.evaluate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"\nEpoch {epoch + 1} 完成 (用时: {epoch_time:.1f}秒)")
            print(f"  训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
            
            # 保存历史
            trainer.history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_macro'],
                'val_predictions': val_metrics['predictions'],
                'val_labels': val_metrics['labels']
            })
            
            # 保存最佳模型
            if val_metrics['f1_macro'] > trainer.best_val_f1:
                trainer.best_val_f1 = val_metrics['f1_macro']
                trainer.best_val_acc = val_metrics['accuracy']
                trainer.save_best_model()
                print(f"  ** 保存最佳模型 (F1: {val_metrics['f1_macro']:.4f})")
            
            # 早停检查
            if trainer.should_early_stop(val_metrics['f1_macro']):
                print(f"\n触发早停! 连续{config.early_stopping_patience}个epoch没有改善")
                break
            
            # 清理GPU缓存
            clean_gpu_memory()
        
        train_time = time.time() - train_start_time
        print(f"\n训练完成! 总用时: {train_time/60:.2f} 分钟")
        
        # ==================== 8. 测试评估 ====================
        print("\n" + "=" * 70)
        print("【步骤8】测试集评估")
        print("=" * 70)
        
        # 加载最佳模型进行测试
        print("加载最佳模型...")
        del model
        clean_gpu_memory()
        
        model = BertForSequenceClassification.from_pretrained(config.best_model_dir)
        model.to(config.device)
        
        # 测试
        print("运行测试...")
        trainer.model = model
        test_metrics = trainer.evaluate(test_loader)
        
        print(f"\n【测试结果】")
        print(f"  准确率: {test_metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        
        print(f"\n【详细分类报告】")
        print(trainer.get_classification_report(
            test_metrics['predictions'],
            test_metrics['labels']
        ))
        
        # ==================== 9. 保存最终模型 ====================
        print("\n" + "=" * 70)
        print("【步骤9】保存模型和结果")
        print("=" * 70)
        
        # 保存最终模型
        os.makedirs(config.output_dir, exist_ok=True)
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        print(f"模型已保存到: {config.output_dir}")
        
        # 保存训练结果
        results = {
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1_macro': float(test_metrics['f1_macro']),
            'test_f1_weighted': float(test_metrics['f1_weighted']),
            'best_val_accuracy': float(trainer.best_val_acc),
            'best_val_f1': float(trainer.best_val_f1),
            'train_time_minutes': float(train_time / 60),
            'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset),
            'config': {
                'batch_size': config.batch_size,
                'gradient_accumulation_steps': config.gradient_accumulation_steps,
                'effective_batch_size': config.batch_size * config.gradient_accumulation_steps,
                'max_length': config.max_length,
                'learning_rate': config.learning_rate,
                'epochs': config.epochs,
                'use_fp16': config.use_fp16,
                'use_cosine_scheduler': config.use_cosine_scheduler,
                'label_smoothing': config.label_smoothing,
                'model_name': config.model_name
            }
        }
        
        results_path = os.path.join(config.output_dir, 'results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {results_path}")
        
        # 打印最终总结
        total_time = time.time() - total_start_time
        print("\n" + "=" * 70)
        print("【训练完成总结】")
        print("=" * 70)
        print(f"  总用时: {total_time/60:.2f} 分钟")
        print(f"  训练用时: {train_time/60:.2f} 分钟")
        print(f"  最佳验证F1: {trainer.best_val_f1:.4f}")
        print(f"  测试准确率: {test_metrics['accuracy']:.4f}")
        print(f"  测试F1: {test_metrics['f1_macro']:.4f}")
        print(f"  模型保存路径: {config.output_dir}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
