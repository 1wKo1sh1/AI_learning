"""
标准Softmax	  O(V)	 准确，但极慢
负采样	    O(k)k 通常 5~20	    简单、高效，对高频词和罕见词都友好
负采样 => 最大化正样本接近1，最小化负样本接近0


大幅加速训练：每次更新只需计算 k+1个输出向量（1 正 + k 负）的梯度。
提高词向量质量：迫使模型不仅学会预测正确上下文，还要学会区分错误上下文，从而学到更鲁棒的表示。
对高频词自动降采样：因为负采样分布基于词频，高频词频繁被用作负样本，模型不会过度关注它们。
"""
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
import re

# 数据输入
corpus = [
    "jack like dog", "jack like cat", "jack like animal",
    "dog cat animal", "banana apple cat dog like", "dog fish milk like",
    "dog cat animal like", "jack like apple", "apple like", "jack like banana",
    "apple banana jack movie book music like", "cat dog hate", "cat dog like"
]
# 将句子分词，并转换为小写
def tokenize(sentence):
    # \b 单词的边界
    # \w+ 匹配一个或者多个单词字符（字母，数字，下划线）
    # \[,.!?] 匹配逗号、句号、感叹号和问号
    word_list = []
    for word in re.findall(r"\b\w+\b|[,.!?]", sentence):
        word_list.append(word.lower())
    return word_list
# words为全部单词的集合
words = []
for sentence in corpus:
    for word in tokenize(sentence):
        words.append(word)

word_counts = Counter(words)
# 去重排序进行编码
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
# {"like": 1, "dog": 2}
# 创建词汇表到索引的映射
vocab2int = {word: ii for ii, word in enumerate(vocab, 1)}
# 索引->词汇
int2vocab = {ii: word for ii, word in enumerate(vocab, 1)}
# 将所有单词集合中的单词变成索引状态
word2index = [vocab2int[word] for word in words]
window = 1
center = []
context = []
num_negative = 2
vocab_size = len(vocab2int) + 1     #seq-len或者词汇表大小 = 13+1
embedding_dim = 2  # 嵌入维度
negative_samples = []

# 枚举，从下标window开始
for i, target in enumerate(word2index[window: -window], window):
    # print(i, target)
    # 数据：3 1 2 3 1 4
    # 索引：0 1 2 3 4 5
    center.append(target)
    context.append(word2index[i-window: i] + word2index[i+1: i+1+window])
    # 负样本
    con = word2index[i-window: i] + word2index[i+1: i+1+window]
    nagative_samples_i = []

    for _ in range(num_negative):
        negative_sample = np.random.choice(vocab_size)
        while negative_sample == target or negative_sample in con:
            negative_sample = np.random.choice(vocab_size)
        nagative_samples_i.append(negative_sample)
        negative_samples.append(nagative_samples_i)

torch.manual_seed(0)
# 特殊标记：0，用于填充或者标记未知单词
# <SOS>: 句子起始标识符
# <EOS>：句子结束标识符
# <PAD>：补全字符
# <MASK>：掩盖字符
# <SEP>：两个句子之间的分隔符
# <UNK>：低频或未出现在词表中的词

class CBOWModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super().__init__()
        # 使用embedding类直接实例化这个嵌入矩阵
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # 初始化嵌入矩阵padding=0为遇到索引为0自动填充
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        # context.shape -> (bs=4, 2,2)定义banchsize为4，每个bs内的数据为2*2形状（2行上下文，同时嵌入维度为2）
        context_emb =  self.embedding(context)  # 取出嵌入矩阵的对应参数
        # 内置功能=>自动处理输入的索引值查找相应嵌入向量，支持padding索引用于处理变长序列，提供权重初始化选项
        avg_emb = torch.mean(context_emb, dim=1, keepdim=True).squeeze(1)  # 求平均，去除行数维度，因为两行求平均变成一行了
        y = self.linear(avg_emb)
        return y

model = CBOWModel(vocab_size, embedding_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

bs = 4
epochs = 2000
for epoch in range(1, epochs + 1):
    total_loss = 0
    for batch_index in range(0, len(context), bs):
        # 上下文的tensor
        context_tensor = torch.tensor(context[batch_index: batch_index + bs]) # (4 2)
        # 中心词的tensor
        center_tensor = torch.tensor(center[batch_index: batch_index + bs]) # (4 1)
        # 负样本tensor
        negative_tensor = torch.tensor(negative_samples[batch_index: batch_index + bs])  # (4 2)

        # -----------正样本的损失positive
        context_emb = model.embedding(context_tensor) # (4 2 2)
        center_emb = model.embedding(center_tensor) # (4 2)
        avg_context_emb = torch.mean(context_emb, dim =1) # 平均(4 2)
        # 内积
        positive_scores = torch.matmul(avg_context_emb, center_emb.t())
        positive_labels = torch.eye(bs)

        positive_loss = criterion(positive_scores, positive_labels)

        # ----------负样本的损失negatibe
        negative_emb = model.embedding(negative_tensor)
        negative_scores = torch.matmul(avg_context_emb.unsqueeze(1), negative_emb.permute(0,2,1).squeeze(1))
        negative_labels = torch.zeros_like(negative_scores)
        negative_loss = criterion(negative_scores, negative_labels)

        loss = positive_loss + negative_loss
        # output = model(context_tensor)
        # loss = criterion(output, center_tensor)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(context)
    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}]  Loss {avg_loss:.4f}")
        # 每个单词的embedding向量
        word_vec = model.embedding.weight.detach().numpy()
        # print(word_vec)
        x = word_vec[:, 0]
        y = word_vec[:, 1]
        selected_word = ["dog", "cat", "milk","like","animal","fish","banana","apple"]
        selected_word_index = [vocab2int[word] for word in selected_word]
        selected_word_x = x[selected_word_index]
        selected_word_y = y[selected_word_index]
        plt.cla()
        plt.scatter(selected_word_x, selected_word_y, color="blue")
        # 将每个点的标注加上
        for word, x, y in zip(selected_word, selected_word_x, selected_word_y):
            plt.annotate(word, (x, y), textcoords="offset points", xytext=(0, 10))
        plt.pause(0.5)

plt.show()