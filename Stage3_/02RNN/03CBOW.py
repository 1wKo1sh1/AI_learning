"""
onehot有着两个缺点，维数灾难和语义鸿沟
Embedding过程类似linear的过程，通过训练w矩阵将onehot编码的稀疏矩阵转化为稠密矩阵，解决维数灾难。而自己w本身就是可学习的，可以解决语义鸿沟问题
最基础的embedding为nntorch下自带的embedding方法，通过训练得到结果。现展示最经典的其中一个Word2Vec算法来进行实现

Word2Vec有两种网络结构:1为CBOW即连续词袋模型=>通过上下文预测中心词
                     2为skip_gram=>通过中心词预测上下文

既然onehot通过w变成embedding是学习的过程，现在舍弃onehot这个过程，直接定义一个初始随机的embedding矩阵(行为seq-len，列为emb-dim)进行学习
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
# 枚举，从下标window开始
for i, target in enumerate(word2index[window: -window], window):
    # print(i, target)
    # 数据：3 1 2 3 1 4
    # 索引：0 1 2 3 4 5
    center.append(target)
    context.append(word2index[i-window: i] + word2index[i+1: i+1+window])

torch.manual_seed(0)
# 特殊标记：0，用于填充或者标记未知单词
# <SOS>: 句子起始标识符
# <EOS>：句子结束标识符
# <PAD>：补全字符
# <MASK>：掩盖字符
# <SEP>：两个句子之间的分隔符
# <UNK>：低频或未出现在词表中的词
vocab_size = len(vocab2int) + 1     #seq-len或者词汇表大小 = 13+1
embedding_dim = 2  # 嵌入维度
class CBOWModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim)) # 初始化嵌入矩阵
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        # context.shape -> (bs=4, 2,2)定义banchsize为4，每个bs内的数据为2*2形状（2行上下文，同时嵌入维度为2）
        context_emb =  self.embedding[context]  # 取出嵌入矩阵的对应参数
        avg_emb = torch.mean(context_emb, dim=1, keepdim=True).squeeze(1)  # 求平均，去除行数维度，因为两行求平均变成一行了
        y = self.linear(avg_emb)
        return y

model = CBOWModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

bs = 4
epochs = 2000
for epoch in range(1, epochs + 1):
    total_loss = 0
    for batch_index in range(0, len(context), bs):
        # 上下文的tensor
        context_tensor = torch.tensor(context[batch_index: batch_index + bs])
        # 中心词的tensor
        center_tensor = torch.tensor(center[batch_index: batch_index + bs])

        output = model(context_tensor)
        loss = criterion(output, center_tensor)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(context)
    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}]  Loss {avg_loss:.4f}")
        # 每个单词的embedding向量
        word_vec = model.embedding.data.numpy()
        # print(word_vec)
        x = word_vec[:, 0]
        y = word_vec[:, 1]
        selected_word = ["dog", "cat", "milk"]
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