import pandas as pd
from sklearn.model_selection import train_test_split # 训练集划分
from sklearn.preprocessing import StandardScaler,LabelEncoder # 标准化, 编码分类变量的类别标签变成数值内容
from sklearn.naive_bayes import GaussianNB # 正态分布
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report # 准确率，混淆矩阵，分类报告 => 计算直指标
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1.数据处理
dataset = pd.read_csv('../dataset/iris.data')
# 列名
column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class' ]
dataset.columns = column_names

# 查看前几行
print(dataset.head())

# 数据特征值
X = dataset.iloc[:,:4]
Y = dataset.iloc[:,-1]

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# 初始化朴素贝叶斯分类器
nb_classifier = GaussianNB()

# 根据划分去训练
nb_classifier.fit(X_train, Y_train)

# 测试集上预测
Y_pred = nb_classifier.predict(X_test)

# 准确率：计算分类模型的准确率，还可以计算准确个数
accuracy = accuracy_score(Y_test,Y_pred)

# 混淆矩阵：分类模型有用的工具，直观展示分类模型在各个类别上的分类情况
# 返回一个二维数组，行为真实类别，列为预测类别
conf_matrix = confusion_matrix(Y_test,Y_pred)

"""
              原始0   原始其他
预测为0       真正     假正
预测为其他     假反     真反

准确率：该类被正确预测(真正例TP)的样本数与所有被预测为该类(假正例FP+真正例TP)的样本数比例，即正类样本种有多少是真的正类，得分高=很少误判(希望预测垃圾邮件确实为垃圾邮件，不要把正常的误判为垃圾)
召回率：该类被正确预测(真正例TP)的样本数与所有实际属于该类(假反例FN+真正例TP)的样本数比例，即正类中有多少被真的预测出来了，得分高=很少漏掉(希望找出全部癌症患者，正常的误判为癌症可以接受)
f1值：精准率和召回率的调和平均 2 * pre * recall / (pre + recall)，反应模型整体性能
支持度：每个类在真实标签中出现的样本数量(某个样本里的样本数比较少可能会有偶然性，需要谨慎对待这些样本数少的结果)
宏平均：各个类别指标(re,recall,f1)的简单平均
加权平均：根据指标的支持度进行加权平均，更好反应数量不均衡样本的指标
"""
report = classification_report(Y_test,Y_pred)

print(f'Accuracy:{accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{report}')

# 可视化，但是原来为4维，所以使用t-sne算法进行降维到2维然后可视化
# t-sne随机邻域嵌入算法 => 高维点的数据映射到低维基于点和点之间的概率分布尽可能保持相似度关系
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(X_test_scaler)

# 字符串编码为数值进行绘图
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(Y_pred) # fit和transform是两个函数也可以一起运行
# 编码结果为列表保存0 1 2整数信息，即三类对应的编码

# 绘图
plt.figure(figsize=(8,6))
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_test_numeric, cmap='viridis')
plt.legend(*scatter.legend_elements(),title="Classes")
plt.show()