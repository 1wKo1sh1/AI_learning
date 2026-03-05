import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

# 读取数据
df = pd.read_excel("./source.xlsx")
print(df.head())

# 填充异常成绩
df = df.fillna(0)

# 提取平时成绩核考试成绩
attendance = df['attendance'].values
exam = df['exam'].values

# 使用NumPy的round函数计算最终成绩，考试分数占70%，出勤分数占30%
finally_data = np.round(exam * 0.7 + attendance * 0.3)

# 将计算得到的最终成绩添加到DataFrame的新列'finally'中
df['finally'] = finally_data

# 使用apply函数根据最终成绩判断是否通过（60分及以上），并创建新列'pass'
df['pass'] = df['finally'].apply(lambda x: 'yes' if x >= 60 else 'no')

# df.to_excel('./source1.xlsx')

# 设置直方图的区间边界，从0到110，步长为10
bins = np.arange(0, 111, 10)
print(bins)
# 使用np.histogram函数计算每个区间的学生人数
hist, bin_edges = np.histogram(df['finally'], bins=bins)
print(hist)
print(bin_edges)

# 创建一个图表实例
fig = plt.figure()
# 计算条形图的宽度
bar_width = (bin_edges[1] - bin_edges[0])
# 绘制条形图，x轴是分数区间，y轴是学生人数
plt.bar(bin_edges[:-1], hist, width=bar_width, align='edge')

# 在每个条形图上添加文本，显示该区间内的学生人数
for i in range(len(hist)):
    if hist[i]:  # 如果该区间有学生，则添加文本
        plt.text(bin_edges[i] + bar_width / 2, hist[i] + 0.1, str(hist[i]), ha='center')

# 设置图表的标题和坐标轴标签
plt.title('finally')
plt.xlabel('score')
plt.ylabel('number')

# 设置x轴的刻度，显示分数区间的边界
plt.xticks(bin_edges[:-1])

# 显示条形图
plt.show()

# 创建一个新的图表实例
fig = plt.figure()

# 使用value_counts方法统计通过和未通过的学生数量
pass_count = df['pass'].value_counts()

# 绘制饼图，显示通过和未通过的比例
plt.pie(pass_count, labels=pass_count.index, autopct='%1.1f%%')
# 设置饼图的标题
plt.title('Distribution of Passing Status')
# 显示饼图
plt.show()