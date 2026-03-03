import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 第一张绘图区域，分为四个区块
fig = plt.figure(figsize=(12, 6)) # 图形大小 12英寸*6英寸
gs = gridspec.GridSpec(2, 2) # 划分为4个区块

# 线图
x1 = np.linspace(-2*np.pi, 2*np.pi, 10000)
y1 = np.tan(x1)
y1[np.abs(y1)>10] = np.nan

ax1 = fig.add_subplot(gs[0, 0]) # 展示在区块00上
ax1.grid(True)
ax1.plot(x1,y1,'-',) # 绘制

# 散点图
x2 = np.linspace(-2*np.pi, 2*np.pi, 100)
y2 = np.sin(x2)
colors = y2

ax2 = fig.add_subplot(gs[1, 0])
ax2.grid()
ax2.scatter(x2,y2,c=colors)

# 三维图  可以不显式 mpl_toolkits.mplot3d import Axes3D（通过 projection='3d' 自动触发)
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sin(X)+np.sin(Y)

ax3 = fig.add_subplot(gs[:, 1], projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis')

# 条形图
fig2 = plt.figure()
labels = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 33]

plt.bar(labels, values,
        width=0.3,       # 条形的宽度
        bottom=0,
        align='edge',  # 条形与x位置的对齐方式
        data=None,
        color='r',  # 条形的填充颜色, 和facecolor等价
        edgecolor='r',   # 条形边缘的颜色
        # facecolor = 'g', # 填充颜色
        linewidth=2,     # 条形边缘的线宽
        linestyle='-',   # 条形边缘的线型
        alpha=0.7,       # 条形的透明度
        hatch='x',       # 条形的填充图案
        log = False,     # 条形的高度不以对数尺度表示。
        label='test'     # 为条形创建图例时使用的标签
       )
# 1.data参数plt.bar(x='categories', height='values', data=data),即通过关键字直接索引字典
# 2.piker关键字
'''
def on_pick(event):
    # 获取被点击的条形
    if isinstance(event.artist, plt.Rectangle):  # 只对条形进行操作
        event.artist.set_facecolor('red')  # 改变被点击条形的填充颜色
        plt.draw()  # 更新图形
# 连接事件
plt.gcf().canvas.mpl_connect('pick_event', on_pick)
'''
# 饼图
fig3 = plt.figure()
sizes = [25, 35, 20, 21]
labels = ['A', 'B', 'C', 'D']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

plt.pie(sizes,                   # 饼图中每个扇形的尺寸
        explode=[0.5, 0, 0, 0],    # 用于指定每个扇形是否突出显示
        labels=labels,           # 用于指定每个扇形的标签
        colors=colors,           # 用于指定每个扇形的颜色
        autopct='%.1f%%',       # 用于在饼图上显示每个扇形的百分比
        startangle=0,          # 饼图开始的角度，默认为 0（即从 x 轴正方向开始）
        shadow=True,            # 用于指定是否为饼图添加阴影
        radius=1,                # 饼图的半径
        wedgeprops=dict(edgecolor='black', linewidth=2, linestyle='-'),  # 指定饼图中每个扇形的属性
        textprops=dict(color='red', weight='bold'),  # 用于指定饼图中标签的文本属性，这里指定文本的颜色和字体的粗细为粗体
        center=(0, 0),            # 用于指定饼图的中心位置
        frame=False,              # 用于指定是否为饼图添加一个坐标轴
        normalize = True ,         # 为True则x中的值将被归一化以使它们的总和等于1
        hatch = 'x'                # 图案填充
)


plt.show()