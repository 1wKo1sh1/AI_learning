import math
x = [1, 2]
y = [4, 6]

# 欧氏 l2
def euclidean_distance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
# def l2(a,b):
#     return norm(np.array(a) - np.array(b))

print("欧氏距离:", euclidean_distance(x, y))
'''
缺点:使用此度量前需要标准化数据,随着维度增加,欧氏距离用处也就越小
d = ((x1-x2)**2 + (y1-y2)**2 + ...)**(1/2)
实际情况可以去掉开方运算
'''

# 曼哈顿 l1
def manhattan_distance(x, y):
    return sum([abs(a - b) for a, b in zip(x, y)])

print("曼哈顿距离:", manhattan_distance(x, y))
''' 
缺点:不是最短路径,能给出更高的距离值,维度增加用处变小
 d = abs(x1-x2) + abs(y1-y2)+...
'''

# 切比雪夫
def chebyshev_distance(x, y):
    return max([abs(a - b) for a, b in zip(x, y)])

print("切比雪夫距离:", chebyshev_distance(x, y))
'''
缺点:用于特殊例,很难像欧式通用
d = max(abs(x1-x2), abs(y1-y2)), ...)
'''

# 余弦相似度
def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = math.sqrt(sum(a ** 2 for a in x)) * math.sqrt(sum(b ** 2 for b in y))
    return numerator / denominator
# def cos_sim(x,y):
#     return dot(x,y)/(norm(x)*norm(y))
print("余弦相似度:", cosine_similarity(x, y))
'''
两组数判断,即两个向量方向差距;NLP自然语言领域用的比较多即TOKEN转化为向量
缺点:无法捕捉幅度信息,只考虑方向,不考虑大小
t = (x1y1+x2y2+...) / ([(x1**2+x2**2+...)**(1/2)]*[(y1**2+y2**2+...)**(1/2)])
实际向量计算使用np库
'''

# 汉明
def hamming_distance(x_str, y_str):
    return sum(a != b for a, b in zip(x_str, y_str))

x_str = "101100"
y_str = "111000"
print("汉明距离:", hamming_distance(x_str, y_str))
'''
两组等长字符串相同位置上不同字符串个数
缺点:字符串长度不等很难用汉明来衡量
d = int(x1!=y1)+int(x2!=y2)+...
'''

# 闵可夫斯基
def minkovski_distance(x, y, p):
    return sum(abs(a - b) ** p for a, b in zip(x, y)) ** (1 / p)

p = 100
print("闵可夫斯基距离:", minkovski_distance(x, y, p))
'''
欧式p=2,曼哈顿p=1,切比雪夫的p=∞的泛化
缺点:使用参数p实际十分麻烦
d = ((x1-x2)**p + (y1-y2)**p + ...)**(1/p)
'''

# Jaccard指数
def jaccard_index(x_set, y_set):
    intersection = len(set(x_set & y_set))
    union = len(set(x_set | y_set))
    return intersection / union

x_set = {1, 2, 3}
y_set = {2, 3, 4}

print("jaccrd指数:", jaccard_index(x_set, y_set))
'''
计算集合相似度
缺点:数据大小很大影响,大型数据集可能对指数产生很大影响,因为数据量大可能显著增加并集,同时交集不变
D = |X ∩ Y|/|X ∪ Y| 
'''

# 半正矢距离
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # 地球半径，km

    # 将十进制转成弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # 得到经纬度的差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    d = 2 * R * math.asin(math.sqrt(math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2))

    return d

lat1, lon1 = 52.2296756, 21.0122287  # 华沙的经纬度
lat2, lon2 = 51.5073509, -0.1277583  # 伦敦的经纬度

print("半正矢距离:", haversine_distance(lat1, lon1, lat2, lon2))
'''
地理系统两个经纬点的距离(球体)
缺点:很难通用
d = 2 R arcsin(|sin((t2-t1)/2)|) + cost1 cost2 sin((l2-l1)/2)**2
R球半径,t两点纬度,l为经度的弧度表示

'''
