import torch
import torch.nn as nn


'''
state_dict() 返回模型的状态字典包含所有参数
save保存到路径下的文件夹中
torch.save(model.state_dict(), 'model.pth')

直接保存模型全部结构和参数
torch.save(model, 'entire_model.pth')
'''

# # 加载模型参数
# model1 = nn.Linear(1,1)
# model1.load_state_dict(torch.load('model.pth'))
# # 评估模式
# model1.eval()
# # 定义测试
# x_test = torch.tensor([[1.0],[2.5]], dtype=torch.float32)
# with torch.no_grad():
#     y_pred = model1(x_test)
# print(y_pred)

"""
方案一：如果你完全信任文件来源（风险较低）
这是最简单的恢复方法。在加载时显式设置 weights_only=False，沿用旧版本的行为。
注意：此方法会执行 pickle 文件中的任意代码。请仅在你 100% 确定文件来自可靠来源（如自己保存的、官方发布的）时使用。
"""
# 整个模型
# model2 = torch.load('entire_model.pth',weights_only=False)
"""
方案二:在保持安全模式 (weights_only=True) 下允许特定类
# 方法 A：使用上下文管理器（推荐，作用范围更精确）
# 列出文件中所有需要允许的类
safe_classes = [nn.Linear, nn.Conv2d, nn.BatchNorm1d] # 根据你的模型补充

with torch.serialization.safe_globals(safe_classes):
    model2 = torch.load('entire_model.pth')

# 方法 B：使用全局添加（影响之后所有的 torch.load，谨慎使用）
# torch.serialization.add_safe_globals([nn.Linear, nn.Conv2d])
# model2 = torch.load('entire_model.pth')
"""
safe_classes = [nn.Linear]
with torch.serialization.safe_globals(safe_classes):
    model2 = torch.load('entire_model.pth')

model2.eval()

x_test = torch.tensor([[1.0],[2.5]], dtype=torch.float32)
with torch.no_grad():
    y_pred = model2(x_test)
print(y_pred)
# 即使加载整个模型也不能注释掉上文定义的自定义类（多个函数拼接的列表等），因为模型内部还是在用这些