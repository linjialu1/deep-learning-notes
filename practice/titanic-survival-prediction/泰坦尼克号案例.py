"""
泰坦尼克号生存预测 —— PyTorch 处理"脏"表格数据的完整流程

和鸢尾花/手机价格数据集最大的区别:
    前两个是纯数字表, astype(np.float32) 一行就能进 PyTorch
    这个是真实采集的数据, 有缺失值、有字符串、有无用列, 必须先清理

核心观念: 数据清理 100% 在 pandas 阶段做完, PyTorch 只负责接收数值 tensor。
"""

import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# 可视化前置设置: 让matplotlib能显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']       # 指定黑体(Windows自带), 否则中文显示成方块
plt.rcParams['axes.unicode_minus']=False          # 修复坐标轴负号显示为方块的问题
def create_dataset():
    data=pd.read_csv('./data/train.csv')
    data=data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'])
    #将性别列的字符串转换为数字
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    # print(data.isnull().sum())
    #使用中位数填充Age列
    data['Age'] = data['Age'].fillna(data['Age'].median())
    # print(data.isnull().sum())
    x = data.iloc[:, 1:].values  # 特征：Pclass, Sex, Age, SibSp, Parch, Fare
    y = data.iloc[:, 0].values  # 标签：Survived
    x=x.astype(np.float32)
    y=y.astype(np.float32)
    #做数据切割
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=10)
    #标准化: Age 0~80而Fare 0~512, 不缩放梯度会被Fare主导
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)
    #数据分装成张量数据集
    data_set_train = TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y))
    data_set_test = TensorDataset(torch.from_numpy(test_x),torch.from_numpy(test_y))
    return data_set_train,data_set_test
#搭建神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(6,32)
        self.linear2=nn.Linear(32,128)
        self.linear3=nn.Linear(128,64)
        self.linear4=nn.Linear(64,16)
        self.output=nn.Linear(16,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x=self.relu(self.input(x))
        x=self.relu(self.linear2(x))
        x=self.relu(self.linear3(x))
        x=self.relu(self.linear4(x))
        x=self.output(x)
        return x.squeeze(-1)
def train_model(data_set_train):
    train_loader = DataLoader(data_set_train,batch_size=32,shuffle=True)
    model = Net()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    # 训练

    loss_list = []  # 收集每一轮的平均loss, 画曲线用
    epochs = 200
    for epoch in range(epochs):
        # 记录损失值和每批的次数
        total_loss, batch_num = 0.0, 0
        start = time.time()
        for x, y in train_loader:
            # 模型状态
            model.train()
            # 预测值
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 梯度清0，反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录每轮的损失值
            total_loss += loss.item()
            batch_num += 1
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}s')
        loss_list.append(total_loss / batch_num)
    torch.save(model.state_dict(), './model/titanic.pth')

    # 可视化: 训练loss曲线
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), loss_list, color='tab:orange', label='训练loss')
    plt.xlabel('epoch (训练轮数)')
    plt.ylabel('loss (BCE损失)')
    plt.title('Titanic 训练loss曲线')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('./titanic_loss_curve.png', dpi=150)
    print('loss曲线已保存: ./titanic_loss_curve.png')
    plt.show()
    return model
#评估
def evaluate_model(data_set_test, model):
    #创建测试集的数据加载器
    evaluate_loader = DataLoader(data_set_test, batch_size=8, shuffle=False)
    #定义变量记录正确样本个数
    correct = 0
    #记录所有预测概率和真实值, 画图用
    all_prob = []
    all_true = []
    for x, y in evaluate_loader:
        #切换模型状态
        model.eval()
        #模型预测值(原始输出是logit, 用sigmoid转成概率)
        y_pred = model(x)
        y_prob = torch.sigmoid(y_pred)        # logit -> 概率, 范围0~1
        y_pred_label = (y_prob >= 0.5).int()   # 概率>=0.5预测为1(存活), 否则为0(死亡)
        correct += (y_pred_label == y.int()).sum()
        all_prob.extend(y_prob.tolist())
        all_true.extend(y.tolist())
    acc = correct / len(data_set_test)
    print(f'准确率(Accuracy): {acc:.4f}')

    # 可视化: 预测概率分布图
    # 按真实标签分成两组, 看模型给的概率分布
    prob_dead = [p for p, t in zip(all_prob, all_true) if t == 0]
    prob_alive = [p for p, t in zip(all_prob, all_true) if t == 1]
    plt.figure(figsize=(8,5))
    #排序后画线, 更平滑
    plt.plot(sorted(prob_dead), color='tab:blue', label=f'实际死亡 ({len(prob_dead)}人)')
    plt.plot(sorted(prob_alive), color='tab:red', label=f'实际存活 ({len(prob_alive)}人)')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='分类阈值(0.5)')
    plt.xlabel('样本序号(按概率排序)')
    plt.ylabel('预测存活概率')
    plt.title(f'Titanic 预测概率分布 (Accuracy={acc:.4f})')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('./titanic_prob_dist.png', dpi=150)
    print('概率分布图已保存: ./titanic_prob_dist.png')
    plt.show()

if __name__ == '__main__':
    data_set_train,data_set_test=create_dataset()
    model=train_model(data_set_train)
    evaluate_model(data_set_test, model)