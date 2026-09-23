"""背景:
    基于手机的20列特征 -> 预测手机的价格区间(4个区间), 可以用机器学习做, 也可以用 深度学习做(推荐)

ANN案例的实现步骤:
    1. 构建数据集.
    2. 搭建神经网络.
    3. 模型训练.
    4. 模型测试.

优化思路:
    1. 优化方法从 SGD -> Adam
    2. 学习率从 0.001 -> 0.0001
    3. 对数据进行标准化.
    4. 增加网络的深度, 每层的神经元数量
    5. 调整训练的轮数
    6. ......
"""

# 导包
import torch                                    # PyTorch框架, 封装了张量的各种操作
from torch.utils.data import TensorDataset      # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader         # 数据加载器.
import torch.nn as nn                           # neural network, 封装了神经网络的各种操作
import torch.optim as optim                     # 优化器
from sklearn.model_selection import train_test_split    # 训练集和测试集的划分
import matplotlib.pyplot as plt                 # 绘图
import numpy as np                              # 数组(矩阵)操作
import pandas as pd                             # 数据处理
import time                                     # 时间模块
from torchsummary import summary                # 模型结构可视化

def create_dataset():
    data=pd.read_csv('./data/手机价格预测.csv')   #获取数据
    print(data.head())
    x,y=data.iloc[:,:-1].values,data.iloc[:,-1].values   #提取特征, 并转成numpy数组(DataFrame不能直接转张量)
    # print(x.shape,y.shape)
    x=x.astype(np.float32)                     #将x转换成浮点型
    y=y.astype(np.int64)                       #分类任务的标签用整型, 便于后续 CrossEntropyLoss 使用
    train_x,test_x,train_y,test_y=train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)  #切割数据 固定随机种子
    # 5. 把数据集封装成 张量数据集.  思路: 数据 -> 张量Tensor -> 数据集TensorDataSet -> 数据加载器DataLoader
    train_dataset=TensorDataset(torch.tensor(train_x),torch.tensor(train_y))
    test_dataset=TensorDataset(torch.tensor(test_x),torch.tensor(test_y))
    return train_dataset,test_dataset,train_x.shape[1], len(np.unique(y))
#搭建神经网络
class Net(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear1=nn.Linear(input_dim,128)
        self.linear2=nn.Linear(128,256)
        self.output=nn.Linear(256,output_dim)
        #反向传播
    def forward(self,x):
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=self.output(x)
        return x


#训练
def train(train_dataset,input_dim,output_dim):
    #创建优化器
    dataLoader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #损失函数对象，因为是多分类 所以使用多分类交叉熵损失函数CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    #创建优化器
    optimizer=optim.SGD(model.parameters(),lr=0.001)
    #训练
    epochs=50
    for epoch in range(epochs):
        total_loss,batch_num=0.0,0
        start=time.time()
        for x,y in dataLoader:
            #切换模型(状态)
            model.train() #训练模式.    model.eval()   # 测试模式
            y_pred = model(x) #模型预测
            #计算损失
            loss=criterion(y_pred,y)
            #梯度清0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()   # 把本轮的每批次(16条)的 平均损失累计起来. 第1批次的平均损失 + 第2批次的平均损失 + ...
            batch_num += 1
        #至此，结束第一轮的训练
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}s')
    #至此，多轮训练结束，保存模型(参数) 参1：模型对象的参数(权重矩阵, 偏置矩阵)  参2：模型保存的文件名.
    torch.save(model.state_dict(),'./model/model.pth') #保存模型参数
#模型测试
def eluate(test_dataset,input_dim,output_dim):
    #创建神经网络分类对象
    model=Net(input_dim,output_dim)
    #加载模型
    model.load_state_dict(torch.load('./model/model.pth'))
    dataLoader=DataLoader(test_dataset,batch_size=16,shuffle=False) #封装测试集 参1：数据集对象  参2：批次大小
    #定义变量 记录正确样本的个数
    cerr=0
    for x,y in dataLoader:
        #切换模型状态
        model.eval()
        #模型预测
        y_pred=model(x)
        #根据加权求和进行分类
        y_pred=torch.argmax(y_pred,dim=1)
        print(y_pred)
        cerr+=(y_pred==y).sum()
        # 6.走到这里, 模型预测结束, 打印准确率即可.
    print(f'准确率(Accuracy): {cerr / len(test_dataset):.4f}')


if __name__ == '__main__':
    train_dataset,test_dataset,input_dim,output_dim=create_dataset()
    # print(f'训练集 数据集对象: {train_dataset}')
    # print(f'测试集 数据集对象: {test_dataset}')
    # print(f'输入特征数: {input_dim}')    # 20
    # print(f'输出标签数: {output_dim}')   # 4

    model = Net(input_dim,output_dim)
    #计算模型参数 summary
    #参1：输入批次，每批16条，输出特征
    # summary(model,input_size=(16,input_dim,),device='cpu')
    train(train_dataset,input_dim,output_dim)
    eluate(test_dataset,input_dim,output_dim)