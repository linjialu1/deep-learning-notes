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

# ★ 可视化前置设置: 让matplotlib能显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']      # 指定黑体(Windows自带), 否则中文显示成方块
plt.rcParams['axes.unicode_minus']=False        # 修复坐标轴负号显示为方块的问题

#1获取数据
def create_dataset():
    data=pd.read_csv("./data/Iris.csv")
    # print(date)
    print(f'data: {data.shape}')
    data = data.drop(columns=['Id'])  #丢弃ID列 这一列是不需要的
    #将字符串映射成数字 这样评估的时候方便分类
    data['Species'] = data['Species'].map(
        {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})  # 字符串标签 -> 整数
    x,y=data.iloc[:,:-1].values,data.iloc[:,-1].values
    x=x.astype(np.float32)   # ★ 转成float32, 与模型权重的dtype保持一致, 否则前向传播报dtype不匹配
    y=y.astype(np.int64)
    #做数据切割
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=3)
    #将数据分装成张量 把数据集封装成 张量数据集.  思路: 数据 -> 张量Tensor -> 数据集TensorDataSet -> 数据加载器DataLoader
    data_set_train=TensorDataset(torch.tensor(train_x),torch.tensor(train_y))
    data_set_test=TensorDataset(torch.tensor(test_x),torch.tensor(test_y))
    # ★ 额外返回完整的x,y(150条全量数据, 未切割), 供数据散点图使用
    return data_set_train,data_set_test,train_x.shape[1], len(np.unique(y)), x, y

# ★ 可视化1: 数据分布散点图(选区分度最高的两个特征: 花瓣长/花瓣宽)
def plot_data(x,y):
    plt.figure(figsize=(8,6))
    colors=['tab:red','tab:green','tab:blue']
    names=['山鸢尾 setosa(0)','变色鸢尾 versicolor(1)','维吉尼亚鸢尾 virginica(2)']
    for i in range(3):   # 分别画出三个类别的样本
        plt.scatter(x[y==i,2], x[y==i,3], c=colors[i], label=names[i], alpha=0.7)
    plt.xlabel('花瓣长 PetalLengthCm (cm)')
    plt.ylabel('花瓣宽 PetalWidthCm (cm)')
    plt.title('Iris 数据分布: 三类鸢尾花的散点图')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('./iris_scatter.png', dpi=150)   # 保存成图片留档
    print('散点图已保存: ./iris_scatter.png')
    plt.show()   # 弹窗显示, 关掉窗口后程序继续往下执行

#2搭建神经网络
class IrisDataset(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear1=nn.Linear(input_dim,16)
        self.output=nn.Linear(16,output_dim)
    #定义方向传播
    def forward(self,x):
        #隐藏层1多分类任务使用Relu
        x=torch.relu(self.linear1(x))
        x = self.output(x)
        return x
#3训练
def train_model(data_set_train,input_dim,output_dim):
    #1.构建数据加载器
    train_loader=DataLoader(data_set_train,batch_size=8,shuffle=True)
    # 2. 创建神经网络模型.
    model=IrisDataset(input_dim,output_dim)
    # 3. 定义损失函数, 因为是多分类, 这里用的是: 多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #4.创建优化器
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    #训练

    loss_list=[]   # ★ 收集每一轮的平均loss, 画曲线用
    epochs=50
    for epoch in range(epochs):
        #记录损失值和每批的次数
        total_loss, batch_num = 0.0, 0
        start = time.time()
        for x, y in train_loader:
            #模型状态
            model.train()
            #预测值
            y_pred = model(x)
            #计算损失
            loss = criterion(y_pred, y)
            #梯度清0，反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #记录每轮的损失值
            total_loss += loss.item()
            batch_num += 1
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}s')
        loss_list.append(total_loss / batch_num)   # ★ 记录本轮的平均loss
    torch.save(model.state_dict(),'./model/Iris.pth')

    # ★ 可视化2: 训练loss曲线
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), loss_list, marker='o', color='tab:orange', label='训练loss')
    plt.xlabel('epoch (训练轮数)')
    plt.ylabel('loss (交叉熵损失)')
    plt.title('Iris 训练loss曲线')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('./loss_curve.png', dpi=150)
    print('loss曲线已保存: ./loss_curve.png')
    plt.show()
#4评估
def evaluate_model(data_set_test,input_dim,output_dim):
    #1.创建神经网络分类对象
    model=IrisDataset(input_dim,output_dim)
    #2.加载模型
    model.load_state_dict(torch.load('./model/Iris.pth'))
    # 3. 创建测试集的 数据加载器对象.
    evaluate_loader = DataLoader(data_set_test,batch_size=8,shuffle=False)
    #4.定义变量记录正确样本个数
    correct=0
    #训练
    for x, y in evaluate_loader:
        #切换模型状态
        model.eval()
        #模型预测值
        y_pred = model(x)
        # 根据加权求和, 得到类别, 用argmax()获取最大值对应的下标, 就是类别.
        y_pred = torch.argmax(y_pred, dim=1)  # dim=1 表示逐行处理.
        correct += (y_pred == y).sum()
    print(f'准确率(Accuracy): {correct / len(data_set_test):.4f}')

if __name__ == '__main__':
    data_set_train,data_set_test,input_dim,output_dim,x,y=create_dataset()
    plot_data(x,y)   # ★ 先看看数据长什么样(关掉弹窗后继续往下)
    train_model(data_set_train,input_dim, output_dim)     # 训练结束会弹出loss曲线
    evaluate_model(data_set_test,input_dim, output_dim)