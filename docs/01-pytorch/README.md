# PyTorch 框架使用

---

## 1. 张量创建

### 1.1 什么是张量

张量是 PyTorch 中的核心数据抽象，就是元素为同一种数据类型的多维矩阵，与 NumPy 数组类似。

PyTorch 中，张量以"类"的形式封装起来，对张量的一些运算、处理的方法（数值计算、矩阵操作、自动求导）被封装在类中。

张量维度关系：
- 多个二维张量组成三维张量
- 多个三维张量组成四维张量
- 多个四维张量组成五维张量

### 1.2 基本创建方式

张量中默认的数据类型是 **float32 (torch.FloatTensor)**

**torch.tensor(data=, dtype=)**：根据指定数据创建张量

```python
import torch
import numpy as np

# 1. 创建标量
data = torch.tensor(10)

# 2. 从 numpy 数组创建
data = np.random.randn(2, 3)
data = torch.tensor(data)

# 3. 从列表创建
data = [[10., 20., 30.], [40., 50., 60.]]
data = torch.tensor(data)
```

**torch.Tensor(size=)**：根据形状创建张量

```python
# 1. 创建 2 行 3 列的张量
data = torch.Tensor(2, 3)

# 2. 如果传递列表，则创建包含指定元素的张量
data = torch.Tensor([10])
data = torch.Tensor([10, 20])
```

**指定类型的张量创建**：

```python
# int32
data = torch.IntTensor(2, 3)

# 其他类型
data = torch.ShortTensor()   # int16
data = torch.LongTensor()    # int64
data = torch.FloatTensor()   # float32
data = torch.DoubleTensor() # float64
```

### 1.3 线性和随机张量

**固定步长/固定元素数线性张量**：

```python
# arange: [start, end, step) 左闭右开，按步长生成
data = torch.arange(0, 10, 2)

# linspace: [start, end, steps] 左闭右闭，按元素个数生成
# step = (end-start) / (steps-1)
data = torch.linspace(0, 9, 10)
```

**随机张量**：

```python
# 随机浮点张量
data = torch.randn(2, 3)

# 随机整数张量，左闭右开
data = torch.randint(0, 10, (2, 3))

# 随机种子设置
torch.manual_seed(100)
```

### 1.4 指定值张量

```python
# 全 0 张量
data = torch.zeros(2, 3)
data = torch.zeros_like(data)

# 全 1 张量
data = torch.ones(2, 3)
data = torch.ones_like(data)

# 全为指定值的张量
data = torch.full([2, 3], 10)
data = torch.full_like(data, 20)
```

### 1.5 类型转换

```python
data = torch.full([2, 3], 10)

# 方式一: type()
data = data.type(torch.DoubleTensor)

# 方式二: 快捷方法
data = data.double()
data = data.float()
data = data.int()
data = data.long()
```

---

## 2. 张量类型转换

### 2.1 张量转换为 NumPy 数组

使用 `tensor.numpy()` 可以将张量转换为 ndarray 数组，**默认共享内存**，可以使用 copy 函数避免共享。

```python
data_tensor = torch.tensor([2, 3, 4])

# 转换（共享内存）
data_numpy = data_tensor.numpy()

# 修改其中一个，另一个也会改变
data_numpy[0] = 100

# 避免共享内存
data_numpy = data_tensor.numpy().copy()
```

### 2.2 NumPy 数组转换为张量

```python
import numpy as np
data_numpy = np.array([2, 3, 4])

# 方式一: torch.from_numpy()，默认共享内存
data_tensor = torch.from_numpy(data_numpy)

# 方式二: torch.tensor()，默认不共享内存
data_tensor = torch.tensor(data_numpy)
```

### 2.3 提取标量张量的数值

对于只有一个元素的张量，使用 `item()` 函数将该值从张量中提取出来。

```python
data = torch.tensor([30,])
print(data.item())

data = torch.tensor(30)
print(data.item())
```

---

## 3. 张量数值计算

### 3.1 基本运算

- 加减乘除取负号：`+`、`-`、`*`、`/`、`-`
- 对应方法：`add()`、`sub()`、`mul()`、`div()`、`neg()`
- 带下划线的版本（如 `add_()`）会**直接修改原数据**

```python
data = torch.randint(0, 10, [2, 3])

# 不修改原数据
new_data = data.add(10)

# 直接修改原数据
data.add_(10)
```

### 3.2 点乘运算

点乘（Hadamard）也称为元素级乘积，指的是相同形状的张量对应位置的元素相乘，使用 `mul()` 和运算符 `*` 实现。

```python
data1 = torch.tensor([[1, 2], [3, 4]])
data2 = torch.tensor([[5, 6], [7, 8]])

# 两种方式等价
data = torch.mul(data1, data2)
data = data1 * data2
```

### 3.3 矩阵乘法运算

矩阵乘法要求第一个矩阵 shape: (n, m)，第二个矩阵 shape: (m, p)，结果 shape: (n, p)。

```python
data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
data2 = torch.tensor([[5, 6], [7, 8]])

# 方式一: @ 运算符
data3 = data1 @ data2

# 方式二: matmul
data4 = torch.matmul(data1, data2)
```

---

## 4. 张量运算函数

常用统计函数：

- `tensor.mean(dim=)`: 平均值
- `tensor.sum(dim=)`: 求和
- `tensor.min/max(dim=)`: 最小值/最大值
- `tensor.pow(exponent=)`: 幂次方
- `tensor.sqrt()`: 平方根
- `tensor.exp()`: 指数 e^x
- `tensor.log()`: 对数（以 e 为底）

> dim=0 按列计算，dim=1 按行计算
> 注意：mean 要求张量必须为 Float 或 Double 类型

```python
data = torch.randint(0, 10, [2, 3], dtype=torch.float64)

print(data.mean())
print(data.mean(dim=0))  # 按列
print(data.mean(dim=1))  # 按行

print(data.sum())
print(torch.pow(data, 2))
print(data.sqrt())
print(data.exp())
print(data.log())
```

---

## 5. 张量索引操作

```python
data = torch.randint(0, 10, [4, 5])

# 1. 简单行列索引
print(data[0])       # 第 0 行
print(data[:, 0])    # 第 0 列

# 2. 列表索引
print(data[[0, 1], [1, 2]])  # (0,1)、(1,2) 两个位置的元素

# 3. 范围索引
print(data[:3, :2])  # 前 3 行的前 2 列

# 4. 布尔索引
print(data[data[:, 2] > 5])  # 第三列大于 5 的行

# 5. 多维索引
data = torch.randint(0, 10, [3, 4, 5])
print(data[0, :, :])   # 0 轴上的第一个数据
print(data[:, 0, :])   # 1 轴上的第一个数据
print(data[:, :, 0])   # 2 轴上的第一个数据
```

---

## 6. 张量形状操作

### 6.1 reshape

保证张量数据不变的前提下改变数据的维度。

```python
data = torch.tensor([[10, 20, 30], [40, 50, 60]])

# 查看形状
print(data.shape)
print(data.size())

# 修改形状
new_data = data.reshape(1, 6)
```

### 6.2 squeeze 和 unsqueeze

- **squeeze**：删除指定位置形状为 1 的维度（降维），不指定位置则删除所有形状为 1 的维度
- **unsqueeze**：在指定位置添加形状为 1 的维度（升维）

```python
mydata1 = torch.tensor([1, 2, 3, 4, 5])  # 1 维

# 升维
mydata2 = mydata1.unsqueeze(dim=0)   # 1*5
mydata3 = mydata1.unsqueeze(dim=1)   # 5*1
mydata4 = mydata1.unsqueeze(dim=-1)  # 5*1

# 降维
mydata5 = mydata4.squeeze()
```

### 6.3 transpose 和 permute

- **transpose**：交换张量形状的指定维度（一次交换两个）
- **permute**：一次交换更多的维度

```python
data = torch.tensor(np.random.randint(0, 10, [3, 4, 5]))

# 交换 1 和 2 维度
mydata2 = torch.transpose(data, 1, 2)

# permute 直接指定新维度顺序
mydata5 = torch.permute(data, [1, 2, 0])
mydata6 = data.permute([1, 2, 0])
```

### 6.4 view 和 contiguous

- **view**：修改张量形状，只能用于**连续张量**
- **is_contiguous()**：判断张量是否连续
- **contiguous()**：将不连续张量转为连续张量

> 张量经过 transpose 或 permute 处理后，底层数据在内存中的存储顺序与逻辑顺序不一致，变成不连续张量，此时不能直接使用 view。

```python
data = torch.tensor([[10, 20, 30], [40, 50, 60]])

print(data.is_contiguous())  # True

# view 修改形状
mydata2 = data.view(3, 2)

# transpose 后变成不连续
mydata3 = torch.transpose(data, 0, 1)
print(mydata3.is_contiguous())  # False

# 先转成连续，再用 view
mydata4 = mydata3.contiguous().view(2, 3)
```

---

## 7. 张量拼接操作

### 7.1 cat / concat

沿着**现有维度**连接一系列张量。所有输入张量除了指定的拼接维度外，其他维度必须匹配。

```python
data1 = torch.randint(0, 10, [1, 2, 3])
data2 = torch.randint(0, 10, [1, 2, 3])

# 按指定维度拼接
new_data = torch.cat([data1, data2], dim=0)
```

### 7.2 stack

在**新的维度**上连接一系列张量，会增加一个新维度，并且所有输入张量的形状必须**完全相同**。

```python
data1 = torch.randint(0, 10, [2, 3])
data2 = torch.randint(0, 10, [2, 3])

# 在新维度上拼接
new_data = torch.stack([data1, data2], dim=0)
```

---

## 8. 自动微分模块

自动微分就是自动计算梯度值，也就是计算导数。

- **对函数求导得到的值就是梯度**
- **梯度就是上山下山最快的方向**
- **反向传播传播的是梯度**：利用链式法则不断从后向前求导
- **链式法则中，梯度相乘，就是梯度传播**

PyTorch 内置 `torch.autograd` 微分模块，支持任意计算图的自动梯度计算。

### 8.1 梯度基本计算

> PyTorch 不支持向量张量对向量张量的求导，只支持标量张量对向量张量的求导。x 如果是张量，y 必须是标量才可以进行求导。

```python
# 定义张量，requires_grad=True 表示需要计算梯度
x = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 定义函数
y = 2 * x ** 2

# 计算梯度（y 必须是标量）
y.sum().backward()

# 查看 x 的梯度值
print(x.grad)
```

### 8.2 梯度下降法求最优解

梯度下降法公式：`w = w - r * grad`（r 是学习率，grad 是梯度值）

```python
# 定义点
x = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 循环迭代求最优解
for i in range(1, 1001):
    # 前向计算
    y = x ** 2 + 20

    # 梯度清零（默认会累加历史梯度）
    if x.grad is not None:
        x.grad.zero_()

    # 反向传播
    y.sum().backward()

    # 梯度更新（修改 data，不影响计算图）
    x.data = x.data - 0.01 * x.grad
```

### 8.3 注意事项

- 不能将自动微分的张量直接转换成 numpy 数组，需要先调用 `detach()`
- `detach()` 产生一个新的张量作为叶子结点，与原张量共享数据，但不会自动微分

```python
x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)

# 通过 detach() 转换
x2 = x1.detach()

# 现在可以转 numpy 了
print(x2.numpy())
```

---

## 9. 构建线性回归模型

PyTorch 中进行模型构建的整个流程一般分为四个步骤：
1. 准备训练集数据
2. 构建要使用的模型
3. 设置损失函数和优化器
4. 模型训练

用到的 API：
- `nn.MSELoss()`：平方损失函数
- `DataLoader`：数据加载器
- `optim.SGD`：优化器
- `nn.Linear`：线性层

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.datasets import make_regression

# 构造数据集
def create_dataset():
    x, y, coef = make_regression(n_samples=100, n_features=1, noise=10, coef=True, bias=14.5, random_state=0)
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x, y, coef

# 训练模型
def train():
    x, y, coef = create_dataset()

    # 数据集和数据加载器
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    # 模型
    model = nn.Linear(in_features=1, out_features=1)

    # 损失函数
    criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.SGD(params=model.parameters(), lr=1e-2)

    # 训练
    epochs = 100
    for _ in range(epochs):
        for train_x, train_y in dataloader:
            y_pred = model(train_x.type(torch.float32))
            loss = criterion(y_pred, train_y.reshape(-1, 1).type(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(model.weight)
    print(model.bias)
```
