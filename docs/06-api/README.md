# PyTorch 常用 API 手册

---

## 1. 张量创建与基础操作

### 1.1 torch.tensor()
**作用**：从 Python 列表或数组创建张量

**参数**：
- `data`：输入数据（列表、numpy 数组等）
- `dtype`：数据类型（如 torch.float32、torch.int64）
- `device`：设备（'cpu' 或 'cuda'）
- `requires_grad`：是否需要自动求导，默认 False

---

### 1.2 torch.zeros()
**作用**：创建全 0 张量

**参数**：
- `*size`：张量的形状（如 3, 4 表示 3行4列）
- `dtype`：数据类型
- `device`：设备

---

### 1.3 torch.ones()
**作用**：创建全 1 张量

**参数**：
- `*size`：张量的形状
- `dtype`：数据类型
- `device`：设备

---

### 1.4 torch.randn()
**作用**：创建服从标准正态分布（均值0，方差1）的随机张量

**参数**：
- `*size`：张量的形状
- `dtype`：数据类型
- `device`：设备

---

### 1.5 tensor.view()
**作用**：改变张量的形状，不改变数据，必须保证元素总数一致

**参数**：
- `*shape`：目标形状（如 (3, 4) 或 (12, -1)，-1 表示自动计算）

---

### 1.6 tensor.reshape()
**作用**：改变张量形状，和 view 类似，但可以处理不连续的张量

**参数**：
- `*shape`：目标形状

---

### 1.7 tensor.item()
**作用**：将单元素张量转换为 Python 标量

**参数**：无

---

### 1.8 tensor.backward()
**作用**：自动计算梯度（反向传播）

**参数**：
- `gradient`：同形状的梯度（一般单损失函数不用传）

---

## 2. 神经网络层

### 2.1 nn.Module
**作用**：所有神经网络模块的基类，自定义模型都要继承它

**常用方法**：
- `__init__()`：初始化网络层
- `forward(x)`：前向传播，必须重写

---

### 2.2 nn.Linear()
**作用**：全连接层，对输入做线性变换

**参数**：
- `in_features`：输入神经元数量
- `out_features`：输出神经元数量
- `bias`：是否使用偏置，默认 True

---

### 2.3 nn.ReLU()
**作用**：ReLU 激活函数，负值置 0，正值不变

**参数**：
- `inplace`：是否原地操作，默认 False

---

### 2.4 nn.Sigmoid()
**作用**：Sigmoid 激活函数，将输入压缩到 (0,1) 区间

**参数**：无

---

### 2.5 nn.Tanh()
**作用**：Tanh 激活函数，将输入压缩到 (-1,1) 区间

**参数**：无

---

### 2.6 nn.Dropout()
**作用**：Dropout 正则化，训练时随机失活一部分神经元，防止过拟合

**参数**：
- `p`：失活概率，默认 0.5
- `inplace`：是否原地操作

---

### 2.7 nn.BatchNorm2d()
**作用**：二维批量归一化，用于卷积层输出，加速训练、稳定收敛

**参数**：
- `num_features`：输入通道数
- `eps`：数值稳定性的小常数，默认 1e-5
- `momentum`：滑动平均动量，默认 0.1

---

## 3. 损失函数

### 3.1 nn.MSELoss()
**作用**：均方误差损失，用于回归任务

**参数**：
- `reduction`：损失计算方式，'mean'（默认）、'sum'、'none'

---

### 3.2 nn.CrossEntropyLoss()
**作用**：交叉熵损失，用于多分类任务，内部包含 Softmax

**参数**：
- `weight`：各类别的权重
- `reduction`：损失计算方式

---

### 3.3 nn.BCEWithLogitsLoss()
**作用**：带 Sigmoid 的二元交叉熵损失，用于二分类任务，数值更稳定

**参数**：
- `pos_weight`：正样本权重，处理类别不平衡
- `reduction`：损失计算方式

---

## 4. 优化器

### 4.1 torch.optim.SGD()
**作用**：随机梯度下降优化器

**参数**：
- `params`：模型参数（model.parameters()）
- `lr`：学习率
- `momentum`：动量，默认 0
- `weight_decay`：权重衰减（L2 正则化）

---

### 4.2 torch.optim.Adam()
**作用**：Adam 优化器，自适应学习率，收敛快，常用默认选项

**参数**：
- `params`：模型参数
- `lr`：学习率，默认 1e-3
- `betas`：一阶和二阶矩估计的指数衰减率，默认 (0.9, 0.999)
- `weight_decay`：权重衰减

---

### 4.3 torch.optim.RMSprop()
**作用**：RMSprop 优化器，适合循环神经网络

**参数**：
- `params`：模型参数
- `lr`：学习率
- `alpha`：平滑常数，默认 0.99
- `weight_decay`：权重衰减

---

## 5. 学习率调度器

### 5.1 torch.optim.lr_scheduler.StepLR()
**作用**：每隔固定步长衰减学习率

**参数**：
- `optimizer`：优化器实例
- `step_size`：每隔多少 epoch 衰减一次
- `gamma`：衰减倍率，默认 0.1

---

### 5.2 torch.optim.lr_scheduler.MultiStepLR()
**作用**：在指定的 epoch 节点衰减学习率

**参数**：
- `optimizer`：优化器实例
- `milestones`：指定 epoch 节点列表（如 [10, 20, 30]）
- `gamma`：衰减倍率

---

### 5.3 torch.optim.lr_scheduler.ExponentialLR()
**作用**：每个 epoch 都按指数衰减学习率

**参数**：
- `optimizer`：优化器实例
- `gamma`：衰减倍率（如 0.95 表示每轮乘 0.95）

---

### 5.4 torch.optim.lr_scheduler.ReduceLROnPlateau()
**作用**：当指标（如验证损失）不再下降时自动降低学习率

**参数**：
- `optimizer`：优化器实例
- `mode`：'min'（监控指标下降）或 'max'（监控指标上升）
- `factor`：学习率衰减倍率，默认 0.1
- `patience`：耐心轮数，多少轮没改善就衰减
- `verbose`：是否打印学习率变化

---

## 6. CNN 相关 API

### 6.1 nn.Conv2d()
**作用**：二维卷积层，提取图像局部特征

**参数**：
- `in_channels`：输入通道数（RGB 图像为 3）
- `out_channels`：输出通道数（卷积核数量）
- `kernel_size`：卷积核大小（如 3 表示 3×3，元组表示不同高宽）
- `stride`：步长，默认 1
- `padding`：填充数量，默认 0；'same' 表示输出尺寸和输入一致

---

### 6.2 nn.MaxPool2d()
**作用**：最大池化层，取窗口内最大值，降维减参

**参数**：
- `kernel_size`：池化窗口大小
- `stride`：步长，默认和 kernel_size 相同
- `padding`：填充数量

---

### 6.3 nn.AvgPool2d()
**作用**：平均池化层，取窗口内平均值

**参数**：
- `kernel_size`：池化窗口大小
- `stride`：步长
- `padding`：填充数量

---

## 7. NLP 文本预处理 API（jieba）

### 7.1 jieba.lcut()
**作用**：中文分词，返回列表

**参数**：
- `sentence`：要分词的文本
- `cut_all`：是否全模式，默认 False（精确模式）

---

### 7.2 jieba.lcut_for_search()
**作用**：搜索引擎模式分词，对长词再次切分

**参数**：
- `sentence`：要分词的文本

---

### 7.3 jieba.load_userdict()
**作用**：加载用户自定义词典，识别领域专有名词

**参数**：
- `file_name`：自定义词典文件路径（每行格式：词语 词频 词性）

---

### 7.4 jieba.posseg.lcut()
**作用**：带词性标注的分词

**参数**：
- `sentence`：要分词的文本

**返回**：每个元素是 (词, 词性) 的元组
