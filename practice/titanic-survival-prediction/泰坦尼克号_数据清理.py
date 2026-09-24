"""
泰坦尼克号 —— 数据清理模块(只负责清理, 模型部分你自己写)

对外接口就一个函数: clean(df, stats=None)
    训练集:  x_df, y, stats = clean(train_df)              # stats 从本份数据算
    验证集:  x_df, y, _    = clean(valid_df, stats=stats)   # 沿用训练集的统计量
    测试集:  x_df, y, _    = clean(test_df,  stats=stats)   # test.csv 没有 Survived, y 返回 None

返回值中的 x_df 已经是纯 float32 的 DataFrame, 可以直接 .values 取出来用。
"""

import numpy as np
import pandas as pd

# 登船港口的三个取值, 写死成常量
# 目的: 保证 train / valid / test tr5后的列数一致(见 clean() 第 5 步的说明)
EMB_CLASSES = ['C', 'Q', 'S']

# 要扔掉的列
DROP_COLS = ['PassengerId', 'Name', 'Ticket', 'Cabin']


def clean(df, stats=None):
    """
    清理一份泰坦尼克号 DataFrame

    参数:
        df:    原始 DataFrame
        stats: 填补用的统计量字典。None 表示"这是训练集, 我自己算";
               传字典表示"这是验证集/测试集, 用训练集算好的"

    返回:
        (x_df, y, stats)
        x_df  -> 清理后的特征, 全部 float32, 9 列
        y     -> 标签 Series(int64); 如果输入没有 Survived 列(如 test.csv), 返回 None
        stats -> 统计量字典, 训练集模式下生成, 要传给后续的 valid/test
    """
    df = df.copy()
    is_train = stats is None

    # --- 1. 扔掉无用列 ---------------------------------------------
    # PassengerId 纯编号 / Name 姓名 / Ticket 票号 / Cabin 缺失 77% 补不回来
    df = df.drop(columns=DROP_COLS)

    # --- 2. 拆出标签(测试集没有这一列, 就跳过) ----------------------
    if 'Survived' in df.columns:
        y = df['Survived'].astype(np.int64)
        df = df.drop(columns=['Survived'])
    else:
        y = None

    # --- 3. 算统计量(只在训练集上算, 防止数据泄漏) -------------------
    if is_train:
        stats = {
            'age_median': df['Age'].median(),      # 中位数比均值抗极端值(Age 最大 80)
            'fare_median': df['Fare'].median(),
            'emb_mode': df['Embarked'].mode()[0],  # mode() 返回 Series, 取第一个
        }

    # --- 4. 补缺失值 ------------------------------------------------
    # Age      缺 177/891 (19.9%) -> 中位数
    # Fare     train 不缺, 但 test.csv 缺 1 个, 一起补
    # Embarked 缺 2 个            -> 众数
    df['Age'] = df['Age'].fillna(stats['age_median'])
    df['Fare'] = df['Fare'].fillna(stats['fare_median'])
    df['Embarked'] = df['Embarked'].fillna(stats['emb_mode'])

    # --- 5. 字符串转数字 --------------------------------------------
    # 5a. 二分类 -> 直接 map 成 0/1, 一个维度就够
    #     只有两个取值时, 0/1 只是两个"名字", 不存在大小关系问题
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # 5b. 三分类 -> 必须独热
    #     若 map 成 S=0 C=1 Q=2, 模型会以为 Q(2) > S(0), 凭空造出虚假的顺序关系
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Emb')

    # 坑: 若某份数据里恰好没有 Q 港的人, get_dummies 只生成 2 列,
    #     与训练集的 3 列对不上, 模型第一层会维度报错。这里强制补齐。
    for c in EMB_CLASSES:
        col = f'Emb_{c}'
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype(np.float32)

    # --- 6. 统一转 float32 ------------------------------------------
    # PyTorch 默认浮点是 float32; float64 多占一倍显存且更慢
    # 这行同时充��检查: 若还有没处理干净的字符串列, 这里会直接报错
    df = df.astype(np.float32)

    return df, y, stats


# ====================================================================
# 下面是清理完之后怎么接上 PyTorch, 你照着改就行
# ====================================================================
if __name__ == '__main__':
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset

    df = pd.read_csv('./data/train.csv')

    # 顺序不能反: 先划分, 再清理, 最后 fit 标准化
    # stratify 保证两边生还比例一致
    train_df, valid_df = train_test_split(
        df, train_size=0.8, random_state=88, stratify=df['Survived'])

    x_train_df, y_train, stats = clean(train_df)
    x_valid_df, y_valid, _ = clean(valid_df, stats=stats)

    # 标准化: Age 0~80 而 Fare 0~512, 差 6 倍, 不缩放梯度会被 Fare 主导
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df.values).astype(np.float32)
    x_valid = scaler.transform(x_valid_df.values).astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.tensor(y_valid.values))

    print('特征列:', list(x_train_df.columns))
    print('特征维度:', x_train_df.shape[1])
    print('train:', x_train.shape, ' valid:', x_valid.shape)
