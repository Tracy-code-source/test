# DeepLearning

### 层在神经网络的解释

```python
from keras.models import Sequential
from keras.layers import Dense, Activation
# Dense 是一种层类型，表示全连接层（密集层）。全连接层是神经网络中最基本的层之一，它将输入数据的所有特征与输出神经元完全连接
# activation 是神经网络层的一个参数，用于指定该层的激活函数。激活函数的作用是对神经元的输出进行非线性变换，从而使神经网络能够学习复杂的模式和特征
model = Sequential([
    # 第一层是层内节点或神经元数量
    Dense(5, input_shape = (3,), activation = 'relu'),
    Dense(2, activation = 'softmax'),
])
```

顺序模型只有第一层需要输入形状（input_shape， 3,  隐式输入层， 实际上有三层， 代码显示两层）

------



### 激活函数及使用

神经元的激活函数定义了一组输入的神经元的输出， 如以上， 取每个连接的加权和， 这些连接指向下一层中的同一个神经元， 把加权和传递给激活函数， 激活函数将总和转换为介于某个下限和某个上限之间的数字

*Example*   Sigmoid函数
$$
Sigmoid(x) = 1 / (1 + e^-x)
$$
当x趋于负无穷， 函数值趋于零； 当x趋于正无穷， 函数值趋于一， 此时上下限分别为1和0

**激活功能在生物学上指受大脑活动启发， 不同神经元被不同的刺激激活**

当今最广泛的激活函数之一， 称为ReLU （整流线性单元的缩写）
$$
f(x) = max(0, x)
$$

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(5, input_shape = (3,), activation = 'relu')
])
```

```python
model = Sequential()
model.add(Dense(3, input_shape = (3,)))
model.add(Activation('relu')) # 激活层

```

以上是两种指定激活函数的方法

------



### 训练神经网络

我们提供特征给到整个网络， 模型将提供输出

算法一定程度上影响了模型的准确性 

**模型根据我们提供的数据进行学习和更新权重参数**

------



### 神经网络是如何学习的

*初始化*

创建模型时给定的权重（weights）是随机的，通过传递的数据更新权重

*误差*

拟定损失函数， 模型计算损失及预测值与标签的误差

**不断更新权重，不断减小损失函数的值， 称之为模型学习**

例如

```python
import keras
from keras import backend as K # 导入Keras后端接口， 用于低级操作
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam # 导入Adam优化器， 用于模型训练
from keras.metrics import categorical_crossentropy # 导入分类交叉熵指标， 用于评估模型性能
```

```python
model = Sequential([
    Dense(16, input_shape = (1,), activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(2, activation = 'softmax')
])
```

**第一层**

1. **Dense(16)** :  这一层有 16 个神经元
2. **input_shape=(1,)**：输入数据的形状为一维，即每个样本只有一个特征
3. **activation='relu'**：使用 ReLU 激活函数

**第二层**

1. **Dense(32)**：这一层有32个神经元
2. **activation='relu'**：使用 ReLU 激活函数

**第三层**：

1. **Dense(2)**：这一层有两个神经元， 对应分类问题的两个类别
2. **activation='softmax'**: 使用Softmax激活函数， 将输出转换为概率分布

```python
model.compile(Adam(learning_rate = .0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(scaled_train_samples, train_labels, batch_size = 10, epochs = 20, shuffle = True, verbose = 2)
```

1. **model.compile**: Keras中用于配置模型训练参数的方法
2. **Adam（learning——rate）**: Adam优化器， lr为学习率， Adam是一种常用的优化算法，能够自适应地调整学习率
3. **loss='sparse_categorical_crossentropy'**: 指定损失函数为稀疏分类交叉熵， 用于标签为整数的情况
4. **model.fit**： Keras中用于训练模型的方法
5. **scaled_train_samples**：训练数据，通常是经过预处理（如归一化）的特征数据
6. **train_labels**：训练数据对应的标签
7. **batch_size=10**：每次更新模型参数时使用的样本数量为 10
8. **epochs=20**：整个训练数据集将被完整遍历 20 次
9. **shuffle=True**：在每个 epoch 开始前，训练数据会被随机打乱，以提高模型的泛化能力
10. **verbose=2**：控制训练过程的日志输出详细程度，**verbose=2** 表示每个 epoch 输出一行日志

    ------

### 损失在神经网络的解释

在神经网络中，损失（Loss）是指衡量模型预测值与真实值之间差异的指标

例如均方误差（MSE）：过计算预测值与真实值之间差值的平方的平均值来衡量模型的预测准确性

*算法不同， 计算损失的标准也不同*

**损失随着模型权重的更新而改变**

------

### 学习率在神经网络的解释

学习率（Learning Rate）是一个重要的超参数，它控制着模型在训练过程中更新权重的步长大小

**控制更新步长， 避免迭代过程中越过Loss的最小值**

```python
model.optimizer.learning_rate = 0.01
```

可以通过以上语句更改模型学习率

------

### 关于训练集，测试集， 验证集的解释

1. 训练集（train）：用于训练模型， 过程中权重会不断更新
2. 验证集 （validation）：用更新好的权重进行验证， 不更改权重
3. 测试集 （test）： 在模型经过训练集及验证集检验后， 在我们对模型相对自信时开始用测试集评估模型

**过拟合：在训练集上表现良好， 但在验证集上表现不佳**

```python
model.fit(scaled_train_samples, train_labels, validation_split = 0.20, batch_size = 10, epochs = 20, shuffle = True, verbose = 2)
```

以上代码说明我们并不一定要单独列出验证集， 可以通过分割训练集获得

------



### 通过神经网络预测的解释

**训练集及验证集传入模型的参数包含样本特征及样本各自的标签， 而测试集仅传入特征**

```python
predition = model.predict(scaled_test_samples, batch_size = 10, verbose = 0)
```

------

### 神经网络中过拟合的解释

如何判断：训练集和验证集性能差异过大

*如何解决*

**增加训练数据， 数据越多， 模型学习自然越好**

**Data Augmentation （数据增强）**：它通过对现有数据进行变换或生成新数据来增加数据集的大小和多样性，其核心目的在于提高模型的泛化能力，减少过拟合的风险

**减少模型层数或者神经元个数**：简化模型

**Dropout**：通过随机失活一部分神经元，使得每次训练迭代中网络的结构不同，相当于训练多个不同的神经网络

------

### 神经网络中欠拟合的解释

欠拟合（Underfitting）是机器学习和深度学习中模型无法充分学习数据特征，导致模型在训练数据和新数据（如测试数据）上都表现不佳的现象

*原因*：可能是模型过于简单

如何解决

**提高模型复杂度， 这点与防止过拟合相反**：适当增加层数、神经元

**增加样本特征**

**减少dropout的比例**

------

### 监督学习的解释

**利用已标注的训练数据来训练模型，使模型能够对新的、未标注的数据进行预测或分类**

Example：将带标签的训练集交给模型学习， 这就是一种监督学习

```python
import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
model = Sequential([
    Dense(16, input_shape = (2, ), activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(2, activation = 'sigmoid')
])
model.compile(Adam(learning_rate = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# weight, height
train_samples = np.array([[150, 67], [130, 60], [200, 65], [125, 52], [230, 72], [181, 70]])
# 0 male
# 1 female 
train_labels = np.array([1, 1, 0, 1, 0, 0])
model.fit(x = train_samples, y = train_labels, batch_size = 3, epochs = 10, shuffle = True, verbose = 2)
```

```python
Epoch 1/10
2/2 - 0s - 225ms/step - accuracy: 0.5000 - loss: 26.9419
Epoch 2/10
2/2 - 0s - 13ms/step - accuracy: 0.5000 - loss: 26.6583
Epoch 3/10
2/2 - 0s - 13ms/step - accuracy: 0.5000 - loss: 26.3752
Epoch 4/10
2/2 - 0s - 12ms/step - accuracy: 0.5000 - loss: 26.0891
Epoch 5/10
2/2 - 0s - 13ms/step - accuracy: 0.5000 - loss: 25.8177
Epoch 6/10
2/2 - 0s - 13ms/step - accuracy: 0.5000 - loss: 25.5369
Epoch 7/10
2/2 - 0s - 13ms/step - accuracy: 0.5000 - loss: 25.2547
Epoch 8/10
2/2 - 0s - 12ms/step - accuracy: 0.5000 - loss: 24.9781
Epoch 9/10
2/2 - 0s - 12ms/step - accuracy: 0.5000 - loss: 24.6516
Epoch 10/10
2/2 - 0s - 13ms/step - accuracy: 0.5000 - loss: 24.3439
```

------

### 无监督学习的解释

同样是机器学习的一种类型，它从无标注数据中自动发现潜在的模式和结构

**不依赖于预先定义的标签**

1. K-means算法:将类似的数据聚集形成簇， 无监督学习的一种

2. 自编码器（Autoencoder）:是一种人工神经网络，用于无监督学习，主要目的是学习如何高效地表示数据。它通过将输入数据压缩到一个低维的编码表示（编码器），然后尽可能准确地重构原始输入（解码器）来实现这一目标； 

   Example: 去噪（Denoising）：指从被噪声污染的数据中恢复原始信号或数据的过程

------

### 半监督学习的解释

介于上面两者之间

**伪标签学习（Pseudo-Label Learning）：一种半监督学习方法，旨在利用未标注数据来提高模型性能 **

*将带伪标签数据和原始带标签数据结合， 重新训练模型*

------

### 数据增强的解释

简单来说，在已有数据集的条件下，不引入新的数据， 实现扩大数据集的规模

------

### 独热编码的解释

是一种将分类变量转换为数值型表示的方法，常用于机器学习和数据处理中。它通过将每个分类值转换为一个二进制向量，从而使得分类变量能够在模型中被正确处理

*使模型读到的并不是标签本身（如字符串）， 而是被转化为二进制的编码*

------

### 卷积神经网络的解释

基础：**卷积层**

**卷积层接收输入并转换输入并传给下一层**

CNN在处理图像等数据时具有优势

------

