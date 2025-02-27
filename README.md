# From federated learning to unlearning（联邦忘却学习）
![](https://img.shields.io/badge/python-3.6-blue) ![](https://img.shields.io/badge/pytorch-1.8.1-green) ![](https://img.shields.io/badge/cuda-10.1-blue)

通过联邦忘却学习撤销用户数据对模型更新的方式解决联邦学习中存在的隐私泄露问题。联邦忘却学习的一般架构图：

<img src="/doc/structure.png" width="50%" height="50%">

具体流程；
+ 联邦学习。中心服务器协同用户进行联邦学习，训练得到最优模型参数。
+ 数据更新撤销。用户可以在任意时刻向中心服务器发送撤销数据请求。中心服务器根据用户的请求使用忘却学习算法重新训练得到的模型参数。
+ 用户将数据从联邦学习系统中删除，防止数据被重新训练。
+ 重新开始联邦学习任务，恢复全局模型的准确率。

## 忘却请求
用户在不同场景下提出的联邦忘却学习请求有所不同，主要体现在联邦忘却学习对全局模型的忘却粒度。
+ 样本忘却。样本忘却是从联邦学习模型中撤销特定数据样本对模型的训练更新，是细粒度的忘却学习请求，也是联邦忘却学习中常见的请求之一。
+ 类别忘却。类别忘却用于分类任务中撤销单个或多个类别的数据对全局模型的训练更新。
+ 任务忘却。在多任务训练模式下，模型学习多个任务时可以因任务之间的相关性获得性能上的收益。任务忘却是粗粒度的忘却学习请求，用于撤销某个任务所有数据对模型的训练更新，但会因大量的数据撤销而产生灾难性的忘却。

## 联邦忘却学习算法
### 1. 面向全局模型的联邦忘却学习算法
面向全局模型的联邦忘却学习算法通过直接修改全局模型参数来实现对目标数据的忘却。
#### 1.1 重新训练
重新训练能够完全撤销特定用户对模型的训练更新，是将全局模型训练后的参数初始化为随机值，并在剩余数据集上重新训练。重新训练基本架构图：

<img src="/doc/retraining.png" width="30%" height="30%">

#### 1.2 用户贡献删除
针对重复性的再训练工作，用户贡献删除算法可以减少用户重复训练的时间和通信开销。重新训练因将参数回退而需要大量时间开销用于再训练。用户贡献删除基本架构图：

<img src="/doc/contribution-remove.png" width="30%" height="30%">

面向全局模型的联邦忘却学习算法优缺点、适用模型、发起者和忘却学习请求类型的不同如下：

<img src="/doc/table1.png" width="30%" height="30%">

### 2. 面向局部模型的联邦忘却学习算法
面向局部模型的联邦忘却学习算法利用历史训练信息进行模型忘却，其主要在现有模型的基础上增加一定的联邦学习训练，训练过程中对局部模型进行修正，通过直接聚合的方式实现忘却学习。
#### 2.1 训练更新校正
训练更新校正算法避免重新训练的过程，具体过程：

<img src="/doc/training-update.png" width="30%" height="30%">

#### 2.2 训练梯度校正
训练梯度校正算法的思想是增加联邦学习训练，修改部分用户的训练方法，通过直接聚合来更新全局模型参数。训练梯度校正算法基本架构图：

<img src="/doc/training-gradient.png" width="30%" height="30%">

面向局部模型的代表算法在模型优缺点、适用模型、发起者和忘却学习请求类型的不同如下：

<img src="/doc/table2.png" width="30%" height="30%">

### 3. 面向特定结构的联邦忘却学习算法
面向特定结构的联邦忘却学习算法解决特定模型结构的忘却学习问题，该类算法通过结构信息计算用户数据贡献的参数位置，准确地删除用户数据对模型的贡献。
#### 3.1 传统机器学习模型
决策树、支持向量机等传统机器学习模型在工业场景得到了广泛的应用，但是也存在着因模型记忆而导致的利益冲突。考虑到这个问题，研究者将联邦忘却学习应用于传统机器学习模型。
#### 3.2 深度神经网络
深度神经网络中各层上的参数通过加权、乘积等方式构建了输入与输出之间的映射。目前的研究针对卷积层等特定结构，探索参数对模型输入输出的贡献，实现了高效的联邦忘却学习。

### 4. 不同种类联邦忘却学习算法的对比
+ 面向全局模型的联邦忘却学习算法虽然能够有效地删除目标用户对全局模型的训练更新，但模型准确率会在短时间内大幅降低，而且恢复后的模型难以在短时间内达到撤销模型更新之前的映射效果，主要适用于不考虑用户延迟、高度关注用户隐私等场景。
+ 面向局部模型的联邦忘却学习算法利用模型训练产生的数据信息，在此基础上通过训练实现联邦忘却。因此，面向局部模型的联邦忘却学习算法可以防止模型准确率的急剧下降。面向局部模型算法适用于服务器具有较高计算能力和空间等场景。
+ 面向特定结构的联邦忘却学习算法可以利用模型的结构信息和训练产生数据信息，因此在特定的模型上往往表现出良好的忘却学习效果，同时能够尽可能地减少准确率损失，但其仅适合特定的模型结构，存在着一定限制。
不同种类联邦忘却学习算法的优点、缺点、解决问题以及适应场景，具体如下：

<img src="/doc/table3.png" width="30%" height="30%">


