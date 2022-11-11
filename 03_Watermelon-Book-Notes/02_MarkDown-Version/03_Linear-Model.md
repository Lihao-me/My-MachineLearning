# 03线性模型

## 3.1基本形式

### 线性模型(linear model):试图学得一个通过属性的线性组合来进行预测的函数

- (3.1)
  
  $$
  f(\vec x) = {\omega _1}{x_1} + {\omega _2}{x_2} +  \cdots  + {\omega _d}{x_d} + b
  $$
  
  
- (3.2)
  
  $$
  f(\vec x) = {\vec \omega ^T}\vec x + b
  $$

### 线性模型特点

- 1.形式简单，易于建模，蕴含机器学习基本思想
- 2.是非线性模型(nonlinear model)的基础
- 3.有很好的解释性(comprehensibility)/可理解性(understandability)

## 3.2线性回归

### 线性回归(linear regression):试图学得一个线性模型以尽可能准确地预测实值输出标记

### 离散属性存在序(order)关系可通过连续化转化为连续值

- 身高：{高，矮}——>{1.0,0.0}
- 高度：{高，中，低}——>{1.0,0.5,0.0}
- 子主题 3

### 属性不存在序关系，转化为k维向量

- 瓜类：{西瓜，南瓜，黄瓜}——>{(0,0,1),(0,1,0),(1,0,0)}

### 参数的求解：均方误差最小化

- 均方误差的几何意义：欧氏距离(Euclidean distance)
- 最小二乘法(least square method):基于均方误差最小化来进行模型求解。对于线性回归，即找到一条直线，使所有样本到直线上的欧式距离之和最小。

### 线性回归模型的最小二乘参数估计(parameter estimation):求解参数使均方误差最小化的过程

- (3.5)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-5.png)
  
  - (3.5)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-5i.jpg)

- (3.6)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-6.png)
  
  - (3.6)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-6i.jpg)

- (3.7)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-7.png)
  
  - (3.7)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-7i.jpg)

- (3.8)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-8.png)

### 多元线性回归(multivariate linear regression)

- 由多特征量求解回归函数，如(3.2)式

- 参数
  
  - (3.9)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-9.png)

- 求导得最小值参数
  
  - (3.10)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-10.png)
    
    - (3.10)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-10i.jpg)
  
  - 当X'X是满秩矩阵(full-rank matrix)或正定矩阵(positive definite matrix)时
    
    - (3.11)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-11.png)
      
      - (3.11)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-11i.jpg)
  
  - 当不满秩时，比如变量数大于样例数时，选择哪一个解由学习算法的归纳偏好决定，常规做法为引入正则化(regularization)

- 多元线性回归模型
  
  - (3.12)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-12.png)

### 对数线性回归(log-linear regression)

- 在形式上仍是线性回归，实质上是在求取输入空间到输出空间的非线性函数映射
- (3.14)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-14.png)
- 对数回归示意图(图3.1)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3_1.png)

### 广义线性模型(generalized linear model)

- (3.15)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-15.png)

## 3.3对数几率回归

### 假设函数

- 单位阶跃函数(unit-step)(3.16)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-16.png)

- 对数几率函数(logistic function)
  
  - (3.17)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-17.png)
    
    - (图3.2)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3_2.png)

- 对数几率(log odds)
  
  - (3.19)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-19.png)

- 对数几率回归(logistic regression):用线性回归模型的预测结果去逼近真实标记的对数几率
  
  - 优点
    
    - 1.直接对分类可能性建模，无需事先假设数据分布，避免假设分布不准确带来的问题
    - 2.不仅预测出类别，可得到近似概率的预测，对利用概率辅助决策的任务有用
    - 3.对率函数是任意阶可导凸函数，有很好的数学性质，现有的许多数值优化算法可直接用于求取最优解

### 参数确定

- 对率回归模型最大化“对数似然”(log-likelihood)
  
  - (3.25)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-25.png)
    
    - (3.27)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-27.png)
      
      - (3.27)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-27i.jpg)

- 求最优解的方法
  
  - 梯度下降法(gradient descent mothed)
  
  - 牛顿法(Newton method)
    
    - 最优解(3.28)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-28.png)
    
    - 更新公式(3.29)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-29.png)
    
    - 一阶导数(3.30)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-30.png)
      
      - (3.30)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-30i.jpg)
    
    - 二阶导数(3.31)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-31.png)
      
      - (3.31)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-31i.jpg)

## 3.4线性判别分析

### 线性判别分析(Linear Discriminant Analysis, LDA):给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离；在对新样本进行分类时，将其投影到这条直线上，再根据投影点的位置来确定新样本的类别。

- LDA的二维示意图(图3.3)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3_3.png)

### 二分类

- 最大化目标(3.32)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-32.png)
  
  - (3.32)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-32i.jpg)
  
  - (3.35)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-35.png)
    
    - 类内散度矩阵(within-class scatter matrix)(3.33)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-33.png)
    - 类间散度矩阵(between-class scatter matrix)(3.34)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-34.png)
  
  - (3.39)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-39.png)
    
    - (3.39)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-39i.jpg)

### 多分类

- 全局散度矩阵(3.40)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-40.png)

- 类内散度矩阵(3.41)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-41.png)

- 类间散度矩阵(3.43)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-43.png)
  
  - (3.43)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-43i.jpg)

- 优化目标(3.44)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-44.png)
  
  - (3.44)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-44i.jpg)

- 参数求解(3.45)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-45.png)
  
  - (3.45)推导![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-45i.jpg)

## 3.5多分类学习

### 拆分策略

- 一对一(One vs. One, OvO)
  
  - 将N个类别两两配对，从而产生N(N-1)/2个二分类任务，最后把被预测得最多的类别作为最终分类的结果
    
    - 存储开销和测试时间开销比OvR更大，但每个分类器只用到两个样例。类别很多时，其训练时间开销比OvR更小

- 一对其余(One vs. Rest, OvR)
  
  - 每次将一个类的样例作为正例、其他所有类的样例作为反例来训练N个分类器，在测试时若仅有一个分类器预测为正类，则对应的类别标记为最终分类的结果。若有多个分类器预测为正类，则通常考虑各分类器的预测置信度，选择置信度最大的类别标记作为分类结果。是MvM的特例
    
    - 每个分类器使用到了所有样例

- 多对多(Many vs. Many)
  
  - 每次将若干类别作为正类，若干个其他类作为反类。正、反类构造必须有特殊设计，不能随意取。
    
    - 纠错输出码(Error Correcting Output Codes, ECOC)
      
      - 编码：对N个类别做M次划分，每次划分将一部分类别化为正类，一部分划为反类，从而形成一个二分类训练集；这样一共产生M个训练集，可训练出M个分类器
      
      - 解码：M个分类器分别对测试样本进行预测，这些预测标记组成一个编码，将这个预测编码与每个类别各自的编码进行比较，返回其中距离最小的类别作为最终预测结果
      
      - 纠错：ECOC编码越长，纠错能力越强，但是编码越长所需训练的分类器也越多，计算、开销会增大；对有限类别数，可能组合数目有限，码长超过一定范围后失去意义。
        
        - 码距越远纠错能力越强，但码长稍大难以有效确定最优编码——NP难问题
        - 非最优编码往往足以产生较好的分类器
        - 并不是编码的理论性质越好，分类性能就越好。

## 3.6类别不平衡问题

### 类别不平衡(class-imbalance):分类任务中不同类别的训练样例数目差别很大的情况

- 基本策略——“再缩放”(rescaling)
  
  - (3.48)![My-MachineLearning/3-5.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/3-48.png)

- 再缩放的操作不简单：训练集是真实样本总体的无偏采样这个假设往往并不成立
  
  - 1.直接对训练集里的反类样例进行“欠采样”(undersampling)，去除一些反例
    
    - 开销小于过采样，但不能随机丢弃反例，可能丢失一些重要信息
  
  - 2.对训练集里的正类样例进行“过采样”(oversampling)，增加一些正例
    
    - 不能简单对初始正例样本进行重复采样，否则会严重过拟合
  
  - 3.直接基于原始训练集学习，但在用训练好的分类器进行预测时，将式(3.48)嵌入到其决策过程中进行阈值移动(threshold-moving)

## 李浩 Li Hao 2022.11.11
