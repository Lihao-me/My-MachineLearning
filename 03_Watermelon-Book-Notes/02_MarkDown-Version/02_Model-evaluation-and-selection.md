# 02模型评估与选择

## 2.1经验误差与过拟合

### 误差

- 错误率(error rate)
  
  - 分类错误的样本占总样本数的比例
    
    - $E = \frac{a}{m}$

- 精度(accuracy)
  
  - 精度=1-错误率
    
    - $1 - \frac{a}{m}$

- 误差(error)
  
  - 学习器的实际预测输出与样本的真实输出之间的差异
    
    - 训练误差(training error)/经验误差(empirical error)
      
      - 学习器在训练集上的误差
    
    - 泛化误差(generalization)
      
      - 学习器在新样本上的误差

### 拟合

- 过拟合(overfitting)/过配
  
  - 把训练样本自身的一些特点当作所有潜在样本的一般性质，导致泛化性能下降
    
    - 原因：学习能力过于强大，把训练样本包含的不太一般特性学到
      
      - 无法彻底避免，只能缓解

- 欠拟合(underfitting)/欠配
  
  - 对训练样本的一般性质尚未学好
    
    - 原因：学习能力低下

## 2.2评估方法

### 测试集(testing set)

- 用于测试学习器对新样本的判别能力
  
  - 测试集应尽可能与训练集互斥

### 测试误差(testing error)

- 一般将其作为泛化误差的近似

### 2.2.1留出法(hold-out)

- 概念：将数据集D划分为两个互斥的集合：训练集D和测试集T
  
  - 用S训练出模型，用T评估测试误差

- 注意
  
  - 1.训练/测试集的划分要尽可能保持数据分布的一致性，避免因数据划分过程中引入额外的偏差对最终结果产生影响
  - 2.在使用留出法时一般采用若干次随机划分、重复进行实验评估后取平均值作为评估结果

- 训练集与测试集的比例的矛盾即为训练效果和评估结果的矛盾
  
  - 一般将大约2/3~4/5的样本用于训练，剩下的作为测试集

### 2.2.2交叉验证法(cross validation)/k折交叉验证(k-fold cross validation)

- 步骤：先将数据集D划分为k个大小相似的互斥子集D_i，每个子集保持数据分布一致性(分层采样)，然后每次用k-1个子集的并集作为训练集，余下的子集作测试集。进行k次训练和测试，最终返回k个测试结果的均值。

- 为减小因样本划分不同引入差别，通常需要随机使用不同的划分重复p次，最终评估结果为p次k折交叉验证的均值
  
  - 10次10折交叉验证

- 留一法(Leave-One-Out, LOO)
  
  - 数据集中有m个样本，令k=m进行交叉验证
    
    - 优点：不受随机样本划分方式的影响，评估结果比较准确
    - 缺陷：数据集较大时训练m个模型的计算开销极大

### 2.2.3自助法(bootstrapping)

- 基础：自助采样法(bootstrap sampling)/可重复采样/有放回采样

- 步骤：每次随机从D中挑选一个样本将其拷贝放入 D‘ 然后再将该样本放回初始数据集D中，使得该样本在下次采样时仍有可能被采到;这个过程重复执行m次后就得到了包含m个样本的数据集D',初始数据集中约有 36.8% 的样本未出现在采样数据集 D'中,于是可将 D' 用作训练集, D\D' 用作测试集

- 适用情况：数据集较小、难以有效划分训练集和测试集
  
  - 优点：能从初始数据集产生多个不同训练集，对集成学习等方法有好处
  - 缺点：其产生的数据集改变了初始数据集的分布，引入估计偏差

### 2.2.4调参与最终模型

- 参数(parameter)
  
  - 超参数
    
    - 算法的参数，数目在10以内，通常由人工设定多个参数候选值产生模型
  
  - 模型的参数
    
    - 数目可能很多，通过学习来产生多个候选模型

- 调参(parameter tuning)
  
  - 对算法参数进行设定
  - 现实中常对每个参数选定范围和变化步长

- 验证集(validation set)
  
  - 模型评估与选择中用于评估测试的数据集

## 2.3性能度量

### 性能度量(performance measure)

- 衡量模型泛化能力的评价标准
  
  - 回归任务：均方误差(mean squared error)
    
    - (2.2)
      
      $$
      E(f;D) = \frac{1}{m}\sum\limits_{i = 1}^m {{{(f({x_i}) - {y_i})}^2}}
      $$
      
      
    - (2.3)
      
      $$
      E(f;{\cal D}) = \int_{x \sim \cal D} {{{(f(x) - y)}^2}p(x)dx}
      $$
      
      

### 2.3.1错误率与精度

- 错误率
  
  - 分类错误的样本数占样本总数的比例
    
    - (2.4)![2.4](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-4.png)
    - (2.6)![2.6](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-6.png)

- 精度
  
  - 分类正确的样本数占样本总数的比例
    
    - (2.5)![2.6](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-5.png)
    - (2.7)![2.7](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-7.png)

### 2.3.2查准率、查全率与F1

- 查准率(precision)/准确率
  
  - 检索出的信息有多少比例使用户感兴趣的
    
    - (2.8)
      
      $$
      P = \frac{{TP}}{{TP + FP}}
      $$
      
      

- 查全率(recall)/召回率
  
  - 用户感兴趣的信息中有多少被检索出来了
    
    - (2.9)
      
      $$
      R = \frac{{TP}}{{TP + FN}}
      $$
      
      

- P-R曲线
  
  - ![ROC-AUC](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2_P-R.png)

- 真正例(true positive)、假正例(false positive)、真反例(true negative)、假反例(false negative)
  
  - TP+FP+TN+FN=样例总数

- 混淆矩阵(confusion matrix)
  
  - ![confusion-matrix](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2_confusion-matrix.png)

- 查准率和查全率为一对矛盾的度量
  
  - 增加选瓜数量能尽可能多地将好瓜选出，但查准率低
  - 只挑选最有把握的瓜来使选出的瓜中好瓜比例尽可能高，但会漏掉不少好瓜，查全率低

- 查准率-查全率曲线(P-R曲线)
  
  - 1.一个学习器的曲线被另一个完全包住，则前者性能优于后者
  
  - 2.平衡点：查准率=查全率时的取值
    
    - (2-10)
      
      $$
      F1 = \frac{{2 \times P \times R}}{{P + R}} = \frac{{2 \times TP}}{{样例总数 + TP - TN}}
      $$
      
      
      
      - (2.10)推导![2.10i](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-10i.jpg)

- 查准率/查全率偏好
  
  - (2.11)
    
    - $$
      {F_\beta } = \frac{{(1 + {\beta ^2}) \times P \times R}}{{({\beta ^2} \times P) + R}}
      $$
    - $$
      \left\{ {\begin{array}{*{20}{c}}
{\beta  < 1}&查准率影响大\\
{\beta  = 1}&退化为标准F1\\
{\beta  > 1}&查全率影响大
\end{array}} \right.
      $$

- 多分类
  
  - 先计算查准率和查全率，再计算平均值
    
    - 宏查准率(macro-P)
      
      $$
      macro - P = \frac{1}{n}\sum\limits_{i = 1}^n {{P_i}}
      $$
      
      
    - 宏查全率(macro-R)
      
      $$
      macro - R = \frac{1}{n}\sum\limits_{i = 1}^n {{R_i}}
      $$
      
      
    - 宏F1
      
      $$
      macro - F1 = \frac{{2 \times macro - P \times macro - R}}{{macro - P + macro - R}}
      $$
      
      
  
  - 先平均再计算
    
    - 微查准率(micro-P)
      
      $$
      micro-P = \frac{{\overline {TP} }}{{\overline {TP}  + \overline {FP} }}
      $$
      
      
    - 微查全率(micro-R)
      
      $$
      micro - R = \frac{{\overline {TP} }}{{\overline {TP}  + \overline {FN} }}
      $$
      
      
    - 微F1(micro-F1)
      
      $$
      micro - F1 = \frac{{2 \times micro - P \times micro - R}}{{micro - P + micro - R}}
      $$

### 2.3.3ROC与AUC

- 分类阈值(threshold)
  
  - 将学习器的预测值与阈值比较，大于则为正类，否则为反类
    
    - 决定了学习器的泛化能力

- 截断点(cut point)
  
  - 根据预测结果对测试样本排序，将截断点前作正例，后一部分作反例
    
    - 重视查准率
      
      - 在靠前位置截断
    
    - 重视查全率
      
      - 在靠后位置截断

- 受试者工作特征(Receiver Operating Characteristic, ROC)曲线
  
  - 纵轴：真正例率(True Positive Rate, TPR)
    
    - $$
      TPR = \frac{{TP}}{{TP + FN}}
      $$
  
  - 横轴：假正例率(False Positive Rate, FPR)
    
    - $$
      FPR = \frac{{FP}}{{TN + FP}}
      $$
      
      ![ROC-AUC](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2_ROC-AUC.png)

- AUC(Area Under ROC Curve)
  
  - ROC曲线下面积
    
    - (2.20)
      
      $$
      AUC = \frac{1}{2}\sum\limits_{i = 1}^{m - 1} {({x_{i + 1}} - {x_i}) \cdot ({y_i} + {y_{i + 1}})}
      $$
      
      
      
      - 反映了样本的排序质量，越大越好

- 排序损失(loss)/排序误差
  
  - (2.21)![2.21](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-21.png)
    
    - 正例预测值小于反例记一个罚分，等于则记0.5个
    
    - (2.22)![2.22](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-22.png)
      
      - (2.22)推导![2.22i](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-22i.jpg)

### 2.3.4代价敏感错误率与代价曲线

- 非均等代价(unequal cost)
  
  - 权衡不同类型错误所造成的不同损失
    
    - 期望最小化总体代价(total cost)而不是错误数

- 二分类：代价矩阵(cost matrix)
  
  - ![cost-matrix](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2_cost-matrix.png)

- 代价敏感(cost-sensitive)错误率
  
  - (2.23)![2.23](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-23.png)
    
    - （2.23）演进![2.23i](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-23i.jpg)

- 代价曲线(cost curve)
  
  - 反映学习器的期望总体代价
    
    - 横轴：正例概率代价[0,1]
      
      - (2.24)
        
        $$
        P( + )cost = \frac{{p \times cost_{{01}}}}{{p \times cost_{{0{\rm{1}}}} + (1 - p) \times cost_{{10}}}}
        $$
        
        
    
    - 规范化(normalization):将不同变化范围的值映射到相同的固定范围
    
    - 纵轴：归一化代价[0,1]
      
      - (2.25)
        
        $$
        cos {t_{norm}} = \frac{{FNR \times p \times cos {t_{01}} + FPR \times (1 - p) \times cos {t_{10}}}}{{p \times cos {t_{0{\rm{1}}}} + (1 - p) \times cos {t_{10}}}}
        $$

## 2.4比较检验

### 学习器性能的比较

- 1.测试集上的性能并不一定等于泛化性能
- 2.不同样例而相同大小的测试集的测试结果也会不同
- 3.相同的参数在一个测试集运行多次的结果也会不同

### 2.4.1假设检验

- 在包含m个样本的测试集上，泛化错误率为$\epsilon$的学习器被测得测试错误率为$\hat \epsilon$的概率为
  
  - (2.26)![2.26](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-26.png)
    
    - 概率在二者相等时最大
      
      - (2.26)推演![2.26i](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-26i.jpg)

- 在1-α概率内能观测到的最大错误率
  
  - (2.27)
    
    $$
    \overline \epsilon   = \min \epsilon 
\begin{array}{*{20}{c}}
{}&{}
\end{array}s.t.\begin{array}{*{20}{c}}
{}&{}
\end{array}\sum\limits_{i = {\epsilon _0} \times m + 1}^m {\left( {\begin{array}{*{20}{c}}
m\\
i
\end{array}} \right)} {\epsilon ^i}{(1 - \epsilon )^{m - i}} < \alpha
    $$
    
    
    
    - （2.27）推导![2.27i](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-27i.jpg)
  
  - 若测试错误率小于临界值，则能以1-α的置信度(confidence)认为学习器的泛化错误率不大于$\epsilon_{0}$

- 多次重复留出法或交叉验证法多次训练和测试，使用t检验(t-test)
  
  - 错误率(2.28)![2.28](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-28.png)
  - 方差(2.29)![2.29](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-29.png)
  - 变量(2.30)![2.30](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-30.png)

### 2.4.2交叉验证t检验

- 基本思想：若两个学习器的性能相同，则它们使用相同的训练集、测试集得到的测试错误率应相同

### 2.4.3McNemar检验

- 对二分类问题使用留出法估计学习器的测试错误率，获得分类结果差别

### 2.4.4Friedmaan检验与Nemenyi检验

![Friedman](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2_Friedman.png)

- ## 2.5偏差与方差

### 偏差-方差分解(bias-variance decomposition)

- 是解释学习算法性能的一种重要工具

### 回归任务

- 学习算法的期望预测
  
  - (2.37)![2.37](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-37.png)

- 相同样本数的不同训练集产生的方差和噪声
  
  - (2.38)![2.38](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-38.png)
    
    - 方差度量了同样大小的训练集的变动所导致的学习性能的变化，刻画了数据扰动所造成的影响
  
  - (2.39)![2.39](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-39.png)
    
    - 噪声表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界，刻画了学习问题本身的难度

- 期望输出与真实标记的差别，也即偏差(bias)
  
  - (2.40)![2.40](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-40.png)
    
    - 度量了学习算法与期望的真实结果的偏离程度，刻画了学习算法本身的拟合能力

- 泛化误差可分解为偏差、方差与噪声之和
  
  - (2.42)![2.42](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2-42.png)
    
    - 泛化性能是由学习算法的能力、数据的充分性以及学习任务本身的难度共同决定的

### 给定任务较好的泛化性能

- 较小的偏差
  
  - 充分拟合数据

- 较小的方差
  
  - 使数据扰动的影响小

### 偏差-方差窘境(bias-variance dilemma)

- 偏差与方差是有冲突的
  
  - 1.训练不足，拟合不强，扰动影响不强，偏差主导——>2.训练加大，拟合增强，扰动被学习，方差主导——>3.训练充足，拟合过强，扰动影响大，局部特性被学到，过拟合
    
    - ![bias-variance](https://github.com/Lihao-me/My-MachineLearning/blob/main/03_Watermelon-Book-Notes/00_Images/2_bias-variance.png)

## 李浩 LiHao 2022.10.29
