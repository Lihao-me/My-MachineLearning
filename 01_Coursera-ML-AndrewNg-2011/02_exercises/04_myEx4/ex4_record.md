# Exercise 4记录与总结

李浩 2022.11.17

---

## 1.神经网络

题目：在上一次作业中，我们用题目中所提供的权重实现了神经网络的前向传播，预测了手写数字。在这一部分我需要完成反向传播算法来“学习”神经网络的参数。主体运行代码在[ex4.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/04_myEx4/ex4/ex4.m)文件中。

### 1.1数据可视化

1.1和1.2部分和上次的作业一样。我会得到一个包含有5000个测试样例的手写数字数据。题目已经将数据上传到矩阵变量中。每一个样例是一个20x20的灰度图像。每一个像素代表了灰度强度的浮点数。所以这20x20的网格像素被展开为400维的向量。那么矩阵X是所有训练样例的集合：

$$
X=\begin{bmatrix}
 { - {{({x^{(1)}})}^T} - }
\\
 { - {{({x^{(2)}})}^T} - }
 \\
\vdots \\
{ - {{({x^{(m)}})}^T} - }
\\
\end{bmatrix}
$$

第二部分的数据集为5000维的标签向量。值得一提的是，数字“0”被标记为“10”。

![4dataset1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/4dataset1.png)

### 1.2模型表征

模型结构和课件中展示的一样，有三层：一层输入层、一层隐藏层和一层输出层。每一个数据由20x20像素的灰度图像构成，那么输入层就需要包含400个单元(除了额外的激活单元)。我也被提供了两个神经网络参数：

$$
(\Theta^{(1)},\Theta^{(2)})
$$

这两个参数分别对应的是包含25个单元的隐藏层和包含10个输出单元的输出层。

其中，数据集的上传题目代码已经给出。

### 1.3前向传播和损失函数

现在我需要完成神经网络中代价值和梯度的计算。首先，在[nnCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/04_myEx4/ex4/nnCostFunction.m)文件中返回损失值。

其中，损失函数的表示为：

$$
J(\theta ) = \frac{1}{m}\sum\limits_{i = 1}^m \sum^{K}_{k=1}
{[ - {y^{(i)}_{k}}\log (({h_\theta }({x^{(i)}}))_{k}) - (1 - {y^{(i)}_{k}})
\log (1 - ({h_\theta }({x^{(i)}}))_{k})]} 
$$

其中，假设函数的内函数为sigmoid函数，K的值为10，代表了标签可能的10个类。值得注意的是，公式中的y为10x1的列向量。

我的答案：

```matlab
%不加正则项的损失函数
%方法一
X = [ones(m, 1) X];
hidden_units = sigmoid(X*Theta1');%计算出隐藏层的单元项
hidden_units = [ones(size(hidden_units,1),1) hidden_units];%添加+1的激活因子
output_units = sigmoid(hidden_units*Theta2');%计算输出层的结果
sum = 0;
for i=1:m
    for k=1:10
    Y = zeros(10,1);
    Y(y(i))=1;
    h = output_units(i,:);
    sum = sum+(-Y(k)*log(h(k))-(1-Y(k))*log(1-h(k)));
    end
end
J = sum/m;
```

输出结果：

```
Feedforward Using Neural Network ...
Cost at parameters (loaded from ex4weights): 0.287629 
(this value should be about 0.287629)
```

可以看到计算的损失值和期望值是一致的。

根据吴恩达老师的建议，采用的是用for循环来计算，但其实我没太搞懂的是log函数计算的变量是0或1这个是怎么计算出来的？不过既然公式没问题也没必要深究。

而如果不用for循环来做的话我也参考了别人的答案，感觉也很妙。通过eye矩阵来把1-10的自然数向量化为只含0和1的向量。

答案

```matlab
%不加正则项的损失函数
%方法二
%h = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
hidden_units = sigmoid(X*Theta1');%计算出隐藏层的单元项
hidden_units = [ones(size(hidden_units,1),1) hidden_units];%添加+1的激活因子
output_units = sigmoid(hidden_units*Theta2');%计算输出层的结果
%[~,h] = max(output_units,[],2);%计算出假设函数的输出值

 %将原来的标签值转化为10x1的向量并合并
 trans2one = eye(num_labels);
 y = trans2one(y,:);

 %求取损失值
 J = sum(sum(-y.*log(output_units)-(1-y).*log(1-output_units)))/m;
```

输出：

```
Feedforward Using Neural Network ...
Cost at parameters (loaded from ex4weights): 0.287629 
(this value should be about 0.287629)
```

 可以看到计算的损失值与期望也是一致的。

### 1.4正则化损失函数

带有正则化项的损失函数为：

$$
J(\theta ) = \frac{1}{m}\sum\limits_{i = 1}^m \sum^{K}_{k=1}
{[ - {y^{(i)}_{k}}\log (({h_\theta }({x^{(i)}}))_{k}) - (1 - {y^{(i)}_{k}})
\log (1 - ({h_\theta }({x^{(i)}}))_{k})]} +
\frac{\lambda}{2m}[\sum^{25}_{j=1}\sum^{400}_{k=1}(\Theta^{(1)}_{j,k})^{2}+
\sum^{10}_{j=1}\sum^{25}_{k=1}(\Theta^{(2)}_{j,k})^{2}]
$$

可以假设我们的神经网络只有三层，但是我们的代码应该适用于任意数目的输入和输出。虽然题目给出了两个参数矩阵的大小，但是我们的代码应该适用于任意大小的参数矩阵。计算出来的损失值应当为0.383770左右。只需要把参数矩阵的所有值相加即可。

我的答案：

```matlab
%加正则化项的损失函数
Theta1(:,1)=[];
Theta2(:,1)=[];
regu_value = (sum(sum(Theta1.^2))+sum(sum(Theta2.^2)))*lambda/(2*m);
J = J+regu_value;
```

输出结果：

```
Checking Cost Function (w/ Regularization) ... 
Cost at parameters (loaded from ex4weights): 0.383770 
(this value should be about 0.383770)
```

可以验证出计算的结果是准确的。

## 2.反向传播

现在我们需要通过反向传播计算神经网络损失函数的梯度值。我需要继续在[nnCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/04_myEx4/ex4/nnCostFunction.m)文件中完成代码，返回梯度值。计算出梯度值后就可以用像fmincg这样的函数来最小化损失函数从而训练模型。

先是没有正则化项的，然后是有正则化项的，一步步来。

### 2.1Sigmoid函数的梯度

我们首先在[sigmoidGradient.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/04_myEx4/ex4/sigmoidGradient.m)文件中要完成sigmoid梯度函数。对于sigmoid函数的梯度可以计算为：

$$
g'(z)=\frac{d}{d z}g(z)=g(z)(1-g(z))
$$

其中，

$$
g(z)=sigmoid(z)=\frac{1}{1+e^{-z}}
$$

完成后，我们可以在命令行输入一些值检查一下，比如调用sigmoidGradient(0),那么算出来的返回值应该为0.25。

我的答案：

```matlab
g = sigmoid(z)*(1-sigmoid(z));
```

输出结果：

```
>> sigmoidGradient(0)

ans =

    0.2500
```

可以验证计算是正确的。

### 2.2随机初始化

在课上已经提到过，当训练神经网络的时候，随机初始化参数是很重要的。应当在$[-\epsilon_{init},\epsilon_{init}]$范围内给$\Theta^{(l)}$随机选取初始值。在这里，我们设定$\epsilon_{init}=0.12$,我需要在[randInitializeWeights.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/04_myEx4/ex4/randInitializeWeights.m)文件中初始化参数。而相关代码已经在题目中给出：

```
% Randomly initialize the weights to small values
epsilon init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
```

### 2.3反向传播

现在需要完成反向传播算法。首先会计算对于所有输入样本的前向传播激活因子，包括输出项。然后计算每一个节点的误差项。具体来说就是下面四步和第五步用来累计m个样本的总误差，作为损失值。

- 1.将样本带入输入层，通过前向传播计算每一层的激活因子。值得注意的是，要我们自己手动添加+1的激活因子。

- 2.对于输出层的每一个单元，设置
  
  $$
  \delta^{(3)}_{k}=(a^{(3)}_{k}-y_{k})
  $$
  
  其中：
  
  $$
  y_{k}\in{\{0,1\}}
  $$
  
  它反映了这个样本属于哪一类。

- 3.对于隐藏层，设置
  
  $$
  \delta^{(2)}=(\Theta^{(2)})^{T}\delta^{(3)}.*g'(z^{(2)})
  $$

- 4.用下面的公式对所有的误差累加。值得注意的是应该跳过或者移走$\delta^{(2)}_{0}$项。
  
  $$
  \Delta^{(l)}=\Delta^{(l)}+\delta^{(l+1)}(a^{(l)})^T
  $$

- 5.从而得到代价函数的梯度
  
  $$
  \frac{\partial}{\partial \Theta^{(l)}_{ij}}J(\Theta)=D^{(l)}_{ij}
=\frac{1}{m}\Delta^{(l)}_{ij}
  $$

我的答案：

```matlab
%计算输出层误差
delta_3 = y-output_units;
%disp(size(delta_3));

%计算隐藏层误差
%disp(size(Theta2));
hidden_units(:,1)=[];
%disp(size(hidden_units));
delta_2 = delta_3*Theta2.*sigmoidGradient(hidden_units);

%计算累计项
Delta_1 = (delta_2)'*X;
hidden_units = [ones(size(hidden_units,1),1) hidden_units];
Delta_2 = (delta_3)'*hidden_units;

Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;
```

输出结果：

```
Evaluating sigmoid gradient...
Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:
  0.196612 0.235004 0.250000 0.235004 0.196612 
```

目前还不能确定代码是否正确，下面通过梯度检验来判断。

### 2.4梯度检验

在我们的神经网络中，需要最小化代价函数。为了检验我们的梯度，可以将参数展开成长向量。

根据课上所讲的微积分中极限的思想，设

$$
\theta^{i+}=\theta+
\begin{bmatrix}
0\\
0\\
\vdots\\
\epsilon\\
\vdots\\
0
\end{bmatrix}
and\
\theta^{i-}=\theta-
\begin{bmatrix}
0\\
0\\
\vdots\\
\epsilon\\
\vdots\\
0
\end{bmatrix}
$$

现在可以近似求出代价函数的梯度：

$$
f_{i}(\theta)=\frac{\partial}{\partial \theta_{i}}J(\theta)\approx
\frac{J(\theta^{(i+)})-J(\theta^{(i-)})}{2\epsilon}
$$

如果计算的结果和梯度接近，那么基本上是没问题的。题目中已经在[computeNumericalGradient.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/04_myEx4/ex4/computeNumericalGradient.m)文件中给出了梯度计算的代码，我们不需要做任何操作。[checkNNGradients.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/04_myEx4/ex4/checkNNGradients.m)文件中已给出了梯度检验的代码。如果我的反向传播算法是正确的，那么可以得到一个不超过 1e-9的误差值。

在这个过程中我发现刚刚的delta3计算写错了，改正后：

```matlab
%计算输出层误差
delta_3 = output_units-y;

%disp(size(delta_3));

%计算隐藏层误差
%disp(size(Theta2));
hidden_units(:,1)=[];
%disp(size(hidden_units));
delta_2 = delta_3*Theta2;
delta_2 = delta_2(:,2:end);
delta_2 = delta_2.*sigmoidGradient(hidden_units);


%disp(size(delta_2));

%计算累计项
Delta_1 = (delta_2)'*X;
hidden_units = [ones(size(hidden_units,1),1) hidden_units];
Delta_2 = (delta_3)'*hidden_units;    

Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;
```

输出结果：

```
If your backpropagation implementation is correct, then 
the relative difference will be small (less than 1e-9). 

Relative Difference: 0.00105872
```

可以看出相关误差是大于1e-9的，说明代码还需改进。

...

调试了好久，最后发现根本不是代码需要改进，而是前面的代码写错了！我根本没分清a_2和z_2的区别，现特此更正！

```matlab
z2 = a1*Theta1';
hidden_units = sigmoid(z2);%计算出隐藏层的单元项
hidden_units = [ones(size(hidden_units,1),1) hidden_units];%添加+1的激活因子
a2 = hidden_units;   


%计算输出层误差
delta_3 = a3-y;

%disp(size(delta_3));

%计算隐藏层误差
%disp(size(Theta2));

%disp(size(hidden_units));
delta_2 = delta_3*Theta2;
delta_2 = delta_2(:,2:end);
delta_2 = delta_2.*sigmoidGradient(z2);


%disp(size(delta_2));

%计算累计项
Delta_1 = (delta_2)'*a1;

Delta_2 = (delta_3)'*a2;

Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;
```

输出结果：

```
Checking Backpropagation... 
   -0.0093   -0.0093
    0.0089    0.0089
   -0.0084   -0.0084
    0.0076    0.0076
   -0.0067   -0.0067
   -0.0000   -0.0000
    0.0000    0.0000
   -0.0000   -0.0000
    0.0000    0.0000
   -0.0000   -0.0000
   -0.0002   -0.0002
    0.0002    0.0002
   -0.0003   -0.0003
    0.0003    0.0003
   -0.0004   -0.0004
   -0.0001   -0.0001
    0.0001    0.0001
   -0.0001   -0.0001
    0.0002    0.0002
   -0.0002   -0.0002
    0.3145    0.3145
    0.1111    0.1111
    0.0974    0.0974
    0.1641    0.1641
    0.0576    0.0576
    0.0505    0.0505
    0.1646    0.1646
    0.0578    0.0578
    0.0508    0.0508
    0.1583    0.1583
    0.0559    0.0559
    0.0492    0.0492
    0.1511    0.1511
    0.0537    0.0537
    0.0471    0.0471
    0.1496    0.1496
    0.0532    0.0532
    0.0466    0.0466

The above two columns you get should be very similar.
(Left-Your Numerical Gradient, Right-Analytical Gradient)

If your backpropagation implementation is correct, then 
the relative difference will be small (less than 1e-9). 

Relative Difference: 2.37276e-11
```



调了一下午终于发现问题了。现在可以看到相关误差比1e-9要小得多，从而验证了算法的准确性。

### 2.5神经网络正则化

完成反向传播算法后给梯度增加正则化项。其中，

$$
\frac{\partial}{\partial \Theta^{(l)}_{ij}}J(\Theta)=D^{(l)}_{ij}
=\frac{1}{m}\Delta^{(l)}_{ij},
\
for \ j=0
\\
\frac{\partial}{\partial \Theta^{(l)}_{ij}}J(\Theta)=D^{(l)}_{ij}
=\frac{1}{m}\Delta^{(l)}_{ij}+\frac{\lambda}{m}\Theta^{(l)}{ij},
\
for \ j\geq 0
$$

经过梯度检验后的相关误差依然不应超过1e-9。

我的答案：

```matlab
theta1 = Theta1;
theta1(:,1) = 0;
theta2 = Theta2;
theta2(:,1) = 0;

Theta1_grad = Theta1_grad + theta1*lambda/m;
Theta2_grad = Theta2_grad + theta2*lambda/m;
```

输出结果：

```
Checking Backpropagation (w/ Regularization) ... 
   -0.0093   -0.0093
    0.0089    0.0089
   -0.0084   -0.0084
    0.0076    0.0076
   -0.0067   -0.0067
   -0.0168   -0.0168
    0.0394    0.0394
    0.0593    0.0593
    0.0248    0.0248
   -0.0327   -0.0327
   -0.0602   -0.0602
   -0.0320   -0.0320
    0.0249    0.0249
    0.0598    0.0598
    0.0386    0.0386
   -0.0174   -0.0174
   -0.0576   -0.0576
   -0.0452   -0.0452
    0.0091    0.0091
    0.0546    0.0546
    0.3145    0.3145
    0.1111    0.1111
    0.0974    0.0974
    0.1187    0.1187
    0.0000    0.0000
    0.0337    0.0337
    0.2040    0.2040
    0.1171    0.1171
    0.0755    0.0755
    0.1257    0.1257
   -0.0041   -0.0041
    0.0170    0.0170
    0.1763    0.1763
    0.1131    0.1131
    0.0862    0.0862
    0.1323    0.1323
   -0.0045   -0.0045
    0.0015    0.0015

The above two columns you get should be very similar.
(Left-Your Numerical Gradient, Right-Analytical Gradient)

If your backpropagation implementation is correct, then 
the relative difference will be small (less than 1e-9). 

Relative Difference: 2.26976e-11


Cost at (fixed) debugging parameters (w/ lambda = 3.000000): 0.576051 
(for lambda = 3, this value should be about 0.576051)
```

从输出结果来看显然相关误差小于1e-9，而且正则化的代价值和题目的期望一致，证明结果是准确的。

### 2.6用fmincg函数学习参数

在我们成功实现了神经网络代价函数和梯度的计算，下面用fmincg函数进行模型的训练。在训练完后，代码会计算准确率，同时由于是随机初始化的参数，所以可能会有1%左右的浮动，但大致是95.3%的准确率。可以通过增加迭代次数来提高准确率的。可以尝试增加迭代次数，比如加到400以及修改正则化参数来看变化。如果得到了正确的参数，是可能通过神经网络来完美拟合训练集的。

输出结果：

```
Training Neural Network... 
Iteration     1 | Cost: 3.305129e+00
Iteration     2 | Cost: 3.245789e+00
Iteration     3 | Cost: 3.219558e+00
Iteration     4 | Cost: 2.432228e+00
Iteration     5 | Cost: 2.143689e+00
Iteration     6 | Cost: 2.000052e+00
Iteration     7 | Cost: 1.776712e+00
Iteration     8 | Cost: 1.664510e+00
Iteration     9 | Cost: 1.515927e+00
Iteration    10 | Cost: 1.452427e+00
Iteration    11 | Cost: 1.364958e+00
Iteration    12 | Cost: 1.276418e+00
Iteration    13 | Cost: 1.243959e+00
Iteration    14 | Cost: 1.108027e+00
Iteration    15 | Cost: 1.057941e+00
Iteration    16 | Cost: 1.008834e+00
Iteration    17 | Cost: 9.511477e-01
Iteration    18 | Cost: 8.910541e-01
Iteration    19 | Cost: 8.606464e-01
Iteration    20 | Cost: 8.176194e-01
Iteration    21 | Cost: 7.801373e-01
Iteration    22 | Cost: 7.661776e-01
Iteration    23 | Cost: 7.471502e-01
Iteration    24 | Cost: 7.281490e-01
Iteration    25 | Cost: 7.205453e-01
Iteration    26 | Cost: 7.192330e-01
Iteration    27 | Cost: 7.073116e-01
Iteration    28 | Cost: 7.005243e-01
Iteration    29 | Cost: 6.975506e-01
Iteration    30 | Cost: 6.866026e-01
Iteration    31 | Cost: 6.815932e-01
Iteration    32 | Cost: 6.799253e-01
Iteration    33 | Cost: 6.742676e-01
Iteration    34 | Cost: 6.690619e-01
Iteration    35 | Cost: 6.631016e-01
Iteration    36 | Cost: 6.305451e-01
Iteration    37 | Cost: 5.873235e-01
Iteration    38 | Cost: 5.562399e-01
Iteration    39 | Cost: 5.489734e-01
Iteration    40 | Cost: 5.463247e-01
Iteration    41 | Cost: 5.439009e-01
Iteration    42 | Cost: 5.330093e-01
Iteration    43 | Cost: 5.267779e-01
Iteration    44 | Cost: 5.234289e-01
Iteration    45 | Cost: 5.158683e-01
Iteration    46 | Cost: 5.109927e-01
Iteration    47 | Cost: 5.082554e-01
Iteration    48 | Cost: 5.046228e-01
Iteration    49 | Cost: 5.016468e-01
Iteration    50 | Cost: 4.986272e-01  


Training Set Accuracy: 95.060000
```

可以看到准确率是在95.3%附近的。

## 3.隐藏层可视化

一种更好地理解神经网络是如何工作的方法就是对隐藏层可视化。考虑到隐藏层的维度和输入层的维度不同，所以可以通过向量维度的重构来进行显示。显示的图像如下：

![4hiddenUnits](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/4hiddenUnits.png)

### 3.1选做练习

如上一个题目结尾所说，可以通过改变迭代次数和正则化因子的大小来看看神经网络的更好工作效果。

神经网络是一种构造复杂决策边界的强大的模型。如果没有正则化，神经网络可能会过拟合从而使其在训练集上的准确率极为逼近100%，但是对没有训练过的样本的效果不一定好。所以可尝试自主改变迭代次数和正则项因子来看看。
