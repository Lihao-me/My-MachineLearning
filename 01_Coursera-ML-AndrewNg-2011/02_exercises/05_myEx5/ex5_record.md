# Exercise 5记录与总结

李浩 2022.11.23

---

## 1.线性回归正则化

题目：在这一次作业的上半部分，我需要通过水平面的改变预测水库的水量，在后半部分则需要测试算法的偏差与方差的影响。章节的主代码在[ex5.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/ex5.m)文件中。

### 1.1数据可视化

首先，将水库中历史的水位线x和水流量y的数据集可视化。

数据集包含三部分：

- 我们需要模型学习的训练集

- 用于设置正则化参数的交叉验证集

- 用于评估模型的测试集 

下一步将会绘制训练集。

![5trainingdata](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5trainingdata.png)

### 1.2正则化线性回归的损失函数

根据正则化线性回归的损失函数：

$$
J(\theta)=\frac{1}{2m}(\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^{2})+
\frac{\lambda}{2m}(\sum^{n}_{j=1}\theta^{2}_{j})
$$

其中$\lambda$为正则化因子，用来控制正则化的程度，决定对参数权重的惩罚。但是值得注意的是，不应该将$\theta_{0}$算进去。

现在我的任务是完成[linearRegCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/linearRegCostFunction.m)文件中的代码，计算正则化的线性回归的损失函数。题目的要求是，尽量尝试使用向量而不是for循环来完成。而我在第一次的作业中已经使用向量了。

我的答案：

```matlab
Theta = theta;
Theta(1,:)=[];
J = sum((X*theta-y).^2)/(2*m)+sum(Theta.^2)*lambda/(2*m);
```

输出结果：

```
Cost at theta = [1 ; 1]: 303.993192 
(this value should be about 303.993192)
```

可以看到对输入的测试θ值，算出的损失值和期望一致，说明代码没有问题。

### 1.3正则化线性回归梯度

相应地，损失函数的偏导数为：

$$
\frac{\partial J(\theta)}{\partial \theta_{0}}=\frac{1}{m}
\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})
x^{(i)}_{j},for \ j=0;

\\
\frac{\partial J(\theta)}{\partial \theta_{0}}=(\frac{1}{m}
\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j})+
\frac{\lambda}{m}\theta_{j},for \ j \ge 1
$$

继续在[linearREgCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/linearRegCostFunction.m)文件中完成梯度的计算。

我的答案：

```matlab
grad = (sum((X*theta-y).*X))'/m+Theta*lambda/m;
```

输出结果：

```
Gradient at theta = [1 ; 1]:  [-15.219682; 598.250744] 
(this value should be about [-15.303016; 598.250744])
```

通过与测试θ的期望比对可知符合预期，代码是正确的。

### 1.4线性回归拟合

一旦我们的代价函数和梯度写对了，那么主文件会调用[trainLinearReg.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/trainLinearReg.m)文件来计算最优的θ值。而训练函数用fmincg函数来优化代价函数。

在这一部分，我们先将正则化因子设为0。因为对于一个低维的案例，正则化因子的设置对θ的求取不会有太大的帮助。而在之后的题目中，我们将会用带有正则化的多项式回归解决问题。

输出和绘制的拟合图像如下：

```
Iteration     1 | Cost: 1.052435e+02
Iteration     2 | Cost: 2.237391e+01
```

![5linearpattern](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5linearpattern.png)

从图像可以看出来这并不是一个能将训练集拟合得很好的模型，因为显然这个数据集的变化为一个非线性的变化趋势。

在下一部分中，我需要完成学习曲线的绘制来帮助我们改善学习算法。

## 2.偏差与方差

机器学习中一个重要的概念就是偏差-方差的权衡。高偏差的模型可能由于不能很好地拟合数据而欠拟合，而高方差的模型会产生过拟合问题。在这部分练习，我需要绘制训练集和测试集在学习曲线上的误差来判断偏差-方差问题。

### 2.1学习曲线

学习曲线是以训练集大小为自变量、以训练集误差和交叉验证集误差为因变量的函数的图像。我需要完成[learningCurve.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/learningCurve.m)文件中的代码，返回训练误差和验证误差。

训练集应该是所有数据集的子集。我们可以用trainLinearReg函数找到最优的θ值，而正则化因子被省略。

训练误差被定义为：

$$
J_{train}(\theta)=\frac{1}{2m}[{\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^{2}}]
$$

我的答案：

```matlab
for i = 1:m
[theta] = trainLinearReg(X(1:i,:), y(1:i), lambda);
error_train(i,:) = linearRegCostFunction(X(1:i,:), y(1:i), theta, 0);
error_val(i,:) = linearRegCostFunction(Xval, yval, theta, 0);
end
```

输出结果：

```
# Training Examples    Train Error    Cross Validation Error
      1        0.000000    210.522449
      2        0.000000    110.300366
      3        3.286595    45.010231
      4        2.842678    48.368911
      5        13.154049    35.865165
      6        19.443963    33.829962
      7        20.098522    31.970986
      8        18.172859    30.862446
      9        22.609405    31.135998
      10        23.261462    28.936207
      11        24.317250    29.551432
      12        22.373906    29.433818
```

![5learningcurve](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5learningcurve.png)

通过和题目示例图的比对，可知运算是正确的。第一次的代码出现了报错，是因为我没注意到主文件已经把1序列加到X变量中了，后来改过来了就好了。

## 3.多项式回归

很显然，用线性模型对数据进行拟合太简单了，从而出现了欠拟合(高偏差)问题。在这部分，我需要通过增加更多特征变量来解决。

为了使用多项式回归，我们的假设函数如下：

$$
\begin{align*}
h_{\theta}(x)
&=
\theta_{0}+\theta_{1}\times(waterLevel)+\theta_{2}\times(waterLevel)^2+
\cdots+\theta_{p}\times(waterLevel)^p
\\
&=
\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+\cdots+\theta_{p}x_{p}
\end{align*}
$$

通过增加水位的幂次来得到多变量的线性回归假设函数。我的任务是在[polyFeatures.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/polyFeatures.m)文件中将原来的数据集X映射为更高阶的。特别地，数据集X的大小为mx1，那么这个函数应该返回一个mxp大小的。其中，我们不需要补全1序列。

我的答案：

```matlab
for i=1:p
    X_poly(:,i)=X.^(i);
end
```

测试结果：

```
>> p=8;
>> X

X =

  -15.9368
  -29.1530
   36.1895
   37.4922
  -48.0588
   -8.9415
   15.3078
  -34.7063
    1.3892
  -44.3838
    7.0135
   22.7627

>> a = polyFeatures(X,p) 
-15.9367581337854    253.980259814775    -4047.62197142405    64505.9723755808    -1028016.07993427    16383243.6235547    -261095791.075474    4161020472.11920
-29.1529792172381    849.896197240719    -24777.0061749684    722323.546084234    -21057883.3271154    613900034.994422    -17897014961.6541    521751305227.703
36.1895486266625    1309.68342980157    47396.8521683381    1715270.68629681    62074871.9096272    2246461595.46730    81298431147.0937    2942153527269.12
37.4921873319951    1405.66411093742    52701.4221731280    1975891.59277748    74080497.7441274    2777439899.07027    104132296999.300    3904147586408.72
-48.0588294525701    2309.65108835122    -110999.127750014    5334488.14992196    -256369256.213855    12320806361.2639    -592123531634.123    28456763821657.8
-8.94145793804976    79.9496700579130    -714.866611983785    6391.94974236915    -57153.3497635217    511034.272929175    -4569391.45629806    40857021.5089730
15.3077928892261    234.328523139441    3587.05250025678    54909.8567567113    840548.714808808    12866945.6395984    196964538.967903    3015092369.04255
-34.7062658113225    1204.52488656617    -41804.5608895186    1450880.20235725    -50354633.9633961    1747621311.16547    -60653409762.8408    2105053361592.21
1.38915436863589    1.92974985990018    2.68072044825483    3.72393452178476    5.17311990945130    7.18626212169158    9.98282742051048    13.8676883225403
-44.3837598516869    1969.91813857222    -87432.3735898713    3880577.47267582    -172234618.633110    7644419951.55883    -339288099335.431    15058881521439.1
7.01350208240411    49.1892114598868    344.988637005732    2419.57852404546    16969.7190169331    119017.159663073    834727.097138785    5854360.23402201
22.7627489197113    518.142738381818    11794.3530583570    268471.897337809    6111158.39109906    139106764.065175    3166452343.44909    72077159660.1632
```

可以看出计算符合预期。

### 3.1多项式回归的学习

现在我需要用线性回归的代价函数对多项式回归模型进行训练。因为我们把一个多项式回归转化为了一个线性回归。由于我们的特征变量个数为8了，各变量的范围有很大的差别，所以需要将变量范围标准化或者说归一化。在计算参数θ之前会调用featureNormalize函数来对训练集标准化。题目中已经给出了标准化的代码，所以不需要我自己写了。

输出的标准化样例：

```
Normalized Training Example 1:
  1.000000  
  -0.362141  
  -0.755087  
  0.182226  
  -0.706190  
  0.306618  
  -0.590878  
  0.344516  
  -0.508481  
```

代码运行到后面我发现了[linearRegCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/linearRegCostFunction.m)文件中的一个问题，那就是在对theta的第0位进行正则化处理时直接置空对二变量是没问题的，但是增加了特征造成矩阵维度不匹配。特此更正：

```matlab
Theta = theta;
Theta(1,:)=0;
J = sum((X*theta-y).^2)/(2*m)+sum(Theta.^2)*lambda/(2*m);

grad = (sum((X*theta-y).*X))'/m+Theta*lambda/m;
```

学习完参数后绘制拟合的图像如下：

![5polynomialFit](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5polynomialFit.png)

从图中可以看出曲线对数据集拟合得很好，训练误差很小。但是，在极端情况下，比如水位很低和很高的时候，曲线都成下降的趋势，极端情况下水量都很少，这显然和实际不符，说明发生了过拟合。它的泛化能力并不算好。

为了更好地理解非正则化的模型，我们可以观察它的学习曲线。

![5Plearningcurve](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5Plearningcurve.png)

上图显示训练误差一直能保持很低，但是交叉验证误差很好，这是很明显的高方差问题。

一种解决过拟合问题的方法就是对模型正则化。在下一部分的练习中，我需要尝试不同的正则化因子来找到一个更好的模型。

### 3.2选做练习：判断正则化因子

在这部分，我将观察正则化因子是如何影响正则化的多项式回归中偏差-方差的。我可以尝试将ex5.m文件中的$\lambda$修改为1或100，然后观察拟合的曲线和学习曲线。

下图分别是λ为1时的拟合曲线和学习曲线。

![5fitlambda1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5fitlambda1.png)

![5learningcurveLambda1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5learningcurveLambda1.png)

可以从拟合曲线看出并没有发生高偏差或高误差问题，它在二者之间权衡得很好。

下图分别为λ为100时的拟合曲线和学习曲线。

![5fitlambda100](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5fitlambda100.png)

![5learningcurvelambda100](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5learningcurvelambda100.png)

可以看出来拟合的效果不是很好，而且由于正则化因子设置得太大发生了欠拟合。

### 3.3用交叉验证集选择λ

从上面的练习中我们可以看出正则化因子对训练集和验证集的结果影响至关重要。特别地，一个没有正则化项的模型能对训练集拟合得很好，但是泛化性能较差；相反地，一个模型的正则化因子太大也会造成无法拟合数据集。选择一个合适的λ很关键。

在这一部分，我们需要自动选一个合适的正则化因子。具体地，我将使用交叉验证集来评估每一个设置的λ，然后用测试集检验模型的泛化性能。

我的任务是在[validationCurve.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/validationCurve.m)文件中补全代码。其中，用trainLinearReg函数来训练模型然后计算训练误差和交叉验证误差。可以尝试的λ范围如下：

$$
\{0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10\}
$$

而后程序会绘制每一个λ对应的训练误差和验证误差的学习曲线。

我的答案：

```matlab
for i = 1:length(lambda_vec)
          lambda = lambda_vec(i);
          [theta] = trainLinearReg(X, y, lambda);
          error_train(i,:) = linearRegCostFunction(X, y, theta, 0);
          error_val(i,:) = linearRegCostFunction(Xval, yval, theta, 0);  
end
```

输出结果：

```
lambda		Train Error	Validation Error
 0.000000	0.141605	17.247758
 0.001000	0.155194	16.705538
 0.003000	0.181312	18.324212
 0.010000	0.222331	17.071663
 0.030000	0.281879	12.830739
 0.100000	0.459318	7.587013
 0.300000	0.921760	4.636833
 1.000000	2.076188	4.260625
 3.000000	4.901351	3.822907
 10.000000	16.092213	9.945509
```

![5validationLambda](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5validationLambda.png)

所绘制图像和题目示例一致。第一次运行的时候代码写错了，当时只是照搬了前面[learningCurve.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/learningCurve.m)文件中求取误差的代码，忽略了此处应当把所有训练集都投入训练。更改后是正确的。

从图像可以看出来，使验证误差最小的λ在3左右。由于训练集和验证集划分的不可测性，所以有可能出现验证误差小于训练误差的情况。

### 3.4选做练习：计算测试集误差

为了更好地测试模型在实际应用中地性能表现，用测试集来评估其泛化性能很重要。在这一部分，我们应当设置λ为3.然后计算测试误差，预期结果应该为3.8599。

我的代码：

```matlab
%% =========== Part 9: Computing test set error =============

lambda = 3;
[theta] = trainLinearReg(X_poly, y, lambda);

close all;

error_test = linearRegCostFunction(X_poly_test, ytest, theta, 0);

fprintf('lambda\t\tTest Error\n');
fprintf(' %f\t%f\t%f\n', lambda, error_test);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
```

输出结果：

```
lambda		Test Error
 3.000000	3.859888	
```

可以看出测试误差和题目的预期结果一致，故计算正确。

### 3.5选做练习：绘制随机选择样例的学习曲线

在练习中，特别是对于小的训练集，随机选取训练集和交叉验证集的样本计算误差对调整算法是有很大帮助的。

同样地，我们可以随机地从训练集和验证集选取样本，然后学习参数θ，最后计算误差。上述过程应该重复很多次，比如50次，然后计算平均的误差作为最终的训练误差和交叉验证误差。

题目中给出了学习曲线的示例，而我们自己绘制的可能略有出入，因为随机选择的不确定性。

为了方便计算，我创建了一个名为[randomlyLearningCurve.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/randomlyLearningCurve.m)的函数文件，文件内容如下：

```matlab
function [error_train, error_val] = ...
    randomlyLearningCurve(X, y, Xval, yval, lambda)


% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================

for i = 1:m
    error_train_sum = 0;
    error_val_sum = 0;
    for k = 1:50
        random_index = randi(m,i,1);
        [theta] = trainLinearReg(X(random_index,:), y(random_index), lambda);
        error_train_one = linearRegCostFunction(X(random_index,:), y(random_index), theta, 0);
        error_val_one = linearRegCostFunction(Xval, yval, theta, 0);
        error_train_sum = error_train_sum + error_train_one;
        error_val_sum = error_val_sum + error_val_one;
    end
    error_train(i,:) = error_train_sum/50;
    error_val(i,:) = error_val_sum/50;
end

% -------------------------------------------------------------

% =========================================================================

end
```

主要的思路还是依照[learningCurve.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/learningCurve.m)文件，只不过加入一个50次的内循环，每一次生成随机数对样例随机取值。

主文件[ex5.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/05_myEx5/ex5/ex5.m)中的相应函数调用和绘图代码为：

```matlab
%% ==== Part 10: Plotting learning curves with randomly selected examples ====

lambda = 0.01;
m = size(X_poly,1);
[error_train, error_val] = randomlyLearningCurve(X_poly,y,X_poly_val,yval,0.01);

plot(1:m, error_train, 1:m, error_val);
title('Polynomial Regression Learning Curve (lambda=0.010000)')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
```

输出结果为：

```
# Training Examples	Train Error	Cross Validation Error
  	1		0.093415	1297.429719
  	2		0.001664	103.385417
  	3		0.014509	75.738684
  	4		0.017133	28.228395
  	5		0.027429	28.696438
  	6		0.036255	24.018164
  	7		0.057890	21.721267
  	8		0.057099	18.109760
  	9		0.071976	19.636329
  	10		0.073860	16.831972
  	11		0.094452	23.004649
  	12		0.088394	20.089562
```

由于每一次都进行了50次计算并求取平均，所以比原先直接取i=1:m消耗了成倍的时间。绘制的图像如下：

![5randomLearningCurve](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/5randomLearningCurve.png)

可以看到趋势和误差值大致符合题目所给样例，计算应该是没问题的。



第五次作业“正则化线性回归和偏差-误差权衡”完毕。
