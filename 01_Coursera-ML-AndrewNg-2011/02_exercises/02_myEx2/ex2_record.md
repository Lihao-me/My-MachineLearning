# Exercise 2记录与总结

李浩 2022.11.02

---

## 1.逻辑回归

题目：构建一个逻辑回归模型来预测一个学生是否获得大学的录取资格。假设我是一所大学机构的管理人员，我需要根据申请者两门考试的结果来决定是否录取他们。目前我手中有过去申请者的历史数据，我需要利用这些数据训练一个逻辑回归的模型来判断每一个申请者获得录取的概率。

### 1.1数据可视化

我需要完成[plotData.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/plotData.m)文件中绘图函数的编写，将已有数据进行绘制。

我的答案：

```matlab
%分别找到正、负样本
pos = find(y==1);
neg = find(y==0);

%绘制样本图像
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7);
```

输出：

![2training-data](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/2training-data.png)

### 1.2执行

#### 1.2.1热身练习：sigmoid函数

如课上所讲，假设函数为：

$$
{h_\theta }(x) = g({\theta ^T}x)
$$

其中，g函数就是sigmoid函数，为

$$
g(z) = \frac{1}{{1 + {e^{ - z}}}}
$$

我需要在[sigmoid.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/sigmoid.m)文件中完成sigmoid函数，使调用它可以计算向量中每一个元素的函数值。

我的答案：

```matlab
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
g = 1./(1+exp(-z));
```

在命令行测试代码的准确性：

```
>> sigmoid([-10000 0 10000])

ans =

         0    0.5000    1.0000
```

和函数的期望结果相符，且能够输入、输出向量，故而能够实现。

#### 1.2.2损耗函数和梯度下降

我需要在[costFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/costFunction.m)文件中完成损耗和梯度下降的计算，其中，损耗函数为：

$$
J(\theta ) = \frac{1}{m}\sum\limits_{i = 1}^m {[ - {y^{(i)}}\log ({h_\theta }({x^{(i)}})) - (1 - {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}}))]}
$$

而其参数的更新策略和线性回归是类似的，只是假设函数不同。

求偏导得：

$$
\frac{{\partial J(\theta )}}{{\partial {\theta _j}}} = \frac{1}{m}\sum\limits_{i = 1}^m {({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}} 
$$

更新策略为：

$$
{\theta _j} = {\theta _j} - \alpha \frac{1}{m}\sum\limits_{i = 1}^m {({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}}
$$

从而由此计算损耗值和导数值。

我的答案：

```matlab
h = sigmoid(X*theta);%经过假设函数计算后为mx1的向量
J = (-y'*log(h)-(1-y')*log(1-h))/m;%损耗函数值
grad = ((h-y)'*X)'./m;%对参数求导值
```

输出结果：

```
Cost at initial theta (zeros): 0.693147
Expected cost (approx): 0.693
Gradient at initial theta (zeros): 
 -0.100000 
 -12.009217 
 -11.262842 
Expected gradients (approx):
 -0.1000
 -12.0092
 -11.2628

Cost at test theta: 0.218330
Expected cost (approx): 0.218
Gradient at test theta: 
 0.042903 
 2.566234 
 2.646797 
Expected gradients (approx):
 0.043
 2.566
 2.647
```

可知跟期望结果相吻合，从而证明代码无误。

#### 1.2.3用fminunc函数求最佳参数

在解决线性回归的问题中，我们通过梯度下降的方法求得使损耗函数局部最小的参数$\theta$,而在这里可以调用Matlab函数库中的fminunc来进行求取。

具体的调用代码题目已给出：

```matlab
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
    fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```

在该步骤我不需要补充任何代码，保证上一步骤的函数准确，运行即可。主函数也会调用[plotDecisionBoundary.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/plotDecisionBoundary.m)文件进行决策边界的绘制。

输出结果：

```
Cost at theta found by fminunc: 0.203498
Expected cost (approx): 0.203
theta: 
 -25.161343 
 0.206232 
 0.201472 
Expected theta (approx):
 -25.161
 0.206
 0.201
```

![2decision-boundary](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/2decision-boundary.png)

可知计算结果与期望相吻合。

#### 1.2.4评估逻辑回归

此时就可以对某一学生能否被录取进行预测。

题目样例的输出为：

```
For a student with scores 45 and 85, we predict an admission probability of 0.776291
Expected value: 0.775 +/- 0.002
```

可知计算结果是准确的。

现在我需要完成[predict.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/predict.m)文件来计算模型的准确率。

我的答案：

```matlab
y_p = X*theta;
p = y_p>=0.5;
```

输出结果：

```
Train Accuracy: 89.000000
Expected accuracy (approx): 89.0
```

和期望一致，证明计算准确。

## 2.逻辑回归正则化

题目：我需要构建正则化的逻辑回归来预测是否通过微芯片工厂的质量验证。其中，每一块芯片都要通过数项检测来确保其功能的完善。假设我是工厂的产品制造经历，我的手里掌握了一些芯片的测试结果。我需要根据这些数据来决定芯片是否合格。

### 2.1数据可视化

同第一部分一样，为了将数据内容可视化处理，我们需要在[plotData.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/plotData.m)文件中对数据进行绘制。

输出结果：

![2training-data1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/2training-data1.png)

从图中可以看出来，我们不能像第一部分一样通过一条直线将两类划分开来。因此，简单的线性逻辑回归不适用于这一模型。

### 2.2特征映射

一种更好地拟合数据的办法是对每一个数据定义更多的特征量。在mapFeature.m文件中题目给出了对$x_{1}$和$x_{2}$两特征量的六次项特征量。

$$
{mapFeature(x)=\begin{bmatrix}
 \\1
 \\x_{1}
 \\x_{2}
 \\x_{1}^{2}
 \\x_{1}x_{2}
 \\x_{2}^{2}
 \\x_{1}^{3}
 \\\vdots 
 \\x_{1}x_{2}^{5}
 \\x_{2}^{6}
\end{bmatrix}}
$$

如此一来，特征量被转变为一个28维的向量。用逻辑回归分类器训练后会产生一个极为复杂的决策边界。由于特征量过多，可能会产生过拟合的问题。所以为了避免这种情况的发生，我需要通过构架正则化的逻辑回归来更好地拟合模型。

### 2.3损失函数和梯度下降

我需要在[costFunctionReg.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/costFunctionReg.m)文件中完成损失的计算和导数值。

损失函数为：

$$
J(\theta ) = \frac{1}{m}\sum\limits_{i = 1}^m {[ - {y^{(i)}}\log ({h_\theta }({x^{(i)}})) - (1 - {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}}))]} + \frac{\lambda }{{2m}}\sum\limits_{j = 1}^n {\theta _j^2}
$$

导数值为：

$$
\frac{\partial J(\theta)}{\partial \theta_{0}}=\left\{\begin{matrix}
 \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)},&j=0  \\
 (\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)})+\frac{\lambda}{m}\theta_{j},&j\geq 1  \\
\end{matrix}\right.
$$

我的答案：

```matlab
h = sigmoid(X*theta);%经过假设函数计算后为mx1的向量
J = (-y'*log(h)-(1-y')*log(1-h))/m + sum(theta.^2)*lambda/(2*m);%损耗函数值
theta(1)=0;
grad = ((h-y)'*X)'./m + theta*lambda/m;%对参数求导值
```

输出结果：

```
Cost at initial theta (zeros): 0.693147
Expected cost (approx): 0.693
Gradient at initial theta (zeros) - first five values only:
 0.008475 
 0.018788 
 0.000078 
 0.050345 
 0.011501 
Expected gradients (approx) - first five values only:
 0.0085
 0.0188
 0.0001
 0.0503
 0.0115

Program paused. Press enter to continue.

Cost at test theta (with lambda = 10): 3.206882
Expected cost (approx): 3.16
Gradient at test theta - first five values only:
 0.346045 
 0.161352 
 0.194796 
 0.226863 
 0.092186 
Expected gradients (approx) - first five values only:
 0.3460
 0.1614
 0.1948
 0.2269
 0.0922
```

从输出结果与题目的期望值的比较可以看出代码是比较准确的。

#### 2.3.1用fminunc函数学习参数

和前一部分一样，我们要用fminunc函数学习最佳的参数值$\theta$。从上面的输出结果也可以看出参数学习得比较好。

### 2.4绘制决策边界

在[plotDecisionBoundary.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/plotDecisionBoundary.m)中进行决策边界的绘制，其中$\lambda=1$。

输出结果：

![2decision-boundary1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/2decision-boundary1.png)

### 2.5选做练习

在这部分，我需要尝试不同的正则化因子来更好地理解正则化是怎么防止过拟合的。

当$\lambda=1$时，输出结果：

```
Train Accuracy: 76.271186
Expected accuracy (with lambda = 1): 83.1 (approx)
```

此处的预测值和题目的期望值显然不符，经过debug后，我发现[predict.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/predict.m)函数写错了，忘记写g函数了。而在上一部分的运行中却没有什么问题，我发现这是因为上一部分X*theta的值经过sigmoid与否并不影响判决，而当多变量的情况就会产生影响了。

故特此更正如下：

```matlab
%y_p = X*theta;
y_p = sigmoid(X*theta);
p = y_p>=0.5;
```

修改后的输出：

```
Train Accuracy: 83.050847
Expected accuracy (with lambda = 1): 83.1 (approx)
```

此时的预测值显然是正确的。

现在我们分别改变$\lambda$的大小，分别令其为0，1，10，100观察输出结果。

![2lambda0](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/2lambda0.png)

![2lambda10](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/2lambda10.png)

![2lambda100](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/2lambda100.png)

上图依次为$\lambda=0,10,100$的决策边界的图像，从图像可以看出，当正则化因子太小时会发生过拟合现象，而随着因子的增大，决策边界趋于良好，但是当设置过大后会发生欠拟合，此时的判决预测的错误率非常高，符合吴恩达课上老师所讲。
