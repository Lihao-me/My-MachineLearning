# Exercise 3记录与总结

李浩 2022.11.12

---

## 1.多分类

题目：用逻辑回归和神经网络来辨认手写数字(0到9)。自动化的手写数字识别广泛应用在今天的邮政中。

首先，我需要扩展原有的逻辑回归模型，并将其应用到一对多的分类中。

### 1.1数据集

我会得到一个包含有5000个测试样例的手写数字数据。题目已经将数据上传到矩阵变量中。每一个样例是一个20x20的灰度图像。每一个像素代表了灰度强度的浮点数。所以这20x20的网格像素被展开为400维的向量。那么矩阵X是所有训练样例的集合：

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

### 1.2数据可视化

题目已经给出数据可视化的代码，运行displayData函数后会显示出20x20像素的灰度图像。

![My-MachineLearning/3dataset.png at main · Lihao-me/My-MachineLearning (github.com)](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/3dataset.png)

### 1.3逻辑回归向量化

我需要用很多一对多的逻辑回归模型构建多分类的分类器。既然有10个类别，，那么我需要训练10个单独的逻辑回归分类器。为了保证训练的高效性，需要确保我们的代码是向量化的。所以在这部分，我需要用不包括for循环的向量化的代码来完成逻辑回归。

#### 1.3.1代价函数向量化

非正则化的代价函数：

$$
J(\theta ) = \frac{1}{m}\sum\limits_{i = 1}^m {[ - {y^{(i)}}\log ({h_\theta }({x^{(i)}})) - (1 - {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}}))]} 
$$

其中，假设函数：

$$
h_{\theta}(x^{i})=g(\theta^{T}x^{(i)})
$$

其中，sigmoid函数：

$$
g(z)=\frac{1}{1+e^{-z}}
$$

对参数定义如下：

$$
\theta=\begin{bmatrix}
 {\theta_{0} }
\\
 { \theta_{1}  }
 \\
\vdots \\
{ \theta_{n} }
\\
\end{bmatrix}
$$

那么二者相乘的结果为：

$$
X\theta=\begin{bmatrix}
 { - {{({x^{(1)}})}^T}\theta - }
\\
 { - {{({x^{(2)}})}^T}\theta - }
 \\
\vdots \\
{ - {{({x^{(m)}})}^T}\theta - }
\\
\end{bmatrix}
=\begin{bmatrix}
 { - \theta^{T}{{({x^{(1)}})}^T} - }
\\
 { - \theta^{T}{{({x^{(2)}})}^T} - }
 \\
\vdots \\
{ - \theta^{T}{{({x^{(m)}})}^T} - }
\\
\end{bmatrix}
$$

我所需要做的是，在文件[lrCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/03_myEx3/ex3/lrCostFunction.m)中补全非正则化的代价函数。

我的答案：

```matlab
h = sigmoid(theta'*X);
J = ((-y'*log(h))-(1-y')*log(1-h))/m;
```

事实上这种向量化的计算方法我在前两次作业中已经使用过了，直接搬过来就行。

#### 1.3.2梯度向量化

非正则化的梯度方程：

$$
\frac{\partial J}{\partial \theta_{j}}=\frac{1}{m}\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)})
$$

对各参数的梯度组成的向量为：

$$
\begin{align*}
\begin{bmatrix}
 {\frac{\partial J}{\partial \theta_{0}}}
\\
 { \frac{\partial J}{\partial \theta_{1}}  }
 \\
\frac{\partial J}{\partial \theta_{2}}\\
\vdots \\
{ \frac{\partial J}{\partial \theta_{n}}}
\\
\end{bmatrix}
&=
\frac{1}{m}
\begin{bmatrix}
 {\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})x_{0}^{(i)}}
\\
 {\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})x_{1}^{(i)}  }
 \\
{\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})x_{2}^{(i)}}\\
\vdots \\
{\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})x_{n}^{(i)}}
\\
\end{bmatrix}\\
&=
\frac{1}{m}\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})x^{(i)})\\
&=
\frac{1}{m}X^{T}(h_{\theta}(x)-y)
\end{align*}
$$

其中，

$$
h_{\theta}(x)-y=
\begin{bmatrix}
 {h_{\theta}(x^{(1)})-y^{(1)}}
\\
 { h_{\theta}(x^{(2)})-y^{(2)}  }
 \\

\vdots \\
{ h_{\theta}(x^{(m)})-y^{(m)}}
\\
\end{bmatrix}
$$

继续在[lrCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/03_myEx3/ex3/lrCostFunction.m)文件中补全梯度的计算。

我的答案：

```matlab
grad = ((h-y)'*X)'./m;
```

输出结果：

```
Gradients:
 0.146561 
 0.051442 
 0.124722 
 0.198003 
Expected gradients:
 0.146561
 -0.548558
 0.724722
 1.398003
```

可以看到输出结果不符合期望值，这是正常的，因为题目中的期望值是进行了正则化的。

#### 1.3.3正则化的逻辑回归向量化

对代价函数和梯度加上正则化项。

代价函数：

$$
J(\theta ) = \frac{1}{m}\sum\limits_{i = 1}^m {[ - {y^{(i)}}\log ({h_\theta }({x^{(i)}})) - (1 - {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}}))]} + \frac{\lambda }{{2m}}\sum\limits_{j = 1}^n {\theta _j^2}
$$

梯度：

$$
\frac{\partial J(\theta)}{\partial \theta_{0}}=\left\{\begin{matrix}
 \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)},&j=0  \\
 (\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)})+\frac{\lambda}{m}\theta_{j},&j\geq 1  \\
\end{matrix}\right.
$$

我们继续把正则化项加到[lrCostFunction.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/03_myEx3/ex3/lrCostFunction.m)文件中。

我的答案：

```matlab
h = sigmoid(X*theta);%经过假设函数计算后为mx1的向量
J = (-y'*log(h)-(1-y')*log(1-h))/m + sum(theta.^2)*lambda/(2*m);%损耗函数值
theta(1)=0;
grad = ((h-y)'*X)'./m + theta*lambda/m;%对参数求导值
```

输出结果：

```
Testing lrCostFunction() with regularization
Cost: 3.734819
Expected cost: 2.534819
Gradients:
 0.146561 
 -0.548558 
 0.724722 
 1.398003 
Expected gradients:
 0.146561
 -0.548558
 0.724722
 1.398003
```

可以看到计算出的代价值和期望不一样。经检查，是由于加入正则化项的代码没有写对，因为正则化项不包含第一个参数，所以应该减去。那么从这里可以看出来，我上次的作业是有点小问题的。但是上次的结果没错可能是由于第一个参数的值的大小对结果影响不大。

---

我也特意在此更正我在上一次作业[costFunctionReg.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/costFunctionReg.m)文件中所犯错误：

更正后的答案：

```matlab
h = sigmoid(X*theta);%经过假设函数计算后为mx1的向量
J = (-y'*log(h)-(1-y')*log(1-h))/m + (sum(theta.^2)-theta(1)^2)*lambda/(2*m);%损耗函数值

theta(1,1) = 0;
grad = ((h-y)'*X)'./m + theta*lambda/m;%对参数求导值
```

更正后的输出：

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

Cost at test theta (with lambda = 10): 3.164509
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

显然，更正后的输出代价值和期望更为接近。

[costFunctionReg.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/02_myEx2/ex2/costFunctionReg.m)于2022.11.14重新上传。

---

更正后的答案：

```matlab
%加入正则化项
h = sigmoid(X*theta);%经过假设函数计算后为mx1的向量
J = (-y'*log(h)-(1-y')*log(1-h))/m + (sum(theta.^2)-theta(1)^2)*lambda/(2*m);%损耗函数值
theta(1)=0;
grad = ((h-y)'*X)'./m + theta*lambda/m;%对参数求导值
```

输出结果：

```
Testing lrCostFunction() with regularization
Cost: 2.534819
Expected cost: 2.534819
Gradients:
 0.146561 
 -0.548558 
 0.724722 
 1.398003 
Expected gradients:
 0.146561
 -0.548558
 0.724722
 1.398003
Program paused. Press enter to continue.
```

此时的计算结果和题目的期望是完全吻合的。

### 1.4一对多分类

下面需要对正则化的分类器进行多次训练实现一对多训练的目的。我们要解决的是分10类的问题，但是我们的代码不应该局限在10。

现在完成oneVsAll.m文件中的代码训练分类器。特别的，代码返回的应该是所有分类器的参数，它的大小是一个矩阵：

$$
\Theta\in \mathbb{R}^{K\times (N+1)}
$$

矩阵的每一行应当是每一类的逻辑回归的参数。

此外，这里会使用fmincg函数，它比fminunc函数更高效、处理更多参数。

我的答案：

```matlab
    % Set Initial theta
    initial_theta = zeros(n + 1, 1);

    % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);

    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost
for c = 1:num_labels
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                initial_theta, options);
     all_theta(c,:)=theta';
end
```

经过多次迭代计算后，迭代过程为：

[IterationCost.txt](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/03_myEx3/IterationCost.txt)

#### 1.4.1一对多预测

训练完一对多分类器后，我们可以用计算出来的参数来预测给出的图片的数字。对于任意的输入，我们计算的都应该是它属于该逻辑回归的可能性，而我们的预测函数则选取最大可能性的那一类作为返回值。

我需要完成[predictOneVsAll.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/03_myEx3/ex3/predictOneVsAll.m)文件中的代码进行预测。

只需要将测试的数据带入到计算出的参数构成的10个方程，然后判断哪个方程算出的概率值最大，选择最大概率对应的数字作为预测值。

我的答案：

```matlab
A = X*all_theta';
[~,p] = max(A,[],2);
```

输出结果：

```
Training Set Accuracy: 94.920000
```

可知预测的准确率为94.92%，达到了题目的预期，是正确的。

## 2.神经网络

在刚刚的练习里，我们用多分类逻辑回归完成了手写数字识别。然而，由于逻辑回归也仅仅是一种线性的分类器，它没法完成更加复杂的假设函数的情形。

在这部分，我们要使用一个神经网络来对相同的数据集实现手写数字识别。我的目标是完成算法的前向传播计算参数值。而反向传播会被下一次的练习用到。

### 2.1模型表征

模型结构和课件中展示的一样，有三层：一层输入层、一层隐藏层和一层输出层。每一个数据由20x20像素的灰度图像构成，那么输入层就需要包含400个单元(除了额外的激活单元)。我也被提供了两个神经网络参数：

$$
(\Theta^{(1)},\Theta^{(2)})
$$

这两个参数分别对应的是包含25个单元的隐藏层和包含10个输出单元的输出层。

其中，数据集的上传题目代码已经给出。

数据集的可视化预览如下：

![3dataset1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/3dataset1.png)

### 2.2前向传播和预测

现在我需要完成神经网络的前向传播，并在[predict.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/03_myEx3/ex3/predict.m)文件中完成代码并返回神经网络的预测值。

我的答案：

```matlab
% Add ones to the X data matrix
X = [ones(m, 1) X];
hidden_units = sigmoid(X*Theta1');%计算出隐藏层的单元项
hidden_units = [ones(m,1) hidden_units];%添加+1的激活因子
output_units = sigmoid(hidden_units*Theta2');%计算输出层的结果
[~,p] = max(output_units,[],2);
```

输出结果：

```
Training Set Accuracy: 97.520000
Program paused. Press enter to continue.

Displaying Example Image

Neural Network Prediction: 3 (digit 3)
Paused - press enter to continue, q to exit:

Displaying Example Image

Neural Network Prediction: 6 (digit 6)
Paused - press enter to continue, q to exit:

Displaying Example Image

Neural Network Prediction: 8 (digit 8)
Paused - press enter to continue, q to exit:q
```

![3eg1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/3eg1.png) ![3rg2](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/3eg2.png) ![3eg3](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/3eg3.png)

可以看出计算出的准确率很高，符合期望值。同时，通过对样例的随机抽取预测，可以看到预测的结果都是正确的。

---

值得说明的是，我最开始的代码计算的准确率只有60%多，经过检查发现是假设函数没写对，少添加了sigmoid函数，更正后就正确的。这也使我意识到在上一题的一对多分类的逻辑回归预测中，我的假设函数也没写对。而准确率计算正确，我认为可能的原因是计算的输出本身就是归一化的，所以是否经过sigmoid函数对结果没有较大影响。故而也特此更正[predictOneVsAll.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/03_myEx3/ex3/predictOneVsAll.m)文件中的答案：

```matlab
A = sigmoid(X*all_theta');
[~,p] = max(A,[],2);
```

输出结果

```
Training Set Accuracy: 94.920000
```

---

本次作业完成，并纠正了前一次作业的错误。
