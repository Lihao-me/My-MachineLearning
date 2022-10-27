# Exercise 1记录与总结

李浩 2022.10.26

---

## 1.Matlab函数的简单使用

在[warmUpExercise.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/01_myEx1/ex1/warmUpExercise.m)文件中填充内容，使主文件调用函数输出一个5x5的单位矩阵。

我的答案：

```matlab
% Instructions: Return the 5x5 identity matrix 
%               In octave, we return values by defining which variables
%               represent the return values (at the top of the file)
%               and then set them accordingly. 
A = eye(5);
```

输出：

    Running warmUpExercise ... 
    5x5 Identity Matrix: 
    
    ans =
    
         1     0     0     0     0
         0     1     0     0     0
         0     0     1     0     0
         0     0     0     1     0
         0     0     0     0     1

## 2.单变量的线性回归

题目：假设我是一个连锁餐馆的老板，而不同的分店分布在各种城市，我手里掌握了各城市的人口和分店利润数据。我将利用这些数据选择去哪一个城市经营连锁店。

### 2.1数据绘制

为了将数据内容可视化处理，我们需要在[plotData.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/01_myEx1/ex1/plotData.m)文件中对数据进行绘制。作出“人口-利润“散点图。

我的答案：

```matlab
plot(x,y,'rx','MarkerSize',10);         %Plot the data
ylabel('Profit in $10,000s');           %Set the y-axis label
xlabel('Population of City in 10,000s');%Set the x-axis label
```

输出结果：

![1training-data](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/1training-data.png)

### 2.2梯度下降

找出适于模型的参数值$\theta$。

#### 2.2.1更新方程

主要任务是最小化代价函数

$$
J(\theta ) = \frac{1}{{2m}}\sum\limits_{i = 1}^m {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}}
$$

而线性的假设函数为：

$$
% MathType!MTEF!2!1!+-
% feaahqart1ev3aaatCvAUfeBSjuyZL2yd9gzLbvyNv2CaerbuLwBLn
% hiov2DGi1BTfMBaeXatLxBI9gBaerbd9wDYLwzYbItLDharqqtubsr
% 4rNCHbWexLMBbXgBd9gzLbvyNv2CaeHbl7mZLdGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaabaabauaakeaacaWGObWaaSbaaSqaaiabeI7aXbqabaGc
% caGGOaGaamiEaiaacMcacqGH9aqpcqaH4oqCdaahaaWcbeqaaiaads
% faaaGccaWG4bGaeyypa0JaeqiUde3aaSbaaSqaaiaaicdaaeqaaOGa
% ey4kaSIaeqiUde3aaSbaaSqaaiaaigdaaeqaaOGaamiEamaaBaaale
% aacaaIXaaabeaaaaa!522F!
{h_\theta }(x) = {\theta ^T}x = {\theta _0} + {\theta _1}{x_1}
$$

在梯度下降中，更新参数：

$$
{\theta _j} = {\theta _j} - \alpha \frac{1}{m}\sum\limits_{i = 1}^m {({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}}
$$

需要注意的是，需要同时更新两个参数。

#### 2.2.2执行

将X，y以及参数等初始化，设置迭代次数为1500次，学习率为0.01.

#### 2.2.3计算损失函数

在[cumputeCost.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/01_myEx1/ex1/computeCost.m)文件中将损失函数输入，返回损耗值。

我的答案：

```matlab
J = sum((X*theta - y).^2)/(2*m);%The cost function
```

输出结果：

```
Testing the cost function ...
With theta = [0 ; 0]
Cost computed = 32.072734
Expected cost value (approx) 32.07

With theta = [-1 ; 2]
Cost computed = 54.242455
Expected cost value (approx) 54.24
Program paused. Press enter to continue.
```

#### 2.2.4梯度下降

在[gradientDescent.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/01_myEx1/ex1/gradientDescent.m)文件中更新参数，不断迭代调用函数计算损耗。得到最终的参数值后，将对预测曲线进行绘制。

我的答案：

```matlab
temp1 = theta(1) - alpha*sum((X*theta-y).*X(:,1))/m;
temp2 = theta(2) - alpha*sum((X*theta-y).*X(:,2))/m;
theta = [temp1; temp2];
```

输出结果：

    Running Gradient Descent ...
    Theta found by gradient descent:
    -3.630291
    1.166362
    Expected theta values (approx)
     -3.6303
     1.1664

![1pridict](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/1pridict.png)

### 2.3调试

### 2.4可视化损耗函数

在该部分不需要自主写任何代码，题目中已给出相应代码。通过绘制关于$\theta_{0}$和$\theta_{1}$的二维图像和等高线，我们可以更好地理解代价函数和梯度下降。

![1surface](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/1surface.png)

![1contour](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/1contour.png)

## 3.多变量线性回归

题目：假设我要卖一套房子，而我又需要知道现在的市场行情。所以我需要收集最近的房市信息然后作一个预测模型。现在已有当地很多房子的面积、卧室数量及售价的数据。

### 3.1特征量标准化

我需要在featureNormalize.m中完成特征量的标准化。根据具体的标准化方法：

- 将原特征量减去该特征下所有数据的均值

- 然后除以范围，也即最大值减去最小值

此外，由于标准差是衡量变量偏离的程度的量，我也需要计算变量的标准差。

我的答案：

```matlab
mu = [mean(X(:,1)) mean(X(:,2))];%计算样本中两特征量的均值
mu_rep = repmat(mu,size(X,1),1);%将行向量垂直方向复制m次，水平方向复制1次
scale = [max(X(:,1))-min(X(:,1)) max(X(:,2))-min(X(:,2))];%计算各特征量范围
scale_rep = repmat(scale,size(X,1),1);
X_norm = (X - mu_rep)./scale_rep;%特征量标准化处理
sigma = std(X);%计算特征变量的标准差
```

输出结果：

```
Loading data ...
First 10 examples from the dataset: 
 x = [2104 3], y = 399900 
 x = [1600 3], y = 329900 
 x = [2400 3], y = 369000 
 x = [1416 2], y = 232000 
 x = [3000 4], y = 539900 
 x = [1985 4], y = 299900 
 x = [1534 3], y = 314900 
 x = [1427 3], y = 198999 
 x = [1380 3], y = 212000 
 x = [1494 3], y = 242500 
Program paused. Press enter to continue.
Normalizing Features ...
```

通过查看变量值可以看到所计算的参数值：

$\mu$:

|     | 1          | 2      |
| --- | ---------- | ------ |
| 1   | 2.0007e+03 | 3.1702 |

$\sigma$:

|     | 1        | 2      |
| --- | -------- | ------ |
| 1   | 794.7024 | 0.7610 |

### 3.2梯度下降

同单变量的梯度下降类似，我需要在[computeCostMulti.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/01_myEx1/ex1/computeCostMulti.m)文件和[gradientDescentMulti.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/01_myEx1/ex1/gradientDescentMulti.m)文件中分别计算损失值和进行梯度下降的迭代运算找到使损耗最小的参数值。与单变量线性回归的梯度下降类似，而为了保证参数的同步更新，所以在迭代时的语法会有变动，将公式各项进行向量化计算会很方便。题目中给出了一种提示：

$$
J(\theta ) = \frac{1}{{2m}}{(X\theta - \overrightarrow y )^T}(X\theta - \overrightarrow y )
$$

可以对其进行数学推导进而验证是正确的等价式，而我没有采用这种做法，而是依然用课上推导的带有导数的表达式进而向量化运算的。

我的答案：

```matlab
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
J = sum((X*theta - y).^2)/(2*m);%The cost function
```

```matlab
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
theta = theta - ((alpha/m)*(X*theta-y)'*X)';
```

值得注意的是，此时的学习率$\alpha = 0.01$,对应的参数输出为：

```
Running gradient descent ...
Theta computed from gradient descent: 
 334302.063993 
 82063.900182 
 34705.251476 
```

作出该学习率下的损失函数曲线：

![1cost-function1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/1cost-function1.png)

对于一所面积1650平方英尺、3个卧室的房子进行预测：

我的答案：

```matlab
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this

price = sum(sum(theta*[1 1650 3]));
```

输出结果：

```
Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
 $746071790.688464
```

#### 3.2.1选做练习：选择学习率

通过改变学习率$\alpha$的值，观察损耗函数的变化，按照课上讲的原则选择一个较好的学习率。

我的答案：

```matlab
% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold on
% Choose some alpha value1
alpha = 0.03;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2);
hold on

% Choose some alpha value2
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:numel(J_history), J_history, '-k', 'LineWidth', 2);
hold on

% Choose some alpha value3
alpha = 0.3;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);
hold on

% Choose some alpha value4
alpha = 1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:numel(J_history), J_history, '-y', 'LineWidth', 2);
hold on

% % % Choose some alpha value5
% % alpha = 3;
% % num_iters = 400;
% % 
% % % Init Theta and Run Gradient Descent 
% % theta = zeros(3, 1);
% % [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
% % 
% % % Plot the convergence graph
% % plot(1:numel(J_history), J_history, '-m', 'LineWidth', 2);
% % legend('\alpha = 0.01','\alpha = 0.03','\alpha = 0.1','\alpha = 0.3','\alpha = 1')
% hold off
```

通过将学习率依次设为0.01，0.03，0.1，0.3，1后，绘制的损耗函数曲线如下：

![1cost-function2](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/1cost-function2.png)

而当学习率为3时，绘制的损耗函数曲线如下：

![1cost-function3](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/1cost-function3.png)

从图像看出来，随着学习率的升高，损耗函数收敛得越快，当达到某一数值后，损耗函数不再收敛。在这次实验中，我们有理由认为学习率为1是一个使损耗函数下降得越快，且计算得也很好的学习率。在该学习率下，计算相应的参数值$\theta$:

```
Theta computed from gradient descent: 
 340412.659574 
 504610.299533 
 -34736.666557 
```

而此参数模型所计算的1650平方英尺、3个卧室的房子的售价预测：

```
Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
 $1340213527.879051
```

### 3.3正规方程

现在我们用正规方程法来解决这个问题。根据求得最小参数值的方程：

$$
\theta  = {({X^T}X)^{ - 1}}{X^T}\overrightarrow y 
$$

上式即为该方法的核心，有理论表明这种方法计算的参数值是最准确的。

我需要在[normalEqn.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/01_myEx1/ex1/noemalEqn.m)文件中计算最小的$\theta$值。

我的答案：

```matlab
theta = pinv(X'*X)*X'*y;
```

输出答案：

```
Theta computed from the normal equations: 
 89597.909544 
 139.210674 
 -8738.019113 

Predicted price of a 1650 sq-ft, 3 br house (using normal equations):
 $133972513.227792
```

通过与梯度下降法计算的结果对比，可知我们对学习率的选择是正确的。其结果是很接近正规方程法所计算的参数值和预测值的。
