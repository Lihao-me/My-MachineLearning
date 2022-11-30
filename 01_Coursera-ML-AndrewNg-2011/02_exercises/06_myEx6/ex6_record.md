# Exercise 6记录与总结

李浩 2022.11.26

---

## 1.支持向量机

题目：在这一次作业的上半部分，我将使用支持向量机和2维的数据集样本构建学习模型，从而更好地学会使用和理解使用高斯核函数的支持向量机。在练习的后半部分用我们的模型搭建一个分类器。章节的主代码在[ex6.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/ex6.m)文件中。

### 1.1数据集1的样例

我们拥有一个可以被一个线性决策边界划分的二维数据集样本。主程序会把训练数据绘制出来。

数据集如下所示：

![6trainingdata1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/6trainingdata1.png)

从图中可以看出，正样例和负样例之间有一条明显的间隙，但是在(0.1,1.4)位置处有一个偏离整体的正样本。我们将从这个例子中体会这个偏离点对支持向量机边界划分的影响。

首先，我需要尝试不同的C值。在课上已经讲过，C控制的是对训练样本错误分类的惩罚力度。一个很大的C值迫使支持向量机正确对样本进行分类。C和之前逻辑回归中的λ为一个近似反比的关系。SVM算法已经被题目提供在svmTrain.m文件中。当C=1时，可以看到绘制的决策边界如下：

![6svmC1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/6svmC1.png)

值得注意的是，大部分的SVM软件都自动额外提供了x_0=1给我们，所以一般不需要自己加。

现在把C改为100再试试。

![6svmC100](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/6svmC100.png)

可以看到决策边界的拟合对数据的拟合并不自然。

### 1.2带高斯核函数的支持向量机

在这一部分，我需要通过带高斯核函数的支持向量机进行非线性的分类。

#### 1.2.1高斯核函数

为了用支持向量机找到非线性的决策边界，我们需要首先写好高斯核函数。我们可以把高斯核函数看作是两个样本点$(x^{(i)},x^{(j)})$之间的距离度量。而高斯核函数的影响因子σ也至关重要，决定了最高点向周围下降的剧烈程度。

我需要在[gaussianKernel.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/gaussianKernel.m)文件中计算两个样本点间的高斯核函数。高斯核函数定义如下：

$$
K_{\text {gaussian }}\left(x^{(i)}, x^{(j)}\right)=\exp \left(-\frac{\left\|x^{(i)}-x^{(j)}\right\|^{2}}{2 \sigma^{2}}\right)=\exp \left(-\frac{\sum_{k=1}^{n}\left(x_{k}^{(i)}-x_{k}^{(j)}\right)^{2}}{2 \sigma^{2}}\right)
$$

题目期望的测试计算值是0.324652

我的答案：

```matlab
sim = exp(-sum((x1-x2).^2)/(2*sigma^2));
```

输出结果：

```
Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2.000000 :
    0.324652
(for sigma = 2, this value should be about 0.324652)
```

可以看到计算的核函数值和预期一致，所以代码应该是正确的。

#### 1.2.2数据集2的样例

在接下来的部分，对数据2的样本进行上传和绘制。绘制的图像如下：

![6trainingdata2](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/6trainingdata2.png)

从图像可以看出，不易用线性的模型来对分类决策边界进行拟合。然而，通过使用支持向量机可以进行非线性的决策边界拟合。

继续运行题目的代码可以看到绘制的非线性的决策边界。

![6svm2](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/6svm2.png)

上图显示了用高斯核函数后通过支持向量机绘制的非线性的决策边界。显然这个决策边界能较好地将正负样本区分开来。

#### 1.2.3数据集3的样例

下面上传数据集3并绘制如下：

![6trainingdata3](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/6trainingdata3.png)

接下来我需要用交叉验证集来确定最佳的C值和σ。我需要尝试写额外的代码来确定。题目建议可以试试0.01,0.03,0.1,0.3,1,3,10,30等。然后将确定的最好的参数填到[dataset3Params.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/dataset3Params.m)文件中，绘制出最佳的决策边界。

我的答案：

```matlab
value = [0.01 0.03 0.1 0.3 1 3 10 30];%尝试的参数
num = length(value);
final_error=100;%通过与一个较大值比较从而找到最小的错误率
for i = 1:num
    for j = 1:num
        C_temp = value(i);
        sigma_temp = value(j);
        %训练模型
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        %预测模型
        predictions = svmPredict(model, Xval);
        %计算误差
        error = mean(double(predictions ~= yval));
        if(error < final_error)
            %更新最佳参数
            final_error = error;
            C = C_temp;
            sigma = sigma_temp;
        end
    end    
end
```

用最佳参数绘制的边界图像为：

![6bestParameter](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/6bestParameter.png)

可以看出与题目中给的参考示例的图像是一样的。

然后我把此时最佳的参数打印如下：

```
Optimal parameter C = 1.000000 ,sigma = 0.100000
```

可以看到计算的是C=1和σ=0.1时为最佳参数。

## 2.垃圾邮件分类

题目：现在很多电子邮箱都能够提供很好的垃圾邮件过滤。在这部分，我需要用支持向量搭建自己的垃圾邮件过滤器。

我需要对一封邮件x判断，它时垃圾邮件(y=1)还是不是垃圾邮件(y=0)。特别地，我需要将每一封邮件转换成一个特征向量。下面会详细解释如何对其进行转化。

章节的主代码在[ex6_spam.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/ex6_spam.m)文件中。对于这个练习，我只需要使用邮件的主体部分，而将头内容排除在外。

### 2.1邮件预处理

在开始一个机器学习任务前，一般将数据集转换为可视化的。一个典型的邮件包含URL码，地址(在末尾)，数字，钱数等等内容。因此，通常对邮件预处理的方法是把这些值标准化，所以URL码和所有的数字都被看作是没有区别的。例如，把所有的URL替换为特殊的字符串“httpaddr”。它所起到的效果是让邮件分类器判断是否存在网站链接而不是判断是否存在一个特定的网站链接。因为垃圾邮件经常随机化URL，所以这种办法能显著提升垃圾邮件分类器的性能。

在[processEmail.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/processEmail.m)文件中，题目已经完成了邮件预处理和标准化的步骤，包括：把所有字母转换为小写，移除HTML标签，将所有的URL替换为字符串，用文本替换掉所有邮件地址，将所有数字替换为文本，将所有的美元符号替换为文本，将所有的同源词汇替换为原词，移除掉所有的非文字和标点。

#### 2.1.1词汇表

在对邮件预处理后，可以得到一个词汇表。下一步就是选择用哪些词在分类器，把哪些词丢掉。

在这部分，题目已经选择了最频繁的词构成词汇的集合。既然在很少的邮件中出现的很少的词汇在训练集中可能导致模型的过拟合，所以将其摒除。完整的词汇表在[vocab.text](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/vocab.txt)文件中。这个词汇表中包含了在垃圾邮件中至少出现了100次的词汇，包含了1899个单词。而在实际的应用中，词汇表大概能包含10000到50000个单词。

有了词汇表，就可以对预处理的邮件进行映射，从而转换成一个包含索引序号的表。

我的任务是完成[processEmail.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/processEmail.m)文件完成这种映射。在代码中，我有一个str变量代表邮件中的单个单词。我需要将这个词和词汇表比对，找到是否在词汇表后找到它的索引。如果词汇表没有这个单词，我就可以跳过这个单词。

文本替换后的内容：

```
anyon know how much it cost to host a web portal well it depend on how mani 
visitor you re expect thi can be anywher from less than number buck a month 
to a coupl of dollarnumb you should checkout httpaddr or perhap amazon ecnumb 
if your run someth big to unsubscrib yourself from thi mail list send an 
email to emailaddr 
```

我的答案：

```matlab
list_length = length(vocabList);
for i =1:list_length
    if strcmp(str,vocabList(i))
        word_indices = [word_indices;i];
        break;
    end
end
```

对内容映射词汇表的索引为：

```
86 916 794 1077 883 370 1699 790 1822 1831 883 431 1171 794 1002 1893 1364 592 1676 238 162 89 688 945 1663 1120 1062 1699 375 1162 479 1893 1510 799 1182 1237 810 1895 1440 1547 181 1699 1758 1896 688 1676 992 961 1477 71 530 1699 531
```

可以看到是和题目的示例一致，代码应该是无误的。

### 2.2从邮件提取特征

现在需要通过特征提取把邮件转化为一个特征向量。在这部分，我要用n=#单词表中的词。特别地，用0或1作为邮件的单词是否出现在字典的标记。

那么对于一个典型的邮件，特征向量类似这样：

$$
x=\left[\begin{array}{c}
0 \\
\vdots \\
1 \\
0 \\
\vdots \\
1 \\
0 \\
\vdots \\
0
\end{array}\right] \in \mathbb{R}^{n}
$$

我需要完成[emailFeatures.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/emailFeatures.m)文件中的代码，通过word_indices变量构建特征向量。如果代码正确，可以看到特征向量的长度为1899，其中有45个非零元素。

我的答案：

```matlab
x(word_indices) = 1;
```

直接将序号作为索引即可。

输出结果：

```
Length of feature vector: 1899
Number of non-zero entries: 45
```

可以看到与期望的值一致，估计代码没有问题。

### 2.3为垃圾邮件分类器训练支持向量机

在完成邮件的特征提取后，下一步就是把预处理的数据用于训练支持向量机。[spamTrain.mat](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/spamTrain.mat)数据文件中包含了4000个垃圾与非垃圾邮件的训练样本，而[spamTest.mat](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/06_myEx6/ex6/spamTest.mat)则包含了1000个测试样本。每一个源邮件已经被预处理转换成特征向量了。

当训练完成后，我可以看到分类器的训练准确度大概为99.8%以及测试准确度大概为98.5%。

```
Training Accuracy: 99.825000

Evaluating the trained Linear SVM on a test set ...
Test Accuracy: 98.700000
```

计算是正确的。

### 2.4最具预测性的单词

为了更好地理解垃圾邮件分类器是怎么工作的，可以检查一下哪些单词是分类器判决的最具预测性的。下一步打印出了分出垃圾邮件中出现次数较多的单词。

```
Top predictors of spam: 
 our             (0.499181) 
 click           (0.460766) 
 remov           (0.416599) 
 guarante        (0.389294) 
 visit           (0.373303) 
 basenumb        (0.344319) 
 dollar          (0.325891) 
 will            (0.269737) 
 price           (0.262506) 
 most            (0.261105) 
 pleas           (0.258648) 
 nbsp            (0.253214) 
 lo              (0.250590) 
 ga              (0.241385) 
 dollarnumb      (0.241215) 
```

可以看到，如果邮件中包含guarantee,remove,dollar,price这种单词，那么它很有可能被分类器当作垃圾邮件。

### 2.5选做练习：尝试自己的邮件

可以尝试将自己的邮件作为测试样例测试刚刚训练的模型。

### 2.6选做练习：构建自己的数据集

从公开网站构建自己的词汇表。
