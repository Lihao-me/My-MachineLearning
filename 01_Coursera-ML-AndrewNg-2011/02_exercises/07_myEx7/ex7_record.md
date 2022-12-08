# Exercise 7记录与总结

李浩 2022.12.7

---

## 1.K均值聚类

题目：在第一部分，我需要完成k均值算法并且用它应用于图像压缩。我将首先用一个二维的数据集对K均值算法进行直观感受。然后通过减少图像中的颜色数量来对图像进行压缩。该部分的主代码在[ex7.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/ex7.m)文件中。

### 1.1实现K均值

K均值算法是一种将相似的数据样本自动聚类的算法。具体地，我会被给到m个数据样本，然后通过算法进行聚类。算法的具体过程就是首先给出聚类中心，然后通过不断地迭代更新聚类中心，最后根据聚类中心确定所聚的类。

K均值算法的代码实现如下：

```matlab
% Initialize centroids
centroids = kMeansInitCentroids(X, K);
for iter = 1:iterations
% Cluster assignment step: Assign each data point to the
% closest centroid. idx(i) corresponds to cˆ(i), the index
% of the centroid assigned to example i
idx = findClosestCentroids(X, centroids);
% Move centroid step: Compute means based on centroid
% assignments
centroids = computeMeans(X, idx, K);
end
```

代码中的循环主要包括两个步骤：(i)将每个样本分配给距离它最近的聚类中心。(ii)用已分配的点重新计算均值得到新的聚类中心。对于不同的初始化聚类中心，算法的计算时间也不同。迭代过程中，也是通过最小化代价函数来得到最佳分配和选择。

#### 1.1.1找到最近的聚类中心

在“聚类中心分配”这一步中，算法将每一个样本点分配给离它最近的聚类中心。特别地，对于第i个样本，我们定义：

$$
c^{(i)}:=j \quad \text { that minimizes } \quad\left\|x^{(i)}-\mu_{j}\right\|^{2}
$$

其中$c^{(i)}$是离样本点最近的聚类中心的索引，而$\mu_{j}$是第j个聚类中心的位置。

我的任务是在[findClosestCentroids.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/findClosestCentroids.m)文件中完成代码找到最近的聚类中心。

我的答案：

```matlab
m = size(X,1);
for i = 1:m
    dis = [];%存放样本和中心点的距离
    for j = 1:K
        dis = [dis sum((X(i,:) - centroids(j,:)).^2)];
    end
    [value,c] = min(dis);%找到最小距离和相应的索引
    idx(i) = c;
end
```

输出结果：

```
Closest centroids for the first 3 examples: 
 1 3 2
(the closest centroids should be 1, 3, 2 respectively)
```

和题目的预期结果一致，可以判断代码大致正确。

#### 1.1.2计算均值中心点

将样本点分配给聚类中心点后，第二步就是计算已有聚类的均值中心点。特别地，对于第k个中心点我们定义：

$$
\mu_{k}:=\frac{1}{\left|C_{k}\right|} \sum_{i \in C_{k}} x^{(i)}
$$

其中，$C_{k}$是被分配给第k个中心点的样本点集合。具体地，假设有两个样本点$x^{(3)}$和$x^{(5)}$被分配给第2个中心点，那么我应该更新：

$$
\mu_{2}=\frac{1}{2}\left(x^{(3)}+x^{(5)}\right)
$$

现在我需要在[computeCentroids.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/computeCentroids.m)文件中完成计算。

我的答案：

```matlab
for k = 1:K
    count = 0;%记录相同类别样本个数
    sum_value = zeros(1,n);
    for i = 1:m
        if idx(i) == k
            sum_value = X(i,:)+sum_value;
            count = count+1;
        end
    end
    centroids(k,:) = sum_value./count;
end
```

输出结果：

```
Centroids computed after initial finding of closest centroids: 
 2.428301 3.157924 
 5.813503 2.633656 
 7.119387 3.616684 

(the centroids should be
   [ 2.428301 3.157924 ]
   [ 5.813503 2.633656 ]
   [ 7.119387 3.616684 ]
```

计算结果与预期的中心点结果一致，从而判断代码大致正确。

### 1.2样本数据集

在完成了算法的两个关键步骤后，下一步就是运行算法绘制2D数据集来帮助我们理解K均值算法是怎么工作的。对每一步的迭代过程进行可视化，最后可以得到最终绘制的聚类样本点如下：

![7Kmeans10](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7Kmeans10.png)

### 1.3随机初始化

在实际的应用中，一种好的初始化聚类中心点的策略就是在训练集样本中随机挑选样本点作为中心点。

在这一部分中我需要用下面的代码完成[kMeansInitCentroids.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/kMeansInitCentroids.m)文件：

```matlab
% Initialize the centroids to be random examples
% Randomly reorder the indices of examples
randidx = randperm(size(X, 1));
% Take the first K examples as centroids
centroids = X(randidx(1:K), :);
```

随机选择避免了两次选择相同样本的风险。

### 1.4用K均值算法进行图像压缩

在这一部分，我将应用K均值算法于图像压缩。对于一幅有色的且每一个颜色为24比特的图像，其每一个像素点由三个8bit的无符号整型数构成，分别代表了红、绿、蓝三种基础色的强度值。这是一种RGB的编码方法。一幅图像包含成千上万种颜色，而简单起见，这部分的练习会将颜色数量减少到16种。

下面可以通过高效的方式进行图像压缩。特别地，我只需要存储16种被选择的RGB值，而对于图像的每一个像素点，我只需要存储每一种颜色的索引。

通过使用K均值算法来选择用于压缩的16种颜色。具体地，我要将图像中的每一个像素点视作一个样本数据，然后用K均值算法将三维的RGB空间样本进行16个聚类。一旦算法运行完成后就可以用这16个颜色来代表各像素点进行原图像的表示。

#### 1.4.1像素点表征

在Matlab中，读取图像的代码如下：

```matlab
% Load 128x128 color image (bird small.png)
A = imread('bird small.png');
% You will need to have installed the image package to used
% imread. If you do not have the image package installed, you
% should instead change the following line to
%
% load('bird small.mat'); % Loads the image into the variable A
```

![bird_small](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/bird_small.png)

创建了一个三维向量A来存储图像。A的前两个索引表示像素点位置，而最后一个索引表示红色、绿色还是蓝色。例如，A(50,33,3)表示图像第50行第33列处的像素点的蓝色强度。

将数据集构建好后，通过调用finClosedtCentroids函数来找到最近的中心点。原来的图像为128x128像素，每一个像素点的大小是24bit，那么原始图像的大小为128x128x24=393,216比特。而用一个含16个颜色的字典对图像进行表征后，每一个像素点仅仅需要4bit来索引颜色位置。也就是说总共大小为16x24+128x128x4=65,920比特。通过聚类表征减少了图像的颜色，使图像大小大大减少。

在这一部分题目已经给出了代码，压缩前后的图像如图所示：

![7compressed](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7compressed.png)

### 1.5选做练习：使用自己的图像

将题目中的图像替换为自己的，不过如果图像很大的话耗费时间很长。通过自己的图像也可以帮助我们理解K均值算法。同时也可以尝试不同的K值来判断对压缩的影响。

![7myImage](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7myImage.png)

## 2.主成分分析

题目：在这一部分，我需要使用主成分分析(PCA)实现降维。首先通过二维的数据集样例直观感受PCA如何工作，而后用一个5000的人脸数据集进行实操。

我需要对一封邮件x判断，它时垃圾邮件(y=1)还是不是垃圾邮件(y=0)。特别地，我需要将每一封邮件转换成一个特征向量。下面会详细解释如何对其进行转化。该部分的主代码在[ex7_pca.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/ex7_pca.m)文件中。

### 2.1数据集样例

为了更好地理解PCA是怎么工作的，首先实现对二维数据降一维的处理。训练集绘制如下。

![7dataset1](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7dataset1.png)

### 2.2实现PCA

PCA包含两个计算步骤：首先需要计算数据的协方差矩阵。然后需要用Matlab中的SVD函数计算出特征向量。

在使用PCA前，对数据进行标准化处理是很重要的。对数据减去平均值，然后将它们约束到一个相同的范围内。在题目中已经给出了数据标准化的函数[featureNormalize.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/featureNormalize.m)。

对数据标准化处理后，我需要在[pca.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/pca.m)文件中计算数据集的主成分。首先，计算数据的协方差矩阵：

$$
\Sigma=\frac{1}{m} X^{T} X
$$

其中，m是样本个数。

在计算出协方差矩阵后，可以运行SVD函数计算主成分。

我的答案：

```matlab
Sigma = X'*X./m;%计算协方差矩阵
[U, S, V] = svd(Sigma);%计算主成分
```

输出结果：

```
Top eigenvector: 
 U(:,1) = -0.707107 -0.707107 

(you should expect to see -0.707107 -0.707107)
```

计算的特征向量与预期一致。绘制主成分如下图：

![7eigenvetors](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7eigenvectors.png)

### 2.3用PCA降维

在计算完主成分后，可以使用他们实现对数据集特征空间的降维。而在这一部分，主要是对2维的数据降维为1维。

降维后的数据投入学习机中的速度会更快一些。

#### 2.3.1用主成分投影数据

我需要在[projectData.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/projectData.m)文件中完成相关代码。具体来说，我已经有了数据集X和主成分特征向量U，希望降低到的维度为K。我需要通过主成分U将X降低到K维。当完成代码后，运行的示例输出应当为1.481。

我的答案：

```matlab
m = size(X,1);
for i = 1:m
    x = X(i,:)';
    projection_k = x' * U(:, K);
    Z(i,:) = projection_k; 
end
```

输出结果：

```
Projection of the first example: 1.481274

(this value should be about 1.481274)
```

数据计算是正确的。

#### 2.3.2数据的粗略重建

在将数据进行降维处理后，我可以将数据粗略地向原始数据空间还原。我需要在[recoverData.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/recoverData.m)文件中完成数据的粗略还原。完成后计算的值应该为[-1.047 -1.047]。

我的答案：

```matlab
m = size(Z,1);
j = size(U,1);
for i = 1:m
    v = Z(i, :)';
    recovered_j = v' * U(j, 1:K)';
    X_rec(i,:) = recovered_j;
end
```

输出结果：

```
Approximation of the first example: -1.047419 -1.047419

(this value should be about  -1.047419 -1.047419)
```

与题目预期值一致，可知计算正确。

#### 2.3.3投影可视化

在完成数据的投影和重建后可以分别绘制前后的数据从而将投影的过程可视化，如下图所示：

![7project](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7project.png)

上图中，蓝色圆圈的是原始数据，红色圆圈的是投影后的数据。

### 2.4人脸图像数据集

在这部分，通过主成分分析看看它是怎么对人脸图像进行降维处理的。[ex7faces.mat](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/ex7faces.mat)包含32x32的人脸灰度图像。对于一张人脸图像的的每一行由X反映。代码对人脸数据进行上传并抽取100个显示如下：

![7facesDataset](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7facesDataset.png)

#### 2.4.1对面部的主成分分析

为了将PCA应用到人脸数据集，我们首先需要将数据集进行标准化处理，依然是对数据向量的每一个特征减去均值。运行PCA后，可以得到一个主成分的数据集。每一个主成分U的行向量长度为n=1024，我们可以将其重组为一个32x32的矩阵可视化。对前36个图像的主成分的可视化如下：

![7PC](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7PC.png)

在这一步发现前面程序有问题，经过排查是[projectData.m](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/02_exercises/07_myEx7/ex7/projectData.m)文件中的投影计算错误，特此更正如下：

```matlab
m = size(X,1);
Ureduce = U(:,1:K);
for i = 1:m
    x = X(i,:)';
    projection_k = x' * Ureduce;
    Z(i,:) = projection_k; 
end
```

下面将每一个人脸图像降到100维。原始图像和粗略恢复图像如下图所示：

![7orFaces](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7orFaces.png)

可以观察到恢复后的图像丢失了一些细节。但是通过PCA处理能够在一定程度上提升学习器的学习速率。

### 2.5选做练习：PCA可视化

在这一部分对3维投影处理到2维的数据进行可视化。如下：

![7orIn3D](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7orIn3D.png)

![7PCA2D](https://github.com/Lihao-me/My-MachineLearning/blob/main/01_Coursera-ML-AndrewNg-2011/00_images/7PCA2D.png)
