function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
mu = [mean(X(:,1)) mean(X(:,2))];%计算样本中两特征量的均值
mu_rep = repmat(mu,size(X,1),1);%将行向量垂直方向复制m次，水平方向复制1次
scale = [max(X(:,1))-min(X(:,1)) max(X(:,2))-min(X(:,2))];%计算各特征量范围
scale_rep = repmat(scale,size(X,1),1);
X_norm = (X - mu_rep)./scale_rep;%特征量标准化处理
sigma = std(X);%计算特征变量的标准差










% ============================================================

end
