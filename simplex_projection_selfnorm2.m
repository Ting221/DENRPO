function w = simplex_projection_selfnorm2(v, b)
%{
This function is the simplex projection function exploited by Peak Price Tracking (PPT)[1]
and Adaptive Input and Composite Trend Representation (AICTR)[2]. It
originates from [4][5]. 

For any usage of this function, the following papers should be cited as
reference:

[1] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. “A peak price tracking 
based learning system for portfolio selection”, 
IEEE Transactions on Neural Networks and Learning Systems, 2017. Accepted.
[2] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang.  “Radial basis functions 
with adaptive input and composite trend representation for portfolio selection”, 
IEEE Transactions on Neural Networks and Learning Systems, 2018. Accepted.
[3] Pei-Yi Yang, Zhao-Rong Lai*, Xiaotian Wu, Liangda Fang. “Trend Representation 
Based Log-density Regularization System for Portfolio Optimization”,  
Pattern Recognition, vol. 76, pp. 14-24, Apr. 2018.

At the same time, it is encouraged to cite the following papers that
propose this method and the original code:

[4] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, “Efficient
projections onto the \ell_1-ball for learning in high dimensions,” in
Proceedings of the International Conference on Machine Learning (ICML 2008), 2008.
[5] B. Li, D. Sahoo, and S. C. H. Hoi. Olps: a toolbox for on-line portfolio selection.
Journal of Machine Learning Research, 17, 2016.


Inputs:
v                  -a d-dimensional vector，d维向量
b                  -the "size" of the simplex, default=1，单纯形的"大小"，默认1

Outputs:
w                  -the output vector on the simplex，输出在单纯形的向量

%}

while(max(abs(v))>1e6)
v=v/10;
end

u = sort(v,'descend');     %对v降序排序

sv = cumsum(u);            %cumsum(u) 返回一个向量，该向量中第m行的元素是u中第1行到第m行的所有元素累加和；
rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');     %找最后一个不为零元素的索引值
theta = (sv(rho) - b) / rho;
w = max(v - theta, 0);
end