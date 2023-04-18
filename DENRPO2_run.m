function S = DENRPO2_run(data,lambda,gamma,rho,eta,tao,win_size)
%% Statement
%{ 
The source code is provided "as is" without warranty of any kind, and its author disclaims any 
and all warranties, including but not limited to any implied warranties of merchantability 
and fitness for a particular purpose, and any warranty or non infringement. The user assumes all responsibilities 
and obligations for the use of this source code, and the author is not responsible for any kind of damage 
caused by the use of this source code. Without limiting the generality of the above, the author 
does not guarantee that the source code will be error-free, will operate without interruption, 
or will meet the needs of the user.
  
This function is the main code for portfolio optimization in a non-zero transaction cost environment 
based on the linearized augmented Lagrangian method. Based on some empirical financial principles 
and optimization strategies, it uses elastic-net regularization terms to consider the correlation 
between transaction costs and variables, thereby maximizing the cumulative wealth in the entire investment 
while reducing transaction costs.

For the usage of this function, you can refer to the following papers:
[1]B. Li, J. Wang, D. Huang, and S. C. H. Hoi, "Transaction cost optimization 
for online portfolio selection," Quantitative Finance, pp.1¨C14, 2017.
[2]Zhao-Rong Lai, Pei-Yi Yang, Liangda Fang and Xiaotian Wu. "Short-term Sparse 
Portfolio Optimization based on Alternating Direction Method of Multipliers", 
Journal of Machine Learning Research, 2018.

%}
%% Parameters Description
%{ 
Inputs:
data                  -data with price relative sequences
gamma                 -transaction cost
lambda,eta            -elastic net regularization parameters for transaction cost
                       set lambda=10*gamma,eta=0.00025
tao                   -elastic net regularization parameter for portfolio variable
                       set tao=0.00005
rho                   -parameter controlling the convergence of the algorithm
                       set rho=0.618
win_size              -window size, set win_size=4 in our experiments

output:
S                     -cumulative wealth
%}

%% Variables Initialization
[T,m]=size(data);
b0_hat=zeros(m,1);      %initialize the portfolio before transaction
b1=ones(m,1)/m;         %initialize the portfolio in t=1
S=zeros(T,1);           %record the cumulative wealth in each iteration
s0=1;                   %initialize the wealth before transaction
w0=w(b0_hat,b1,gamma);  %net wealth proportion, which from the formula 1=w_t-1+gamma*||b_t-1_hat-b_t*w_t-1||_1(refer paper[1])
b_t_hat=zeros(m,T);     %Record the proportion of assets after the end of the day's trading

%% main
%compute the cumulative wealth in t=1        
x1=data(1,:)';

run_ret =s0*w0; 
s1=run_ret*(b1'*x1);
S(1)=s1;   

b1_hat=(b1.*x1)/(b1'*x1);
b_t_hat(:,1)=b1_hat;  

%compute the cumulative wealth in t=2:T 
for t=2:T
    b_tcut1_hat=b_t_hat(:,t-1);
    s_tcut1=S(t-1);
    b_t=DENRPO2_fun(data,t-1,b_tcut1_hat,rho,lambda,eta,tao,win_size); 
    b_t=b_t/norm(b_t,1);
    w_tcut1=w(b_tcut1_hat,b_t,gamma);
    run_ret =s_tcut1*w_tcut1;
    x_t=data(t,:)';
    s_t=run_ret*(b_t'*x_t);
    S(t)=s_t;
    b_t_hat(:,t)=(b_t.*x_t)/(b_t'*x_t);
end
% if S(end)<10000
%     fprintf('\t %.2f \n',S(end));       
% else
%     fprintf('\t %.2e \n',S(end));         
% end
S=S(end);