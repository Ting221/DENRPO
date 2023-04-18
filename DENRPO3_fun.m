function [b_tplus1]=DENRPO3_fun(data,tday,b_t_hat,rho,lambda,eta,tao)
%% algorithm parameter and iteration setting
max_iter = 1e8;
ABSTOL = 1e-8;
m=size(data,2);

%% Price prediction
%acquire history price information
x_t=data(tday,:)';
%price prediction
f1=1./x_t;

%% main
%initialize
b=ones(m,1)/m;
d=b;
cathy=zeros(m,1)/m;

b_old=ones(m,1)/m;
c1=tao+rho;
c2=eta+rho;

%iterate
for iter=1:max_iter
    %update b
    b=(f1-cathy+rho*d)/c1;
    b=simplex_projection_selfnorm2(b,1);
    
    %update d
    d_temp=(rho*b+cathy-rho*b_t_hat)/c2;
    d=b_t_hat+wthresh(d_temp,'s',lambda/c2);
    
    %update cathy
    cathy=cathy+rho*(b-d);
    
    prim_res=norm(b-b_old,2)/norm(b,2);  
    b_old=b;
  
    if (prim_res)<ABSTOL        
         break;
    end  
    
end

b_tplus1=b;  
end
