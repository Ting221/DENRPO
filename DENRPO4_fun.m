function [b_tplus1]=DENRPO4_fun(data,tday,b_t_hat,rho,lambda,eta,tao,win_size)
%% algorithm parameter and iteration setting
max_iter = 1e8;
ABSTOL = 1e-8;
m=size(data,2);

%% Price prediction
x_wan=1;
temp=1;
for k=1:win_size-1
    if tday-k+1<=0
        k=k-1;
        break
    end
    x_tcutkplus1=data(tday-k+1,:)';
    x_wan=x_wan+temp./x_tcutkplus1;
    temp=temp./x_tcutkplus1;
end
f2=x_wan/(k+1);


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
    b=(f2-cathy+rho*d)/c1;
    b=simplex_projection_selfnorm2(b,1);
    
    %update d
    d_temp=(rho*b+cathy-rho*b_t_hat)/c2;
    d=b_t_hat+wthresh(d_temp,'s',lambda/c2);
    
    %update cathy
    cathy=cathy+rho*(b-d);
    
    prim_res=norm(b-b_old,2)/norm(b,2); %²Ð²î      
    b_old=b;
  
    if (prim_res)<ABSTOL        
         break;
    end  
    
end

b_tplus1=b;  

end
