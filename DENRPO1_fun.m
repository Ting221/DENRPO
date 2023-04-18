function [b_tplus1]=DENRPO1_fun(data,tday,b_t_hat,rho,lambda,eta,tao)
%% algorithm parameter and iteration setting
max_iter = 1e8;                          %the maximum number of iterations
ABSTOL = 1e-8;                           %iteration tolerance

m=size(data,2);
I=ones(m,1);
z=zeros(m,1);                            %project onto R_m^+

alpha=0.999/(rho*m);                     %parameter controlling the convergence of the algorithm
cathy=10;                                %initialize the Lagrange Multiplier
b=ones(m,1)/m;                           %initialize portfolio in the subproblem in each iteration
b_old=ones(m,1)/m;                       %initialize old portfolio in the subproblem in each iteration
C=tao+eta+1/alpha;

%% Price prediction
x_t=data(tday,:)';
f1=1./x_t;

%% main
for iter=1:max_iter
    b_temp=(eta/C)*b_t_hat-b_t_hat+((1/alpha)/C)*b-(rho/C)*I*(I'*b-1)+(1/C)*f1-(1/C)*I*cathy;
    b=b_t_hat+wthresh(b_temp,'s',lambda/C);
    b(b<0)=z(b<0);                                        %project onto R_m^+
    cathy=cathy+rho*(I'*b-1);                             %update the Lagrange Multiplier
    prim_res=norm(b-b_old,2)/norm(b,2);                   %compute the iteration tolerance      
    b_old=b;

    if (prim_res)<ABSTOL        
         break;
    end  
    
end

b_tplus1=b;  

end
