function g=w(b_tcut1_hat,b_t,gamma)
    function y= w(w_tcut1)
        y=w_tcut1+gamma*norm(b_tcut1_hat-b_t.*w_tcut1,1)-1;
    end
    g=fzero(@w,0);
end



