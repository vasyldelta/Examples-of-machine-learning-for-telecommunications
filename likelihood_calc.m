function [res,res_arr]=likelihood_calc(x,alpha,mu,sigma)
[d,N]=size(x);
M=length(alpha);

res=0;
res_arr=zeros(1,N);
for i=1:N
    if norm(x(1,i))~=0
        p=0;
        for l=1:M
            p=p+alpha(l)*det(sigma(:,:,l))^(-0.5)*exp(-0.5*(x(:,i)-mu(:,l))'*inv(sigma(:,:,l))*(x(:,i)-mu(:,l)));
            %p=p+alpha(l)*exp(-0.5*(x(:,i)-mu(:,l))'*sigma(:,:,l)*(x(:,i)-mu(:,l)));
        end
        res=res+log(p);
        res_arr(i)=log(p);
    end
end
res=res/N;