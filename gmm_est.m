function [alpha,mu,sigma]=gmm_est(x,M,num_iter)
%M is a number of gaussians
[d,N]=size(x);

if 0
    alpha=zeros(1,M);
    mu=zeros(d,M);
    for j=1:M
        sigma(:,:,j)=eye(d,d);
    end
end

%alpha=ones(1,M)/M;
[mu,D,num,sigma]=k_means(x,M,num_iter);
%mu_init=km_init_uniform(x,M);
%[mu,D,num,sigma]=k_means(x,M,num_iter,mu_init);
alpha=num/sum(num);


h=zeros(N,M);
mu_prev=mu;
for nn=1:num_iter
    %disp([num2str(nn) ' iteration'])

    %calculation of h

    t=zeros(N,M);
    for i=1:N
        for l=1:M
            t(i,l)=det(sigma(:,:,l))^(-0.5)*exp(-0.5*(x(:,i)-mu(:,l))'*inv(sigma(:,:,l)+1e-6*eye(d))*(x(:,i)-mu(:,l)));
        end
    end

    for i=1:N
        for j=1:M
            s=0;
            for l=1:M
                s=s+t(i,l);
            end
            h(i,j)=t(i,j)/s;
        end
    end

    %calculation of alpha
    for j=1:M
        alpha(j)=mean(h(:,j));
    end

    %calculation of mu
    for j=1:M
        s=0;
        for i=1:N
            s=s+h(i,j)*x(:,i);
        end
        mu(:,j)=s/sum(h(:,j));
    end

    %calculation of sigma
    for j=1:M
        s=0;
        for i=1:N
            %s=s+h(i,j)*(x(:,i)-mu(:,j))*(x(:,i)-mu(:,j))';
            s=s+h(i,j)*(x(:,i)-mu_prev(:,j))*(x(:,i)-mu_prev(:,j))';
            
        end
        sigma(:,:,j)=s/sum(h(:,j));
    end
mu_prev=mu;
end