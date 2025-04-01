function [cent,D,num,assign]=k_means_lsf(z,L,num_iter,cent)
%INPUT VARIABLES
%z is an array of training frames
%L is a size of codebook
%num_iter is a number of iterations

%OUTPUT VARIABLES
%cent is an array of centroids
%D(n) is an average distortion produced by obtained codebook 
%							on the n-th iteration
%num(k) is a number of traning vectors in k-th cell

[N,M]=size(z);%M is a length of training sequence
%N is a length of one training frame
D=zeros(1,num_iter);

% %Initial conditions

if nargin==3
    cent=zeros(N,L);
    for k=1:L
        cent(:,k)=z(:,k*floor(M/L));
    end
end
num=ones(1,L);

assign=zeros(1,M); 
for nn=1:num_iter
    %disp([num2str(nn) ' iteration'])
    sm=zeros(N,L);
    num=zeros(1,L);
    
    %sorting of vectors
    for j=1:M
        [k,d]=neighbour_2(z(:,j),cent);
        D(nn)=D(nn)+d/M;
        sm(:,k)=sm(:,k)+z(:,j);
        num(k)=num(k)+1;
        assign(j)=k;
    end
    
    %correction of centroids
    for k=1:L 
        if num(k)~=0
            cent(:,k)=sm(:,k)/num(k);
        end
    end
end