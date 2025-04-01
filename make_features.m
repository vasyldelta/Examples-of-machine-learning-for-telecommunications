function [X,Y]=make_features(bits,y,nb,m,mod_type)%,mod_type)
%by now - bpsk
N=length(bits);
num_blocks=N/nb;
Y=[];
X=[];
for k=1:num_blocks
    Y=[Y;bi2de(bits((k-1)*nb+1:k*nb),'left-msb')];
    X=[X;[real(y((k-1)*nb/mod_type*m+1:(k-1)*nb/mod_type*m+nb/mod_type*m)) imag(y((k-1)*nb/mod_type*m+1:(k-1)*nb/mod_type*m+nb/mod_type*m))]];
end