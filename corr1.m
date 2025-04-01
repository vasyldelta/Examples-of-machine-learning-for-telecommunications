function h=corr1(x,x_ref,d)
%estimation of one-dimension corrector
%8.11.02
%x_ref is a reference signal with sampling rate 1 
%x is a current signal with sampling rate d
N=length(x_ref);
sum1=complex(0);sum2=complex(0);
for k=1:N
    sum1=sum1+x_ref(k)*x(d*k-d+1)';
    sum2=sum2+abs(x(d*k-d+1))^2;
end
h=sum1/sum2;