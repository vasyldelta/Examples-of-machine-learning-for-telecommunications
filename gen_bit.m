function [y,symb]=gen_bit(bits,a1,phi1,w1,tau1,alpha1,type,mod_type,m)
if nargin==8
    m=2;
end



if mod_type==1
    symb=1-2*bits;
elseif mod_type==2
    symb=[1-2*bits(2:2:end)+1i*(1-2*bits(1:2:end))]/sqrt(2);
elseif mod_type==21%oqpsk
    symb1=1-2*bits(1:2:end);
    symb2=1-2*bits(2:2:end);

elseif mod_type==3
    symb=mapping_8PSK(bits);
elseif mod_type==31
    symb=mapping_8QAM(bits);
elseif mod_type==4
    symb=mapping_16QAM(bits);
end

%m=2;
M=33;
b = rcosdesign(alpha1, M, m,type);
b=b/max(b);
if mod_type~=21
    yy = upfirdn(symb, b, m);
    yy=[yy(M+1:m*length(symb)+M)];
else%mod_type==21
    yy1 = upfirdn(symb1, b, m);
    yy1=[yy1(M+1:m*length(symb1)+M)];

    yy2 = upfirdn(symb2, b, m);
    yy2=[yy2(M+1:m*length(symb2)+M)];

    yy2=[zeros(1,m/2) yy2];
    yy=yy1+i*yy2(1:end-m/2);
end
if tau1==0
    sincVector=[0 0 0 0 0 1 0 0 0 0 0];
else
    sincVector  = sinc_([5:-1:-5]-tau1);
end
yy = conv(yy,sincVector);
yy = yy(6:end-5);
y=yy;

y=a1*y.*exp(1i*w1*[0:length(y)-1]+1i*phi1);