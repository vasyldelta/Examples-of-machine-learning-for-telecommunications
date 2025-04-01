function [n2,d,a0]=neighbour_2(b,cent);
%We are looking for approximation of b by quants contained in cent
%the function returns the number of the nearest element
[n,c_b_size]=size(cent);
a=sum(abs(b*ones(1,c_b_size)-cent).^2);
[n1 n2] = min(a);
d = a(n2);
a0=cent(:,n2);

