function farrow_out = farrow(mu,BUF_X)

Z1= [ 1/6  -1/2  1/2  -1/6]*BUF_X.';
Z2= [   0   1/2   -1   1/2]*BUF_X.';
Z3= [ -1/6    1  -1/2 -1/3]*BUF_X.';
Z4= [   0     0    1     0]*BUF_X.';

farrow_out = mu*(mu*( mu*Z1+Z2 ) +Z3 ) + Z4;