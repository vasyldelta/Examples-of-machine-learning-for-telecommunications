function [kp,ki] = loop_filter_coef(bT,N_samp_inp,Kd)

%% -----------расчет коэф. петлевого фильтра -----------
ksi= 1/sqrt(2); % демпинг-фактор
Omega_n=bT/(ksi+1/(4*ksi))/N_samp_inp;
kp = -4*ksi*Omega_n/(1+2*ksi*Omega_n+Omega_n^2)/Kd;
ki = -4*Omega_n^2/(1+2*ksi*Omega_n+Omega_n^2)/Kd;