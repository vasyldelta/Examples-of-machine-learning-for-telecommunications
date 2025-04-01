function [out_burst_soft,out_burst_complex,out_demod_bits,SNR_UW,err_uw,SNR0,corr_ok,f0_est]=...
    demod_my(y,m,mod_type,uw_bits,UW,S0)

UW_struct = struct(...
    'UW_bits',1-uw_bits,...
    'UW_symb',UW.',... % symbols in 1 sps
    'UW_mod',S0.',... % symbols in 2 sps
    'UW_mod_cnj_norm',conj(S0).',...% conj of symbols in 2 sps div at norm
    'UW_soft_len',length(uw_bits));

flag_LR_on=0;
flag_mf_on=0;
h=1;
Fd=1;N_samp_inp=m;
[out_burst_soft,out_burst_complex,out_demod_bits,SNR_UW,err_uw,SNR0,corr_ok,f0_est]=...
    demodulator_mpsk_new_ver0_mex([complex(zeros(2,1));y.'],length(y),mod_type,mod_type,UW_struct,Fd,N_samp_inp,flag_mf_on,h,flag_LR_on);

if mod_type==1
    out_demod_bits=1-out_demod_bits';
elseif mod_type==2
    out_demod_bits=demapping(out_burst_complex,2,1);
end


out_demod_bits(1:length(uw_bits))=uw_bits;

