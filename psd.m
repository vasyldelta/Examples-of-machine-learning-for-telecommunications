function [h,f]=psd(x,Ndfft,Fs)
% [h,f]  = psd(x,Ndfft,Fs);
if Ndfft < length(x)
    len_window = Ndfft;
else
    len_window = 2^fix(log2(length(x)));
end
[h,f] = pwelch(x, hanning(len_window), [], Ndfft, Fs);
h      = h/max(h);
if nargout==0
    plot(f,10*log10(abs(h)));grid;shg;
    title('power spectral density'); xlabel('Hz'); ylabel('dB');
end