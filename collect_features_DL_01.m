clear
rand('state',1);randn('state',0);
warning off
mod_type=1;
type='normal';
m=2;
snr_arr=0:2.5:15;
uw_bits=[1 0 1 0 0 0 0 0 0 1 0 1 1 1 1 0 1 1 0 1];
[S0,symb0]=gen_bit(uw_bits,1,0,0,0,0.25,type,mod_type,m);
len_uw=length(uw_bits);
pos11=1;pos12=pos11+len_uw-1;
num_bits=120;
num_samples=1000;
nb=1;

if 1
    X_train_total=[];
    X_test_total=[];
    y_train_total=[];
    y_test_total=[];

    X_total=[];
    y_total=[];
    tic
    for ind_snr=1:7
        rand('state',0);randn('state',0);
        SNR=2.5*(ind_snr-1);
        disp(SNR)
        for n=1:num_samples%500
            %disp(n)
            rand('state',n);randn('state',n);
            a1=1;phi1=pi/6*(2*rand-1);tau1=0.45*(2*rand-1);alpha1=0.25;w1=0*2e-3*(2*rand-1);
            bits1_0=round(rand(1,num_bits));
            bits1_0(1:len_uw)=uw_bits;
            [y1,symb1]=gen_bit(bits1_0,a1,phi1,w1,tau1,alpha1,type,mod_type,m);


            y2=zeros(size(y1));

            y=y1+y2;

            noise_var=10^(-0.1*SNR)*mean(abs(y).^2)/2;
            y=y+sqrt(noise_var)*(randn(size(y))+1i*randn(size(y)));

            h1=corr1(y(m*(pos11-1)/mod_type+1:m*(pos12)/mod_type),S0,1);
            y=y*h1;


            [X,Y]=make_features(bits1_0,y,nb,m,mod_type);
            X_total=[X_total;X];
            y_total=[y_total;Y];




            if 0
                [P,ff]=psd1(y.^4,2^14,1);
                [~,iw]=max(P);
                f0=ff(iw)/4;
                if size(y,2)<size(y,1)
                    y=y.';
                end
                y=y.*exp(-2*pi*1i*f0*[0:length(y)-1]/1);
            end

        end
    end

    toc
    %save features_DL_demod_BPSK_01 X_total y_total

    %save arrays_total0 X_train_total X_test_total y_train_total y_test_total
    N=length(y_total);
    ind=randperm(N);
    X_total=X_total(ind,:);
    y_total=y_total(ind);
    X_train=X_total(1:N/2,:);
    y_train=y_total(1:N/2);
    X_test=X_total(N/2+1:end,:);
    y_test=y_total(N/2+1:end);

    [B,dev,stats] = mnrfit(X_train,categorical(y_train));%,'model','ordinal');

    y_test1=zeros(size(y_test));
    for nn=1:length(y_test)
        [~,y_test1(nn)]=max(mnrval(B,X_test(nn,:),stats));
        y_test1(nn)=y_test1(nn)-1;
    end
    %subplot(212)
    plot(y_test-y_test1,'.');shg
    length(find(y_test~=y_test1))/length(y_test)

    ii=N/2+find(y_test~=y_test1);

    ind1=zeros(size(ind));for i=1:N j=find(ind==i);ind1(i)=j;end;
    op=ind1(ii);
    orig_positions = arrayfun(@(num) find(ind == find(1:N == num, 1)), ii);

    ind_=ceil(op/num_samples/(num_bits/nb));
end


if 1%testing
    %load B_nb4 B stats
    bits1_est=zeros(1,num_bits);
    num_bits=1000;num_samples=100;
    err_arr=zeros(1,7);
    err_arr_=zeros(1,7);
    for ind_snr=1:7
        %rand('state',0);randn('state',0);
        SNR=2.5*(ind_snr-1);
        disp(SNR)
        for n=1:num_samples%500
            %disp(n)
            %rand('state',n);randn('state',n);
            a1=1;phi1=pi/6*(2*rand-1);tau1=0.45*(2*rand-1);alpha1=0.25;w1=0*2e-3*(2*rand-1);
            bits1_0=round(rand(1,num_bits));
            bits1_0(1:len_uw)=uw_bits;
            [y1,symb1]=gen_bit(bits1_0,a1,phi1,w1,tau1,alpha1,type,mod_type,m);
            y2=zeros(size(y1));
            y=y1+y2;

            noise_var=10^(-0.1*SNR)*mean(abs(y).^2)/2;
            y=y+sqrt(noise_var)*(randn(size(y))+1i*randn(size(y)));

            h1=corr1(y(m*(pos11-1)/mod_type+1:m*(pos12)/mod_type),S0,1);
            y=y*h1;


            [X,Y]=make_features(bits1_0,y,nb,m,mod_type);
            Y1=zeros(size(Y));
            for nn=1:length(Y1)
                [~,Y1(nn)]=max(mnrval(B,X(nn,:),stats));
                Y1(nn)=Y1(nn)-1;
                bits1_est((nn-1)*nb+1:nn*nb)=de2bi(Y1(nn),nb,'left-msb');
            end
                            
            bits1_est((nn-1)*nb+1:nn*nb)=de2bi(Y1(nn),nb,'left-msb');
            [z] = demodulator_qpsk_lite(y,2^mod_type,2,1);
            h=corr1(z((pos11-1)/mod_type+1:pos12/mod_type),symb0,1);
            z=z*h;

            bits1_est_=(1-sign(real(z.')))/2;


            num_err=length(find(bits1_0~=bits1_est));
            err_arr(ind_snr)=err_arr(ind_snr)+num_err;

            if length(bits1_est_)<length(bits1_est)
                bits1_est_=[bits1_est_ zeros(1,length(bits1_est)-length(bits1_est_))];
            end

            num_err_=length(find(bits1_0~=bits1_est_));
            err_arr_(ind_snr)=err_arr_(ind_snr)+num_err_;



            if 0
                [P,ff]=psd1(y.^4,2^14,1);
                [~,iw]=max(P);
                f0=ff(iw)/4;
                if size(y,2)<size(y,1)
                    y=y.';
                end
                y=y.*exp(-2*pi*1i*f0*[0:length(y)-1]/1);
            end

        end
    end
    err_arr=err_arr/num_bits/num_samples;
err_arr_=err_arr_/num_bits/num_samples;
semilogy(2.5*[0:6],err_arr,2.5*[0:6],err_arr_);grid;shg

end

if 0
UW_struct = struct(...
                'UW_bits',uw_bits,...
                'UW_symb',symb0.',... % symbols in 1 sps
                'UW_mod',S0.',... % symbols in 2 sps
                'UW_mod_cnj_norm',conj(S0).',...% conj of symbols in 2 sps div at norm
                'UW_soft_len',uw_bits);
end