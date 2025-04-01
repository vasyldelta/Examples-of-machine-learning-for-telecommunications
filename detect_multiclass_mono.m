clear

DL_mode=0;
MNR_mode=0;
NN_mode=0;
GMM_mode=1;

if MNR_mode
    load B_5_mono
    %load B_5_big B
elseif DL_mode
    load trainedNet_5mod_mono.mat
    rx=zeros(1,1024,2,1);
elseif NN_mode
    load cent_5mod_1024_mono
    n_q=size(cent,2)/5;
elseif GMM_mode
    load gmm_5mod_16_mono.mat
    n_q=size(mu,2)/5;
end

path_str = 'bin\';decim=2;files_all_dir = dir([path_str '**/DelCy(0)1.bin']);



N=length(files_all_dir);

num_err=0;
n_total=0;
num=zeros(1,6);
y_pred=[];
y_test=[];
num_correct=zeros(1,6);


zz=zeros(1,5);
X_test=[];
%load gmm_8 alpha1 mu1 sigma1 alpha2 mu2 sigma2 alpha3 mu3 sigma3
%load nn_8.mat;n_q=length(cent)/3;
for n=1:N
    disp('-----------')
    fname=[files_all_dir(n).folder '\' files_all_dir(n).name];
    if contains(fname,"QPSK") & ~contains(fname,"BPSK") & ~contains(fname,"8PSK") & ~contains(fname,"OQPSK")
        num(2)=num(2)+1;
        flag=2;disp('QPSK')
    elseif contains(fname,"BPSK") & ~contains(fname,"QPSK") & ~contains(fname,"8PSK")
        flag=1;disp('BPSK')
        num(1)=num(1)+1;
    elseif contains(fname,"8PSK") & ~contains(fname,"QPSK") & ~contains(fname,"BPSK")
        flag=3;disp('8PSK')
        num(3)=num(3)+1;
    elseif contains(fname,"OQPSK") 
        flag=4;disp('OQPSK')
        num(4)=num(4)+1;
    elseif contains(fname,"8QAM")
        flag=5;disp('8QAM')
        num(5)=num(5)+1;
    elseif contains(fname,"16QAM")
        flag=6;disp('16QAM')
        num(6)=num(6)+1;
    else
        n=n;
        continue
    end
    n_total=n_total+1;

    x=fread_int16(fname,Inf,0);z=x(1:2:end)+1i*x(2:2:end);
    y=z(1:decim:end);
    y=y(1:2e5);%!!!!!!!!!!!!!

    y=y/std(y);

    if 1%flag~=3
        [P,ff]=psd1(y.^4,2^14,1);
        [~,iw]=max(P);
        f0=ff(iw)/4;
        if size(y,2)<size(y,1)
            y=y.';
        end
        y=y.*exp(-2*pi*1i*f0*[0:length(y)-1]/1);
    end

    if ~DL_mode
        [z] = demodulator_qpsk_lite(y,4,2,1);
        z=z(round(0.25*length(z)+1)+1:end);
        x=calc_features(z);
        X_test=[X_test;x];
    end
    %x=x([1:10 12:end]);

    if MNR_mode
        p_arr=zeros(1,5);
        num1=fix(length(z)/1024);
        for kk=1:num1
            x=calc_features(z((kk-1)*1024+1:kk*1024));
            for k=1:4
                zz(k)=x*B(2:end,k)+B(1,k);
            end
            zz(end)=0;
            e=exp(zz);
            p=e/sum(e);
            p_arr=p_arr+p;
        end
        p=p_arr;
        p=[p(1:3) -Inf p(4:5)];
        [~,iw]=max(p);
    elseif DL_mode
        %y_test=double(classify(trainedNet,rxValidFrames));
        num1=fix(length(y)/1024);
        p=zeros(1,5);
        for kk=1:num1
            rx(1,:,1,1)=real(y((kk-1)*1024+1:kk*1024));
            rx(1,:,2,1)=imag(y((kk-1)*1024+1:kk*1024));
            yy=double(classify(trainedNet,rx));
            p(yy)=p(yy)+1;
        end
        p=[p(1:3) -Inf p(4:5)];
        [~,iw]=max(p);
    elseif NN_mode
        p=zeros(1,5);
        num1=fix(length(z)/1024);
        for kk=1:num1
            x=calc_features(z((kk-1)*1024+1:kk*1024));
            x=x(1:9);
            n2=neighbour_2(x',cent);
            iw=ceil((n2)/n_q);
            p(iw)=p(iw)+1;
        end
        p=[p(1:3) -Inf p(4:5)];
        [~,iw]=max(p);

    elseif GMM_mode
        p=zeros(1,5);
        num1=fix(length(z)/1024);
        for kk=1:num1
            x=calc_features(z((kk-1)*1024+1:kk*1024));
            x=x(1:9);
            for n_=1:5
                res(n_)=likelihood_calc(x',alpha((n_-1)*n_q+1:n_*n_q),mu(:,(n_-1)*n_q+1:n_*n_q),sigma(:,:,(n_-1)*n_q+1:n_*n_q));
            end
            if 0
                [~,iw]=max(res);
                p(iw)=p(iw)+1;
            else
                p=p+res;
            end
        end
        p=[p(1:3) -Inf p(4:5)];
        [~,iw]=max(p);


        
    end

    if 1%iw==1 | iw==2 | iw==3
        h=psd(abs(resample(y,2,1)).^2,2^16,4);%plot(log10(abs(h)),'o');grid;shg
        pf=calc_peakfactor(h,16385);
        if pf<5
            iw=4;
        end
    end
    y_pred=[y_pred iw-1];
    y_test=[y_test flag-1];

    if iw==flag
        disp('Right')
        num_correct(flag)=num_correct(flag)+1;
    else
        disp('Wrong')
    end

    disp('-----------')
end
disp(num)
disp(['Accuracy: ' num2str(1-length(find(y_pred~=y_test))/length(y_test))])
disp(num_correct./num)
mean(num_correct./num)


function p=calc_features(s)
M20=cumul(s,2,0);
M21=cumul(s,2,1);
M22=cumul(s,2,2);
M40=cumul(s,4,0);
M41=cumul(s,4,1);
M42=cumul(s,4,2);
M60=cumul(s,6,0);
M43=cumul(s,4,3);
M61=cumul(s,6,1);
M62=cumul(s,6,2);
M63=cumul(s,6,3);

C20=M20;
C21=M21;
C40=M40-3*M20^2;
C41=M41-3*M20*M21;%???????
C42=M42-abs(M20)^2-2*M21^2;
C60=M60-15*M20*M40+30*M20^3;
C61=M61-5*M21*M40-10*M20*M41+30*M20^2*M21;
C62=M62-6*M20*M42-8*M21*M41-M22*M40+6*M20^2*M22+24*M21^2*M20;
C63=M63-9*M21*M42+12*M21^3-3*M20*M43-3*M22*M41+18*M20*M21*M22;
d=[C20 C21 C40 C41 C42 C60 C61 C62 C63];
d=abs(d);
d=d.^[2./[2 2 4 4 4 6 6 6 6]];
M=length(d);
if 1
    p=[d d.^2];
    %p=[d d.^2];
    for i=1:M
        for j=i+1:M
            p=[p d(i)*d(j)];
        end
    end
elseif 0
    p=d;
end
end




function res=cumul(s,p,q)
res=sum((s.^(p-q)).*(conj(s).^q));
res=res/length(s);
end

