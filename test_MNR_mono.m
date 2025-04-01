%TRAINING PART
clear
close all
rand('state',1)
randn('state',1)
mask=[1 2 3 5 6];
y=[];
X=[];
n_=0;
for k=1:6
    if ismember(k,mask)
        n_=n_+1;
        if k==1
            load features\features_arr_bpsk_1sps_new2 features_arr
        elseif k==2
            load features\features_arr_qpsk_1sps_new2 features_arr
        elseif k==3
            load features\features_arr_8psk_1sps_new2 features_arr
        elseif k==4
            load features\features_arr_oqpsk_1sps_new2 features_arr
        elseif k==5
            load features\features_arr_8qam_1sps_new2 features_arr
        elseif k==6
            load features\features_arr_16qam_1sps_new2 features_arr
        end

        X=[X;features_arr];
        m_=size(features_arr,1);
        y=[y (n_-1)*ones(1,m_)];
    end
end

m=size(X,1);

ratio_train=0.5;
ind=randperm(m);ind=ind(1:ratio_train*m);
X_train=X(ind,:);
y_train=y(ind);
ind_test=sort(setdiff(1:m,ind));
X_test=X(ind_test,:);
y_test=y(ind_test);

%CHECK
flag_scale=0;
if flag_scale
    X_train0=X_train;
    [X_train,mu,var]=scale(X_train);
end

[B,dev,stats] = mnrfit(X_train,categorical(y_train));%,'model','ordinal');

if flag_scale
    for nn=1:length(mask)-1
        [w,b]=unscale_param(B(2:end,nn)',B(1,nn),mu,var);
        B(1,nn)=b;B(2:end,nn)=w';
    end
    X_train=X_train0;
end
%=================================================
%TEST PART

%X_test=X_train;
%y_test=y_train;
y_test1=zeros(size(y_test));

zz=zeros(1,length(mask));
num=zeros(1,length(mask));
num_correct=zeros(1,length(mask));
for n=1:length(y_test1)
    for k=1:length(mask)-1
        zz(k)=X_test(n,:)*B(2:end,k)+B(1,k);
    end
    zz(end)=0;
    e=exp(zz);
    p=e/sum(e);
    [~,y_test1(n)]=max(p);

    num(y_test(n)+1)=num(y_test(n)+1)+1;
    flag=y_test(n)+1;

    if (y_test1(n))==flag
        num_correct(flag)=num_correct(flag)+1;
    end
    y_test1(n)=y_test1(n)-1;
end
1-length(find(y_test~=y_test1))/length(y_test)
disp([num_correct./num])
%save B_5 B


