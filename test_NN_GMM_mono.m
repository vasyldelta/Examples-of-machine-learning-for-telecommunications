%TRAINING PART
clear
warning off %!!!!!!!!!!!!!!!!!!!!!!!!!
close all
rand('state',1)
randn('state',1)
NN_mode=1;
GMM_mode=0;
for n_q=[4 16 64 256 1024]
    disp('____________________')
    disp(['n_q: ' num2str(n_q) ])
mask=[1 2 3 5 6];
y=[];
X=[];
n_=0;
dim_x=9;
cent=zeros(dim_x,n_q*length(mask));
alpha=zeros(1,n_q*length(mask));
mu=zeros(dim_x,n_q*length(mask));
sigma=zeros(dim_x,dim_x,n_q*length(mask));
ratio_train=0.5;
X_test=[];
y_test=[];
X_train0=[];
y_train0=[];
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
        features_arr=features_arr(:,1:dim_x);
        %features_arr=randn(size(features_arr));
        %features_arr = isomap(features_arr, dim_x);%, k)
        %features_arr = lle(features_arr, dim_x);%, k)
        %[features_arr1,mapping,M] = pca(features_arr, dim_x);%, k)
        m_=size(features_arr,1);

        ind=randperm(m_);ind=ind(1:ratio_train*m_);
        X_train=features_arr(ind,:);
        ind_test=sort(setdiff(1:m_,ind));
        X_test=[X_test;features_arr(ind_test,:)];
        y_test=[y_test (n_-1)*ones(1,m_*ratio_train)];

        X_train0=[X_train0;X_train];
        y_train0=[y_train0 (n_-1)*ones(1,m_*ratio_train)];

        if NN_mode
            cent(:,(n_-1)*n_q+1:n_*n_q)=k_means_lsf(X_train',n_q,5);
        elseif GMM_mode
            [alpha((n_-1)*n_q+1:n_*n_q),mu(:,(n_-1)*n_q+1:n_*n_q),sigma(:,:,(n_-1)*n_q+1:n_*n_q)]=...
                gmm_est(X_train',n_q,5);
        end

    end
end


%TEST PART
y_test1=zeros(size(y_test));
num=zeros(1,length(mask));
num_correct=zeros(1,length(mask));
res=zeros(1,length(mask));


for n=1:length(y_test1)

    if NN_mode
        iw=neighbour_2(X_test(n,:)',cent);
        y_test1(n)=ceil((iw)/n_q);

    elseif GMM_mode
        for n_=1:length(mask)
            res(n_)=likelihood_calc(X_test(n,:)',alpha((n_-1)*n_q+1:n_*n_q),mu(:,(n_-1)*n_q+1:n_*n_q),sigma(:,:,(n_-1)*n_q+1:n_*n_q));
        end
        [~,y_test1(n)]=max(res);
    end

    num(y_test(n)+1)=num(y_test(n)+1)+1;
    flag=y_test(n)+1;

    if (y_test1(n))==flag
        num_correct(flag)=num_correct(flag)+1;
    end
    y_test1(n)=y_test1(n)-1;
end
1-length(find(y_test~=y_test1))/length(y_test)
disp([num_correct./num])
end