%TRAINING PART
clear 
close all
%modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
%  "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", ...
%  "B-FM", "DSB-AM", "SSB-AM"]);

modulationTypes = categorical(["BPSK","QPSK","8PSK","8QAM","16QAM"]);
%modulationTypes = categorical(["BPSK+BPSK","QPSK+QPSK"]);
num_mod=length(modulationTypes);

sps = 1;                % Samples per symbol
spf = 1024;             % Samples per frame
modClassNet = helperModClassCNN(modulationTypes,sps,spf);
maxEpochs = 25;
miniBatchSize = 1024;

Y1=importdata('features\features_arr_bpsk_1sps_new2.mat','y_total');Y1=Y1.y_total;


[~,~,~,M]=size(Y1);
TrainLabels=zeros(num_mod*M,1);
TrainFrames=zeros(1,1024,2,num_mod*M);

aa=importdata('features\features_arr_bpsk_1sps_new2.mat','y_total');TrainFrames(:,:,:,1:M)=aa.y_total;
aa=importdata('features\features_arr_qpsk_1sps_new2.mat','y_total');TrainFrames(:,:,:,M+1:2*M)=aa.y_total;
aa=importdata('features\features_arr_8psk_1sps_new2.mat','y_total');TrainFrames(:,:,:,2*M+1:3*M)=aa.y_total;
aa=importdata('features\features_arr_8qam_1sps_new2.mat','y_total');TrainFrames(:,:,:,3*M+1:4*M)=aa.y_total;
aa=importdata('features\features_arr_16qam_1sps_new2.mat','y_total');TrainFrames(:,:,:,4*M+1:5*M)=aa.y_total;

for k=1:num_mod
    TrainLabels((k-1)*M+1:k*M)=k-1;
end

ind=randperm(num_mod*M);
ind1=ind(1:0.5*num_mod*M);ind2=setdiff(1:num_mod*M,ind1);

rxTrainFrames=TrainFrames(:,:,:,ind1);
rxTrainLabels=TrainLabels(ind1);
rxValidFrames=TrainFrames(:,:,:,ind2);
rxValidLabels=TrainLabels(ind2);

rxTrainLabels=categorical(rxTrainLabels);
rxValidLabels=categorical(rxValidLabels);


options = helperModClassTrainingOptions(maxEpochs,miniBatchSize,...
    numel(rxTrainLabels),rxValidFrames,rxValidLabels);

if 0
    trainedNet = trainNetwork(rxTrainFrames,rxTrainLabels,modClassNet,options);%[1 1024 2 10000]
else
    load trainedNet_5mod_mono trainedNet
end

%TEST PART
y_test1=double(classify(trainedNet,rxValidFrames));
y_test=double(rxValidLabels);
num=zeros(1,num_mod);
num_correct=zeros(1,num_mod);
for n=1:length(y_test1)
    num(y_test(n))=num(y_test(n))+1;
    flag=y_test(n);

    if (y_test1(n))==flag
        num_correct(flag)=num_correct(flag)+1;
    end
end
1-length(find(y_test~=y_test1))/length(y_test)
disp([num_correct./num])
%nnz(y_test-double(rxValidLabels))/length(rxValidLabels)
%nnz(y_test-double(rxTrainLabels))/length(rxTrainLabels)

if 0
    classLabels = {'BPSK', 'QPSK', '8PSK', '8QAM', '16QAM'};
    y_test_categorical = categorical(y_test, 1:length(classLabels), classLabels);
    y_test1_categorical = categorical(y_test1, 1:length(classLabels), classLabels);
    confusionchart(y_test_categorical, y_test1_categorical); % No need for additional class labels
end
function modClassNet = helperModClassCNN(modulationTypes,sps,spf)
%helperModClassCNN Modulation classification CNN
%   CNN = helperModClassCNN(MODTYPES,SPS,SPF) creates a convolutional
%   neural network, CNN, for modulation classification. MODTYPES is the
%   modulation types the network can identify, SPS is the samples per
%   symbol, and SPF is the samples per frame.
%
%   The CNN consists of six convolution layers and one fully connected
%   layer. Each convolution layer except the last is followed by a batch
%   normalization layer, rectified linear unit (ReLU) activation layer, and
%   max pooling layer. In the last convolution layer, the max pooling layer
%   is replaced with an average pooling layer. The output layer has softmax
%   activation. 
%   
%   See also ModulationClassificationWithDeepLearningExample.

%   Copyright 2019 The MathWorks, Inc.

numModTypes = numel(modulationTypes);
netWidth = 1;
filterSize = [1 sps];
poolSize = [1 2];
modClassNet = [
  imageInputLayer([1 spf 2], 'Normalization', 'none', 'Name', 'Input Layer')
  
  convolution2dLayer(filterSize, 16*netWidth, 'Padding', 'same', 'Name', 'CNN1')
  batchNormalizationLayer('Name', 'BN1')
  reluLayer('Name', 'ReLU1')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool1')
  
  convolution2dLayer(filterSize, 24*netWidth, 'Padding', 'same', 'Name', 'CNN2')
  batchNormalizationLayer('Name', 'BN2')
  reluLayer('Name', 'ReLU2')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool2')
  
  convolution2dLayer(filterSize, 32*netWidth, 'Padding', 'same', 'Name', 'CNN3')
  batchNormalizationLayer('Name', 'BN3')
  reluLayer('Name', 'ReLU3')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool3')
  
  convolution2dLayer(filterSize, 48*netWidth, 'Padding', 'same', 'Name', 'CNN4')
  batchNormalizationLayer('Name', 'BN4')
  reluLayer('Name', 'ReLU4')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool4')
  
  convolution2dLayer(filterSize, 64*netWidth, 'Padding', 'same', 'Name', 'CNN5')
  batchNormalizationLayer('Name', 'BN5')
  reluLayer('Name', 'ReLU5')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool5')
  
  convolution2dLayer(filterSize, 96*netWidth, 'Padding', 'same', 'Name', 'CNN6')
  batchNormalizationLayer('Name', 'BN6')
  reluLayer('Name', 'ReLU6')
  
  averagePooling2dLayer([1 ceil(spf/32)], 'Name', 'AP1')
  
  fullyConnectedLayer(numModTypes, 'Name', 'FC1')
  softmaxLayer('Name', 'SoftMax')
  
  classificationLayer('Name', 'Output') ];

end

function options = helperModClassTrainingOptions(maxEpochs,miniBatchSize,...
  trainingSize,rxValidFrames,rxValidLabels)
%helperModClassTrainingOptions Modulation classification training options
%   OPT = helperModClassTrainingOptions(MAXE,MINIBATCH,NTRAIN,Y,YLABEL)
%   returns the training options, OPT, for the modulation classification
%   CNN, where MAXE is the maximum number of epochs, MINIBATCH is the mini
%   batch size, NTRAIN is the number of training frames, Y is the
%   validation frames and YLABEL is the labels.
%
%   This function configures the training options to use an SGDM solver.
%   By default, the 'ExecutionEnvironment' property is set to 'auto', where
%   the trainNetwork function uses a GPU if one is available or uses the
%   CPU, if not. To use the GPU, you must have a Parallel Computing Toolbox
%   license. Set the initial learning rate to 3e-1. Reduce the learning
%   rate by a factor of 0.75 every 6 epochs. Set 'Plots' to
%   'training-progress' to plot the training progress.
%   
%   See also ModulationClassificationWithDeepLearningExample.

%   Copyright 2019-2023 The MathWorks, Inc.

validationFrequency = floor(trainingSize/miniBatchSize);
options = trainingOptions('sgdm', ...
  InitialLearnRate = 3e-1, ...
  MaxEpochs = maxEpochs, ...
  MiniBatchSize = miniBatchSize, ...
  Shuffle = 'every-epoch', ...
  Plots = 'training-progress', ...
  Verbose = false, ...
  ValidationData = {rxValidFrames,rxValidLabels}, ...
  ValidationFrequency = validationFrequency, ...
  LearnRateSchedule = 'piecewise', ...
  LearnRateDropPeriod = 6, ...
  LearnRateDropFactor = 0.75, ...
  OutputNetwork='best-validation-loss'); 
end