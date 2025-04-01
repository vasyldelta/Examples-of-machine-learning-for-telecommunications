clear
rand('state',1);randn('state',0);
warning off
mod_type=2;
type='normal';
m=2;
snr_arr=0:2.5:15;
uw_bits=[1 0 1 0 0 0 0 0 0 1 0 1 1 1 1 0 1 1 0 1];
if mod_type==3
    uw_bits=[uw_bits 1 1 0 1];
end
[S0,symb0]=gen_bit(uw_bits,1,0,0,0,0.25,type,mod_type,m);
len_uw=length(uw_bits);
pos11=1;pos12=pos11+len_uw-1;
pos21=41;pos22=pos21+len_uw-1;
num_bits=120;
num_samples=100;
nb=2;

% Add NN option
use_nn = true; % Set to true to enable Neural Network classification
include_h=1;
if 1
    X_train_total = [];
    X_test_total = [];
    y_train_total = [];
    y_test_total = [];

    X_total = [];
    y_total = [];
    tic
    for ind_snr = 1:7
        rand('state', 0); randn('state', 0);
        SNR = 2.5 * (ind_snr - 1);
        disp(SNR)
        for n = 1:num_samples
            rand('state', n); randn('state', n);
            a1 = 1; phi1 = pi * (2 * rand - 1); tau1 = 0.45 * (2 * rand - 1); alpha1 = 0.25; w1 = 0 * 2e-3 * (2 * rand - 1);
            bits1_0 = round(rand(1, num_bits));
            bits1_0(1:len_uw) = uw_bits;
            [y1, symb1] = gen_bit(bits1_0, a1, phi1, w1, tau1, alpha1, type, mod_type, m);

            if 0
                y2 = zeros(size(y1));
            else
                a2 = 0.1+0*(0.5+0.5*rand); phi2 = pi * (2 * rand - 1); tau2 = 0.45 * (2 * rand - 1); alpha1 = 0.25; w2 = 2e-3 * (2 * rand - 1);
                bits2_0 = round(rand(1, num_bits));
                bits2_0(pos21:pos22) = uw_bits;

                %bits1_0(1:len_uw) = uw_bits;
                [y2, symb2] = gen_bit(bits2_0, a2, phi2, w2, tau2, alpha1, type, mod_type, m);
            end
            y = y1 + y2;

            noise_var = 10^(-0.1 * SNR) * mean(abs(y).^2) / 2;
            y = y + sqrt(noise_var) * (randn(size(y)) + 1i * randn(size(y)));

            h1 = corr1(y(m * (pos11 - 1) / mod_type + 1:m * (pos12) / mod_type), S0, 1);
            if ~include_h
                y = y * h1;
            end

            [X, Y] = make_features(bits1_0, y, nb, m, mod_type);
            if include_h
                X=[X real(h1)*ones(size(X,1),1) imag(h1)*ones(size(X,1),1)];
            end
            X_total = [X_total; X];
            y_total = [y_total; Y];
        end
    end

    toc
    N = length(y_total);
    %ind = randperm(N);
    ind=1:N;
    X_total = X_total(ind, :);
    y_total = y_total(ind);
    X_train = X_total(1:N/2, :);
    y_train = y_total(1:N/2);
    X_test = X_total(N/2+1:end, :);
    y_test = y_total(N/2+1:end);

    if use_nn
        % Define the Neural Network
        numFeatures = size(X_train, 2);  % Number of input features
        numClasses = length(unique(y_train));  % Number of output classes

        layers = [
            featureInputLayer(numFeatures, 'Name', 'input')            % Input layer
            %fullyConnectedLayer(128, 'Name', 'fc1')                    % First hidden layer
            %reluLayer('Name', 'relu1')                                % Activation for hid
            fullyConnectedLayer(32, 'Name', 'fc1')                    % First hidden layer
            reluLayer('Name', 'relu1')                                % Activation for hidden layer
            fullyConnectedLayer(16, 'Name', 'fc2')                    % Second hidden layer
            reluLayer('Name', 'relu2')                                % Activation for hidden layer
            fullyConnectedLayer(numClasses, 'Name', 'fc3')            % Output layer
            softmaxLayer('Name', 'softmax')                           % Softmax for classification
            classificationLayer('Name', 'output')                    % Classification layer
            ];

        % Training Options
        options = trainingOptions('adam', ...
            'MaxEpochs', 20, ...
            'MiniBatchSize', 32, ...
            'Verbose', false, ...
            'Plots', 'training-progress');  % Visualize training

        % Train the Neural Network
        y_train_categorical = categorical(y_train); % Convert labels to categorical
        net = trainNetwork(X_train, y_train_categorical, layers, options);

    else
        % Multinomial Regression
        [B, dev, stats] = mnrfit(X_train, categorical(y_train));
    end
end

if 1 % Testing
    %load B_nb4 B stats
    bits1_est = zeros(1, num_bits);
    err_arr = zeros(1, 7);
    err_arr_ = zeros(1, 7);
    for ind_snr = 7
        %rand('state', 0); randn('state', 0);
        SNR = 2.5 * (ind_snr - 1);
        disp(SNR)
        for n = 1:num_samples
            %rand('state', n); randn('state', n);
            a1 = 1; phi1 = pi/6 * (2 * rand - 1); tau1 = 0.45 * (2 * rand - 1); alpha1 = 0.25; w1 = 0 * 2e-3 * (2 * rand - 1);
            bits1_0 = round(rand(1, num_bits));
            bits1_0(1:len_uw) = uw_bits;
            [y1, symb1] = gen_bit(bits1_0, a1, phi1, w1, tau1, alpha1, type, mod_type, m);

            if 0
                y2 = zeros(size(y1));
            else
                a2 = 0.1+0*(0.5+0.5*rand); phi2 = pi * (2 * rand - 1); tau2 = 0.45 * (2 * rand - 1); alpha1 = 0.25; w2 = 2e-3 * (2 * rand - 1);
                bits2_0 = round(rand(1, num_bits));
                %bits1_0(1:len_uw) = uw_bits;
                [y2, symb2] = gen_bit(bits2_0, a2, phi2, w2, tau2, alpha1, type, mod_type, m);
            end
            
            y=y1+y2;

            noise_var = 10^(-0.1 * SNR) * mean(abs(y).^2) / 2;
            y = y + sqrt(noise_var) * (randn(size(y)) + 1i * randn(size(y)));

            h1 = corr1(y(m * (pos11 - 1) / mod_type + 1:m * (pos12) / mod_type), S0, 1);
            if ~include_h
                y = y * h1;
            end

            [X, Y] = make_features(bits1_0, y, nb, m, mod_type);

            Y1 = zeros(size(Y));
            for nn = 1:length(Y1)
                if use_nn
                    % Neural Network Prediction
                    if ~include_h
                    [~, Y1(nn)] = max(predict(net, X(nn, :))); % Get the predicted class
                    else
                        [~, Y1(nn)] = max(predict(net, [X(nn, :) real(h1) imag(h1)])); % Get the predicted class
                    end
                    Y1(nn) = Y1(nn) - 1; % Convert to zero-based indexing

                else
                    % Multinomial Regression Prediction
                    [~, Y1(nn)] = max(mnrval(B, X(nn, :), stats));
                    Y1(nn) = Y1(nn) - 1;
                end
                bits1_est((nn - 1) * nb + 1:nn * nb) = de2bi(Y1(nn), nb, 'left-msb');
            end
            %[out_burst_soft,out_burst_complex,bits1_est,SNR_UW,err_uw,SNR0,corr_ok,f0_est]=demod_my(y,m,mod_type,uw_bits,symb0,S0);


            num_err = length(find(bits1_0 ~= bits1_est));
            err_arr(ind_snr) = err_arr(ind_snr) + num_err;
        end
    end
    err_arr = err_arr / num_bits / num_samples;
    semilogy(2.5 * [0:6], err_arr); grid; shg
end
