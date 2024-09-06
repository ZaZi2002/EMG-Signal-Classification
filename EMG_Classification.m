clc
clear all
close all

%% Loading data
load("DB2_s1\S1_E1_A1.mat");  S = 1;   % Subject 1 data
%load("DB2_s2\S2_E1_A1.mat");  S = 2; % Subject 2 data
%load("DB2_s3\S3_E1_A1.mat");  S = 3; % Subject 3 data

%% Filtering 
fs = 2000;  % Sampling frequency
activation_length = 5;  % Activation time(s)  
rest_length = 3;    % Rest time (s)

[b1,a1] = butter(6,500/(fs/2),'low'); % Butterworth lowpass filter of order 6 & 500Hz
[b2,a2] = butter(2,1/(fs/2),'high'); % Butterworth highpass filter of order 2 & 1Hz
[b3,a3] = iirnotch(50/(fs/2), 0.01); % Notch filter for 50Hz
[b4,a4] = iirnotch(100/(fs/2), 0.01); % Notch filter for 100Hz

% Filtering channels
for i = 1:12
    filtered_emg(:,i) = filter(b1,a1,emg(:,i));
    filtered_emg(:,i) = filter(b2,a2,filtered_emg(:,i));
    filtered_emg(:,i) = filter(b3,a3,filtered_emg(:,i));
    filtered_emg(:,i) = filter(b4,a4,filtered_emg(:,i));
end

% FFT of first channel befor and after filtering
X = fftshift(fft(emg(:,1)));
Y = fftshift(fft(filtered_emg(:,1)));
N = length(X);
f = (-N/2+1:N/2)*fs/N;

% Plotting spectrums
figure;

subplot(2,1,1);
plot(f,abs(X));
grid on
xlim ([0,1000]);
title("Not filtered signal of channel 1");
xlabel("Frequency(Hz)");

subplot(2,1,2);
plot(f,abs(Y));
grid on
xlim ([0,1000]);
title("Filtered signal of channel 1");
xlabel("Frequency(Hz)");

%% Normalizing (Z-Score)
for i = 1:12
    normalized_emg(:,i) = normalize(filtered_emg(:,i),'zscore');
end

%% Data windows
% Finding windows number per subject
switch S
    case 1
        N = 90396;
    case 2
        N = 90165;
    case 3
        N = 89737;
end

% windows
window_size = 400;
window_overlap = 20;
windows_1 = zeros(N,400,12);
j = 0;
for i = 1:N
    % Finding mode label
    M = mode(stimulus((i-1)*window_overlap +1 : (i-1)*window_overlap +window_size ));
    % Deleting 0 labels
    if (M~=0)
        j = j + 1;
        windows_1(j,:,:) = normalized_emg( ((i-1)*window_overlap +1 : (i-1)*window_overlap +window_size) ,:);
        windows_labels(j) = M;
    end
    i
end
windows = windows_1((1:j),:,:); % Main windows

windows_labels = windows_labels.';
windows_number = j; % Not rest windows number

%% Freature extraction

features_number = 7;
features = zeros(windows_number,12,features_number);
for i = 1:windows_number
    for j = 1:12
        features(i,j,1) = mean(abs(windows(i,:,j)));    % MAV
        features(i,j,2) = std(windows(i,:,j));  % STD
        features(i,j,3) = var(windows(i,:,j));  % VAR
        features(i,j,4) = rms(windows(i,:,j));  % RMS

        wl = 0;
        for t = 2:380
            wl = wl + windows(i,t,j)-windows(i,t-1,j);
        end
        features(i,j,5) = wl;   % WL

        zc = 0;
        for t = 2:380
            if sign((windows(i,t,j)*windows(i,t-1,j))) == -1
                zc = zc + 1;
            end
        end
        features(i,j,6) = zc;   % ZC
        features(i,j,7) = sum(abs(windows(i,:,j)));   % IAV
    end
    i
end

% Making main features matrix (windows*features)
main_features = reshape(features,windows_number,7*12);


%% Training

main_features; % Features
windows_labels; % Labels

% Putting 70% of windows for training and 30% of windows for testing
cv = cvpartition(windows_labels,'HoldOut',0.3);

% Defining train & test data
data_train = main_features(cv.training,:);
label_train = windows_labels(cv.training,:);
data_test = main_features(cv.test,:);
label_test = windows_labels(cv.test,:);

% KNN Model
KNN_model = fitcknn(data_train,label_train,'NumNeighbors',5,'Distance','euclidean');

% Random Forest Model
RF_model = TreeBagger(100,data_train,label_train,'Method','classification');

% Predicting labels of test data (KNN & RF)
predicted_labels_knn = predict(KNN_model,data_test);
predicted_labels_rf = predict(RF_model,data_test);
predicted_labels_rf = cellfun(@str2num, predicted_labels_rf);

%% Validation
clc
% Confusion matrix
C_knn = confusionmat(label_test, predicted_labels_knn);
C_rf = confusionmat(label_test, int8(predicted_labels_rf));
labels = ["G1","G2","G3","G4","G5","G6","G7","G8","G9","G10","G11","G12","G13","G14","G15","G16","G17"];
pred_labels = ["Pred. G1","Pred. G2","Pred. G3","Pred. G4","Pred. G5","Pred. G6","Pred. G7","Pred. G8","Pred. G9","Pred. G10","Pred. G11","Pred. G12","Pred. G13","Pred. G14","Pred. G15","Pred. G16","Pred. G17"];

figure;
heatmap(pred_labels, labels, C_knn);
title("Confusion Matrix of KNN");

figure;
heatmap(pred_labels, labels, C_knn);
title("Confusion Matrix of RF");


% Compute accuracy of models
accuracy_knn = sum(predicted_labels_knn == label_test) / numel(label_test);
accuracy_rf = sum(predicted_labels_rf == label_test) / numel(label_test);

display("Accuracy with KNN model is : " + accuracy_knn*100 + "%");
display("Accuracy with RF model is : " + accuracy_rf*100 + "%");
