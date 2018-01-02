%%
% 
% Data is first centered using the mean of the training set, and then
% normalized dividing by the standard deviation of the training set, so 
% that x_norm = (x-mean_trn)/(sigma_trn^2)
%
% Input
% - training_filename: string containing the training data file path
% - validation_filename: string containing the validation data file path
% - test_filename: string containing the test data file path
%
% 
function [X_trn_norm, y_trn, X_val_norm, y_val, X_tst_norm, y_tst] = ReadNormalizedOptdigitsDataset(training_filename, validation_filename, test_filename)

content = dlmread(training_filename);
content_val = dlmread(validation_filename);
content_test = dlmread(test_filename);


% Slice up data
% Training
X_trn = content(:,1:end-1);

mu_trn = mean(X_trn);
std_trn = std(X_trn);

[N_trn,M] = size(X_trn);
X_trn_norm = zeros(N_trn,M);
for x=1:N_trn
    for y=1:M
        X_trn_norm(x,y) = ((X_trn(x,y) - mu_trn(y))/std_trn(y));
    end
end

%Replace NaN with 0
X_trn_norm(isnan(X_trn_norm))=0;

y_trn = content(:,end);

% Validation Calcualted mean and sd from Training set!!!!!!!!!
X_val = content_val(:,1:end-1);
X_val_norm = zeros(N_trn,M);
for x=1:N_trn
    for y=1:M
        X_val_norm(x,y) = ((X_val(x,y) - mu_trn(y))/std_trn(y));
    end
end
%
%% Replace NaN with 0
%%
X_val_norm(isnan(X_val_norm))=0;

y_val = content_val(:,end);

% Testing Calculated mean and sd from TRAINING SET!!!!!

X_tst = content_test(:,1:end-1);
X_tst_norm = zeros(N_trn,M);
for x=1:N_trn
    for y=1:M
        X_tst_norm(x,y) = ((X_tst(x,y) - mu_trn(y))/std_trn(y));
    end
end

%Replace NaN with 0
X_tst_norm(isnan(X_tst_norm))=0;

y_tst = content_test(:,end);
%%%%
end

