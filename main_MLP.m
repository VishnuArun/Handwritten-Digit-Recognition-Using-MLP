

% read data: 
[X_trn_norm, y_trn, X_val_norm, y_val, X_tst_norm, y_tst] = ReadNormalizedOptdigitsDataset("optdigits_train.txt", "optdigits_valid.txt", "optdigits_test.txt");
%%%%

Hs = [4,8,12,16,20,24];
training_error = zeros(length(Hs),1);
validation_error = zeros(length(Hs),1);

% check training and validation error for each option of H
for i=1:length(Hs)
    H = Hs(i);

    % train MLP using current H using MLPTrain

    [Y_pred,Z,W,V] = MLPTrain(X_trn_norm, y_trn, H);
    %%%%

    % calculate error rate for Y predicted to the training set

    training_error(i) = CalculateErrorRate(Y_pred,y_trn);
    %%%%

    fprintf('Training set error rate when H=%d: %f\n', H, training_error(i));
    

    [Y,Z] = ForwardPropagation(X_val_norm, W, V);
    %%%%

    % calculate error rate for Y predicted to the validation set
 
    validation_error(i) = CalculateErrorRate(Y,y_val);
    %%%%
    
    fprintf('Validation set error rate when H=%d: %f\n', H, validation_error(i));
    
end

% Plot training and validation error

PlotTrainingValidationError(Hs,training_error, validation_error);
%%%% 

%%% compile our errors and H values
Errors = [Hs.',training_error,validation_error];
%Min H based on Min training error
[ans,ind] = min(Errors(:,2));
MinH = Errors(ind,1);


% train MLP using the best number of hidden units MLPTrain

[Y_pred,Z,W,V] = MLPTrain(X_trn_norm, y_trn, MinH);
%%%%

% Predict Y for the test set

[Y,Z] = ForwardPropagation(X_tst_norm, W, V);
%%%%

% calculate error rate for Y predicted to the test set 

test_error = CalculateErrorRate(Y,y_tst);
%%%%

fprintf('Test set error rate when H=%d: %f\n', MinH, test_error);



% Train the MLP with 2 hidden units, using MLPTrain

[Y_pred,Z,W,V] = MLPTrain(X_trn_norm, y_trn, 2);
%%%%

% Predict Y for the validation and test set

[Y_tst,Z_tst] = ForwardPropagation(X_tst_norm, W, V);
[Y_val,Z_val] = ForwardPropagation(X_val_norm, W, V);
%%%%

% 2D scatter showing Z for the training, validation and test set 

figure('Name','Training Set');
PlotZ2DScatter(Z,Y_pred);

figure('Name','Validation Set');
PlotZ2DScatter(Z_val,Y_val);
 
figure('Name','Test Set');
PlotZ2DScatter(Z_tst,Y_tst);
%%%% 

 

% Train the MLP with 3 hidden units, using MLPTrain

[Y_pred,Z,W,V] = MLPTrain(X_trn_norm, y_trn, 3);
%%%%

% Predicting Y for the validation and test set.

[Y_tst,Z_tst] = ForwardPropagation(X_tst_norm, W, V);
[Y_val,Z_val] = ForwardPropagation(X_val_norm, W, V);
%%%%

% 3D scatter showing Z for the training, validation and test set 
figure('Name','Training Set');
PlotZ3DScatter(Z,Y_pred);

figure('Name','Validation Set');
PlotZ3DScatter(Z_val,Y_val);
 
figure('Name','Test Set');
PlotZ3DScatter(Z_tst,Y_tst);
%%%% 
