%%
% Plot error rate for the training and validation set
%
% Inputs
% - Hs (#Hs x 1): Vector with all the H options
% - training_error (#Hs x 1): Training set error rate for each H
% - validation_error (#Hs x 1): Validation set error rate for each H
%
function PlotTrainingValidationError(Hs,training_error, validation_error)


plot(Hs,training_error); hold on; plot(Hs,validation_error);

%%%%
end
