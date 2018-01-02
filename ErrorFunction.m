%%
% Calculates the error function according to the current predictions
%
% Inputs
% - y_label (N x 1): True labels of each data point, where N is the number
% of data points
% - Y_pred (N x K): Output of each output unit, where K=10 (0 to 9)
%
% Outputs
% - E (1 x 1): The value of the error function
%E = ErrorFunction(y_trn,X_trn_norm)
function E = ErrorFunction(y_label,Y_pred)

[N,K] = size(Y_pred);

E =0;

for n=1:N
    for k=1:K
        E = E + ((y_label(n)+1) * log(Y_pred(n,k)));
    end
end
E = -1 * E;




end

