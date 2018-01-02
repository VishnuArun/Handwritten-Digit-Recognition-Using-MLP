%%
% Calculates the error rate (percentage of wrongly classified samples)
%
% Inputs
% - Y_pred (N x K): Output of each output unit
% - y_label (N x 1): True labels of each data point, where N is the number
% of data points
%
% Outputs
% - error_rate (1 x 1): The error rate (between 0 and 1)
%
function error_rate = CalculateErrorRate(Y_pred,y_label)

    error = 0;
    [N,K] = size(Y_pred);
    for i=1:N
        j = y_label(i);
        [val, idx] = max(Y_pred(i,:));
       
        if(j+1 ~= idx)
            error = error + 1;
        end
        
    end
    
    error_rate = error/N;
    
    
end