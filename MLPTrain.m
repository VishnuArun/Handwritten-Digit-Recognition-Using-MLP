%%
% Train the MLP
%
% Input
% - X_trn (N x D): Training datapoints matrix, where N is the 
% number of training data points, and D is the number of features
% - y_trn (N x 1): Vector contains the labels of the 
% training datapoints
% - H (1 x 1): Number of hidden units
%
% Output
% - Y_pred (N x K): Output from the last Forward Propagation
% - Z (N x H+1): Matrix that contains the output from the hidden units,
% including the bias unit z0=+1
% - W (D+1 x H): Weights learned between each input unit and hidden unit
% - V (H+1 x K): Weights learned between each hidden unit and output unit
%
function [Y_pred,Z,W,V] = MLPTrain(X_trn, y_trn, H)
    
    K = 10;
    D = size(X_trn,2);
    maxiter = 1000;
    eta = 0.0005;

    rng(1); % For reproducibility
    W = -0.01 + (0.01+0.01)*rand(D+1,H);
    rng(2); % For reproducibility
    V = -0.01 + (0.01+0.01)*rand(H+1,K);

    % Initial value of Y, Z and the Error function, 
    
    [Y_pred,Z] = ForwardPropagation(X_trn, W, V);
    E = ErrorFunction(y_trn,Y_pred);
    %%%% 
    
    % We are implementing Batch Gradient Descent. We use all the samples
    % at the same time 
    for iter=1:maxiter
        
        % Findinf dW and dV to update W 
        % and V
        
        [dW, dV] = BackwardPropagation(X_trn, y_trn, Y_pred, Z, V, eta);
        V = V + dV;
        W = W + dW;
        %%%%
        
        % Find new Y and Z
        % Calculate new value of error function
        
        [Y_pred,Z] = ForwardPropagation(X_trn, W, V);
        a;
        %%%%
        
        % Check convergence and stop if abs(E_new-E) <= 0.2 : 
        % Converge too quick enough so iter > 50 imposed
       
        if iter > 50 && abs(E - E_new) <= 0.2
            break
        end
        
        E = E_new;
        
        %%%%
        
    end
    
end

