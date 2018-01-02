%%
% Implements the MLP Backward Propagation step
%
% Inputs
% - X (N x D): Training datapoints matrix, where N is the 
% number of training data points, and D is the number of features
% - y_label (N x 1): True labels of each data point
% - Y_pred (N x K): Output of each output unit
% - Z (N x H+1): Matrix that contains the output from the hidden units,
% including the bias unit z0=+1
% - V (H+1 x K): Weights between each hidden unit and output unit
% - eta (1 x 1): The learning rate
%
% Outputs
% - dW (D+1 x H): Updates for the weights in W
% - dV (H+1 x K): Updates for the weights in V
%
function [dW, dV] = BackwardPropagation(X, y_label, Y_pred, Z, V, eta)
    
    [N,D] = size(X);
    [H_1,K] = size(V);
    
    %Convert y_labels from column vecotr to matrix
    [N,K] = size(Y_pred);
    Y_new = zeros(N,K);
    for n=1:N
        i = y_label(n);
        Y_new(n,i + 1) = 1;
    end
    
    y_label = Y_new;
    
            
    
    dV = zeros(H_1,K);
    
    sm = 0;
    for i = 1:K
        for h = 1:H_1
            for t=1:N
                sm = sm + (y_label(t,i) - Y_pred(t,i))*Z(t,h);
         
            end
            dV(h,i) = eta * sm;
            sm = 0;
        end
    end
         
    trn = zeros(N,D+1);
    trn(:,2:end) = X;
    trn(:,1) = 1;
    X = trn;
    
    [N,D_1] = size(X);
    
    dW = zeros(D_1,H_1 - 1);
    %%%% Take into account bias (N x D+1) -> Entire column is sum((y_label - Y_pred(i))*Z(h),2)
    sm = 0;
    tm = 0;
    
    
    
    for d=1:D_1
        for h=1:(H_1 - 1)
            for n=1:N
                for k=1:K
                    sm = sm +(y_label(n,k) - Y_pred(n,k)) * V(h+1,k);
                    
                end
                tm = tm + sm* Z(n,h+1) *(1 - Z(n,h+1)) *X(n,d);
                sm = 0;
                
            end
            dW(d,h) = eta * tm;
            tm = 0;
        end
    end
    
           
end

