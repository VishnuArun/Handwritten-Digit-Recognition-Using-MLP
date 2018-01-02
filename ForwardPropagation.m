%%
% Implements the MLP Forward Propagation step. 
%
% Inputs
% - X (N x D): Training datapoints matrix, where N is the 
% number of training data points, and D is the number of features
% - W (D+1 x H): Weights between each input unit and hidden unit
% - V (H+1 x K): Weights between each hidden unit and output unit
%
% Outputs
% - Y (N x K): Output of each output unit
% - Z (N x H+1): Output of each hidden units, including the bias unit z0=+1
%
function [Y,Z] = ForwardPropagation(X, W, V)


    [N,D] = size(X);
    [D_1,H] = size(W);
    Z= zeros(N,H);
    trn = zeros(N,D+1);
    trn(:,2:end) = X;
    trn(:,1) = 1;
    X = trn;
      
    Z = Sigmoid(X * W);
   
    %%% inserted bias at front
    [N,H] = size(Z);
    bias_z = zeros(N,H+1);
    bias_z(:,2:end) = Z;
    bias_z(:,1) = 1;
    Z = bias_z;
    
    
    Y = Z * V;
    
    %Softmax
    [N,K] = size(Y);
    Y_new = zeros(N,K);
    for n=1:N
        
        for k=1:K
            Y_new(n,k) = ( exp(Y(n,k)) / sum(exp(Y(n,:)))  );
        end
    end
    Y = Y_new;
            
        
end

