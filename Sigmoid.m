%%
% Sigmoid function
%
function Y = Sigmoid(X)
    Y = power(1+exp(-X), -1);
end