function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%fprintf("X size: %d\n", size(X));
h_x = X * theta;
penalty = lambda * sum(theta(2:end, :) .^ 2)/ (2 * m);
J = sum((h_x - y) .^2 )/(2*m) + penalty;

%fprintf("XX size: %d %d\n", size(sum((h_x - y) .* X), 1), size(sum((h_x - y) .* X), 2));
%fprintf("theta size: %d %d\n", size(theta, 1), size(theta, 2));

theta(1) = 0;
grad = (1 / m) * sum((h_x - y) .* X) + (lambda .* theta/ m)';
%grad(1) = sum((h_x - y) .* X(1)) / m;







% =========================================================================

grad = grad(:);

end
