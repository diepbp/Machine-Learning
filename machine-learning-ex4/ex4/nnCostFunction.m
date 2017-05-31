function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end;

% add column 1
X = [ones(m, 1) X];
%fprintf('Size X: %d %d\n', size(X, 1), size(X, 2));
%fprintf('Size Theta1: %d %d\n', size(Theta1, 1), size(Theta1, 2));
z2 = X * Theta1';

a2 = sigmoid(z2);
% add column 1
a2 = [ones(size(a2, 1), 1) a2];

%fprintf('Size a2: %d %d\n', size(a2, 1), size(a2, 2));
%fprintf('Size Theta2: %d %d\n', size(Theta2, 1), size(Theta2, 2));
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;

%fprintf('Size a3: %d %d\n', size(a3, 1), size(a3, 2));
J = 1 / m * sum(sum((0 - Y) .* log(h) - (1 - Y) .* log(1 - h)), 2);

%Theta1 = Theta1(:, 2:end);
%Theta2 = Theta2(:, 2:end);


regularizedCost = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end) .^2, 2)));
%fprintf('regularizedCost: %f\n', regularizedCost);

J = J + regularizedCost; 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


Sigma3 = a3 - Y;
%fprintf('Sigma3: %d %d\n', size(Sigma3, 1), size(Sigma3, 2));
%fprintf('Theta2: %d %d\n', size(Theta2, 1), size(Theta2, 2));
%fprintf('sigmoidGradient z2: %d %d\n', size(sigmoidGradient(z2), 1), size(sigmoidGradient(z2), 2));

Sigma2 = Sigma3 * Theta2(:, 2:end) .* sigmoidGradient(z2);
%fprintf('Sigma2: %d %d\n', size(Sigma2, 1), size(Sigma2, 2));
%fprintf('a2: %d %d\n', size(a2, 1), size(a2, 2));

Delta2 = Sigma3' * a2;
Delta1 = Sigma2' * X;
%fprintf('Delta2: %d %d\n', size(Delta2, 1), size(Delta2, 2));
%fprintf('Delta1: %d %d\n', size(Delta1, 1), size(Delta1, 2));

Theta1_grad = Delta1 ./m;
Theta2_grad = Delta2 ./ m;

%fprintf('Theta1: %d %d\n', size(Theta1, 1), size(Theta1, 2));
%fprintf('Theta2: %d %d\n', size(Theta2, 1), size(Theta2, 2));
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

tmpTheta1 = Theta1(:, 2:end);
tmpTheta1 = [zeros(size(Theta1, 1), 1) tmpTheta1];

tmpTheta2 = Theta2(:, 2:end);
tmpTheta2 = [zeros(size(Theta2, 1), 1) tmpTheta2];

Theta1_grad = Delta1 ./m + lambda * tmpTheta1 / m;
Theta2_grad = Delta2 ./m + lambda * tmpTheta2 / m;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
