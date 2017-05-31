function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% add column 1
X = [ones(m, 1) X];
%Theta1 = [ones(size(Theta1, 1), 1), Theta1];
%Theta2 = [ones(size(Theta2, 1), 1), Theta2];


fprintf("size theta1: %d %d\n", size(Theta1, 1), size(Theta1, 2));
fprintf("size theta2: %d %d\n", size(Theta2, 1), size(Theta2, 2));
fprintf("size X: %d %d\n", size(X, 1), size(X, 2));

% to the hidden layer
hidden = sigmoid(Theta1 * X');

% add column 1
hidden_t = [ones(size(X, 1), 1) hidden'];
% to the ouput layer
fprintf("size hidden_t: %d %d\n", size(hidden_t, 1), size(hidden_t, 2));
output = sigmoid(Theta2 * hidden_t');

fprintf("size ouput: %d %d\n", size(output, 1), size(output, 2));
[x, p] = max(output, [], 1);
% =========================================================================


end
