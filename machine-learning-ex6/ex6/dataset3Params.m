function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
err_vec = [];
for i = 1:length(C_vec), 
    for j = 1:length(sigma_vec), 
        tmpC = C_vec(i);
        tmpSigma = sigma_vec(j);
        model = svmTrain(X, y, tmpC, @(x1, x2) gaussianKernel(x1, x2, tmpSigma)); 
        predictions = svmPredict(model, Xval);
        err_vec = [err_vec mean(double(predictions ~= yval))];
    end;
end;

disp(err_vec);
[val pos] = min(err_vec);
fprintf("val min = %d at pos = %f\n", val, pos);

C = C_vec(ceil(pos / length(C_vec)));
sigma = sigma_vec(mod(pos, length(sigma_vec)));
fprintf("C = %f\n sigma = %f\n", C, sigma);





% =========================================================================

end
