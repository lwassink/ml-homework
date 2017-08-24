function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features

% You need to return the following variables correctly
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

L = eye(n);
L(1, 1) = 0;
d = X*theta - y;
J = 1/(2*m) * d'*d + lambda/(2*m) * theta'*L*theta;
grad = 1/m * X'*d + lambda/m  * L*theta;

% =========================================================================

grad = grad(:);

end
