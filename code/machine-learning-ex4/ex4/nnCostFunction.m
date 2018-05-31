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
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Initialize the y matrix
yNum = zeros(m, num_labels);
for i = 1:m
  yNum(i, y(i)) = 1;
end

% Begin training
a1 = X;

% 5000 x 25
z2 = (Theta1 * [ones(m, 1) a1]')';
a2 = sigmoid(z2);

% 5000 x 10
z3 = (Theta2 * [ones(m, 1) a2]')';
a3 = sigmoid(z3);

for i = 1:m
  J = J + (-yNum(i, :) * log(a3(i, :))' - (1.- yNum(i, :)) * log(1.- a3(i, :))');
end

J = J / m

% Regularization, except for the 1st item (bias unit)

t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);

regularization = lambda / (2 * m) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));

J = J + regularization;

% backpropagation

% 5000 x 10

d3 = a3-yNum;

% 5000 x 25
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

% 10 x 26 Theta2_grad

% 这里为什么要作这样的处理？感觉与公式并不符合，但结果是对的
a2_with_a0 = [ones(m, 1) a2];

D2 = d3' * a2_with_a0;

Theta2_grad = D2 / m;

regularization = lambda / m * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta2_grad = Theta2_grad + regularization;

% 25 x 401 Theta1_grad

a1_with_a0 = [ones(m, 1) a1];

D1 = d2' * a1_with_a0;

Theta1_grad = D1 / m;

regularization = lambda / m * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];

Theta1_grad = Theta1_grad + regularization;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
