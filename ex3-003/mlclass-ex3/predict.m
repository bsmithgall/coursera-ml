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

% In order to do the vectorized implementation of the neural network,
% we need to calculate z1, z3, a2, and a3, and then use the same
% implementation as used in predictManyToOne.m to get the final output

% First, add a column of ones to input X
X = [ones(m,1) X];

% Now calculate z1 (Theta1 is read in with the data)
z1 = Theta1 * X';
% Implement sigmoid function on z1 as per vectorized NN procedure
z1 = sigmoid(z1);
% Now add the bias node to z1 to calculate a2
a2 = [ones(1, size(z1, 2)); z1];
% Now calculate z2 and use that to calculate a3
z2 = Theta2 * a2;
a3 = sigmoid(z2);
% Transpose the matrix and use the procedure from predictOneVsAll.m
a3 = a3';

for i=1:size(a3,1),
    [x, ix] = max(a3(i,:));
    p(i) = ix;
endfor


% =========================================================================


end
