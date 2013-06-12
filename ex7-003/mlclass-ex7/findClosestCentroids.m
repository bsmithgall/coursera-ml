function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% We are going to loop through every row of X
for i=1:size(X,1),

    % Initialize a vector of zeros with length K. These will be
    % populated by a nested loop that calculates the distance from
    % each centroid.
    distVector = zeros(K,1);

    % Loop through all of the values of K to populate distVector as
    % per the equations given on page 3 of ex7.pdf.
    for k=1:K,
	dist = X(i,:) - centroids(k,:);
	distVector(k) = dist*dist';
    endfor

    % Now that we've stored the values, we can use the built-in min
    % function to assign the minimum value's index to idx and complete
    % the function

    [value, index] = min(distVector);
    idx(i) = index;
endfor


% =============================================================

end

