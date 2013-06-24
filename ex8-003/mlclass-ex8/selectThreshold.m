function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % As above, set predictions to a 0 and 1 vector
    predictions = (pval < epsilon);

    % Now, go through and use equality testing to sort out the
    % different important quadrants
    truePositive = sum( (yval==1) & (predictions==1) );
    falsePositive = sum( (yval==0) & (predictions==1) );
    falseNegative = sum( (yval==1) & (predictions==0) );

    % Now we use the formulas for precision and recall as found on
    % page 6 in order to calculate the F1Scores.
    precision = truePositive ./ (truePositive+falsePositive);
    recall = truePositive ./ (truePositive+falseNegative);

    % Now calculate the F1Scores
    F1 = (2 .* precision .* recall) / (precision+recall);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
