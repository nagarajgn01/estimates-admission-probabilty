function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hyp = sigmoid(X*theta);
sec = (1-y).*log(1-hyp); 
fir = y.*log(hyp);
equ = (-1.*fir)-(sec);
partOne=sum(equ)/m;
thetaNew=theta(2:end,:);
partTwo=lambda*sum(thetaNew.^2)/(2*m);
J=partOne+partTwo;

for i = 1:size(theta)
    if i==1
        grad(i,1)=sum((hyp-y).*X(:,i));
    else
        grad(i,1)=(sum((hyp-y).*X(:,i)))+(lambda*theta(i));
    end
end
grad=grad/m;



% =============================================================

end
