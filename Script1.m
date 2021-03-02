data=load('ex2data1.txt');
X=data(:,[1,2]);
y=data(:,3);

plotData(X,y);

xlabel('Exam 1 score')
ylabel('Exam 2 score')

legend('Admitted', 'Not admitted')

%  Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display the initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
disp('Gradient at initial theta (zeros):'); disp(grad);

