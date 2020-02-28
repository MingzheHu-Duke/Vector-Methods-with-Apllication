    % Load data and avoid reloading every time
    if ~exist('mnist_load','var')
        mnist = csvread('mnist_train.csv', 1, 0);
        mnist_load=1;
    end

    clc
    close all
    warning('off')
    
    %% Exercise 1
    % Let U and W be subspaces of R^5 that are spanned, respectively, by
    % the columns of the matrices A and $B$ (shown below)
    % Write a function `altproj(A,B,v0,n)` that returns v_{2n} after 2n 
    % steps of alternating projection onto U and W starting from v_0.
    % Use this function to find the orthogonal projection of v0 (shown 
    % below) onto $ U \cap W $. 
    % How large should n be chosen so that the projection is correct to 4 
    % decimal places (e.g., absolute error at most 0.0001 in each 
    % coordinate)?
    A = [3, 2, 3; 1, 5, 7; 4, 11, 13; 1, 17, 19; 5, 23, 29];
    B = [1, 1, 2.5; 2, 0, 6; 2, 1, 12; 2, 0, 18; 6, -3, 26];
    v0 = [1; 2; 3; 4; 5];
    n = 20;
    [~, err] = altproj(A, B, v0, n);
    
    figure(1)
    semilogy(1:n, err)
    title('Exercise 1')
    
    %% Exercise 2
    % Write a function `kaczmarz(A,b,I)` that returns a matrix X with I 
    % columns corresponding to the Kaczmarz iteration after i = 1, ..., I 
    % full passes through the Kaczmarz algorithm for the matrix A and 
    % right-hand side b (e.g., one full pass equals m steps). 
    % Use this function to find the minimum-norm solution of linear system 
    % Ax = b
    A = [2, 5, 11, 17, 23; 3, 7, 13, 19, 29];
    b = [228; 227];
    I = 500;
    [~, err] = kaczmarz(A, b, I);
    
    figure(2)
    semilogy(1:I, err)
    title('Exercise 2')
    
    %% Exercise 3
    % Repeat the experiment with I=100 for a random system defined by 
    % `A = randn(500,1000)` and `b = A * randn(1000,1)`. 
    % Compare the iterative solution with the true minimum-norm solution 
    % x_hat = A^H {(A A^H)}^{-1} b $.
    A = randn(500, 1000);
    b = A * randn(1000, 1);
    I = 100;
    [X, err] = kaczmarz(A, b, I);
    
    x_hat = 
    diff = 
    figure(3)
    semilogy(1:I, err)
    title(sprintf('Exercise 3, norm of x - x_hat is %.4g', diff))
    
    %% Exercise 4
    % Consider the linear program
    % min c^T x   subject to   A x >= b,   x >= 0
    % with c, A, b given below.
    % Let p^* denote the optimum value of this program.
    % Then, p^* <=e 0 is satisfied if and only if there is a non-negative 
    % x = [x_1, x_2, x_3]^T satisfying
    %      2x_1 -  x_2 +  x_3 >= -1
    %       x_1        + 2x_3 >=  2
    %     -7x_1 + 4x_2 - 6x_3 >=  1
    %     -3x_1 +  x_2 - 2x_3 >=  0
    % where the last inequality restricts the value of the program to be at
    % most 0. One can find the optimum value p and an optimizer x with the 
    % command 
    %     [x,p]=linprog(c,-A,-b,[],[],zeros(1,length(c)),[])
    %
    % Starting from x_0=0, write a program that uses alternating projection
    % onto half spaces (see (6)) to find a non-negative vector satisfying 
    % the above inequalities. 
    %
    % Write a function `lp_altproj(A,b,I)` that uses alternating projection
    % with I passes through entire set of inequality constraints, to find a 
    % non-negative vector x that satisfies A x >= b.
    % 
    % Warning: don?t forget to also project onto the half spaces defined by
    % the non-negativity constraints x_1 >= 0, x_2 >= 0, x_3 >= 0. 
    % 
    % Use the result to find a vector that satisfies all the inequalities. 
    % How many iterations are required so that the absolute error is at 
    % most 0.0001 in each coordinate?
    c = [3; -1; 2];
    A = [2, -1, 1; 1, 0, 2; -7, 4, -6];
    b = [-1; 2; 1];
    
    % Do not forget constraint xi >= 0
    A1 = 
    b1 = 
    I = 100;
    [x, err] = lp_altproj(A1, b1, I);
    
    figure(4)
    semilogy(1:I, err)
    title(sprintf('Exercise 4, min of Ax - b is %.4g, min of x is %.4g', min(A*x - b), min(x)))
    
    %% Exercise 5
    % Consider the ?random? convex optimization problem defined by c, A, b
    % below. Modify A and b (by adding one row and one element) so that 
    % your function can be used to prove that the value of the convex
    % optimization problem, in (3), is at most ?1000. Try using I=1000 
    % passes through all 501 inequality constraints.
    % This type of iteration typically terminates with an ?almost feasible?
    % x. To find a strictly feasible point, try running the same algorithm 
    % with the argument b + epsilon for some small epsilon > 0  (e.g., try 
    % epsilon = 1e-6). Then, the resulting x can satisfy `all(A*x - b > 0)`
    rng(0, 'twister');
    c = randn(1000, 1);
    A = [ones(1, 1000); randn(500, 1000)];
    b = [-1000; A(2:end, :) * rand(1000, 1)];
    
    % Do not forget constraint xi >= 0 and c^T x <= -1000
    A1 = 
    b1 = 
    I = 1000;
    [x, err] = lp_altproj(A1, b1, I);
    
    figure(5)
    semilogy(1:I, err)
    title(sprintf('Exercise 5, min of Ax - b is %.4g, min of x is %.4g, c^T x is %.4g', min(A*x - b), min(x), dot(c, x)))

    %% Exercise 6
    % Repeat the MNIST training exercise from the Least-Squares Handout 
    % using the training method described above. First, extract the indices 
    % of all the 0?s and randomly separate the samples into equal-sized 
    % training and testing groups. Second, do the same for the 1?s. Now, 
    % extend each vector to length 785 by appending a ?1. This will allow 
    % the system to learn a general hyperplane separation.
    %
    % Next, use alternating projections to design a linear classifier to 
    % separate for 0?s and 1?s. For the resulting linear function, report 
    % the classification error rate and confusion matrices for the both the 
    % training and test sets. Also, for the test set, compute the histogram
    % of the function output separately for each class and then plot the 
    % two histograms together. This shows easy or hard it is to separate 
    % the two classes.
    % 
    % Depending on your randomized separation into training and test sets, 
    % the training data may or may not be linearly separable. Comment on 
    % what happens to the test set performance when the error rate does 
    % converge to zero for the training set.
    fprintf('\nExercise 6\n')
    solver = @ (A, b) lp_altproj(A, b + 1e-6, 100);
    mnist_pairwise_altproj(mnist, 0, 1, solver, 0.5, true);

    %% Exercise 7
    % Describe how this approach should be extended to multi-class linear 
    % classification (parameterized by X in R^{n x d}) where the classifier 
    % maps a vector v to class j if the j-th element of X^T v is the 
    % largest element in the vector. Use the implied alternating-projection 
    % solution to design a multi-class classifier for MNIST.
    fprintf('\nExercise 8\n')
    solver = @ (A, b) lp_altproj(A, b + 1e-6, 100);
    mnist_multiclass_altproj(mnist, solver, 0.5);

    %% Exercise 8 (Optional)
    % Let V = R^2 and consider the orthogonal projection of u = [2, -2]^T
    % onto the intersection of
    %     C_1 = { v in V | v_2 >= 0 }
    %     C_2 = { v in V | v_1^2 + (v_2 - sqrt(3)/2)^2 <= 1 }
    % Draw a picture illustrating the algernating projections (without 
    % Dykstra's modification) defined by: 
    %     P_{C_2}( P_{C_1}(u) ) and P_{C_1}( P_{C_2}(u) )
    % Does either give the desired result $ P_{C_1 \cap C_2}(\underline{u}) $?
    % Now, try Dykstra's algorithm using both orders and 4 iterations.
    % Are these approaching $ P_{C_1 \cap C_2}(\underline{u}) $?
    % Define P_C1 and P_C2
    P_C1 = @ (u) proj_HS(u, [0; 1], 0);
    P_C2 = @ (u) proj_NB(u, 1, [0; sqrt(3)/2]);
    Ps = {P_C1, P_C2};
    
    % Trajectory of P_C2(P_C1(u))
    us1 = [1, 1; -2, -2];
    while true
        u = 
        if norm(us1(:, end-1) - u) >= 1e-4
            us1 = [us1, u];
        else
            break
        end
    end
    
    % Trajectory of P_C1(P_C2(u))
    us2 = [1, 1; -2, -2];
    while true
        u = 
        if norm(us2(:, end-1) - u) >= 1e-4
            us2 = [us2, u];
        else
            break
        end
    end
    
    % Trajectory of Dykstra's algorithm
    vs = [1, 1; -2, -2];
    ws = zeros(2);
    while true
        v = 
        ws = 
        if norm(vs(:, end-1) - v) >= 1e-4
            vs = [vs, v];
        else
            break
        end
    end
    
    figure('rend', 'painters', 'pos', [10 10 900 600])
    subplot(1, 2, 1)
    hold on
    plot(us1(1, :), us1(2, :), 'r+')
    plot(us2(1, :), us2(2, :), 'bx')
    plot(vs(1, :), vs(2, :), 'gv')
    
    x = linspace(-0.5, 0.5, 101);
    plot(x, zeros(size(x)), 'y-', 'LineWidth', 2)
    x = linspace(-1, 1, 101);
    y = sqrt(3) / 2 + sqrt(1 - x.^2);
    plot(x, y, 'y-', 'LineWidth', 2)
    x = linspace(-1, -0.5, 101);
    y = sqrt(3) / 2 - sqrt(1 - x.^2);
    plot(x, y, 'y-', 'LineWidth', 2)
    x = linspace(0.5, 1, 101);
    y = sqrt(3) / 2 - sqrt(1 - x.^2);
    plot(x, y, 'y-', 'LineWidth', 2)
    axis equal
    legend({'P_{C_2}(P_{C_1}(u))', 'P_{C_1}(P_{C_2}(u))', 'Dykstra'}, ...
        'location', 'South', 'FontSize', 15)
    
    subplot(1, 2, 2)
    hold on
    plot(us1(1, :), us1(2, :), 'r+', 'MarkerSize', 7)
    plot(us2(1, :), us2(2, :), 'bx', 'MarkerSize', 7)
    plot(vs(1, :), vs(2, :), 'g*', 'MarkerSize', 7)
    
    plot(us1(1, end), us1(2, end), 'ro', 'MarkerSize', 10)
    plot(us2(1, end), us2(2, end), 'bo', 'MarkerSize', 10)
    plot(vs(1, end), vs(2, end), 'go', 'MarkerSize', 10)
    
    x = linspace(-0.5, 0.5, 101);
    plot(x, zeros(size(x)), 'y-', 'LineWidth', 2)
    x = linspace(-1, 1, 101);
    y = sqrt(3) / 2 + sqrt(1 - x.^2);
    plot(x, y, 'y-', 'LineWidth', 2)
    x = linspace(-1, -0.5, 101);
    y = sqrt(3) / 2 - sqrt(1 - x.^2);
    plot(x, y, 'y-', 'LineWidth', 2)
    x = linspace(0.5, 1, 101);
    y = sqrt(3) / 2 - sqrt(1 - x.^2);
    plot(x, y, 'y-', 'LineWidth', 2)
    axis([0.25, 1.05, -0.35, 0.45])
    legend({'P_{C_2}(P_{C_1}(u))', 'P_{C_1}(P_{C_2}(u))', 'Dykstra'}, ...
        'location', 'South', 'FontSize', 15)


function [v, err] = altproj(A, B, v0, n)
% Arguments:
%     A -- matrix whose column span is vector space U
%     B -- matrix whose column span is vector space W
%     v0 -- initialization vector
%     n -- number of sweeps for alternating projection
% Returns:
%     v -- the output after 2n steps of alternating projection
%     err -- the error after each full pass
    % Construct projection matrix
    PU = 
    PW = 
    
    % Compute the exact solution
    basis_UintW = 
    P_UintW = 
    v_star = 
    
    % Apply n full pass of alternating projection
    v = v0;
    err = zeros(1, n);
    for t = 1:n
        v = 
        err(t) = max(v - v_star);
    end
end

function [X, err] = kaczmarz(A, b, I)
% Arguments:
%     A -- matrix defines the LHS of linear equation
%     b -- vector defines the RHS of linear equation
%     I -- number of full passes through the Kaczmarz algorithm
% Returns:
%     X -- the output of all I full passes
%     err -- the error after each full pass
    [m, n] = size(A);
    v = zeros(n, 1);
    X = zeros(n, I);
    err = zeros(1, I);
    for i = 1:I
        for j = 1:m
            aj = 
            bj = 
            v = 
        end
        X(:, i) = v;
        err(i) = 
    end
end

function p = proj_HS(v, w, c)
% Projection on half space defined by {v| <v,w> = c}
% Arguments:
%     v -- vector to be projected
%     w -- norm vector of hyperplane
%     c -- intercept
% Returns:
%     p -- orthogonal projection of x on half-space <v|w> >= c
    
end

function p = proj_NB(v, a, v0)
% Projection on norm ball defined by {v| <v-v0|v-v0> <= a^2}
% Arguments:
%     v -- vector to be projected
%     a -- radius of the norm ball
%     v0 -- center of the norm ball
% Returns:
%     p -- orthogonal projection of x on norm ball <v-v0|v-v0> <= a^2
    
end

function [v, err] = lp_altproj(A, b, I)
% Find a feasible solution for A v >= b using alternating projection
% Arguments:
%     A -- matrix defines the LHS of linear equation
%     b -- vector defines the RHS of linear equation
%     I -- number of full passes through the alternating projection
% Returns:
%     v -- the output after I full passes
%     err -- the error after each full pass
    [m, n] = size(A);
    % Apply I sweeps of alternating projection
    v = zeros(n, 1);
    err = zeros(1, I);
    for i = 1:I
        for j = 1:m
            aj = 
            bj = 
            v = 
        end
        err(i) = 
    end
end

function [X_tr, X_te, y_tr, y_te] = extract_and_split(mnist, d, test_size)
% extract the samples with given lables and randomly separate the samples 
% into equal-sized training and testing groups, extend each vector to 
% length 785 by appending a ?1
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     d -- digit needs to be extracted, can be 0, 1, ..., 9
%     test_size -- the fraction of testing set
% Returns:
%     X_tr -- training set features, a matrix with 785 columns
%             each row corresponds the feature of a sample
%     y_tr -- training set labels, 1d-array
%             each element corresponds the label of a sample
%     X_te -- testing set features, a matrix with 785 columns 
%             each row corresponds the feature of a sample
%     y_te -- testing set labels, 1d-array
%             each element corresponds the label of a sample
    
end

function x_filtered = remove_outlier(x)
% returns points that are not outliers to make histogram prettier
% reference: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564
% Arguments:
%     x -- 1d-array, points to be filtered
% Returns:
%     x_filtered -- 1d-array, filtered points after dropping outlier
    diff = abs(x - median(x));
    modified_z_score = 0.6745 * diff / median(diff);
    x_filtered = x(modified_z_score <= 3.5);
end

function [z_hat, err_tr, err_te] = mnist_pairwise_altproj(mnist, a, b, solver, test_size, verbose)
% Pairwise experiment for applying alternating projection to classify digit 
% a, b
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     a, b -- digits to be classified
%     solver -- solver function to return coefficients of linear classifier
%     test_size -- the fraction of testing set
%     verbose -- whether to print and plot results
% Returns:
%     z_hat -- coefficients of linear classifier
%     err_tr -- training set classification error
%     err_te -- testing set classification error

    % Find all samples labeled with digit a and split into train/test sets
    [Xa_tr, Xa_te, ya_tr, ya_te] = 
    
    % Find all samples labeled with digit b and split into train/test sets
    [Xb_tr, Xb_te, yb_tr, yb_te] = 
    
    % Construct the full training set
    X_tr = 
    y_tr = 
    
    % Construct the full testing set
    X_te = 
    y_te = 
    
    % Run solver on training set to get linear classifier
    A_tilde = 
    z_hat = 
    
    % Compute estimation and misclassification on training set
    y_hat_tr = 
    err_tr = 
    
    % Compute estimation and misclassification on training set
    y_hat_te = 
    err_te = 
    
    if verbose
        fprintf('Pairwise experiment, mapping %d to -1, mapping %d to 1\n', a, b)
        fprintf('training error = %.2f%%, testing error = %.2f%%\n', 100 * err_tr, 100 * err_te)
        
        % Compute confusion matrix for training set
        cm_tr = 
        fprintf('Confusion matrix for training set:\n')
        disp(cm_tr)
        
        % Compute confusion matrix for testing set
        cm_te = 
        fprintf('Confusion matrix for testing set:\n')
        disp(cm_te)
        
        % Compute the histogram of the function output separately for each 
        % class, then plot the two histograms together
        ya_te_hat = 
        yb_te_hat = 
        
        % Remove outlier to make pretty histogram
        ya_te_hat = remove_outlier(ya_te_hat);
        yb_te_hat = remove_outlier(yb_te_hat);
        
        figure()
        hold on
        histogram(ya_te_hat, 50, 'facecolor', 'red')
        histogram(yb_te_hat, 50, 'facecolor', 'blue')
        title(sprintf('histogram of pairwise experiment on %d and %d', a, b))
    end
end

function [Z, err_tr, err_te] = mnist_multiclass_altproj(mnist, solver, test_size)
% Experiment for applying least-square to classify all digits using one-hot 
% encoding
% Arguments:
%     mnist -- the MNIST data set read from csv file
%     solver -- solver function to return coefficients of linear classifier
%     test_size -- the fraction of testing set
% Returns:
%     Z -- coefficients of linear classifier
%     err_tr -- training set classification error
%     err_te -- testing set classification error
    % Split into training/testing set
    
    
    % Construct the training set
    X_tr = 
    y_tr = 
    
    % Construct the testing set
    X_te = 
    y_te = 
    
    % Run alternating projection on training set for each digit
    A_tilde = 
    b_tilde = 
    Z = 
    % Reshape Z to 785 x 10
    Z = 
    
    % Compute estimation and misclassification on training set
    [~, y_hat_tr] = 
    y_hat_tr = 
    err_tr = 
    
    % Compute estimation and misclassification on training set
    [~, y_hat_te] = 
    y_hat_te = 
    err_te = 
    
    fprintf('training error = %.2f%%, testing error = %.2f%%\n', 100 * err_tr, 100 * err_te)
    % Compute confusion matrix
    cm_tr = 
    fprintf('Confusion matrix for training set:\n')
    disp(cm_tr)
    
    cm_te = 
    fprintf('Confusion matrix for testing set:\n')
    disp(cm_te)
    
    figure()
    subplot(1, 2, 1)
    imagesc(cm_tr)
    axis('equal')
    title('Confusion matrix of training set')
    subplot(1, 2, 2)
    imagesc(cm_te)
    axis('equal')
    title('Confusion matrix of testing set')
end
