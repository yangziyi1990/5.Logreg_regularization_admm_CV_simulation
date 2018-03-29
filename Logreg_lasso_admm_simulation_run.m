% L1 regularized logistic regression (not distributed)
clc
clear all;

%% Generate problem data
% rand('seed', 0);
% randn('seed', 0);

n = 500;   % the number of samples;
p = 2000;  % the number of features

beta_int=zeros(1,p)';
beta_int(1)=1;
beta_int(2)=-1;
beta_int(3)=1;
beta_int(4)=-1;
beta_int(5)=1;
beta_int(6)=-1;
beta_int(7)=1;
beta_int(8)=-1;
beta_int(9)=1;
beta_int(10)=-1;
% beta_int = sprandn(p, 1, 0.01);       % N(0,1), 10% sparse
beta_zero = randn(1);                   % random intercept
beta_true = [beta_zero; beta_int];

% mu=zeros(p,1);
% sigma=ones(p,p);
% for i=1:1:p
%     for j=1:1:p
%         if i~=j
%             sigma(i,j)=0.5;
%         end
%     end
% end
% X=mvnrnd(mu,sigma,n);  % add correlationships
X = sprandn(n, p, 0.1);
Y = sign(X*beta_int + beta_zero);
% noise is function of problem size use 0.1 for large problem
% Y = sign(X*beta_int + beta_zero + sqrt(0.1)*randn(n,1)); % labels with noise

A = spdiags(Y, 0, n, n) * X;
ratio = sum(Y == 1)/(n);

%% Setting lambda && Cross validation
lambda_max = 1/n * norm((1-ratio)*sum(A(Y==1,:),1) + ratio*sum(A(Y==-1,:),1), 'inf');
lambda_min=lambda_max * 0.1;
m = 10;
for i=1:m
    lambda(i) = lambda_max*(lambda_min/lambda_max)^(i/m);
    [beta history] = l1_logreg(A, Y, lambda(i), 1.0, 1.0);
    beta_path(:,i)=beta; 
    i
end

[opt,Mse]=CV_Lasso_logistic(A,Y,lambda,beta_path, beta_int, beta_zero);
beta=beta_path(:,opt);

%% without Cross validation
% lambda = 0.1 * 1/n * norm((1-ratio)*sum(A(Y==1,:),1) + ratio*sum(A(Y==-1,:),1), 'inf');
% [beta history] = l1_logreg(A, Y, lambda, 1.0, 1.0);   % (X, Y, lambda, rho, alpha)

%% Solve problem
% generate testing data %
n_test=200;
X_test = randn(n_test, p); 
l_test = X_test * beta_int + beta_zero;         % no noise
prob_test=exp(l_test)./(1 + exp(l_test));
for i=1:n_test
    if prob_test(i)>0.5
        Y_test(i,1)=1;
    else
        Y_test(i,1)=0;
    end
end

% the performance of testing data %
y_validation=X_test * beta(2:end) + beta(1);
prob_validation=exp(y_validation)./(1 + exp(y_validation));
for i=1:n_test
    if prob_validation(i)>0.5
        Y_validation(i,1)=1;
    else
        Y_validation(i,1)=0;
    end
end
error_test=abs(Y_validation-Y_test);
error_number=length(find(nonzeros(error_test)));

%% Performance
[accurancy,sensitivity,specificity]=performance(Y_test,Y_validation);
fprintf('The accurancy of lasso: %f\n' ,accurancy);
fprintf('The sensitivity of lasso: %f\n' ,sensitivity);
fprintf('The specificity of lasso: %f\n' ,specificity);

%% Reporting
% K = length(history.objval);
% 
% h = figure;
% plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
% ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');
% 
% g = figure;
% subplot(2,1,1);
% semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
%     1:K, history.eps_pri, 'k--',  'LineWidth', 2);
% ylabel('||r||_2');
% 
% subplot(2,1,2);
% semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
%     1:K, history.eps_dual, 'k--', 'LineWidth', 2);
% ylabel('||s||_2'); xlabel('iter (k)');
