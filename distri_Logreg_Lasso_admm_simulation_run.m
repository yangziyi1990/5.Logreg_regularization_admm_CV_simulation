% Distributed L1 regularized logistic regression
% (compared against l1_logreg package)
clc;
clear all;
%% Generate problem data
% rand('seed', 0);
% randn('seed', 0);

n = 50;  % the number of samples;
p = 100;  % the number of features
N = 5;  % the number of agent;

% beta_int = sprandn(p, 1, 0.01);       % N(0,1), 10% sparse
beta_int=zeros(1,p)';
beta_int(1)=5;
beta_int(2)=-5;
beta_int(3)=5;
beta_int(4)=-5;
beta_int(5)=5;
beta_int(6)=-5;
beta_int(7)=5;
beta_int(8)=-5;
beta_int(9)=5;
beta_int(10)=-5;
beta_zero = randn(1);                   % random intercept
beta_true = [beta_zero; beta_int];

X0 = sprandn(n*N, p, 0.1);           % data / observations
Y0 = sign(X0*beta_int + beta_zero);
% noise is function of problem size use 0.1 for large problem
% Y0 = sign(X0*beta_int + beta_zero + sqrt(0.1)*randn(n*N, 1)); % labels with noise

% packs all observations in to an m*N x n matrix
A0 = spdiags(Y0, 0, n*N, n*N) * X0;
ratio = sum(Y0 == 1)/(n*N);

%% Setting lambda && Cross validation
lambda_max = 1/(n*N) * norm((1-ratio)*sum(A0(Y0==1,:),1) + ratio*sum(A0(Y0==-1,:),1), 'inf');
lambda_min = lambda_max * 0.1;
m = 10;
for i=1:m
    lambda(i) = lambda_max*(lambda_min/lambda_max)^(i/m);
    [beta history] = distr_l1_logreg(A0, Y0, lambda(i), N, 1.0, 1.0);   % (X, Y, lambda, N, rho, alpha)
    beta_path(:,i)=beta; 
end

[opt,Mse]=CV_distri_Lasso_logistic(A0, Y0, lambda, N, beta_path, beta_int, beta_zero);
beta=beta_path(:,opt);

%% without Cross validation
% lambda = 0.1 * 1/(n*N) * norm((1-ratio)*sum(A0(Y0==1,:),1) + ratio*sum(A0(Y0==-1,:),1), 'inf');
% [beta history] = distr_l1_logreg(A0, Y0, lambda, N, 1.0, 1.0);    % (X, Y, lambda, N, rho, alpha)

%% Solve problem
% generate testing data %
n_test=50;
% X_test = randn(n_test*N, p); 
X_test = sprandn(n*N, p, 0.1); 
l_test = X_test * beta_int + beta_zero;         % no noise
prob_test=exp(l_test)./(1 + exp(l_test));
for i=1:n_test*N
    if prob_test(i)>0.5
        Y_test(i,1)=1;
    else
        Y_test(i,1)=0;
    end
end

% the performance of testing data %
y_validation=X_test * beta(2:end) + beta(1);
prob_validation=exp(y_validation)./(1 + exp(y_validation));
for i=1:n_test*N
    if prob_validation(i)>0.5
        Y_validation(i,1)=1;
    else
        Y_validation(i,1)=0;
    end
end
error_test=abs(Y_validation-Y_test);
error_number=find(nonzeros(error_test));
% [beta history] = distr_l1_logreg(X_test, Y_test, lambda, N, 1.0, 1.0);
% 
% index_nonzero_beta=find(beta~=0);
% number_nonzero_beta=length(nonzeros(beta));
% index_nonzero_betatrue=find(beta_true~=0);
% index_right_beta=intersect(index_nonzero_beta,index_nonzero_betatrue);
% number_right_beta=length(index_right_beta);
% 
% %% Reporting
% 
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
