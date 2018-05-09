% Lhalf regularized logistic regression (not distributed)
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
Y_origin=Y;
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
    [beta history] = lhalf_logreg(A, Y, lambda(i), 1.0, 1.0);
    beta_path(:,i)=beta; 
    i
end

[opt,Mse]=CV_Lhalf_logistic(A,Y,lambda,beta_path,beta_int,beta_zero);
beta=beta_path(:,opt);

%% without Cross validation
% lambda = 0.1 * 1/n * norm((1-ratio)*sum(A(Y==1,:),1) + ratio*sum(A(Y==-1,:),1), 'inf');
% [beta history] = lhalf_logreg(A, Y, lambda, 1.0, 1.0);   % (X, Y, lambda, rho, alpha)

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
error_number=length(find(nonzeros(error_test)))
beta_non_zero=length(nonzeros(beta))

%% Performance
[accurancy,sensitivity,specificity]=performance(Y_test,Y_validation);
fprintf('The accurancy of testing data (lhalf): %f\n' ,accurancy);
fprintf('The sensitivity of testing data (lhalf): %f\n' ,sensitivity);
fprintf('The specificity of testing data (lhalf): %f\n' ,specificity);


%% performance for training data
l1 = X * beta(2:end) + beta(1);
prob1=exp(l1)./(1 + exp(l1)); 
train_size=n;
for i=1:train_size
    if prob1(i)>0.5
        train_y(i)=1;
    else
        train_y(i)=0;
    end
end
Y_origin(find(Y_origin==-1))=0;

error_train=train_y'-Y_origin;
error_number_train=length(nonzeros(error_train))

[accurancy_train,sensitivity_train,specificity_train]=performance(Y_origin,train_y');
fprintf('The accurancy of training data(lhalf): %f\n' ,accurancy_train);
fprintf('The sensitivity of training data (lhalf): %f\n' ,sensitivity_train);
fprintf('The specificity of training data (lhalf): %f\n' ,specificity_train);

%% performance for beta
[accurancy_beta,sensitivity_beta,specificity_beta]=performance_beta(beta_true,beta);
fprintf('The accurancy of beta (lhalf): %f\n' ,accurancy_beta);
fprintf('The sensitivity of beta (lhalf): %f\n' ,sensitivity_beta);
fprintf('The specificity of beta (lhalf): %f\n' ,specificity_beta);
