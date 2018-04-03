function [ Opt,Mse ] = CV_SCAD_logistic(X,y,Lambda,beta_path,beta_int,beta_zero)

%%%%%%%%%%%%%%     K cross validation    %%%%%%%%%%%%%%%%
k=10;            %%% K-fold%%%
[n,p] = size(X);
valida_n=floor(n/k);
sample_sequence=1:n;

for j=1:length(Lambda)
    lambda=Lambda(j);
    beta_ini=beta_path(:,j);
    for i=1:k
      if i<=k-1
          validation_seq=sample_sequence(:,(i-1)*valida_n+1:i*valida_n);
      else
          validation_seq=sample_sequence(:,(i-1)*valida_n+1:n);
      end
      train_seq=setdiff(sample_sequence,validation_seq);
      X_train = X(train_seq,:);
      y_train = y(train_seq);
      X_validation= X(validation_seq, :);
      y_validation = y(validation_seq);
      [b history] = SCAD_logreg(X_train, y_train, lambda, 1.0, 1.0);
%       y_test = sign(X_validation*b(2:end) + b(1));
%       error=abs(y_test-y_validation)/2;
      l_test = X_validation * beta_int + beta_zero;         % no noise
      [n_size,p_size]=size(X_validation);
      prob_test=exp(l_test)./(1 + exp(l_test));
      for t=1:n_size
          if prob_test(t)>0.5
              Y_test(t,1)=1;
          else
              Y_test(t,1)=0;
          end
      end
       % the performance of testing data %
      y_validation=X_validation * b(2:end) + b(1);
      prob_validation=exp(y_validation)./(1 + exp(y_validation));
      for t=1:n_size
          if prob_validation(t)>0.5
              Y_validation(t,1)=1;
          else
              Y_validation(t,1)=0;
          end
      end
      error=abs(Y_validation-Y_test);
      Mse(i,j)=sum(error); 
      y_test=0;
    end
end
min_Mse_index=find(sum(Mse,1)==min(sum(Mse,1)));
Opt=max(min_Mse_index);
% [d,opt_s]=min(sum(Mse,1));
Mse=sum(Mse,1);
end
