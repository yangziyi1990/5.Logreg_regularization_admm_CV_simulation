function [Opt,Mse]=CV_distri_SCAD_logistic(X,Y,Lambda,J,beta_path,beta_int,beta_zero)

    [n,p]=size(X);
    K=5;  % 5 k-folds;
    
    %% divide the data into k parts
    XCV=cell(K,1);
    YCV=cell(K,1);
    m=1;q=1;r=1;t=1;z=1;
    for i=1:n
        if mod(i,K)==1
            XCV{1}(m,:)=X(i,:);        
            YCV{1}(m,:)=Y(i,:);
            m=m+1;
        end
        if mod(i,K)==2
            XCV{2}(q,:)=X(i,:);
            YCV{2}(q,:)=Y(i,:);
            q=q+1;
        end
        if mod(i,K)==3
            XCV{3}(r,:)=X(i,:);
            YCV{3}(r,:)=Y(i,:);
            r=r+1;
        end
        if mod(i,K)==4
            XCV{4}(t,:)=X(i,:);
            YCV{4}(t,:)=Y(i,:);
            t=t+1;
        end
        if mod(i,K)==0
            XCV{5}(z,:)=X(i,:);
            YCV{5}(z,:)=Y(i,:);
            z=z+1;
        end
    end
    
    %%  data in the J subsystem
    n = n / J; 
    XJ=cell(J,1);
    YJ=cell(J,1);
    for i=1:J
        XJ{i,1}=X(1+(i-1)*n:i*n,:);
        YJ{i,1}=Y(1+(i-1)*n:i*n);
    end

     %% Cross validation
    for l=1:length(Lambda)
        lambda=Lambda(l);
        for i=1:K
            Xtrain=cell(J,1);   %每个cell是第j个agent的training data
            Ytrain=cell(J,1);
            Xtest=cell(J,1);    %每个cell是第j个agent的testing data
            Ytest=cell(J,1);
            for j=1:J 
                r=0;q=0;           
                for w=1:n       %循环第j个agent的所有数据，看是否在第k部分里
                    m=0;
                    for v=1:n
                        if XCV{i}(v,:)==XJ{j}(w,:)        %数据既在agent j的data里，又在第k部分里
                            m=m+1;
                        end
                    end
                    %%将agent j的数据分为training和testing两部分
                    if m==0   %在XJ{j}中不在XCV{k}中
                        r=r+1;
                        Xtrain{j}(r,:)=XJ{j}(w,:);
                        Ytrain{j}(r,1)=YJ{j}(w);
                    else
                        q=q+1;
                        Xtest{j}(q,:)=XJ{j}(w,:);
                        Ytest{j}(q,1)=YJ{j}(w);
                    end
                end
            end
            X_train=cell2mat(Xtrain);
            Y_train=cell2mat(Ytrain);
            X_test=cell2mat(Xtest);            
            Y_test=cell2mat(Ytest);
            [b history] = distr_SCAD_logreg(X_train, Y_train, lambda,J,1.0, 1.0);
            
            l_test = X_test * beta_int + beta_zero;         % no noise
            [n_size,p_size]=size(X_test);
            prob_test=exp(l_test)./(1 + exp(l_test));
            for t=1:n_size
                if prob_test(t)>0.5
                    Y_test(t,1)=1;
                else
                    Y_test(t,1)=0;
                end
            end
            % the performance of testing data %
            y_validation=X_test * b(2:end) + b(1);
            prob_validation=exp(y_validation)./(1 + exp(y_validation));
            for t=1:n_size
                if prob_validation(t)>0.5
                    Y_validation(t,1)=1;
                else
                    Y_validation(t,1)=0;
                end
            end
            error=abs(Y_validation-Y_test);
            Mse(i,l)=sum(error); 
            y_test=0;
        end
    end
    min_Mse_index=find(sum(Mse,1)==min(sum(Mse,1)));
    Opt=max(min_Mse_index);
    %[d,Opt]=min(sum(Mse,1));
    Mse=sum(Mse,1);
    
end