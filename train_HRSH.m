function [B,XW,YW] = train_HRSH(XTrain,YTrain,LTrain,NLTrain,Lc,hubid,param)
    
    % parameters
    max_iter = param.iter;
    kdim = param.nbits/param.sf;
    omega = param.omega;
    theta = param.theta;
    beta = param.beta;
    alpha = param.alpha;
    lambda = param.lambda;
    mu = param.mu;
    nbits = param.nbits;
    sel_num = 1000;
    if strcmp(param.db_name, 'NUS-WIDE')
        sel_num = 10000;
    end
    
    n = size(LTrain,1);
    nonhubid = setdiff(1:n,hubid)';
    m = size(nonhubid,1);
    dX = size(XTrain,2);
    dY = size(YTrain,2);
    
    NLTrain1 = NLTrain(nonhubid,:); % m*c
    NLTrain2 = NLTrain(hubid,:);    % (n-m)*c
    Lc1 = Lc(nonhubid,:);           % m*l
    Lc2 = Lc(hubid,:);              % (n-m)*l

    % hash code learning
    if kdim < n
        H1 = sqrt(m*nbits/kdim)*orth(rand(m,kdim));             % m*k
        H2 = sqrt((n-m)*nbits/kdim)*orth(rand((n-m),kdim));     % (n-m)*k
        B1 = rsign(H1,nbits,kdim);                              % m*k
        B2 = rsign(H2,nbits,kdim);                              % (n-m)*k
        B = zeros(n,kdim);
        H = zeros(n,kdim);
        
        for i = 1:max_iter
            % update H1
            Z = omega*B1 + param.t*nbits*NLTrain1*(NLTrain1'*B1) + nbits*NLTrain1*(NLTrain2'*B2) + nbits*alpha*Lc1*(Lc1'*B1) + nbits*alpha*Lc1*(Lc2'*B2);
            [~,Lmd,VV] = svd(Z'*Z);
            index = (diag(Lmd)>1e-6);
            V = VV(:,index); V_ = orth(VV(:,~index));
            U = Z *  (V / (sqrt(Lmd(index,index))));
            U_ = orth(randn(m,kdim-length(find(index==1))));
            H1 = sqrt(m*nbits/kdim)*[U U_]*[V V_]';
            
            clear Z Temp Lmd VV index U U_ V V_
            
            % update H2
            Z = omega*B2 + nbits*alpha*Lc2*(Lc2'*B2) + nbits*alpha*Lc2*(Lc1'*B1);
            [~,Lmd,VV] = svd(Z'*Z);
            index = (diag(Lmd)>1e-6);
            V = VV(:,index); V_ = orth(VV(:,~index));
            U = Z *  (V / (sqrt(Lmd(index,index))));
            U_ = orth(randn(n-m,kdim-length(find(index==1))));
            H2 = sqrt((n-m)*nbits/kdim)*[U U_]*[V V_]';

            % update B1
            B1 = rsign(omega*H1 + param.t*nbits*NLTrain1*(NLTrain1'*H1) + nbits*alpha*Lc1*(Lc1'*H1) + nbits*alpha*Lc1*(Lc2'*H2) + theta*m*nbits/kdim*ones(m,kdim),nbits,kdim);
            
            % update B2
            B2 = rsign(omega*H2 + param.t*nbits*NLTrain2*(NLTrain1'*H1) + nbits*alpha*Lc2*(Lc1'*H1) + nbits*alpha*Lc2*(Lc2'*H2) + theta*(n-m)*nbits/kdim*ones(n-m,kdim),nbits,kdim);
            
            B(nonhubid, :) = B1;
            B(hubid,:) = B2;

        end
    end
    clear Z Temp Lmd VV index U U_ V V_ 

    % hash function learning
    if m < sel_num
        sel_idx = hubid;
    else
        sel_idx = randperm(m,sel_num);
    end
    Bs = B(sel_idx,:);
    YW = rand(dY,kdim);
    
    if mu == 0
        max_iter = 1;
    end
    for i = 1:max_iter
        XW = (XTrain'*XTrain+(lambda)*eye(dX))\(XTrain'*B+ mu*XTrain'*YTrain*YW +((XTrain'*NLTrain)*NLTrain(sel_idx,:)')*Bs*beta*nbits)...
        /((1+mu) * eye(kdim)+Bs'*Bs*beta);
        YW = (YTrain'*YTrain+lambda*eye(dY))\(YTrain'*B+ mu*YTrain'*XTrain*XW +((YTrain'*NLTrain)*NLTrain(sel_idx,:)')*Bs*beta*nbits)...
        /((1+mu) * eye(kdim)+Bs'*Bs*beta);
    end
end