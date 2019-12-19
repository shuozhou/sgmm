classdef SpatialGMM
    % SpatialGMM a clustering algorithm by using both information from
    % feature and spatial domain
    %
    % refer to "EARLY PREDICTION OF A ROCKSLIDE LOCATION VIA A SPATIALLY-
    % AIDED GAUSSIAN MIXTURE MODEL", Shuo Zhou et al for more technical
    % details
    % 
    % usage:
    %   - initialize a sGMM object with 3 clusters, 2 spatial components in
    %   each cluster, use full covariance matrix in the feature domain, and
    %   use diagonal covariance in the spatial domain
    %     sgmm = SpatialGMM(3, 2, 0, 1)
    %   - train the model with feature data X_train \in R^{M*N} and and
    %   spatial data S_train \in R^{M*2}
    %     sgmm = sgmm.train(X_train, S_train)
    %   - get prediction from sgmm
    %     [pred, posterior, log_likelihood] = sgmm.pred(X_test, S_test)
    
    
    properties
        const  % structure for constant
        parms  % struceure for model parameters
        p      % 1d array, p(i) -- proportion of i-th cluster
        q      % 2d array, q(i,j) -- proportion of the j-th spatial 
               % component in the i-th cluster 
        mu_x   % 2d array, mu_x(i,:) -- the mean of the i-th cluster
        cov_x  % 1d cell array, cov_x{i} -- covariance of the i-th cluster
        mu_s   % 3d array, mu_s(i,j,:) -- the mean of the j-th spatial 
               % component in the i-th clusteri-th cluster
        cov_s  % 2d cell array, mu_s{i,j} -- the mean of the j-th spatial 
               % component in the i-th clusteri-th cluster
        pos    % posterior of the training data 
        ll     % log-likelihood at each training step
    end
    
    methods
        function self = SpatialGMM(K, L, cov_type_x, cov_type_s, init)
            % SpatialGMM Construct an instance of this class
            %   input:
            %     - K: number of clusters
            %     - L: number of spatial components in each cluster
            %     - cov_type_x: type of covariance matrix in feature domain
            %     - cov_type_s: type of covariance matrix in spatial domain
            %     - init: initial paramters of the model
            %   return:
            %     - a sgmm instance
            self.parms.K = K;
            self.parms.L = L;
            self.parms.cov_type_x = cov_type_x;
            self.parms.cov_type_s = cov_type_s;
            self.const.EPSILON32 = 1e-32;
            self.const.EPSILON = 0.1;
            self.const.FULL = 0;
            self.const.DIAGONAL = 1;
            self.const.CIRCLE = 2;
            self.const.MAX_ITER = 100;
            if nargin > 4
                self.p = init{1};
                self.q = init{2};
                self.mu_x = init{3};
                self.cov_x = init{4};
                self.mu_s = init{5};
                self.cov_s = init{6};
                self.parms.need_init = false;
            else
                self.p = [];
                self.q = [];
                self.mu_x = [];
                self.cov_x = cell(self.parms.K);
                self.mu_s = [];
                self.cov_s = cell(self.parms.K, self.parms.L);
                self.parms.need_init = true;
            end
        end
        
        
        function self = sgmm_init(self, X, S)
            % initialize model parameters
            %   input:
            %     - X: feature data
            %     - S: spatial locations
            %   return:
            %     - self: initialized sgmm object
            gm = fitgmdist(X, self.parms.K, 'RegularizationValue', .01, ...
                'Options',statset('Display','off','MaxIter',200,'TolFun',1e-16));
            self.p = gm.ComponentProportion;
            self.mu_x = gm.mu;
            sig = squeeze(num2cell(gm.Sigma, [1,2]));
            if self.parms.cov_type_x == self.const.CIRCLE
                self.cov_x = cellfun(@(sig_k) mean(diag(sig_k)), sig, ...
                    'UniformOutput', 0);
            elseif self.parms.cov_type_x == self.const.DIAGONAL
                self.cov_x = cellfun(@(sig_k) diag(sig_k)', sig, ...
                    'UniformOutput', 0);
            elseif self.parms.cov_type_x == self.const.FULL
                self.cov_x = sig;
            end

            pos = posterior(gm, X);
            [~,C] = max(pos');
            for k=1:self.parms.K
                ind_k = C==k;
                if sum(ind_k) < 4*self.parms.L
                    [~, ind_k] = maxk(pos(:,k), 4*self.parms.L);
                end
                gm = fitgmdist(S(ind_k,:), self.parms.L, 'RegularizationValue', .01, ...
                    'Options',statset('Display','off','MaxIter',200,'TolFun',1e-16));
                self.q(k,:) = gm.ComponentProportion;
                self.mu_s(k,:,:) = gm.mu;
                sig = squeeze(num2cell(gm.Sigma, [1,2]));
                if self.parms.cov_type_s == self.const.CIRCLE
                    self.cov_s(k,:) = cellfun(@(sig_l) mean(diag(sig_l)), sig, ...
                        'UniformOutput', 0);
                elseif self.parms.cov_type_s == self.const.DIAGONAL
                    self.cov_s(k,:) = cellfun(@(sig_k) diag(sig_k)', sig, ...
                        'UniformOutput', 0);
                elseif self.parms.cov_type_s == self.const.FULL
                    self.cov_s(k,:) = sig;
                end
            end
        end

        
        function self = train(self, X_train, S_train)
            % train sgmm model
            %   input:
            %     - X_train: training feature data
            %     - S_train: training spatial locations
            %   return:
            %     - self: trained sgmm object
            X_train = zscore(X_train);
            S_train = zscore(S_train);
            if self.parms.need_init
                self = sgmm_init(self, X_train, S_train);
            end
                         
            for t=1:self.const.MAX_ITER
                [self.pos, self.ll(t)] = estep(self, X_train, S_train);
                self = mstep(self, X_train, S_train);
            end
            [self.pos, self.ll(t+1)] = estep(self, X_train, S_train);
        end
        
        
        function [pred, pos, ll] = predict(self, X_test, S_test)
            % provide predictions given testing data
            %   input:
            %     - X_test: testing feature data
            %     - S_test: training spatial locations
            %   return:
            %     - pred: predicted clustering assignment
            %     - pos: posterior of testing data
            %     - ll: log-likelihood of testing data
            X_test = zscore(X_test);
            S_test = zscore(S_test);       
            [pos, ll] = estep(self, X_test, S_test);
            [~, pred] = max(sum(pos, 3)');
        end
        
        
        function [pos, ll] = estep(self, X, S)
            % E-step of the EM algorithm
            %   input:
            %     - X: feature data
            %     - S: spatial locations
            %   return:
            %     - pos: posterior
            %     - ll: log-likelihood    
            [n_sample, n_feature] = size(X);
            n_coordinate = size(S, 2);
            for k=1:self.parms.K
                cov_xk = self.cov_x{k};
                if size(cov_xk, 1) > 1
                    [~, err] = chol(cov_xk);
                    if err ~= 0
                        cov_xk = make_PD(cov_xk, self.const.EPSILON);
                    end
                elseif self.parms.cov_type_x == self.const.CIRCLE
                    cov_xk = repmat(cov_xk, 1, n_feature);
                end
                for l=1:self.parms.L
                    cov_sl = self.cov_s{k, l};
                    if size(cov_sl, 1) > 1
                        [~, err] = chol(cov_sl);
                        if err ~= 0
                            cov_sl = make_PD(cov_sl, self.const.EPSILON);
                        end
                    elseif self.parms.cov_type_s == self.const.CIRCLE
                        cov_sl = repmat(cov_sl, 1, n_coordinate);
                    end
                    r(:,k,l) = self.p(k) * self.q(k,l) ...
                    .* mvnpdf(X, self.mu_x(k,:), cov_xk) ...
                    .* mvnpdf(S, squeeze(self.mu_s(k,l,:))', cov_sl);
                end
            end 
            ll = sum(log(sum(sum(r,3),2)),1);

            r(isnan(r)) = self.const.EPSILON32;
            r(~r) = self.const.EPSILON32;
            pos = r./sum(sum(r,3), 2); 
        end
        
        
        function self = mstep(self, X, S)
            % M-step of the EM algorithm
            %   input:
            %     - X: feature data
            %     - S: spatial locations
            %   return:
            %     - self: updated sgmm model
            self.p = mean(squeeze(sum(self.pos,3)),1);
            if self.parms.L>1
                self.q = squeeze(sum(self.pos,1)./sum(sum(self.pos,1),3));
            end
            for k=1:self.parms.K
                self.mu_x(k,:) = weighted_mu(sum(self.pos(:,k,:),3), X);
                self.cov_x{k} = weighted_cov(sum(self.pos(:,k,:),3), X, ...
                    self.mu_x(k,:), self.parms.cov_type_x);
                for l=1:self.parms.L
                    self.mu_s(k,l,:) = weighted_mu(self.pos(:,k,l), S);
                    self.cov_s{k, l} = weighted_cov(self.pos(:,k,l), S, ...
                        squeeze(self.mu_s(k,l,:))', self.parms.cov_type_s);
                end
            end
        end
    end
end


function [wmu] = weighted_mu(w, mu)
    % take weighted combination of mu
    %   input:
    %     - w: weights
    %     - mu
    %   return:
    %     - wmu: weighted combination of mu
    wmu = w'*mu./sum(w);
end


function [wcov] = weighted_cov(w, X, mu, cov_type)
    % take weighted combination of mu
    %   input:
    %     - w: weights
    %     - X: input data
    %     - mu: mean
    %     - cov_type: [FULL | DIAGONAL | CIRCLE] 
    %   return:
    %     - wcov: weighted covariance
    w = w./sum(w);
    X_zeromean = X-mu;
    wcov = X_zeromean'*(X_zeromean.*w);
    if cov_type == 2
        wcov = mean(diag(wcov));
    elseif cov_type == 1
        wcov = diag(wcov)';
    end
end


function [S_PD] = make_PD(S, eps)
    % make the covariance as positive-definite
    [V,D] = eig(S);
    d = diag(D);
    d(d < eps) = eps; 
    D_c = diag(d);
    S_PD = V*D_c*V';
    S_PD = real((S_PD+S_PD)/2);
end