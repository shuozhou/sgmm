This folder contains the implementation described in:
    S. Zhou, H. Bondell, A. Todesillas, B.I.P. Rubinstein and J. Bailey. 
    "Early Prediction of a Rockslide Location via a Spatially-aided 
    Gaussian Mixture Model". Annals of Applied Statistics (under review), 2019.

SpatialGMM.m: a class for the Spatially-aided Gaussian Mixture Model.
    % Usage:
    %   - initialize a sGMM object with 3 clusters, 2 spatial components in
    %   each cluster, use full covariance matrix in the feature domain, and
    %   use diagonal covariance in the spatial domain
    %       sgmm = SpatialGMM(3, 2, 0, 1)
    %   - train the model with feature data X_train \in R^{M*N} and and
    %  spatial data S_train \in R^{M*2}
    %       sgmm = sgmm.train(X_train, S_train)
    %   - get prediction from sgmm
    %       [pred, posterior, log_likelihood] = sgmm.pred(X_test, S_test)

demo.m: a demo of using sGMM on synthetic data

nmi.m: function to calculated the nomralized mutual information between two
       random variables
    % Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two 
    % discrete variables x and y.
    % Input:
    %   x, y: two integer vector of the same length 
    % Ouput:
    %   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
    % Written by Mo Chen (sth4nth@gmail.com).
    % https://au.mathworks.com/matlabcentral/fileexchange/29047-normalized-mutual-information