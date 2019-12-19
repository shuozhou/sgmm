%% This is a demo for sGMM on synthetic data
clc;clear;
close all;

n_sample = 1000;
n_cluster = 2;
n_component = 5;

%% create synthetic data
% random parameters for the underlying model
rng(111);

mu_x = rand(n_cluster, 2);
mu_s = zeros(n_cluster, n_component, 2);
mu_s(1,:,1) = linspace(0,1,n_component);
mu_s(1,:,2) = linspace(0,0.5,n_component);
mu_s(2,:,1) = linspace(1,0,n_component)-.1;
mu_s(2,:,2) = linspace(0,0.5,n_component)+.1;
cov_x = zeros(n_cluster, 2, 2);
cov_s = zeros(n_cluster, n_component, 2, 2);
for k=1:n_cluster
    cov_x(k,:,:) = diag(rand(1,2));

    for l=1:n_component
        cov_s(k,l,:,:) = diag(rand(1,2))*.005;
    end
end

p = [0.4,0.6];
q = rand(n_cluster, n_component);
q = q./sum(q,2);

X = zeros(n_sample, 2); % feature data
S = zeros(n_sample, 2); % spatial data
Y = zeros(n_sample, 1); % labels

% generating samples based on the GMM model
idx = 1;
label = 1;
for k=1:n_cluster
    for l=1:n_component
        X(idx:idx+round(n_sample*p(k)*q(k,l))-1,:) = ...
            mvnrnd(squeeze(mu_x(k,:)), squeeze(cov_x(k,:,:)), ...
            round(n_sample*p(k)*q(k,l)));
        S(idx:idx+round(n_sample*p(k)*q(k,l))-1,:) = ...
            mvnrnd(squeeze(mu_s(k,l,:)), squeeze(cov_s(k,l,:,:)), ...
            round(n_sample*p(k)*q(k,l)));
        Y(idx:idx+round(n_sample*p(k)*q(k,l))-1) = label;
        idx = idx+round(n_sample*p(k)*q(k,l));
    end
    label = label+1;
end

%% Clustering samples via KMeans based on different input 
%  (either feature or spatial data)
YX_kmeans = kmeans(X, n_cluster, 'Replicates', 10);
YS_kmeans = kmeans(S, n_cluster, 'Replicates', 10);

%% Clustering samples via sGMM, both feature and spatial data are used
sgmm = SpatialGMM(n_cluster, n_component, 0, 0);
sgmm = sgmm.train(X, S);
[Y_sgmm, ~, ~] = sgmm.predict(X,S);

%% plot results
plot_pairs = {
    X, Y;
    S, Y;
    X, YX_kmeans;
    S, YX_kmeans;
    S, Y_sgmm
    };
titles = [
    "data on the feature domain";
    "data on the spatial domain";
    "KMeans with the feature data";
    "KMeans with the spatial data";
    "sGMM with both feature and spatial data";
    "Clustering accuracy";    
    ];
axis_labels = {
    'X1', 'X2';
    'S1', 'S2';
    'X1', 'X2';
    'S1', 'S2';
    'S1', 'S2';
    };
color = 'rb';
fontsize = 14;

figure('Position', [0, 0, 1200, 1500])
for i=1:length(plot_pairs)
    [X_tmp, Y_tmp] = plot_pairs{i,:};
    [x_label, y_label] = axis_labels{i,:};
    subplot(3,2,i)
    hold on
    for k=1:n_cluster
        scatter(X_tmp(Y_tmp==k,1), X_tmp(Y_tmp==k,2), 20, 'filled', color(k));
    end
    hold off
    title(titles(i))
    xlabel(x_label)
    ylabel(y_label)
    legend('C1', 'C2')
    set(gca, 'FontSize', fontsize)
end

subplot(3,2,6)
Y_pred = {YX_kmeans; YS_kmeans; Y_sgmm};
nmis = cellfun(@(y_pred) nmi(Y, y_pred), Y_pred);
bar([nmis';[0,0,0]])
xlim([.5, 1.5])
xticklabels([])
legend('KMeans on feature data', 'KMeans on spatial data', 'sGMM', 'Location', 'northwest')
ylabel('NMI w.r.t. labels');
title('Clustering quality compared to ground truth');
set(gca, 'FontSize', fontsize)