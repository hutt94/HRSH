close all; clear all; %clc;

bits = [2 4 8 16 32];

param.db_name ='MIRFLICKR';
param.thresh = 0.06;
param.km = 200;

% param.db_name ='IAPRTC-12';
% param.thresh = 0.4;
% param.km = 480;

% param.db_name ='NUSWIDE10';
% param.thresh = 0.04;
% param.km = 50;

param.mu = 3;
param.t = 1;

param

for i = 1:5
param.nbits = bits(i);
param.sf = 0.05;
param.alpha = 1; param.iter = 10;
param.omega = 10; param.beta = 0.01;
param.theta = 10; param.lambda = 0.001;

param.top_K = 2000;
param.pr_ind = [1:50:1000,1000];
param.pn_pos = [1:100:2000,2000];

fprintf('========Load data & Clustering======== \n');
[XTrain,YTrain,LTrain,XTest,YTest,LTest] = load_data(param.db_name);


filename = ['hub_id/', param.db_name, '-hubid-',num2str(bits(i)),'.mat'];
if exist(filename, 'file') == 2
    load(filename);
    temp_id = find(rate>param.thresh);
    hubid = [];
    for j = 1: size(temp_id,1)
        id = find(c==temp_id(j));
        hubid = [hubid;id];
    end
else
    [hubid, rate, c] = find_hub_id(LTrain, param.thresh);
    save(filename, 'hubid', 'rate', 'c');
end

[a,~] = kmeans(LTrain,param.km,'Distance','cosine');
Lc = sparse(1:size(LTrain,1),a,1);
Lc = full(Lc);

fprintf('========%s %d bits start======== \n', 'HRSH',param.nbits);
evaluate_HRSH(XTrain,YTrain,LTrain,XTest,YTest,LTest,Lc,hubid,param);
clearvars -except param bits thresh
end
