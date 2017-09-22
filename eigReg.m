function [T,src2tarEst] = eigReg(srcCloud,tarCloud,overlap,gridStep,Rho)
%This code is the Matlab implimentation of the paper, 
%"Fast Descriptors and Correspondence Propagation for Robust Global Point Cloud Registration,"
%IEEE transactions on Image Processing, 2017.
%This code should be used only for academic research.
%any other useage of this code should not be allowed without Author agreement.
% If you have any problem or improvement idea about this code, please
% contact Huan LEI with hlei.ziyan@gmail.com.

%% parameter configuration for flann search
params.algorithm = 'kdtree';
params.trees = 8;
params.checks = 64;

srcData = srcCloud.Location';
tarData = tarCloud.Location';
radii = (0.5:0.5:2)*gridStep;

srcCloudDown = pcdownsample(srcCloud, 'gridAverage', Rho);
tarCloudDown = pcdownsample(tarCloud, 'gridAverage', Rho);
srcSeed = srcCloudDown.Location';
tarSeed = tarCloudDown.Location';

%% compute descriptors for seed points in the source point cloud
K = length(radii);
srcIdx = rangesearch(srcData',srcSeed',radii(1));
idxSz = cellfun(@length,srcIdx,'uni',true);
srcIdx = srcIdx(idxSz>10);
srcSeed = srcSeed(:,idxSz>10);
M = sum(idxSz>10);
idx = num2cell((1:M)');
[s,n] = cellfun(@(x,y)svdCov(x,y,srcData,srcSeed),srcIdx,idx,'uni',false);
s = cell2mat(s);
n = cell2mat(n);
for k = 2:K
    srcIdx = rangesearch(srcData',srcSeed',radii(k));
    [sk,nk] = cellfun(@(x,y)svdCov(x,y,srcData,srcSeed),srcIdx,idx,'uni',false);
    s = [s cell2mat(sk)];
    n = [n cell2mat(nk)];
end
s = s';
ds = diff(s);
srcDesp = reshape(ds,3*(K-1),[]);
n = mat2cell(n,3*ones(M,1),K);
srcNorm = cellfun(@(x)reshape(x,[],1),n','uni',false);
srcNorm = cell2mat(srcNorm);

%% compute descriptors for seed points in the target point cloud
tarIdx = rangesearch(tarData',tarSeed',radii(1));
idxSz = cellfun(@length,tarIdx,'uni',true);
tarIdx = tarIdx(idxSz>10);
tarSeed = tarSeed(:,idxSz>10);
N = sum(idxSz>10);
idx = num2cell((1:N)');
[s,n] = cellfun(@(x,y)svdCov(x,y,tarData,tarSeed),tarIdx,idx,'uni',false);
s = cell2mat(s);
n = cell2mat(n);
for k = 2:K
    tarIdx = rangesearch(tarData',tarSeed',radii(k));
    [sk,nk] = cellfun(@(x,y)svdCov(x,y,tarData,tarSeed),tarIdx,idx,'uni',false);
    s = [s cell2mat(sk)];
    n = [n cell2mat(nk)];
end
s = s';
ds = diff(s);
tarDesp = reshape(ds,3*(K-1),[]);
n = mat2cell(n,3*ones(N,1),K);
tarNorm = cellfun(@(x)reshape(x,[],1),n','uni',false);
tarNorm = cell2mat(tarNorm);

[srcIdx,dist] = flann_search(srcDesp,tarDesp,1,params); % match with descriptors

%% aggregating each pair of correspondence for finding the best match
M = size(srcSeed,2);
N = size(tarSeed,2);
seedIdx = srcIdx; 
Err = inf(N,1);
tform = cell(1,N); 
ovNum = ceil(overlap*N); 
distThr = 0.2/4*length(radii); 
thetaThr = 10; 
threshold = gridStep*gridStep;
for n = 1:N
    seed = srcSeed(:,seedIdx(n));
    seedNorm = srcNorm(:,seedIdx(n));
    
    % source point cloud
    d = bsxfun(@minus,srcSeed,seed);
    d = sqrt(sum(d.^2,1)); % distance
    inProd = bsxfun(@times,srcNorm,seedNorm);
    inProd = inProd(1:3:end,:) + inProd(2:3:end,:) + inProd(3:3:end,:);
    theta = real(acosd(inProd));  % inner product

    % target point cloud
    r = bsxfun(@minus,tarSeed,tarSeed(:,n));
    r = sqrt(sum(r.^2,1)); % distance
    inProd = bsxfun(@times,tarNorm,tarNorm(:,n));
    inProd = inProd(1:3:end,:) + inProd(2:3:end,:) + inProd(3:3:end,:);
    alpha = real(acosd(inProd));  % inner product   

    IDX = rangesearch(r',d',gridStep/2,'distance','cityblock');
    
    matches = [seedIdx(n) n];
    for m = [1:seedIdx(n)-1 seedIdx(n)+1:M]        
        idx = IDX{m};%find(abs(r-d(m))<gridStep/2);%
        if(isempty(idx))
            continue;
        end
        dTheta = bsxfun(@minus,alpha(:,idx),theta(:,m));
        dTheta = abs(dTheta);
        Tab = dTheta<thetaThr;
        Tab = sum(Tab,1);
        if(all(Tab<size(theta,1)))
            continue;
        end
        sim = mean(dTheta,1);
        sim(Tab<size(theta,1)) = inf;
        [minSim,ol] = min(sim);
        R = norm(srcDesp(:,m)-tarDesp(:,idx(ol)));
        if(minSim<thetaThr && R<distThr)
            matches = [matches; m idx(ol)];
        end
    end

    if(size(matches,1)>10)
        match_srcSeed = srcSeed(:,matches(:,1));
        match_tarSeed = tarSeed(:,matches(:,2));
        CS = ransac(double(match_srcSeed),double(match_tarSeed),threshold);   
        
        if(sum(CS)<3)
            continue;
        end
        
        match_srcSeed = match_srcSeed(:,CS);
        match_tarSeed = match_tarSeed(:,CS);
        [T, Eps] = estimateRigidTransform(match_tarSeed, match_srcSeed);
        tarEst = T*[srcSeed;ones(1,M)];
        tarEst = tarEst(1:3,:);
        tform{n} = T;
        
        [index,dist] = flann_search(tarEst,tarSeed,1,params);
        [dist,ind] = sort(dist);        
        Err(n) = sum(sum((tarEst(:,index(ind(1:ovNum)))-tarSeed(:,ind(1:ovNum))).^2));
    end
 end
[v,idx] = min(Err);
T = tform{idx};
tarEst = T*[srcData;ones(1,length(srcData))];
tarEst = tarEst(1:3,:);

%% trimmed-icp part
%--------------------------------------------------------------------------
[index,dist] = flann_search(tarEst,tarData,1,params);
[dist,ind] = sort(dist);
ovNum = ceil(overlap*length(dist));
tmp = mean(sum((tarEst(:,index(ind(1:ovNum)))-tarData(:,ind(1:ovNum))).^2));
rmsE(1) = sqrt(tmp);

match_srcData = srcData(:,index(ind(1:ovNum)));
match_tarData = tarData(:,ind(1:ovNum));

maxIter = 100; dE = inf; iter = 1;
errThr = 1e-4; rmseThr = 0.001; 
while(dE>errThr && iter<maxIter && rmsE(iter)>rmseThr)
    [T, Eps] = estimateRigidTransform(match_tarData, match_srcData);
    tarEst = T*[srcData;ones(1,length(srcData))];
    tarEst = tarEst(1:3,:);      

    iter = iter + 1;

    [index,dist] = flann_search(tarEst,tarData,1,params);
    [dist,ind] = sort(dist);
    tmp = mean(sum((tarEst(:,index(ind(1:ovNum)))-tarData(:,ind(1:ovNum))).^2));
    rmsE(iter) = sqrt(tmp);

    match_srcData = srcData(:,index(ind(1:ovNum)));
    match_tarData = tarData(:,ind(1:ovNum));
    dE = rmsE(iter - 1) - rmsE(iter);
end
%--------------------------------------------------------------------------

src2tarEst = tarEst;
