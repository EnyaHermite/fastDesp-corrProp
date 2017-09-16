function [s,n] = svdCov(nnIdx, idx, Data, Seed)

nnPt = Data(:,nnIdx);
C = matrixCompute(nnPt,Seed(:,idx));
[U,S,~] = svd(C);
s = diag(S)/sum(diag(S));
n = sign(dot(U(:,3),-Seed(:,idx)))*U(:,3);