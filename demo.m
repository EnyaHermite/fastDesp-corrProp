clc;clear;close all;

addpath('./flann/');
addpath('./estimateRigidTransform');

gridStep = 0.01;
Rho = 0.01;
datapath = './data';

srcFileName = 'data/bun000.ply';
tarFileName = 'data/bun090.ply';
srcCloud = pcread(srcFileName);
tarCloud = pcread(tarFileName);
overlap = 0.5;
tic;
[T,src2tarEst] = eigReg(srcCloud,tarCloud,overlap,gridStep,Rho);
Time = toc
figure,pcshow(tarCloud.Location,'r'),hold on
pcshow(src2tarEst','g')
