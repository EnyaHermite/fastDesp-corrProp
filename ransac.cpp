// ransac.cpp : Defines the initialization routines for the DLL.
//


#include <stdlib.h>
#include <time.h>
#include <vector>
#include "mex.h"


#define M       3
#define maxIter 200

using  namespace std;

// generate random indices for the minimal sample set
vector<int> generate(int N)
{
	vector<int> index(N); //the whole indices
	for (int i = 0; i < N; i++)
	{
		index[i] = i;
	}

	vector<int> vektor(M);

	int in, im = 0;
	for (in = N; in > N - M; in--) 
	{
		int r = rand() % in; /* generate a random number 'r' */
		vektor[im++] = index[r]; /* the range begins from 0 */		
		index.erase(index.begin() + r);
	}

	return vektor;
}



double* estimateTform(double* srcPts, double* tarPts, vector<int> &Idx)
{
	mxArray *rhs[2], *lhs[2];
	rhs[0] = mxCreateDoubleMatrix(3, Idx.size(), mxREAL);
	rhs[1] = mxCreateDoubleMatrix(3, Idx.size(), mxREAL);

	double *X, *Y;
	X = mxGetPr(rhs[0]);
	Y = mxGetPr(rhs[1]);	

	for (int j = 0; j < Idx.size(); j++)
	{
		X[3 * j] = tarPts[3 * Idx[j]];
		X[3 * j + 1] = tarPts[3 * Idx[j] + 1];
		X[3 * j + 2] = tarPts[3 * Idx[j] + 2];

		Y[3 * j] = srcPts[3 * Idx[j]];
		Y[3 * j + 1] = srcPts[3 * Idx[j] + 1];
		Y[3 * j + 2] = srcPts[3 * Idx[j] + 2];
	}	

	lhs[0] = mxCreateDoubleMatrix(4, 4, mxREAL);
	lhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	mexCallMATLAB(2,lhs,2,rhs,"estimateRigidTransform");

	double* T = mxGetPr(lhs[0]);
	return T;
}



void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray *prhs[])
{
	double *srcPts, *tarPts, threshold;

	if (mxGetM(prhs[0]) != 3 || mxGetM(prhs[1]) != 3 || mxGetN(prhs[0]) != mxGetN(prhs[1]))
		mexErrMsgTxt("The input point matrix should be with size 3-by-N!");

	srcPts = mxGetPr(prhs[0]); 
	tarPts = mxGetPr(prhs[1]);
	threshold = mxGetScalar(prhs[2]);
	int N = mxGetN(prhs[0]);
	
	plhs[0] = mxCreateLogicalMatrix(1,N); // indicate either a match is inlier or not
	bool* CS = (bool*)mxGetData(plhs[0]);

	// Main loop
	//---------------------------------------------------------------
	// initializations
	int iter = 0; //number of iterations
	int bestSz = 3; //initial threshold for inlier size of a better model
	vector<int> randIdx(M, 0);
	vector<double> x(3, 0), y_hat(3, 0), y(3, 0);
	vector<bool> thisCS(N, false);

	srand((unsigned)time(NULL)); //set the seed to the current time
	//srand((unsigned)time(0)); //set the seed to 0
	while (iter <= maxIter)
	{
		randIdx = generate(N);
		double* T = estimateTform(srcPts, tarPts, randIdx);

		// to get size of the consensus set
		int inlierSz = 0;
		for (int i = 0; i < N; i++)
		{
			x[0] = srcPts[3 * i]; x[1] = srcPts[3 * i + 1]; x[2] = srcPts[3 * i + 2];
			y[0] = tarPts[3 * i]; y[1] = tarPts[3 * i + 1]; y[2] = tarPts[3 * i + 2];
			
			y_hat[0] = T[0] * x[0] + T[4] * x[1] + T[8] * x[2] + T[12];
			y_hat[1] = T[1] * x[0] + T[5] * x[1] + T[9] * x[2] + T[13];
			y_hat[2] = T[2] * x[0] + T[6] * x[1] + T[10] * x[2] + T[14];

			double thisErr = (y[0] - y_hat[0])*(y[0] - y_hat[0]) + 
						     (y[1] - y_hat[1])*(y[1] - y_hat[1]) + 
							 (y[2] - y_hat[2])*(y[2] - y_hat[2]);

			thisCS[i] = false;
			if (thisErr < threshold)
			{
				inlierSz++;
				thisCS[i] = true;
			}
		}

		if (inlierSz>bestSz)
		{
			bestSz = inlierSz; // update the best model size

			//update the consensus set
			for (int i = 0; i < N; i++)
			{
				CS[i] = thisCS[i];
			}
		}			

		if (bestSz == N)
			break;

		iter++;
	}
	//--------------------------------------------------------------	

	return;
}
