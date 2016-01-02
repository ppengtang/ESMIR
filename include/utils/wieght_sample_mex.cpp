#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <tmwtypes.h>
#include <time.h>

/*
 * Sample randomly without replacement with weights - 
 * this function will sample the indices of the population
 *
 * Usage:
 *  y = wieght_sample_mex(n, w, k)
 *
 * Inputs:
 *   n: number of elements in population (scalar)
 *   w: weights vector - the weights should be a probability,
 *      i.e. w[k]>=0 for all k, and sum(w) = 1. (must be type _DOUBLE_)
 *   k: The number of samples to extract (scalar)
 * Output:
 *   y: First k random samples
 *
 *
 *
 * Copyright (c) Bagon Shai
 * Department of Computer Science and Applied Mathmatics
 * Wiezmann Institute of Science
 * http://www.wisdom.weizmann.ac.il/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, subject to the following conditions:
 *
 * 1. The above copyright notice and this permission notice shall be included in 
 *     all copies or substantial portions of the Software.
 * 2. No commercial use will be done with this software.
 * 3. If used in an academic framework - a proper citation must be included.
 *
 * The Software is provided "as is", without warranty of any kind.
 *
 * Jul. 2007
 *
 */

template<class T>
void GetScalar(const mxArray* x, T& scalar)
{
    if ( mxGetNumberOfElements(x) != 1 )
        mexErrMsgIdAndTxt("weight_sample_mex:GetScalar","input is not a scalar");
    void *p = mxGetData(x);
    switch (mxGetClassID(x)) {
        case mxCHAR_CLASS:
            scalar = *(char*)p;
            break;
        case mxDOUBLE_CLASS:
            scalar = *(double*)p;
            break;
        case mxSINGLE_CLASS:
            scalar = *(float*)p;
            break;
        case mxINT8_CLASS:
            scalar = *(char*)p;
            break;
        case mxUINT8_CLASS:
            scalar = *(unsigned char*)p;
            break;
        case mxINT16_CLASS:
            scalar = *(short*)p;
            break;
        case mxUINT16_CLASS:
            scalar = *(unsigned short*)p;
            break;
        case mxINT32_CLASS:
            scalar = *(int*)p;
            break;
        case mxUINT32_CLASS:
            scalar = *(unsigned int*)p;
            break;
#ifdef A64BITS            
        case mxINT64_CLASS:
            scalar = *(int64_T*)p;
            break;
        case mxUINT64_CLASS:
            scalar = *(uint64_T*)p;
            break;
#endif /* 64 bits machines */            
        default:
            mexErrMsgIdAndTxt("GraphCut:GetScalar","unsupported data type");
    }
}
/* produce random numbers in the range 0-1 */
double MyRand()
{
    return ( (double)(rand())/((double)(RAND_MAX)) );
}


/*
 * main enterance point 
 */
void mexFunction(
    int		  nlhs, 	/* number of expected outputs */
    mxArray	  *plhs[],	/* mxArray output pointer array */
    int		  nrhs, 	/* number of inputs */
    const mxArray	  *prhs[]	/* mxArray input pointer array */
    )
{
    /* check arguments */
    if (nrhs != 3)
        mexErrMsgIdAndTxt("weight_sample_mex:main","wrong number of inputs - should be three");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("weight_sample_mex:main","wrong number of outputs - must be one");

    /* number of samples */
    int n = 0;
    GetScalar(prhs[0], n);
    /* w has to have exactly n elements */
    if ( n != mxGetNumberOfElements(prhs[1]) )
        mexErrMsgIdAndTxt("weight_sample_mex:main","w must have %d elements", n);
    
    double *ow = mxGetPr(prhs[1]);
    /* we are going to change w as we sample - make a copy of it */
    mxArray * copyw = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *w = mxGetPr(copyw);
    unsigned int ni = 0;
    for (ni = 0; ni < n ; ni++)
        w[ni] = ow[ni];
    
    
    int k = 0;
    GetScalar(prhs[2], k);
    
    if ( k>n )
        mexErrMsgIdAndTxt("weight_sample_mex:main","k (%d) is larger than n (%d)", k, n);
    int  dims[2];
    dims[0] = k;
    dims[1] = 1;

    plhs[0] = mxCreateNumericArray(2, dims, mxUINT32_CLASS, mxREAL);
    unsigned int * y = (unsigned int*)mxGetData(plhs[0]);
    
    /* rand seed */
    srand((unsigned)time(0)); 
    
    /* sample */
    double norm = 1;
    double r = 0;
    double s = 0;
    
    for ( unsigned int ki = 0; ki < k ; ki++ ) {
        r = MyRand();
        s = 0;
        for (ni=0; ni < n ; ni++ ) {
            s += w[ni]/norm;
            if ( s >= r ) {
                y[ki] = ni+1; /* add 1 for matlab indexing */
                norm = norm - w[ni];
                w[ni] = 0;
                break;
            }
        }
    }
 }
 