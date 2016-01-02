#include "mex.h"

#include <float.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include "mex.h"

#ifndef M_PI
#define M_PI 3.1415926
#endif

/* Compare a value pointed to by 'ptr' to the 'center' value and
 * increment pointer. Comparison is made by masking the most
 * significant bit of an integer (the sign) and shifting it to an
 * appropriate position. */
#define compab_mask_inc(ptr,shift) { value |= ((unsigned int)(*center - *ptr - 1) & 0x80000000) >> (31-shift); ptr++; }
/* Compare a value 'val' to the 'center' value. */
#define compab_mask(val,shift) { value |= ((unsigned int)(*center - (val) - 1) & 0x80000000) >> (31-shift); }
/* Predicate 1 for the 3x3 neighborhood */
#define predicate 1
/* The number of bits */
#define bits 8

typedef struct
{
	mwIndex x,y;
} integerpoint;

typedef struct
{
	double x,y;
} doublepoint;

void calculate_points(void);
double interpolate_at_ptr(int* upperLeft, mwIndex i, mwSize columns);
void lbp_histogram(int* img, mwSize rows, mwSize columns, int* result, mwSize interpolated);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    mwSize rows, columns; 
    int *img, *result; 
    double *fresult; 
    mwSize interpolated; 
    mwIndex i; 
        
    if (nlhs>1)  
        mexErrMsgTxt("Too many output arguments");
    if (nrhs != 2) 
        mexErrMsgTxt("4 inputs are needed");
    
    img = (int*) mxGetPr(prhs[0]);

    columns =  mxGetM(prhs[0]); 
    rows =  mxGetN(prhs[0]);
    interpolated = (mwSize) mxGetScalar(prhs[1]);
/*
	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<columns; j++)
			mexPrintf("%i\t", img[i*columns+j]); 
		mexPrintf("\n"); 
	}
*/    
    plhs[0] = mxCreateNumericMatrix(1, 256, mxDOUBLE_CLASS, mxREAL);
    fresult = (double*) mxGetPr(plhs[0]);
        
    result = new int[256];
    lbp_histogram(img, rows, columns, result, interpolated);
    
    float sum_val = 1e-20; 
    for(i = 0; i <= 255; i++)
        sum_val += result[i]*result[i]; 
    sum_val = sqrt(sum_val); 
    
    for(i = 0; i <= 255; i++)
        fresult[i] = result[i] / sum_val; 
    
    delete[] result; 
}

integerpoint points[bits];
doublepoint offsets[bits];

/*
 * Get a bilinearly interpolated value for a pixel.
 */
double interpolate_at_ptr(int* upperLeft, int i, int columns)
{
	double dx = 1-offsets[i].x;
	double dy = 1-offsets[i].y;
	return
		*upperLeft*dx*dy +
		*(upperLeft+1)*offsets[i].x*dy +
		*(upperLeft+columns)*dx*offsets[i].y +
		*(upperLeft+columns+1)*offsets[i].x*offsets[i].y;

}
	
/*
 * Calculate the point coordinates for circular sampling of the neighborhood.
 */
void calculate_points(void)
{
	double step = 2 * M_PI / bits, tmpX, tmpY;
	int i;
	for (i=0;i<bits;i++)
		{
			tmpX = predicate * cos(i * step);
			tmpY = predicate * sin(i * step);
			points[i].x = (int)tmpX;
			points[i].y = (int)tmpY;
			offsets[i].x = tmpX - points[i].x;
			offsets[i].y = tmpY - points[i].y;
			if (offsets[i].x < 1.0e-10 && offsets[i].x > -1.0e-10) /* rounding error */
				offsets[i].x = 0;
			if (offsets[i].y < 1.0e-10 && offsets[i].y > -1.0e-10) /* rounding error */
				offsets[i].y = 0;
			
			if (tmpX < 0 && offsets[i].x != 0)
				{
					points[i].x -= 1;
					offsets[i].x += 1;
				}
			if (tmpY < 0 && offsets[i].y != 0)
				{
					points[i].y -= 1;
					offsets[i].y += 1;
				}
		}
}

/*
 * Calculate the LBP histogram for an integer-valued image. This is an
 * optimized version of the basic 8-bit LBP operator. Note that this
 * assumes 4-byte integers. In some architectures, one must modify the
 * code to reflect a different integer size.
 * 
 * img: the image data, an array of rows*columns integers arranged in
 * a horizontal raster-scan order
 * rows: the number of rows in the image
 * columns: the number of columns in the image
 * result: an array of 256 integers. Will hold the 256-bin LBP histogram.
 * interpolated: if != 0, a circular sampling of the neighborhood is
 * performed. Each pixel value not matching the discrete image grid
 * exactly is obtained using a bilinear interpolation. You must call
 * calculate_points (only once) prior to using the interpolated version.
 * return value: result
 */
void lbp_histogram(int* img, int rows, int columns, int* result, int interpolated)
{
	int leap = columns*predicate;
	/*Set up a circularly indexed neighborhood using nine pointers.*/
	int
		*p0 = img,
		*p1 = p0 + predicate,
		*p2 = p1 + predicate,
		*p3 = p2 + leap,
		*p4 = p3 + leap,
		*p5 = p4 - predicate,
		*p6 = p5 - predicate,
		*p7 = p6 - leap,
		*center = p7 + predicate;
	int value;
	int pred2 = predicate << 1; //2
	int r,c;

	memset(result,0,sizeof(int)*256); /* Clear result histogram */
		
	if (!interpolated)
		{
			for (r=0;r<rows-pred2;r++)
				{
					for (c=0;c<columns-pred2;c++)
						{
							value = 0;

							/* Unrolled loop */
							compab_mask_inc(p0,0);
							compab_mask_inc(p1,1);
							compab_mask_inc(p2,2);
							compab_mask_inc(p3,3);
							compab_mask_inc(p4,4);
							compab_mask_inc(p5,5);
							compab_mask_inc(p6,6);
							compab_mask_inc(p7,7);
							center++;

							result[value]++; /* Increase histogram bin value */
						}
					p0 += pred2;
					p1 += pred2;
					p2 += pred2;
					p3 += pred2;
					p4 += pred2;
					p5 += pred2;
					p6 += pred2;
					p7 += pred2;
					center += pred2;
				}
		}
	else
		{
			p0 = center + points[5].x + points[5].y * columns;
			p2 = center + points[7].x + points[7].y * columns;
			p4 = center + points[1].x + points[1].y * columns;
			p6 = center + points[3].x + points[3].y * columns;
				
			for (r=0;r<rows-pred2;r++)
				{
					for (c=0;c<columns-pred2;c++)
						{
							value = 0;

							/* Unrolled loop */
							compab_mask_inc(p1,1);
							compab_mask_inc(p3,3);
							compab_mask_inc(p5,5);
							compab_mask_inc(p7,7);

							/* Interpolate corner pixels */
							compab_mask((int)(interpolate_at_ptr(p0,5,columns)+0.5),0);
							compab_mask((int)(interpolate_at_ptr(p2,7,columns)+0.5),2);
							compab_mask((int)(interpolate_at_ptr(p4,1,columns)+0.5),4);
							compab_mask((int)(interpolate_at_ptr(p6,3,columns)+0.5),6);
							p0++;
							p2++;
							p4++;
							p6++;
							center++;
								
							result[value]++;
						}
					p0 += pred2;
					p1 += pred2;
					p2 += pred2;
					p3 += pred2;
					p4 += pred2;
					p5 += pred2;
					p6 += pred2;
					p7 += pred2;
					center += pred2;
				}
		}
}