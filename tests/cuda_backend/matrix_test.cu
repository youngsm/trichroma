//-*-c-*-

#include "matrix.h"

__device__ Matrix array2matrix(float *a)
{
	return make_matrix(a[0], a[1], a[2],
			   a[3], a[4], a[5],
			   a[6], a[7], a[8]);
}

__device__ void matrix2array(const Matrix &m, float *a)
{
	a[0] = m.a00;
	a[1] = m.a01;
	a[2] = m.a02;
	a[3] = m.a10;
	a[4] = m.a11;
	a[5] = m.a12;
	a[6] = m.a20;
	a[7] = m.a21;
	a[8] = m.a22;
}

extern "C"
{

__global__ void det(float *a, float *dest)
{
	Matrix m = array2matrix(a);
	dest[0] = det(m);
}

__global__ void inv(float *a, float *dest)
{
	Matrix m = array2matrix(a);
	matrix2array(inv(m), dest);
}

__global__ void minusmatrix(float *a, float *dest)
{
	matrix2array(-array2matrix(a), dest);
}

__global__ void matrixadd(float *a, float *b, float *dest)
{
	matrix2array(array2matrix(a)+array2matrix(b), dest);
}

__global__ void matrixsub(float *a, float *b, float *dest)
{
	matrix2array(array2matrix(a)-array2matrix(b), dest);
}

__global__ void matrixmul(float *a, float *b, float *dest)
{
	matrix2array(array2matrix(a)*array2matrix(b), dest);
}

__global__ void multiply(float *a, float3 *x, float3 *dest)
{
	dest[0] = array2matrix(a)*x[0];
}

__global__ void matrixaddfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)+c, dest);
}

__global__ void matrixsubfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)-c, dest);
}

__global__ void matrixmulfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)*c, dest);
}

__global__ void matrixdivfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)/c, dest);
}

__global__ void floataddmatrix(float *a, float c, float *dest)
{
	matrix2array(c+array2matrix(a), dest);
}

__global__ void floatsubmatrix(float *a, float c, float *dest)
{
	matrix2array(c-array2matrix(a), dest);
}

__global__ void floatmulmatrix(float *a, float c, float *dest)
{
	matrix2array(c*array2matrix(a), dest);
}

__global__ void floatdivmatrix(float *a, float c, float *dest)
{
	matrix2array(c/array2matrix(a), dest);
}

__global__ void matrixaddequals(float *a, float *b)
{
	Matrix m = array2matrix(a);
	m += array2matrix(b);
	matrix2array(m,a);
}

__global__ void matrixsubequals(float *a, float *b)
{
	Matrix m = array2matrix(a);
	m -= array2matrix(b);
	matrix2array(m,a);
}

__global__ void matrixaddequalsfloat(float *a, float c)
{
	Matrix m = array2matrix(a);
	m += c;
	matrix2array(m,a);
}

__global__ void matrixsubequalsfloat(float *a, float c)
{
	Matrix m = array2matrix(a);
	m -= c;
	matrix2array(m,a);
}

__global__ void matrixmulequalsfloat(float *a, float c)
{
	Matrix m = array2matrix(a);
	m *= c;
	matrix2array(m,a);
}

__global__ void matrixdivequalsfloat(float *a, float c)
{
	Matrix m = array2matrix(a);
	m /= c;
	matrix2array(m,a);
}

__global__ void outer(float3 a, float3 b, float* dest)
{
	matrix2array(outer(a,b), dest);
}

} // extern "c"
