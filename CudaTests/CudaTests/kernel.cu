/*Author Greg Gutmann */
// Instead of relying on brief documentation and forum questions,
// I have decided to write my own isolated test for different memory access methods and concurrent work/synchronization 
// -because real world examples often have multiple bottlenecks or other features which can make optimization benefits unclear
// The memory tests will be the main part of this code, then they can be run on various hardware to see characteristics
// Tests are also to be used as a learning tool, easier to see what happens than read about it

// Current tests are to test behavior, memory errors and incomplete kernel launch are possible
// <<Will be constantly adding tests as performance questions arise>>

//*This test code was also intended to be a test for creating a simplified allocation and memory deallocation method "managedMem.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"
#include <stdio.h>
#include "conio.h"
#include <random>
//Using Windows for a high-resolution performance counter (also some minor things)
#include <windows.h>
#include "managedMem.h"
#include "cuda_profiler_api.h"

//If false uses 3 data sets <increasing/decreasing kernel data usage>
#define useFourDataSets 0
//With the Float category it is possible to also test with float3
#define useFloat3 0

//High resolution timer
struct timer_ms {
	double PCFreq;
	__int64 CounterStart;
};
//For sorting
struct mapH {
	UINT hashVal;
	UINT orig;
};

//High resolution timer
void StartCounter_ms(timer_ms &t)
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li)) {}

	t.PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	t.CounterStart = li.QuadPart;
}
double GetCounter_ms(timer_ms &t)
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - t.CounterStart) / t.PCFreq;
}

//Data sets used for random indexing
int *randomIndex1, *randomIndex2, *randomIndex3, *randomIndex4;
int *dev_ranIndex1, *dev_ranIndex2, *dev_ranIndex3, *dev_ranIndex4;
cudaTextureObject_t tex_ranIndex1 = 0;

cudaEvent_t start, stop;

//Declarations
template <class T> void memoryTypeAndAccessTesting(unsigned int arrayLength);
template <class T> void streamTests(unsigned int arrayLength);
template <class T> void implicitSync(unsigned int arrayLength);
template <class T> T myrandom(int mod);
template <class T> void addKernel_128bitRequests(T * c, T * a, T * b, T * d, int arrayLength, int threads);

//Math
__device__ __host__ float dot(float3 a)
{
	return (((a.x * a.x) + (a.y * a.y)) + (a.z * a.z));
}
__device__ __host__ float return_inverse(float k) {
	if (k == 0.0f)
		return 0.0f;
	else
		return (1.0f / k);
}
__device__ __host__ float safeDivide(float n, float d) {
	if (d == 0.0f)
		return 0.0f;
	else
		return (n / d);
}
__device__ __host__ float3 normalize(float3 q)
{
	float num2 = dot(q);
	float num = return_inverse(sqrtf(num2));
	float3 norm;
	norm.x = q.x * num;
	norm.y = q.y * num;
	norm.z = q.z * num;
	return norm;
}
__device__ __host__ float3 normalize2(float3 q)
{
	float num2 = dot(q);
	float num = sqrtf(num2);
	float3 norm;
	if (num == 0) {
		norm.x = 0;
		norm.y = 0;
		norm.z = 0;
	}
	else {
		norm.x = q.x / num;
		norm.y = q.y / num;
		norm.z = q.z / num;
	}
	return norm;
}
__device__ __host__ int badd(int n1, int n2) {
	int carry, sum;
	carry = (n1 & n2) << 1; // Find bits that are used for carry
	sum = n1 ^ n2; // Add each bit, discard carry.
	if (sum & carry) // If bits match, add current sum and carry.
		return badd(sum, carry);
	else
		return sum ^ carry; // Return the sum.
}
__device__ __host__ int bsub(int n1, int n2) {
	// Add two's complement and return.
	return badd(n1, badd(~n2, 1));
}

template <class T> __global__ void addKernel(T * __restrict__ d, const T * __restrict__ a, const T * __restrict__ b, const T * __restrict__ c, int arrayLength)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= arrayLength) return;
#if useFourDataSets
    d[i] = a[i] * b[i] * c[i];
#else
	d[i] = a[i] * b[i];
#endif
}
template <class T> __global__ void addKernel_scatter(T * __restrict__ d, const T * __restrict__ a, const T * __restrict__ b, const T * __restrict__ c, const int * __restrict__ idx, int arrayLength)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= arrayLength) return;

	UINT s = idx[i];

#if useFourDataSets
	d[i] = a[s] * b[s] * c[s];
#else
	d[i] = a[s] * b[s];
#endif
}

//Empty work kernels to test overlapping behavior when not all compute units are saturated
//All are the same, they are named differently to see the order more clearly with the Visual Profiler
template <class T> __global__ void addKernel_scatter2(T * __restrict__ a, const T * __restrict__ b, const int * __restrict__ idx, const int arrayLength)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= arrayLength) return;

	UINT s = idx[i];

	//Empty work, j is included in the calculation to make it harder for the compiler to optimize it
	for (int j = 0; j < 100000; j++)
		a[i] = b[s] * (1.0145f + (1/j+1));
}
template <class T> __global__ void addKernel_scatter2B(T * __restrict__ a, const T * __restrict__ b, const int * __restrict__ idx, const int arrayLength)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= arrayLength) return;

	UINT s = idx[i];

	//Empty work, j is included in the calculation to make it harder for the compiler to optimize it
	for (int j = 0; j < 100000; j++)
		a[i] = b[s] * (1.0145f + (1 / j + 1));
}
template <class T> __global__ void addKernel_scatter2C(T * __restrict__ a, const T * __restrict__ b, const int * __restrict__ idx, const int arrayLength)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= arrayLength) return;

	UINT s = idx[i];

	//Empty work, j is included in the calculation to make it harder for the compiler to optimize it
	for (int j = 0; j < 100000; j++)
		a[i] = b[s] * (1.0145f + (1 / j + 1));
}
//Similar to the above
template <class T> __global__ void addKernel2(T * __restrict__ a, const T * __restrict__ b, const int arrayLength)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= arrayLength) return;

	//Empty work, j is included in the calculation to make it harder for the compiler to optimize it
	for (int j = 0; j < 100000; j++)
		a[i] = b[i] * (1.0145f + (1 / j + 1));
}

//Memory access testing
__global__ void addKernel_int4(int4 * __restrict__ d, const int4 * __restrict__ a, const int4 * __restrict__ b, const int4 * __restrict__ c, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;

#if useFourDataSets
	d[i].x = a[i].x * b[i].x * c[i].x;
	d[i].y = a[i].y * b[i].y * c[i].y;
	d[i].z = a[i].z * b[i].z * c[i].z;
	d[i].w = a[i].w * b[i].w * c[i].w;
#else
	d[i].x = a[i].x * b[i].x;
	d[i].y = a[i].y * b[i].y;
	d[i].z = a[i].z * b[i].z;
	d[i].w = a[i].w * b[i].w;
#endif
}
__global__ void addKernel_int4_scatter(int4 * __restrict__ d, const int4 * __restrict__ a, const int4 * __restrict__ b, const int4 * __restrict__ c, const int * __restrict__ idx, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;

	UINT s = idx[i];

#if useFourDataSets
	d[i].x = a[s].x * b[s].x * c[s].x;
	d[i].y = a[s].y * b[s].y * c[s].y;
	d[i].z = a[s].z * b[s].z * c[s].z;
	d[i].w = a[s].w * b[s].w * c[s].w;
#else
	d[i].x = a[s].x * b[s].x;
	d[i].y = a[s].y * b[s].y;
	d[i].z = a[s].z * b[s].z;
	d[i].w = a[s].w * b[s].w;
#endif
}
__global__ void addKernel_float4(float4 * __restrict__ d, const float4 * __restrict__ a, const float4 * __restrict__ b, const float4 * __restrict__ c, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;
#if useFourDataSets
	d[i].x = a[i].x * b[i].x * c[i].x;
	d[i].y = a[i].y * b[i].y * c[i].y;
	d[i].z = a[i].z * b[i].z * c[i].z;
	d[i].w = a[i].w * b[i].w * c[i].w;
#else
	d[i].x = a[i].x * b[i].x;
	d[i].y = a[i].y * b[i].y;
	d[i].z = a[i].z * b[i].z;
	d[i].w = a[i].w * b[i].w;
#endif
}
__global__ void addKernel_float4_scatter(float4 * __restrict__ d, const float4 * __restrict__ a, const float4 * __restrict__ b, const float4 * __restrict__ c, const int * __restrict__ idx, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;

	UINT s = idx[i];

#if useFourDataSets
	d[i].x = a[s].x * b[s].x * c[s].x;
	d[i].y = a[s].y * b[s].y * c[s].y;
	d[i].z = a[s].z * b[s].z * c[s].z;
	d[i].w = a[s].w * b[s].w * c[s].w;
#else
	d[i].x = a[s].x * b[s].x;
	d[i].y = a[s].y * b[s].y;
	d[i].z = a[s].z * b[s].z;
	d[i].w = a[s].w * b[s].w;
#endif
}
__global__ void addKernel_float3(float3 * __restrict__ d, const float3 * __restrict__ a, const float3 * __restrict__ b, const float3 * __restrict__ c, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;
#if useFourDataSets
	d[i].x = a[i].x * b[i].x * c[i].x;
	d[i].y = a[i].y * b[i].y * c[i].y;
	d[i].z = a[i].z * b[i].z * c[i].z;
#else
	d[i].x = a[i].x * b[i].x;
	d[i].y = a[i].y * b[i].y;
	d[i].z = a[i].z * b[i].z;
#endif
}
__global__ void addKernel_float3_scatter(float3 * __restrict__ d, const float3 * __restrict__ a, const float3 * __restrict__ b, const float3 * __restrict__ c, const int * __restrict__ idx, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;

	UINT s = idx[i];

#if useFourDataSets
	d[i].x = a[s].x * b[s].x * c[s].x;
	d[i].y = a[s].y * b[s].y * c[s].y;
	d[i].z = a[s].z * b[s].z * c[s].z;
#else
	d[i].x = a[s].x * b[s].x;
	d[i].y = a[s].y * b[s].y;
	d[i].z = a[s].z * b[s].z;
#endif
}
__global__ void addKernel_double2(double2 * __restrict__ d, const double2 * __restrict__ a, const double2 * __restrict__ b, const double2 * __restrict__ c, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;
#if useFourDataSets
	d[i].x = a[i].x * b[i].x * c[i].x;
	d[i].y = a[i].y * b[i].y * c[i].y;
#else
	d[i].x = a[i].x * b[i].x;
	d[i].y = a[i].y * b[i].y;
#endif
}
__global__ void addKernel_double2_scatter(double2 * __restrict__ d, const double2 * __restrict__ a, const double2 * __restrict__ b, const double2 * __restrict__ c, const int * __restrict__ idx, int arrayLength)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);
	if (i >= arrayLength) return;

	UINT s = idx[i];

#if useFourDataSets
	d[i].x = a[s].x * b[s].x * c[s].x;
	d[i].y = a[s].y * b[s].y * c[s].y;
#else
	d[i].x = a[s].x * b[s].x;
	d[i].y = a[s].y * b[s].y;
#endif
}
template<> void addKernel_128bitRequests<int>(int * a, int * b, int * c, int * d, int arrayLength, int threads)
{
	int newLength = arrayLength / 4;
	int blocks = (newLength / threads);
	addKernel_int4 << <blocks, threads >> >((int4*)d, (int4*)a, (int4*)b, (int4*)c, newLength);
	addKernel_int4_scatter << <blocks, threads >> >((int4*)d, (int4*)a, (int4*)b, (int4*)c, dev_ranIndex4, newLength);
}
template<> void addKernel_128bitRequests<float>(float * a, float * b, float * c, float * d, int arrayLength, int threads)
{
	int newLength = arrayLength / 4;
	int blocks = (newLength / threads);
	addKernel_float4 << <blocks, threads >> >((float4*)d, (float4*)a, (float4*)b, (float4*)c, newLength);
	addKernel_float4_scatter << <blocks, threads >> >((float4*)d, (float4*)a, (float4*)b, (float4*)c, dev_ranIndex4, newLength);
#if useFloat3
	int newLength2 = arrayLength / 3;
	int blocks2 = (newLength2 / threads);
	addKernel_float3 << <blocks2, threads >> >((float3*)d, (float3*)a, (float3*)b, (float3*)c, newLength2);
	addKernel_float3_scatter << <blocks2, threads >> >((float3*)d, (float3*)a, (float3*)b, (float3*)c, dev_ranIndex3, newLength2);
#endif
}
template<> void addKernel_128bitRequests<double>(double * a, double * b, double * c, double * d, int arrayLength, int threads)
{
	int newLength = arrayLength / 2;
	int blocks = (newLength / threads);
	addKernel_double2 << <blocks, threads >> >((double2*)d, (double2*)a, (double2*)b, (double2*)c, newLength);
	addKernel_double2_scatter << <blocks, threads >> >((double2*)d, (double2*)a, (double2*)b, (double2*)c, dev_ranIndex2, newLength);
}

//Very Simple Randoms
template<>int myrandom<int>(int mod) {
	return rand() % mod;
}
template<>float myrandom<float>(int mod) {
	return (rand() % mod)*0.10101f;
}
template<>float3 myrandom<float3>(int mod) {
	float3 ret;
	ret.x = (rand() % mod)*0.10101f;
	ret.y = (rand() % mod)*0.10101f;
	ret.z = (rand() % mod)*0.10101f;
	return ret;
}
template<>double myrandom<double>(int mod) {
	return (rand() % mod)*0.101010101;
}

void merge(struct mapH *mapP, int n, int m) {
	int i, j, k;
	struct mapH *x = (struct mapH*) malloc(sizeof(struct mapH) * n);
	for (i = 0, j = m, k = 0; k < n; k++) {
		if (j == n) {
			x[k] = mapP[i++];
		}
		else if (i == m) {
			x[k] = mapP[j++];
		}
		else if (mapP[j].hashVal < mapP[i].hashVal) {
			x[k] = mapP[j++];
		}
		else {
			x[k] = mapP[i++];
		}
	}
	for (i = 0; i < n; i++) {
		mapP[i] = x[i];
	}
	free(x);
}
void merge_sort(struct mapH *mapP, int n) {
	if (n < 2)
		return;
	int m = n / 2;
	merge_sort(mapP, m);
	merge_sort(mapP + m, n - m);
	merge(mapP, n, m);
}

//<Can ignore this> Test over different math operation methods
int normalizeCPU(unsigned int arrayLength)
{
	float3 *a;
	float3 *b;
	allocateMem(&a, arrayLength, 0, hostMem, "float3_a");
	allocateMem(&b, arrayLength, 0, hostMem, "float3_b");
	unsigned int i;
	for (i = 0; i < arrayLength; i++) {
		a[i] = myrandom<float3>(200);
	}

	timer_ms CPUTimer;
	CPUTimer.CounterStart = 0;
	CPUTimer.PCFreq = 0;

	for (i = 0; i < arrayLength; i++) {
		b[i] = normalize2(a[i]);
	}
	StartCounter_ms(CPUTimer);
	for (i = 0; i < arrayLength; i++) {
		b[i] = normalize(a[i]);
	}
	double totTime = GetCounter_ms(CPUTimer);
	printf("CPUTimer:\t%5.2f ms\n", totTime);
	StartCounter_ms(CPUTimer);
	for (i = 0; i < arrayLength; i++) {
		b[i] = normalize(a[i]);
	}
	totTime = GetCounter_ms(CPUTimer);
	printf("CPUTimer:\t%5.2f ms\n", totTime);
	free(a);
	free(b);
	return 0;
}
//<Can ignore this> Tests with bit wise math (no benefit to implement 'basic' approaches on your own)
// example: non-basic approach of multiplication -> http://www.stoimen.com/blog/2012/05/15/computer-algorithms-karatsuba-fast-multiplication/
int subtractCPU(unsigned int arrayLength)
{
	int *a;
	int *b;
	allocateMem(&a, arrayLength, 0, hostMem, "int_a");
	allocateMem(&b, arrayLength, 0, hostMem, "int_b");
	unsigned int i;
	for (i = 0; i < arrayLength; i++) {
		a[i] = myrandom<int>(200);
		b[i] = myrandom<int>(200);
	}

	timer_ms CPUTimer;
	CPUTimer.CounterStart = 0;
	CPUTimer.PCFreq = 0;

	for (i = 0; i < arrayLength; i++) {b[i] = bsub(a[i], b[i]);}
	StartCounter_ms(CPUTimer);
	for (i = 0; i < arrayLength; i++) {
		b[i] = bsub(a[i], b[i]);
	}
	double totTime = GetCounter_ms(CPUTimer);
	printf("CPUTimer:\t%5.2f ms\n", totTime);

	for (i = 0; i < arrayLength; i++) {
		a[i] = myrandom<int>(200);
		b[i] = myrandom<int>(200);
	}
	for (i = 0; i < arrayLength; i++) { b[i] = a[i] - b[i];}
	StartCounter_ms(CPUTimer);
	for (i = 0; i < arrayLength; i++) {
		b[i] = a[i] - b[i];
	}
	totTime = GetCounter_ms(CPUTimer);
	printf("CPUTimer:\t%5.2f ms\n", totTime);
	free(a);
	free(b);
	return 0;
}

//Creates a list of random numbers than sorts it to create a method of random indexing (worst case memory access)
void createRandomIndex(int * randomIdx, UINT arrayLength) {
	struct mapH *mapP = (struct mapH*) malloc(sizeof(struct mapH) * arrayLength);
	for (unsigned int i = 0; i < arrayLength; i++) {
		mapP[i].hashVal = rand() % 200000; 
		mapP[i].orig = i;
	}
	merge_sort(mapP, arrayLength); //###### SORT ######//
	for (unsigned int i = 0; i < arrayLength; i++)
		randomIdx[i] = mapP[i].orig;
	free(mapP); mapP = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*                                             Main testing code below                                                 */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
	//Intended to be set to { 0 || 1 || 2 }
	int test = 0;
	unsigned int arrayLength = 0;
	
	//Kernels use 256 threads so these numbers are multiples of 256 
	//Nothing is in place to insure accuracy if it is not a multiple of 256, will just ignore a few elements at the end if it is not
	if (test == 0) {
		arrayLength = 4194304; // <- 2^22
	}else {
		arrayLength = 4096; //8192
	}

	//Random indexing lists for different vector lengths
	{
		//Again safeties not in places if the arrayLength is not divisible 
		randomIndex1 = (int*)malloc(arrayLength*sizeof(int));
		randomIndex2 = (int*)malloc((arrayLength / 2)*sizeof(int));
		randomIndex3 = (int*)malloc((arrayLength / 3)*sizeof(int));
		randomIndex4 = (int*)malloc((arrayLength / 4)*sizeof(int));
		createRandomIndex(randomIndex1, arrayLength);
		createRandomIndex(randomIndex2, (arrayLength / 2));
		createRandomIndex(randomIndex3, (arrayLength / 3));
		createRandomIndex(randomIndex4, (arrayLength / 4));
		checkError(cudaMalloc((void**)&dev_ranIndex1, arrayLength*sizeof(int)), "cudaMalloc");
		checkError(cudaMemcpy(dev_ranIndex1, randomIndex1, arrayLength*sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
		checkError(cudaMalloc((void**)&dev_ranIndex2, (arrayLength / 2)*sizeof(int)), "cudaMalloc");
		checkError(cudaMemcpy(dev_ranIndex2, randomIndex2, (arrayLength / 2)*sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
		checkError(cudaMalloc((void**)&dev_ranIndex3, (arrayLength / 3)*sizeof(int)), "cudaMalloc");
		checkError(cudaMemcpy(dev_ranIndex3, randomIndex3, (arrayLength / 3)*sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
		checkError(cudaMalloc((void**)&dev_ranIndex4, (arrayLength / 4)*sizeof(int)), "cudaMalloc");
		checkError(cudaMemcpy(dev_ranIndex4, randomIndex4, (arrayLength / 4)*sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
	}

	if (test == 0) {
		//Designed to be used with Nsight Performance Analysis
		//Have been testing with 4194304
		fprintf(stderr, "memoryTypeAndAccessTesting<int>\n");
		memoryTypeAndAccessTesting<int>(arrayLength);
		fprintf(stderr, "memoryTypeAndAccessTesting<float>\n");
		memoryTypeAndAccessTesting<float>(arrayLength);
		fprintf(stderr, "memoryTypeAndAccessTesting<double>\n");
		memoryTypeAndAccessTesting<double>(arrayLength);
	}
	else if(test == 1){
		//Designed to be used with Nvidia's "Visual Profiler"
		//Test with Small number like 4096 || 8192 so the GPU can do a few kernel at once
		fprintf(stderr, "streamTests<float>\n");
		streamTests<float>(arrayLength);
	}
	else {
		//Designed to be used with Nvidia's "Visual Profiler"
		//Test with Small number like 4096 || 8192 so the GPU can do a few kernel at once
		fprintf(stderr, "implicitSync<float>\n");
		implicitSync<float>(arrayLength);
	}

	//Free the random indexing lists
	free(randomIndex1); free(randomIndex2); free(randomIndex3); free(randomIndex4);
	cudaFree(dev_ranIndex1); cudaFree(dev_ranIndex2); cudaFree(dev_ranIndex3); cudaFree(dev_ranIndex4);

	// cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkError(cudaDeviceReset(), "deviceReset");

	//Uncomment this if using Visual Studio so you can read the end result
	//_getch();
    return 0;
}


template <class T> void memoryTypeAndAccessTesting(unsigned int arrayLength)
{
	T *a, *b, *c, *d;
	allocateMem(&a, arrayLength, 0, hostMem, "T_a");
	allocateMem(&b, arrayLength, 0, hostMem, "T_b");
	allocateMem(&c, arrayLength, 0, hostMem, "T_c");
	allocateMem(&d, arrayLength, 0, hostMem, "T_d");

	for (unsigned int i = 0; i < arrayLength; i++) {
		a[i] = myrandom<T>(200);
		b[i] = myrandom<T>(200);
		c[i] = myrandom<T>(200);
	}
	
	T *dev_a = 0;
    T *dev_b = 0;
    T *dev_c = 0;
	T *dev_d = 0;

	cudaEventCreate(&stop);
	cudaEventCreate(&start);
    // Choose which GPU to run on, change this on a multi-GPU system.
	checkError(cudaSetDevice(0), "cudaSetDevice");

    // Allocate GPU buffers 
	allocateMem(&dev_a, arrayLength, 0, deviceMem, "T_dev_a");
	allocateMem(&dev_b, arrayLength, 0, deviceMem, "T_dev_b");
	allocateMem(&dev_c, arrayLength, 0, deviceMem, "T_dev_c");
	allocateMem(&dev_d, arrayLength, 0, deviceMem, "T_dev_d");
	
	//<<Texture memory tests on random access are to be tested next>>
	allocateMem(&dev_a, arrayLength, 0, textureMem, "tex1", tex_ranIndex1);

    // Copy input vectors from host memory to GPU buffers.
	checkError(cudaMemcpy(dev_a, a, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
	checkError(cudaMemcpy(dev_b, b, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
	checkError(cudaMemcpy(dev_c, c, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");

	int threads = 256; 
	int blocks = (arrayLength / threads);
	//Single value access kernels
	addKernel << <blocks, threads >> >(dev_d, dev_a, dev_b, dev_c, arrayLength);
	addKernel_scatter << <blocks, threads >> >(dev_d, dev_a, dev_b, dev_c, dev_ranIndex1, arrayLength);

    //Vector access kernels (they are wrapped)
	/*timer*/cudaEventRecord(start);
	addKernel_128bitRequests<T>(dev_a, dev_b, dev_c, dev_d, arrayLength, threads);
	/*timer*/cudaEventRecord(stop);
	/*timer*/cudaEventSynchronize(stop);

	checkKernelError("addKernel2w");

	float timer_cu = 0;
	checkError(cudaEventElapsedTime(&timer_cu, start, stop), "cudaEventElapsedTime");
	fprintf(stderr, "kernel Time: %6.4f\n", timer_cu);
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns, any errors encountered during the launch.
	checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	cudaEventDestroy(stop);
	cudaEventDestroy(start);

	deleteAllAllocations();
}
template <class T> void streamTests(unsigned int arrayLength)
{
	cudaEvent_t waitEvent;
	cudaEventCreate(&waitEvent);
	cudaStream_t streamA; cudaStream_t streamB;
	cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking);
	
	T *a, *b, *c, *d, *e, *f;
	allocateMem(&a, arrayLength, 0, hostMem, "T_a");
	allocateMem(&b, arrayLength, 0, hostMem, "T_b");
	allocateMem(&c, arrayLength, 0, hostMem, "T_c");
	allocateMem(&d, arrayLength, 0, hostMem, "T_d");
	allocateMem(&e, arrayLength, 0, hostMem, "T_e");
	allocateMem(&f, arrayLength, 0, hostMem, "T_f");

	for (unsigned int i = 0; i < arrayLength; i++) {
		b[i] = myrandom<T>(200);
		d[i] = myrandom<T>(200);
		f[i] = myrandom<T>(200);
	}

	T *dev_a = 0;
	T *dev_b = 0;
	T *dev_c = 0;
	T *dev_d = 0;
	T *dev_e = 0;
	T *dev_f = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkError(cudaSetDevice(0), "cudaSetDevice");

	// Allocate GPU buffers 
	allocateMem(&dev_a, arrayLength, 0, deviceMem, "T_dev_a");
	allocateMem(&dev_b, arrayLength, 0, deviceMem, "T_dev_b");
	allocateMem(&dev_c, arrayLength, 0, deviceMem, "T_dev_c");
	allocateMem(&dev_d, arrayLength, 0, deviceMem, "T_dev_d");
	allocateMem(&dev_e, arrayLength, 0, deviceMem, "T_dev_e");
	allocateMem(&dev_f, arrayLength, 0, deviceMem, "T_dev_f");

	// Copy input vectors from host memory to GPU buffers.
	checkError(cudaMemcpy(dev_b, b, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
	checkError(cudaMemcpy(dev_d, d, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
	checkError(cudaMemcpy(dev_f, f, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");

	int threads = 256;
	int blocks = (arrayLength / threads);
	
	//Testing wait events <nvidias discription is vague>
	//   NVIDIA "cudaStreamWaitEvent() Makes all future work submitted to stream wait until event reports completion before beginning execution. This synchronization will be performed efficiently on the device.
	//         The stream will wait only for the completion of the most recent host call to cudaEventRecord() on event." 
	//         (The last part: if record is called many times, it waits on the last record call)
	//
	//   Test Reason: Unclear if wait events wait on the last kernel called in the stream before the record or waits on all of the work submitted to the stream before the cudaStreamWaitEvent() call

	//Sleep timers below are to put space between the tests in the Visual Profiler
	cudaProfilerStart();
	if (1) {
		addKernel_scatter2 << <blocks, threads, 0, streamA >> >(dev_a, dev_b, dev_ranIndex1, arrayLength);
		cudaEventRecord(waitEvent, streamA);
		addKernel_scatter2B << <blocks, threads, 0, streamA >> >(dev_c, dev_d, dev_ranIndex1, arrayLength);
		cudaStreamWaitEvent(streamB, waitEvent, 0);
		addKernel_scatter2C << <blocks, threads, 0, streamB >> >(dev_e, dev_f, dev_ranIndex1, arrayLength);

		checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
		Sleep(1);

		addKernel_scatter2 << <blocks, threads, 0, streamA >> >(dev_a, dev_b, dev_ranIndex1, arrayLength);
		addKernel_scatter2B << <blocks, threads, 0, streamA >> >(dev_c, dev_d, dev_ranIndex1, arrayLength);
		cudaEventRecord(waitEvent, streamA);
		cudaStreamWaitEvent(streamB, waitEvent, 0);
		addKernel_scatter2C << <blocks, threads, 0, streamB >> >(dev_e, dev_f, dev_ranIndex1, arrayLength);
		checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	}
	
	if(0){
		Sleep(1);
		addKernel2 << <blocks, threads, 0, streamA >> >(dev_a, dev_b, arrayLength);
		cudaEventRecord(waitEvent, streamA);
		addKernel2 << <blocks, threads, 0, streamA >> >(dev_c, dev_d, arrayLength);
		cudaStreamWaitEvent(streamB, waitEvent, 0);
		addKernel2 << <blocks, threads, 0, streamB >> >(dev_e, dev_f, arrayLength);

		checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
		Sleep(1);

		addKernel2 << <blocks, threads, 0, streamA >> >(dev_a, dev_b, arrayLength);
		addKernel2 << <blocks, threads, 0, streamA >> >(dev_c, dev_d, arrayLength);
		cudaEventRecord(waitEvent, streamA);
		cudaStreamWaitEvent(streamB, waitEvent, 0);
		addKernel2 << <blocks, threads, 0, streamB >> >(dev_e, dev_f, arrayLength);
		checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	}
	
	cudaProfilerStop();
	checkKernelError("addKernel2w");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns, any errors encountered during the launch.
	checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	// Copy output vector from GPU buffer to host memory.
	//checkError(cudaMemcpy(d, dev_d, arrayLength * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");

	cudaEventDestroy(waitEvent); waitEvent = NULL;
	cudaStreamDestroy(streamA); streamA = NULL;
	cudaStreamDestroy(streamB); streamB = NULL;

	deleteAllAllocations();
}
template <class T> void implicitSync(unsigned int arrayLength)
{
	cudaEvent_t waitEvent;
	cudaEventCreate(&waitEvent);
	cudaStream_t streamA; cudaStream_t streamB;
	cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking);

	T *a, *b, *c, *d, *e, *f;
	allocateMem(&a, arrayLength, 0, hostMem, "T_a");
	allocateMem(&b, arrayLength, 0, hostMem, "T_b");
	allocateMem(&c, arrayLength, 0, hostMem, "T_c");
	allocateMem(&d, arrayLength, 0, hostMem, "T_d");
	allocateMem(&e, arrayLength, 0, hostMem, "T_e");
	allocateMem(&f, arrayLength, 0, hostMem, "T_f");

	for (unsigned int i = 0; i < arrayLength; i++) {
		b[i] = myrandom<T>(200);
		d[i] = myrandom<T>(200);
		f[i] = myrandom<T>(200);
	}

	T *dev_a = 0;
	T *dev_b = 0;
	T *dev_c = 0;
	T *dev_d = 0;
	T *dev_e = 0;
	T *dev_f = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkError(cudaSetDevice(0), "cudaSetDevice");

	// Allocate GPU buffers for three vectors (two input, one output)
	allocateMem(&dev_a, arrayLength, 0, deviceMem, "T_dev_a");
	allocateMem(&dev_b, arrayLength, 0, deviceMem, "T_dev_b");
	allocateMem(&dev_c, arrayLength, 0, deviceMem, "T_dev_c");
	allocateMem(&dev_d, arrayLength, 0, deviceMem, "T_dev_d");
	allocateMem(&dev_e, arrayLength, 0, deviceMem, "T_dev_e");
	allocateMem(&dev_f, arrayLength, 0, deviceMem, "T_dev_f");

	// Copy input vectors from host memory to GPU buffers.
	checkError(cudaMemcpy(dev_b, b, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
	checkError(cudaMemcpy(dev_d, d, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
	checkError(cudaMemcpy(dev_f, f, arrayLength * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");

	int threads = 256;
	int blocks = (arrayLength / threads);

	//In the past I have seen CUDA do implicit synchronization based on dependencies 
	// Below is testing memory dependencies between kernels 
	// <Resutlt: no implicit actions here, it will create undefined behaviour> 
	// Good for performance to let the programmer manage this instead of language

	// Future tests with be with various cudaMemcpy and kernels, I have seen implicit syncs with memory in the past. 

	cudaProfilerStart();
	{
		addKernel_scatter2 << <blocks, threads, 0, streamA >> >(dev_a, dev_b, dev_ranIndex1, arrayLength);
		addKernel_scatter2B << <blocks, threads, 0, streamA >> >(dev_c, dev_d, dev_ranIndex1, arrayLength);
		addKernel_scatter2C << <blocks, threads, 0, streamB >> >(dev_e, dev_f, dev_ranIndex1, arrayLength); //Dependant on none

		checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
		Sleep(1);

		addKernel_scatter2 << <blocks, threads, 0, streamA >> >(dev_a, dev_b, dev_ranIndex1, arrayLength); // Kernel 1
		addKernel_scatter2B << <blocks, threads, 0, streamA >> >(dev_c, dev_d, dev_ranIndex1, arrayLength); // Kernel 2
		addKernel_scatter2C << <blocks, threads, 0, streamB >> >(dev_e, dev_a, dev_ranIndex1, arrayLength); //Dependant on Kernel 1

		checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
		Sleep(1);

		addKernel_scatter2 << <blocks, threads, 0, streamA >> >(dev_a, dev_b, dev_ranIndex1, arrayLength); // Kernel 1
		addKernel_scatter2B << <blocks, threads, 0, streamA >> >(dev_c, dev_d, dev_ranIndex1, arrayLength); // Kernel 2
		addKernel_scatter2C << <blocks, threads, 0, streamB >> >(dev_e, dev_c, dev_ranIndex1, arrayLength); //Dependant on Kernel 2
		checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	}
	cudaProfilerStop();
	checkKernelError("addKernel2w");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns, any errors encountered during the launch.
	checkError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	// Copy output vector from GPU buffer to host memory.
	//checkError(cudaMemcpy(d, dev_d, arrayLength * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");

	cudaEventDestroy(waitEvent); waitEvent = NULL;
	cudaStreamDestroy(streamA); streamA = NULL;
	cudaStreamDestroy(streamB); streamB = NULL;

	deleteAllAllocations();
}
