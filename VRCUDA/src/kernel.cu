
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
#include "cuda_texture_types.h"

#include "math_functions.h"
#include "math_functions.hpp"
#include "helper_math.cu"
//#include "cudaGL.h"

#include <stdio.h>


#define opacityThreshold 0.99


texture<float, cudaTextureType3D, cudaReadModeNormalizedFloat>	volume;         // 3D texture
texture<float4, cudaTextureType1D, cudaReadModeNormalizedFloat>	transferFunction; // 1D transfer function texture


typedef struct
{
	float4 m[4];
} float4x4;

__constant__ float4x4 c_invViewMatrix;  // inverse view matrix


__global__ void volumeRenderingKernel(const int width, const int height, float3 * firsHit, float3 * lastHit, float3 * result, const float h){
	
	float2 Pos;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int tpos = y * width + x;
	float u, v;

	if (x < width && y < height){
		Pos.y = height - Pos.y;

		float3 last = firsHit[tpos];
		float3 first = lastHit[tpos];

		//Get direction of the ray
		float3 direction = last - first;
		float D = length(direction);
		direction = normalize(direction);

		float4 color = make_float4(0.0f);
		color.w = 1.0f;

		float3 trans = first;
		float3 rayStep = direction * h;

		for (float t = 0; t <= D; t += h){

			//Sample in the scalar field and the transfer function
			//Need to do tri-linear interpolation here
			float scalar = tex3D(volume, trans.x, trans.y, trans.z);
			//float4 samp = tex1D(transferFunction, scalar);
			//float scalar = 0.1;
			float4 samp = make_float4(0.0f);

			//Calculating alpa
			samp.w = 1.0f - expf(-0.5 * samp.w);

			//Acumulating color and alpha using under operator 
			samp.x = samp.x * samp.w;
			samp.y = samp.y * samp.w;
			samp.z = samp.z * samp.w;

			color.x += samp.x * color.w;
			color.y += samp.y * color.w;
			color.z += samp.z * color.w;
			color.w *= 1.0f - samp.w;

			//Do early termination of the ray
			if (1.0f - color.w > opacityThreshold) break;

			//Increment ray step
			trans += rayStep;
		}

		color.w = 1.0f - color.w;
		result[tpos] = make_float3(color);
	}
}







// Launch a kernel on the GPU with one thread for each element.
	//
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


void kernelwrapper(int *dev_c, const int * dev_a, const int *dev_b, unsigned int size)
{

	addKernel <<< 1, size >>>(dev_c, dev_a, dev_b);
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



	addKernel <<< 1, size >>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}


int CUDAmain(){

	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


	return 0;


}



