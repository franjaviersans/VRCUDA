#ifndef KERNELGPU_H
#define KERNELGPU_H



#include "../include/GL/glew.h"
#include "Definitions.h"
#include "TextureManager.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

//
//#include "cuda_texture_types.h"
#define __CUDA_INTERNAL_COMPILATION__
//#include <math.h>
//#include <math_functions.h>
#include "helper_math.h"
//#include <math_functions.hpp>
#undef __CUDA_INTERNAL_COMPILATION__
//#include <math.h>
#include <float.h>



//#include "cudaGL.h"

#include <stdio.h>



template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		printf("CUDA error at %s : %d \n", file, line);
		printf("%s %s \n", cudaGetErrorString(err), func);
		system("pause");
	}
}


class CUDAClass
{
public:
	CUDAClass(dim3 dim = dim3(16,16,1));
	~CUDAClass();
	

	void destroyObject();
#ifndef NOT_RAY_BOX
	void cudaUpdateMatrix(const float * matrix);
#endif
	void cudaRC();
	void cudaSetVolume(unsigned int width, unsigned int height, unsigned int depth, float diagonal);
	void cudaSetImageSize(unsigned int width, unsigned int height, float NCP, float angle);
	void cudaSetTransferFunction(unsigned int width = 256);
	void Use(GLenum activeTexture);


private:
	int Width, Height;
	//float4 *d_lastHit, *d_FirstHit;
	cudaArray *d_volume, *d_texture, *d_final;
	unsigned int num_vert, num_tri;
	cudaGraphicsResource_t cudaResource_TF, cudaResource_volume, cudaResource_final; // CUDA Graphics Resource
	dim3 block_dim, grid_dim;

#ifdef NOT_RAY_BOX
	cudaArray *d_textureFirst, * d_textureLast;
	cudaGraphicsResource_t cudaResource_First, cudaResource_Last;
#endif
};

#endif