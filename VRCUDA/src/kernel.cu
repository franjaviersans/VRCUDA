


#include "kernel.cuh"

typedef unsigned char VolumeType;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

#define opacityThreshold 0.99


texture<VolumeType, 3, cudaReadModeNormalizedFloat> volume;         // 3D texture
texture<float4, 2, cudaReadModeElementType>         transferFunction; // 1D transfer function texture
surface<void, cudaSurfaceType2D> surf;

/*
texture<VolumeType, cudaTextureType3D, /*cudaReadModeNormalizedFloat cudaReadModeElementType>	volume;         // 3D texture
texture<float4, cudaTextureType1D, /*cudaReadModeNormalizedFloat cudaReadModeElementType>	transferFunction; // 1D transfer function texture
*/

__constant__ float constantH, constantAngle, cosntantNCP;
__constant__ unsigned int constantWidth, constantHeight;
__constant__ float4x4 c_invViewMatrix;  // inverse view matrix





struct Ray
{
	float4 o;   // origin
	float4 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float *tnear, float *tfar)
{
	float3 boxmin = make_float3(-.5f, -.5f, -.5f);
	float3 boxmax = make_float3(.5f, .5f, .5f);
	/*float3 boxmin = make_float3(-1.f, -1.f, -1.f);
	float3 boxmax = make_float3(1.f, 1.f, 1.f);*/
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / make_float3(r.d);
	float3 tbot = invR * (boxmin - make_float3(r.o));
	float3 ttop = invR * (boxmax - make_float3(r.o));

	// re-order intersections to find smallest and largest on each axis

	float3 tmin = make_float3(FLT_MAX);
	float3 tmax = make_float3(0.0f);

	tmin = fminf(tmin, fminf(ttop, tbot));
	tmax = fmaxf(tmax, fmaxf(ttop, tbot));

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}


/**
Kernel to do the volume rendering 
*/
__global__ void volumeRenderingKernel(/*uchar4 * result, const int width, const int height, float3 * firsHit, float3 * lastHit*/){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	

	if (x < constantWidth && y < constantHeight){
		/*float sample = tex3D(volume, float(x) / constantWidth, float(y) / constantHeight, 0.5f);
		float4 color = tex1D(transferFunction, sample);

		result[tpos] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);*/
		//result[tpos] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
		float2 Pos = make_float2(x, y);
		
		//Flip the Y axis
		//Pos.y = constantHeight - Pos.y;

		
		//ok u and v between -1 and 1
		float u = (((x + 0.5f) / (float)constantWidth)*2.0f - 1.0f);
		float v = (((y + 0.5f) / (float)constantHeight)*2.0f - 1.0f);



		// calculate eye ray in world space
		Ray eyeRay;
		float tangent = tan(constantAngle); // angle in radians
		float ar = (float(constantWidth) / constantHeight);
		eyeRay.o = multiplication(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
		eyeRay.d = normalize(make_float4(u * tangent * ar, v * tangent, -cosntantNCP, 0.0f));
		eyeRay.d = multiplication(c_invViewMatrix, eyeRay.d);
		eyeRay.d = normalize(eyeRay.d);

		// find intersection with box
		float tnear, tfar;
		int hit = intersectBox(eyeRay, &tnear, &tfar);  // this must be wrong....anything else seems to be ok now

		float4 color, bg = make_float4(0.15f); //bg color here

		if (hit){

			if (tnear < cosntantNCP)
				tnear = cosntantNCP;     // clamp to near plane


			//float3 last = firsHit[tpos];
			//float3 first = lastHit[tpos];

			float3 first = make_float3(eyeRay.o + eyeRay.d*tnear);
			float3 last = make_float3(eyeRay.o + eyeRay.d*tfar);

			//Get direction of the ray
			float3 direction = last - first;
			float D = length(direction);
			direction = normalize(direction);

			color = make_float4(0.0f);
			color.w = 1.0f;

			float3 trans = first;
			float3 rayStep = direction * constantH;

			for (float t = 0; t <= D; t += constantH){

				//Sample in the scalar field and the transfer function
				//Need to do tri-linear interpolation here
				float scalar = tex3D(volume, trans.x + 0.5f, trans.y + 0.5f, 1.0f - (trans.z + 0.5f)); //convert to texture space
				//float scalar = tex3D(volume, trans.x, trans.y, trans.z);
				float4 samp = tex2D(transferFunction, scalar, 0.5f);
				//float scalar = 0.1;
				//float4 samp = make_float4(0.0f);

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

			//Write to the texture
			uchar4 ucolor = make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
			surf2Dwrite(ucolor, surf, x * sizeof(uchar4), y, cudaBoundaryModeClamp); 

		}else{
			bg.w = 1.0f;

			//Write to the texture
			uchar4 ucolor = make_uchar4(bg.x * 255, bg.y * 255, bg.z * 255, bg.w * 255);
			surf2Dwrite(ucolor, surf, x * sizeof(uchar4), y, cudaBoundaryModeClamp);
		}
		
	}

	
}


CUDAClass::CUDAClass()
{
	d_lastHit = d_FirstHit = NULL;
	d_texture = d_volume = NULL;

	checkCudaErrors(cudaSetDevice(0));
	// Otherwise pick the device with highest Gflops/s


	/*d_pos = NULL;
	d_normal = NULL;
	d_tex = NULL;
	d_id = NULL;
	d_octree = NULL;
	d_texture = NULL;
	d_texW = d_texH = -1;*/
}

CUDAClass::~CUDAClass()
{

/*	destroyObject();*/

	if (d_texture != NULL)
	{
		checkCudaErrors(cudaFreeArray(d_texture));
		cudaUnbindTexture(transferFunction);
	}
	if (d_volume != NULL)
	{
		checkCudaErrors(cudaFreeArray(d_volume));
		cudaUnbindTexture(volume);
	}
	if (d_FirstHit != NULL) checkCudaErrors(cudaFree(d_FirstHit));
	if (d_lastHit != NULL) checkCudaErrors(cudaFree(d_lastHit));

	d_lastHit = d_FirstHit = NULL;
	d_texture = d_volume = NULL;


	checkCudaErrors(cudaDeviceReset());
}


// Helper function for using CUDA to add vectors in parallel.
void CUDAClass::cudaRC(/*, unsigned int width, unsigned int height, float h, float4 *d_buffer, float4 *d_lastHit*/)
{
	
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((Width + blockDim.x) / blockDim.x, (Height + blockDim.y) / blockDim.y, 1);

	volumeRenderingKernel << < gridDim, blockDim >> >();


	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());
}


void CUDAClass::destroyObject()
{
	/*if (d_pos != NULL) checkCudaErrors(cudaFree(d_pos));
	if (d_normal != NULL) checkCudaErrors(cudaFree(d_normal));
	if (d_tex != NULL) checkCudaErrors(cudaFree(d_tex));
	if (d_id != NULL) checkCudaErrors(cudaFree(d_id));
	if (d_octree != NULL) checkCudaErrors(cudaFree(d_octree));
	d_pos = NULL;
	d_normal = NULL;
	d_tex = NULL;
	d_id = NULL;
	d_octree = NULL;*/
}


void CUDAClass::cudaSetVolume(unsigned int width, unsigned int height, unsigned int depth, float diagonal)
{
	// create 3D array
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();

	// set texture parameters
	volume.normalized = true;                      // access with normalized texture coordinates
	volume.filterMode = cudaFilterModeLinear;      // linear interpolation
	volume.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	volume.addressMode[1] = cudaAddressModeClamp;
	volume.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource_volume,
		TextureManager::Inst()->GetID(TEXTURE_VOLUME),
		GL_TEXTURE_3D, cudaGraphicsMapFlagsReadOnly)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_volume, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_volume, cudaResource_volume, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindTextureToArray(volume, d_volume, channelDesc)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_volume, 0)); // Unmap the resource
	

	//Set the step
	float step = 1.f / diagonal;
	checkCudaErrors(cudaMemcpyToSymbol(constantH, &step, sizeof(float)));
}

void CUDAClass::cudaSetTransferFunction(float4 *d_transferFunction, unsigned int width)
{
	//Create channel description for 2D texture
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	// set texture parameters
	transferFunction.normalized = true;
	transferFunction.filterMode = cudaFilterModeLinear;
	transferFunction.addressMode[0] = cudaAddressModeClamp;
	transferFunction.addressMode[1] = cudaAddressModeClamp;
	

	// bind array to 2D texture
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource_TF,
					TextureManager::Inst()->GetID(TEXTURE_TRANSFER_FUNC),
					GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_TF, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_volume, cudaResource_TF, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindTextureToArray(transferFunction, d_volume)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_TF, 0)); // Unmap the resource
	
}


void CUDAClass::cudaSetImageSize(unsigned int width, unsigned int height, float NCP, float angle){

	checkCudaErrors(cudaMemcpyToSymbol(constantWidth, &width, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(constantHeight, &height, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(constantAngle, &angle, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cosntantNCP, &NCP, sizeof(float)));

	Width = width;
	Height = height;

	// bind array to 2D texture
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource_final,
					TextureManager::Inst()->GetID(TEXTURE_FINAL_IMAGE),
					GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_final, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_final, cudaResource_final, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindSurfaceToArray(surf, d_final)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_final, 0)); // Unmap the resource

}


void CUDAClass::cudaUpdateMatrix(const float * matrix){
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, matrix, sizeof(float4x4)));
}


void CUDAClass::Use(GLenum activeTexture){
	
	// copy from pbo to texture
	glActiveTexture(activeTexture);
	TextureManager::Inst()->BindTexture(TEXTURE_FINAL_IMAGE);
}