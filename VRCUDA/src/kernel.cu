


#include "kernel.cuh"

typedef unsigned char VolumeType;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

#define opacityThreshold 0.99


texture<VolumeType, 3, cudaReadModeNormalizedFloat> volume;         // 3D texture
texture<float4, 2, cudaReadModeElementType>         transferFunction; // 1D transfer function texture
#ifdef NOT_RAY_BOX
texture<float4, 2, cudaReadModeElementType>         texFirst, texLast;
#endif
surface<void, cudaSurfaceType2D> surf;



__constant__ unsigned int constantWidth, constantHeight;
__constant__ float constantH;

#ifdef LIGHTING
__constant__ float3 c_diffColor;
__constant__ float3 c_lightDir;
__constant__ float3 c_voxelJump;
#endif

#ifndef NOT_RAY_BOX
__constant__ float constantAngle, cosntantNCP;
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

#endif


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

#ifndef NOT_RAY_BOX
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


		float4 bg = make_float4(0.15f); //bg color here

		if (hit){

			if (tnear < cosntantNCP)
				tnear = cosntantNCP;     // clamp to near plane


			//float3 last = firsHit[tpos];
			//float3 first = lastHit[tpos];

			float3 first = make_float3(eyeRay.o + eyeRay.d*tnear);
			float3 last = make_float3(eyeRay.o + eyeRay.d*tfar);
			first = make_float3(first.x + 0.5f, first.y + 0.5f, 1.0f - (first.z + 0.5f));
			last = make_float3(last.x + 0.5f, last.y + 0.5f, 1.0f - (last.z + 0.5f));

			//Get direction of the ray
			float3 direction = last - first;
			float3 trans = first;
#else
		float4 first = tex2D(texFirst, x, y);
		float4 last = tex2D(texLast, x, y);

		//Get direction of the ray
		float3 direction = make_float3(last) - make_float3(first);
		float3 trans = make_float3(first);
#endif
		float D = length(direction);
		direction = normalize(direction);

		float4 color = make_float4(0.0f);
		color.w = 1.0f;

		float3 rayStep = direction * constantH;

		for (float t = 0; t <= D; t += constantH){

			//Sample in the scalar field and the transfer function
#ifdef NOT_RAY_BOX
			float scalar = tex3D(volume, trans.x, trans.y, trans.z);
#else
			float scalar = tex3D(volume, trans.x, trans.y, trans.z); //convert to texture space
#endif
			float4 samp = tex2D(transferFunction, scalar, 0.5f);
			//float scalar = 0.1;
			//float4 samp = make_float4(0.0f);

			//Calculating alpa
			samp.w = 1.0f - expf(-0.5 * samp.w);

			//Acumulating color and alpha using under operator 
			samp.x = samp.x * samp.w;
			samp.y = samp.y * samp.w;
			samp.z = samp.z * samp.w;

			//calculate lighting for the sample color
#ifdef LIGHTING

			// sample neightbours
			float3 normal;
			//sample right
			normal.x = tex3D(volume, trans.x + c_voxelJump.x, trans.y, trans.z) - scalar;
			normal.y = tex3D(volume, trans.x, trans.y + c_voxelJump.y, trans.z) - scalar;
			normal.z = tex3D(volume, trans.x, trans.y, trans.z - c_voxelJump.z) - scalar;

			//normalize normal
			normal = normalize(normal);

			float d = max(dot(c_lightDir, normal), 0.0f);

			samp.x = samp.x * d * c_diffColor.x;
			samp.y = samp.y * d * c_diffColor.y;
			samp.z = samp.z * d * c_diffColor.z;
#endif

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

#ifndef NOT_RAY_BOX
		}
	else{
		bg.w = 1.0f;

		//Write to the texture
		uchar4 ucolor = make_uchar4(bg.x * 255, bg.y * 255, bg.z * 255, bg.w * 255);
		surf2Dwrite(ucolor, surf, x * sizeof(uchar4), y, cudaBoundaryModeClamp);
	}
#endif
	}
}


CUDAClass::CUDAClass(dim3 dim)
{
	//d_lastHit = d_FirstHit = NULL;
	d_texture = d_volume = NULL;
	block_dim = dim;
	checkCudaErrors(cudaSetDevice(0));
	// Otherwise pick the device with highest Gflops/s


	/*d_pos = NULL;
	d_normal = NULL;
	d_tex = NULL;
	d_id = NULL;
	d_octree = NULL;
	d_texture = NULL;
	d_texW = d_texH = -1;*/

#ifdef LIGHTING
	glm::vec3 diffColor = glm::vec3(1.0, 0.0, 1.0f);
	checkCudaErrors(cudaMemcpyToSymbol(c_diffColor, glm::value_ptr(diffColor), sizeof(float3)));
#endif


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
	/*if (d_FirstHit != NULL) checkCudaErrors(cudaFree(d_FirstHit));
	if (d_lastHit != NULL) checkCudaErrors(cudaFree(d_lastHit));*/

	//d_lastHit = d_FirstHit = NULL;
	d_texture = d_volume = NULL;


	checkCudaErrors(cudaDeviceReset());
}


// Helper function for using CUDA to add vectors in parallel.
void CUDAClass::cudaRC(/*, unsigned int width, unsigned int height, float h, float4 *d_buffer, float4 *d_lastHit*/)
{


	volumeRenderingKernel << < grid_dim, block_dim >> >();


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
		TextureManager::Inst().GetID(TEXTURE_VOLUME),
		GL_TEXTURE_3D, cudaGraphicsMapFlagsReadOnly)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_volume, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_volume, cudaResource_volume, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindTextureToArray(volume, d_volume, channelDesc)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_volume, 0)); // Unmap the resource


	//Set the step
	float step = 1.f / diagonal;
	checkCudaErrors(cudaMemcpyToSymbol(constantH, &step, sizeof(float)));
}

void CUDAClass::cudaSetTransferFunction(unsigned int width)
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
		TextureManager::Inst().GetID(TEXTURE_TRANSFER_FUNC),
		GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_TF, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_volume, cudaResource_TF, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindTextureToArray(transferFunction, d_volume)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_TF, 0)); // Unmap the resource

}


void CUDAClass::cudaSetImageSize(unsigned int width, unsigned int height, float NCP, float angle){

	checkCudaErrors(cudaMemcpyToSymbol(constantWidth, &width, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(constantHeight, &height, sizeof(unsigned int)));

#ifndef NOT_RAY_BOX
	checkCudaErrors(cudaMemcpyToSymbol(constantAngle, &angle, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cosntantNCP, &NCP, sizeof(float)));
#endif

	Width = width;
	Height = height;

	//Update grid dimension
	grid_dim = dim3((Width + block_dim.x) / block_dim.x, (Height + block_dim.y) / block_dim.y, 1);

	// bind array to 2D texture
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource_final,
		TextureManager::Inst().GetID(TEXTURE_FINAL_IMAGE),
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_final, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_final, cudaResource_final, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindSurfaceToArray(surf, d_final)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_final, 0)); // Unmap the resource


#ifdef NOT_RAY_BOX
	// bind array to 2D texture
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource_First,
		TextureManager::Inst().GetID(TEXTURE_FRONT_HIT),
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_First, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_textureFirst, cudaResource_First, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindTextureToArray(texFirst, d_textureFirst)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_First, 0)); // Unmap the resource


	// bind array to 2D texture
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource_Last,
		TextureManager::Inst().GetID(TEXTURE_BACK_HIT),
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)); //Register the texture in a resource

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_Last, 0)); // Map the resource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_textureLast, cudaResource_Last, 0, 0)); //Get the mapped array

	checkCudaErrors(cudaBindTextureToArray(texLast, d_textureLast)); // Map the array to the surface

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_Last, 0)); // Unmap the resource
#endif
}


#ifdef LIGHTING
void CUDAClass::cudaUpdateLight(const float * lightDir){
	checkCudaErrors(cudaMemcpyToSymbol(c_lightDir, lightDir, sizeof(float3)));
}

void CUDAClass::cudaUpdateVoxelSize(const float * voxelJump){
	checkCudaErrors(cudaMemcpyToSymbol(c_voxelJump, voxelJump, sizeof(float3)));
}
#endif

#ifndef NOT_RAY_BOX
void CUDAClass::cudaUpdateMatrix(const float * matrix){
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, matrix, sizeof(float4x4)));
}
#endif

void CUDAClass::Use(GLenum activeTexture){

	// copy from pbo to texture
	glActiveTexture(activeTexture);
	TextureManager::Inst().BindTexture(TEXTURE_FINAL_IMAGE);
}