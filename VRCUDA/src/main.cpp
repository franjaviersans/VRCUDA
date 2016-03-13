#include "kernel.cuh"
#include "Definitions.h"
#include "GLSLProgram.h"
#include "TransferFunction.h"
#include "TextureManager.h"
#include "CubeIntersection.h"
#include "FBOCube.h"
#include "FBOQuad.h"
#include "Volume.h"
#include "FinalImage.h"
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;



namespace glfwFunc
{
	GLFWwindow* glfwWindow;
	int WINDOW_WIDTH = 1024;
	int WINDOW_HEIGHT = 768;
	std::string strNameWindow = "CUDA Volume Ray Casting";

	const float NCP = 1.0f;
	const float FCP = 10.0f;
	const float fAngle = 45.f * (3.14f / 180.0f); //In radians

	//Declare the transfer function
	TransferFunction *g_pTransferFunc;

	char * volume_filepath = "./Raw/volume.raw";
	char * transfer_func_filepath = NULL;
	glm::ivec3 vol_size = glm::ivec3(256, 256, 256);
	glm::ivec2 working_group = glm::ivec2(8, 8);
	dim3 block_dimension(16, 16, 1);

	//Class to wrap cuda code
	CUDAClass * cuda;

	float color[]={1,1,1};
	bool pintar = false;

	glm::mat4x4 mProjMatrix, mModelViewMatrix, mMVP;

	//Variables to do rotation
	glm::quat quater = glm::quat(0.8847f, -.201f, 0.398f, -0.1339f), q2; //begin with a diagonal view
	glm::mat4x4 RotationMat = glm::mat4x4();
	float angle = 0;
	float *vector=(float*)malloc(sizeof(float)*3);
	double lastx, lasty;
	bool pres = false;

#ifdef NOT_RAY_BOX
	CCubeIntersection *m_BackInter, *m_FrontInter;
#endif
	CFinalImage *m_FinalImage;

#ifdef MEASURE_TIME
	std::ofstream time_file("Time.txt", std::ios::out);
	// helper variable
	LARGE_INTEGER temp;
	int num;
	LARGE_INTEGER start_time, end_time;
	double freq, diff_time;
#endif


	Volume *volume = NULL;

	
	///< Callback function used by GLFW to capture some possible error.
	void errorCB(int error, const char* description)
	{
		printf("%s\n",description );
	}


	///
	/// The keyboard function call back
	/// @param window id of the window that received the event
	/// @param iKey the key pressed or released
	/// @param iScancode the system-specific scancode of the key.
	/// @param iAction can be GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
	/// @param iMods Bit field describing which modifier keys were held down (Shift, Alt, & so on)
	///
	void keyboardCB(GLFWwindow* window, int iKey, int iScancode, int iAction, int iMods)
	{
		if (iAction == GLFW_PRESS)
		{
			switch (iKey)
			{
				case GLFW_KEY_ESCAPE:
				case GLFW_KEY_Q:
					glfwSetWindowShouldClose(window, GL_TRUE);
					break;
				case GLFW_KEY_SPACE:
					g_pTransferFunc->isVisible = !g_pTransferFunc->isVisible;
					break;
			}
		}
	}

	inline int TwEventMousePosGLFW3(GLFWwindow* window, double xpos, double ypos)
	{ 
	
		g_pTransferFunc->CursorPos(int(xpos), int(ypos));
		if (pres){
			//Rotation
			float dx = float(xpos - lastx);
			float dy = float(ypos - lasty);

			if (!(dx == 0 && dy == 0)){
				//Calculate angle and rotation axis
				float angle = sqrtf(dx*dx + dy*dy) / 50.0f;

				//Acumulate rotation with quaternion multiplication
				q2 = glm::angleAxis(angle, glm::normalize(glm::vec3(dy, dx, 0.0f)));
				quater = glm::cross(q2, quater);

				lastx = xpos;
				lasty = ypos;
			}
		}
		return false;
	}

	int TwEventMouseButtonGLFW3(GLFWwindow* window, int button, int action, int mods)
	{ 
		pres = false;

		double x, y;   
		glfwGetCursorPos(window, &x, &y);  

		if (!g_pTransferFunc->MouseButton((int)x, (int)y, button, action)){
			
			if(button == GLFW_MOUSE_BUTTON_LEFT){
				if(action == GLFW_PRESS){
					lastx = x;
					lasty = y;
					pres = true;
				}

				return true;
			}else{
				if(action == GLFW_PRESS){			
				}
				
			}
			
			return false;
		}

		return true;
	}
	
	///< The resizing function
	void resizeCB(GLFWwindow* window, int iWidth, int iHeight)
	{

		WINDOW_WIDTH = iWidth;
		WINDOW_HEIGHT = iHeight;

		if(iHeight == 0) iHeight = 1;
		float ratio = iWidth / float(iHeight);
		glViewport(0, 0, iWidth, iHeight);

		mProjMatrix = glm::perspective(float(fAngle), ratio, 1.0f, 10.0f);

		// Update size in some buffers!!!
#ifdef NOT_RAY_BOX
		m_BackInter->SetResolution(iWidth, iHeight);
		m_FrontInter->SetResolution(iWidth, iHeight);
#endif
		m_FinalImage->SetResolution(iWidth, iHeight);
		g_pTransferFunc->Resize(&WINDOW_WIDTH, &WINDOW_HEIGHT);
		
		//Set image Size
		cuda->cudaSetImageSize(iWidth, iHeight, NCP, fAngle/2.0f);
		
	}

	///< Function to warup opencl
	void WarmUP(unsigned int cycles, bool measure = false){

		RotationMat = glm::mat4_cast(glm::normalize(quater));

		mModelViewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, -2.0f)) *
			RotationMat;

		mMVP = mProjMatrix * mModelViewMatrix;

#ifndef NOT_RAY_BOX
		cuda->cudaUpdateMatrix(glm::value_ptr(glm::transpose(glm::inverse(mModelViewMatrix))));
#else
		//Obtain Back hits
		m_BackInter->Draw(mMVP);
		//Obtain the front hits
		m_FrontInter->Draw(mMVP);
#endif


#ifdef MEASURE_TIME
		if (measure){
			QueryPerformanceCounter((LARGE_INTEGER *)&start_time);	//set start time
		}
#endif


		for (int i = 0; i < cycles; ++i) {
			//CUDA volume ray casting
			cuda->cudaRC();
		}

#ifdef MEASURE_TIME
		if (measure){
			QueryPerformanceCounter((LARGE_INTEGER *)&end_time); //end time
			diff_time = (float)(((double)end_time.QuadPart - (double)start_time.QuadPart) / freq); // get total time
			diff_time /= cycles; // get time per cycle
			time_file << diff_time << endl;
			time_file.close();
		}
#endif
	}

	///< The main rendering function.
	void draw()
	{

		GLenum err = GL_NO_ERROR;
		while((err = glGetError()) != GL_NO_ERROR)
		{
		  std::cout<<"INICIO "<< err<<std::endl;
		}

#ifdef MEASURE_TIME
		QueryPerformanceCounter((LARGE_INTEGER *)&start_time);	//set start time
#endif
		
		RotationMat = glm::mat4_cast(glm::normalize(quater));

		mModelViewMatrix =  glm::translate(glm::mat4(), glm::vec3(0.0f,0.0f,-2.0f)) * 
							RotationMat; 


		mMVP = mProjMatrix * mModelViewMatrix;

#ifndef NOT_RAY_BOX
		cuda->cudaUpdateMatrix(glm::value_ptr(glm::transpose(glm::inverse(mModelViewMatrix))));
#else
		//Obtain Back hits
		m_BackInter->Draw(mMVP);
		//Obtain the front hits
		m_FrontInter->Draw(mMVP);
#endif


		//CUDA volume ray casting
		cuda->cudaRC();


		
		

		//Draw the quad with the final result
		glClearColor(0.15f, 0.15f, 0.15f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		//Blend with bg
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		cuda->Use(GL_TEXTURE0); //Use the texture
		m_FinalImage->Draw();
		glDisable(GL_BLEND);



		//Draw the AntTweakBar
		//TwDraw();

		//Draw the transfer function
		g_pTransferFunc->Display();

		glfwSwapBuffers(glfwWindow);

#ifdef MEASURE_TIME
		QueryPerformanceCounter((LARGE_INTEGER *)&end_time); //end time
		diff_time += (float)(((double)end_time.QuadPart - (double)start_time.QuadPart) / freq); // get total time
		++num;
#endif

		while((err = glGetError()) != GL_NO_ERROR)
		{
		  std::cout<<"Swap "<< err<<std::endl;
		  //exit(0);
		}

		
	}
	


	///
	/// Init all data and variables.
	/// @return true if everything is ok, false otherwise
	///
	bool initialize()
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glFrontFace(GL_CCW);
		glCullFace(GL_BACK);

		//Init GLEW
		glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK)
		{
			printf("- glew Init failed :(\n");
			return false;
		}
		printf("OpenGL version: %s\n", glGetString(GL_VERSION));
		printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
		printf("Vendor: %s\n", glGetString(GL_VENDOR));
		printf("Renderer: %s\n", glGetString(GL_RENDERER));

		cuda = new CUDAClass(block_dimension);


		//Init the transfer function
		g_pTransferFunc = new TransferFunction();
		g_pTransferFunc->InitContext(glfwWindow, &WINDOW_WIDTH, &WINDOW_HEIGHT, transfer_func_filepath, -1, -1);

		cuda->cudaSetTransferFunction(256);

		
		// send window size events to AntTweakBar
		glfwSetWindowSizeCallback(glfwWindow, resizeCB);
		glfwSetMouseButtonCallback(glfwWindow, (GLFWmousebuttonfun)TwEventMouseButtonGLFW3);
		glfwSetCursorPosCallback(glfwWindow, (GLFWcursorposfun)TwEventMousePosGLFW3);
		glfwSetKeyCallback(glfwWindow, (GLFWkeyfun)keyboardCB);


		//Create volume
		volume = new Volume();
		volume->Load(volume_filepath, vol_size.x, vol_size.y, vol_size.z);

		cuda->cudaSetVolume(vol_size.x, vol_size.y, vol_size.z, volume->m_fDiagonal);

		

#ifdef NOT_RAY_BOX
		m_BackInter = new CCubeIntersection(false, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_FrontInter = new CCubeIntersection(true, WINDOW_WIDTH, WINDOW_HEIGHT);
#endif
		m_FinalImage = new CFinalImage(WINDOW_WIDTH, WINDOW_HEIGHT);

#ifdef MEASURE_TIME
		// get the tick frequency from the OS
		QueryPerformanceFrequency((LARGE_INTEGER *)&temp);
		freq = ((double)temp.QuadPart) / 1000.0; //convert to the time needed
		num = 0;
#endif


		return true;
	}


	/// Here all data must be destroyed + glfwTerminate
	void destroy()
	{
#ifdef NOT_RAY_BOX
		delete m_BackInter;
		delete g_pTransferFunc;
#endif
		TextureManager::Inst()->UnloadAllTextures();
		glfwTerminate();
		glfwDestroyWindow(glfwWindow);
	}
}

int main(int argc, char** argv)
{

	if (argc == 7 || argc == 8) {

		//Copy volume file path
		glfwFunc::volume_filepath = new char[strlen(argv[1]) + 1];
		strncpy_s(glfwFunc::volume_filepath, strlen(argv[1]) + 1, argv[1], strlen(argv[1]));

		//Volume size
		int width = atoi(argv[2]), height = atoi(argv[3]), depth = atoi(argv[4]);
		glfwFunc::vol_size = glm::ivec3(width, height, depth);

		glfwFunc::block_dimension.x = atoi(argv[5]);
		glfwFunc::block_dimension.y = atoi(argv[6]);

		//Copy volume transfer function path
		if (argc == 8){
			glfwFunc::transfer_func_filepath = new char[strlen(argv[7]) + 1];
			strncpy_s(glfwFunc::transfer_func_filepath, strlen(argv[7]) + 1, argv[7], strlen(argv[7]));
		}

	}
	else if (argc > 6) {
		printf("Too many arguments supplied!!!! \n");
	}

	


	glfwSetErrorCallback(glfwFunc::errorCB);
	if (!glfwInit())	exit(EXIT_FAILURE);
	glfwFunc::glfwWindow = glfwCreateWindow(glfwFunc::WINDOW_WIDTH, glfwFunc::WINDOW_HEIGHT, glfwFunc::strNameWindow.c_str(), NULL, NULL);
	if (!glfwFunc::glfwWindow)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(glfwFunc::glfwWindow);
	if (!glfwFunc::initialize()) exit(EXIT_FAILURE);
	glfwFunc::resizeCB(glfwFunc::glfwWindow, glfwFunc::WINDOW_WIDTH, glfwFunc::WINDOW_HEIGHT);	//just the 1st time

	//WarmUP!!!!
	glfwFunc::WarmUP(20);


#ifndef NOT_DISPLAY
	// main loop!
#ifndef MEASURE_TIME
	while (!glfwWindowShouldClose(glfwFunc::glfwWindow))
	{
#else
	while (!glfwWindowShouldClose(glfwFunc::glfwWindow) && glfwFunc::num <= NUM_CYCLES)
	{
#endif

		if(glfwFunc::g_pTransferFunc->updateTexture) // Check if the color palette changed    
		{
			glfwFunc::g_pTransferFunc->UpdatePallete();
			glfwFunc::g_pTransferFunc->updateTexture = false;
		}
		glfwFunc::draw();
		glfwPollEvents();	//or glfwWaitEvents()
	}

#ifdef MEASURE_TIME
	glfwFunc::diff_time /= glfwFunc::num; // get time per cycle
	glfwFunc::time_file << glfwFunc::diff_time << endl;
	glfwFunc::time_file.close();
#endif
#else

	glfwFunc::WarmUP(NUM_CYCLES, true);

#endif

	glfwFunc::destroy();
	return EXIT_SUCCESS;
}
