// $Id: $
//
// Author: Francisco Sans franjaviersans@gmail.com
//
// Complete history on bottom of file

#define FILE_REVISION "$Revision: $"

#include "Volume.h"
#include <fstream>
#include <iostream>
#include "TextureManager.h"

using namespace std;

/**
* Default constructor
*/
Volume::Volume()
{
	Init();
	volume = NULL;
}


/**
* Default destructor
*/
Volume::~Volume()
{
	if (volume != NULL) delete [] volume;
}


void Volume::Init()
{
	
}

void Volume::Load(string filename, GLuint width, GLuint height, GLuint depth)
{
	//Read texture from file
	std::ifstream textureFile(filename, std::ios::binary);
	
	if(!textureFile.is_open()){
		cout<<"Couldn't load file"<<endl;
		exit(0);
	}

	int length = -1;
	textureFile.seekg (0, textureFile.end);
	length = int(textureFile.tellg());

	if (length != width * height * depth){
		cout << "Bad volume size or rong file!" << endl;
		exit(0);
	}

    textureFile.seekg (0, textureFile.beg);

	volume = new char[length];
	textureFile.read(volume, length);

	textureFile.close();

	m_fWidht = (float)width;
	m_fHeigth = (float)height;
	m_fDepth = (float)depth;
	m_fDiagonal = sqrtf(float(width * width + height * height + depth * depth));

	//std::cout<<width<<"  "<<height<<" "<<depth<<" "<<length<<" "<<width* height* depth<<std::endl;

	//for(int i=0; i<length; ++i) memtexture[i] = 255;

	//Create Texture
	TextureManager::Inst()->CreateTexture3D(TEXTURE_VOLUME, width, height, depth, GL_RED, GL_RED, GL_FLOAT, GL_LINEAR, GL_LINEAR, volume);
}

void Volume::Use(GLenum activeTexture)
{
	glActiveTexture(activeTexture);
	TextureManager::Inst()->BindTexture(TEXTURE_VOLUME);
}


#undef FILE_REVISION

// Revision History:
// $Log: $
// $Header: $
// $Id: $