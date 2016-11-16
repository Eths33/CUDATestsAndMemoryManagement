/*Author Greg Gutmann*/
// Putting all of the code in a header file is an easy way to add it to projects (not the usual way though)
#ifndef __managedAlloc_INCLUDED__   // if x.h hasn't been included yet...
#define __managedAlloc_INCLUDED__   //   #define this so the compiler knows it has been included

#include "cuda_runtime.h"
#include <stdio.h>
//core of Linked list from http://www.thegeekstuff.com/2012/08/c-linked-list-example


#define PRINT_NON_ERROR 1
//Defines used to make the function calls to error messages purple to stand out in code
//This removes the mouse over information though, so they are set to something that resembles their inputs
#define checkKernelError const_char
#define checkError cudaError_t__const_char
//Could change this to be dynamic at run time by adding an extra parameter to allocate function
#define MEM_ALIGNMENT 32

enum memTypes {
	hostMem,
	hostAllignedMem,
	hostPinnedMem,
	deviceMem,
	textureMem,
	NONE
};

//Gets very messy if trying to save a <void**> pointer to a cudaTextureObject_t, so a special pointer was added to support textures. 
// Would be good to only use void** to have a single struct with no unused values for all cases. 
struct managedMem {
	struct managedMem * next;
	void ** mem;
	cudaTextureObject_t * texMem;
	memTypes type;
	int GPU;
	char name[32];
};

struct managedMem *head = NULL;
struct managedMem *curr = NULL;
template <class T> void allocateMem(T** dataForTexture, int length, int gpu, memTypes memType, const char * memName, cudaTextureObject_t texIndx);

//Error Checking that gets the English version of the error codes
void checkKernelError(const char * thing) {
	cudaError_t err = cudaSuccess;
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		// Convert a basic_string string to a wide character 
		// wchar_t* string. You must first convert to a char* for this to work.
		const size_t newsizew = strlen(thing) + 1;
		size_t convertedChars = 0;
		wchar_t *strMessage = new wchar_t[newsizew];
		mbstowcs_s(&convertedChars, strMessage, newsizew, thing, _TRUNCATE);

		// Convert char* string to a wchar_t* string.
		const char * errorCu = cudaGetErrorString(err);
		const size_t newsize = strlen(errorCu) + 1;
		wchar_t * cudaString = new wchar_t[newsize];
		convertedChars = 0;
		mbstowcs_s(&convertedChars, cudaString, newsize, errorCu, _TRUNCATE);

		wchar_t errorStr[300];
		cudaGetErrorString(err);
		swprintf(errorStr, 300, L"Failed to %ls \nError code %ls!", strMessage, cudaString);
		MessageBox(0, errorStr, L"Error", MB_ICONERROR);
		fwprintf(stderr, L"Failed to %ls (error code %ls)!\n", strMessage, cudaString);

	}
}
cudaError_t checkError(cudaError_t err, const char * thing) {
	if (err != cudaSuccess)
	{
		// Convert a basic_string string to a wide character 
		// wchar_t* string. You must first convert to a char* for this to work.
		const size_t newsizew = strlen(thing) + 1;
		size_t convertedChars = 0;
		wchar_t *strMessage = new wchar_t[newsizew];
		mbstowcs_s(&convertedChars, strMessage, newsizew, thing, _TRUNCATE);

		// Convert char* string to a wchar_t* string.
		const char * errorCu = cudaGetErrorString(err);
		const size_t newsize = strlen(errorCu) + 1;
		wchar_t * cudaString = new wchar_t[newsize];
		convertedChars = 0;
		mbstowcs_s(&convertedChars, cudaString, newsize, errorCu, _TRUNCATE);

		wchar_t errorStr[300];
		cudaGetErrorString(err);
		swprintf(errorStr, 300, L"Failed to %ls \nError code %ls!", strMessage, cudaString);
		MessageBox(0, errorStr, L"Error", MB_ICONERROR);
		fwprintf(stderr, L"Failed to %ls (error code %ls)!\n", strMessage, cudaString);

	}
	return err;
}


//Creates a head node, doesn't hold memory pointer
struct managedMem* create_list()
{
#if PRINT_NON_ERROR
	fprintf(stderr, "Creating list with headnode as [%d]\n", NONE);
#endif
	struct managedMem *ptr = (struct managedMem*)malloc(sizeof(struct managedMem));
	if (NULL == ptr)
	{
		fprintf(stderr, "\n Node creation failed \n");
		return NULL;
	}
	
	ptr->next = NULL;
	ptr->mem = NULL;
	ptr->type = NONE;
	ptr->GPU = 0;
	const char * memName = "headNode";
	strcpy(ptr->name, memName);

	head = curr = ptr;
	return ptr;
}
//Basic add to list & Special case add for textures
struct managedMem* add_to_list(void** memptr, int gpu, memTypes memTy, const char * memName)
{
	if (NULL == head)
	{
		create_list();
	}
#if PRINT_NON_ERROR
	fprintf(stderr, "Adding node [%d] %s\n", memTy, memName);
#endif
	struct managedMem *ptr = (struct managedMem*)malloc(sizeof(struct managedMem));
	if (NULL == ptr)
	{
		fprintf(stderr, "\n Node creation failed \n");
		return NULL;
	}
	ptr->next = NULL;
	ptr->mem = memptr;
	ptr->texMem = 0;
	ptr->type = memTy;
	ptr->GPU = gpu;
	strcpy(ptr->name, memName);

	curr->next = ptr;
	curr = ptr;

	return ptr;
}
struct managedMem* addTex_to_list(cudaTextureObject_t* memptr, int gpu, memTypes memTy, const char * memName)
{
	if (NULL == head)
	{
		create_list();
	}
#if PRINT_NON_ERROR
	fprintf(stderr, "Adding node [%d] %s\n", memTy, memName);
#endif
	struct managedMem *ptr = (struct managedMem*)malloc(sizeof(struct managedMem));
	if (NULL == ptr)
	{
		fprintf(stderr, "\n Node creation failed \n");
		return NULL;
	}
	ptr->next = NULL;
	ptr->mem = 0;
	ptr->texMem = memptr;
	ptr->type = memTy;
	ptr->GPU = gpu;
	strcpy(ptr->name, memName);

	curr->next = ptr;
	curr = ptr;

	return ptr;
}


//Free for basic memory types & Special case free for textures
template <class T> void freeMem(T** mem, int gpu, int memType, const char * memName) {
	if (memType == hostMem) {
		free(*mem); 
		*mem = NULL;
	}
	else if (memType == hostAllignedMem) {
		_aligned_free(*mem); 
		*mem = NULL;
	}
	else if (memType == hostPinnedMem) {
		cudaSetDevice(gpu);
		cudaError_t err = cudaFreeHost(*mem); 
		checkError(err, memName);
		*mem = NULL;
		cudaSetDevice(0);
	}
	else if (memType == deviceMem) {
		cudaSetDevice(gpu);
		cudaError_t err = cudaFree(*mem);
		checkError(err, memName);
		*mem = NULL;
		cudaSetDevice(0);
	}
}
void freeTex(cudaTextureObject_t* mem, int gpu, int memType, const char * memName) {
	cudaSetDevice(gpu);
	cudaError_t err = cudaDestroyTextureObject((cudaTextureObject_t)*mem);
	checkError(err, memName);
	(cudaTextureObject_t)*mem = 0;
	cudaSetDevice(0);
}


//Basic memory allocates & Special case for allocating textures due to complexity differences
template <class T> void allocateMem(T** mem, int length, int gpu, memTypes memType, const char * memName) {
	if (memType == hostMem) {
		*mem = (T*)malloc(sizeof(T)*length);
		if (!*mem) {
			fprintf(stderr, "Error %s\n", memName);
		}
		else {
			add_to_list((void**)mem, gpu, memType, memName);
		}
	}
	else if (memType == hostAllignedMem) {
		*mem = (T*)_aligned_malloc(sizeof(T)*length, MEM_ALIGNMENT);
		if (!*mem) {
			fprintf(stderr, "Error %s\n", memName);
		}
		else {
			add_to_list((void**)mem, gpu, memType, memName);
		}
	}
	else if (memType == hostPinnedMem) {
		cudaSetDevice(gpu);
		cudaError_t err = cudaHostAlloc(mem, sizeof(T)*length, cudaHostAllocPortable);
		if (checkError(err, memName) == cudaSuccess) {
			add_to_list((void**)mem, gpu, memType, memName);
		}
		cudaSetDevice(0);
	}
	else if (memType == deviceMem) {
		cudaSetDevice(gpu);
		cudaError_t err = cudaMalloc(mem, sizeof(T)*length);
		if (checkError(err, memName) == cudaSuccess) {
			add_to_list((void**)mem, gpu, memType, memName);
		}
		cudaSetDevice(0);
	}
}
//For Texture: If the last parameter is forgotten the function above will be called and ignore the attempt
template<> void allocateMem<float>(float** dataForTexture, int length, int gpu, memTypes memType, const char * memName, cudaTextureObject_t texIndx) {
	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = *dataForTexture;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = length*sizeof(float);
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	// create texture object: we only have to do this once!
	cudaError_t err = cudaCreateTextureObject(&texIndx, &resDesc, &texDesc, NULL);
	if (checkError(err, memName) == cudaSuccess) {
		addTex_to_list((cudaTextureObject_t*)&texIndx, gpu, memType, memName);
	}
	cudaSetDevice(0);
}
template<> void allocateMem<int>(int** dataForTexture, int length, int gpu, memTypes memType, const char * memName, cudaTextureObject_t texIndx) {
	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = *dataForTexture;
	resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = length*sizeof(int);
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	// create texture object: we only have to do this once!
	cudaError_t err = cudaCreateTextureObject(&texIndx, &resDesc, &texDesc, NULL);
	if (checkError(err, memName) == cudaSuccess) {
		addTex_to_list((cudaTextureObject_t*)&texIndx, gpu, memType, memName);
	}
	cudaSetDevice(0);
}
template<> void allocateMem<double>(double** dataForTexture, int length, int gpu, memTypes memType, const char * memName, cudaTextureObject_t texIndx) {
	//The hardware doesn't support double precision float as a texture format, but it is possible to use int2 and cast 
	//    it to double as long as you don't need interpolation:
#if PRINT_NON_ERROR
	fprintf(stderr, "Error double texture not supported currently\n");
#endif
}


//Calling this will clear all memory and set pointers to null(0) in the linked list structure
void deleteAllAllocations()
{
	struct managedMem *ptr = head; //Get first alloc
#if PRINT_NON_ERROR
	fprintf(stderr, "\n -------Printing list Start------- \n");
#endif
	while (ptr != NULL)
	{
#if PRINT_NON_ERROR
		fprintf(stderr, "deleting [%d] %s\n", ptr->type, ptr->name);
#endif
		if(ptr->type == textureMem)
			freeTex(ptr->texMem, ptr->GPU, ptr->type, ptr->name);
		else
			freeMem(ptr->mem, ptr->GPU, ptr->type, ptr->name);
		//Delete List Items
		struct managedMem *del = ptr;
		ptr = ptr->next;
		free(del); del = NULL;
	}
#if PRINT_NON_ERROR
	fprintf(stderr, "\n -------Printing list End------- \n");
#endif

	head = NULL;
	curr = NULL;

	return;
}


#endif