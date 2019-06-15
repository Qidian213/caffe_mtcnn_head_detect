#pragma once

#ifdef THIS_CAFFE_DLL
#define EXPORT_CAFFE_DLL  __declspec(dllexport) 
#else
#define EXPORT_CAFFE_DLL 
#endif


EXPORT_CAFFE_DLL void caffe_headSetModelPath(char * path);
EXPORT_CAFFE_DLL int  caffe_headDetectMarks(char * rgbdata, int w, int h,float* fRect);


