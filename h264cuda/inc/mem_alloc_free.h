/* ##################################################################### */ 
/*                                                                       */ 
/* Notice:   COPYRIGHT (C) GPU,GROUP. 2010                  */ 
/*           THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPU GROUP        */ 
/*           THE PROGRAM MAY ONLY     */ 
/*           BE USED IN A MANNER EXPLICITLY SPECIFIED IN THE GPU,       */ 
/*           WHICH INCLUDES LIMITATIONS ON COPYING, MODIFYING,           */ 
/*           REDISTRIBUTION AND WARANTIES. UNAUTHORIZED USE OF THIS      */ 
/*           PROGRAM IS SCTRICTLY PROHIBITED.                     */ 
/* ##################################################################### */ 

#include "h264_common.h"
#ifndef _MEM_ALLOC_FREE_H
#define _MEM_ALLOC_FREE_H

#define MAKE_MEM_ALIGN_32(addr) (((addr) + 31) & ~31)  
 
void *getmem_1d_void(int size);
void free_1D(void *ptr);
E_ERR alloc_yuv_frame(yuv_frame_t * frame, int width, int height, int img_width, int img_height);
void alloc_empty_yuv_frame(yuv_frame_t *frame, int width, int height, int img_width, int img_height);
void free_yuv_frame(yuv_frame_t * frame);
#endif

