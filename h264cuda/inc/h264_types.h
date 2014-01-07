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


#ifndef _TYPES_H
#define _TYPES_H

#define TRUE  1
#define FALSE 0

typedef short PIX_COEFF_TYPE;
typedef unsigned short UINT16;
typedef unsigned int UINT32;
typedef short SINT16;
typedef int SINT32;
typedef unsigned char PIXEL;

typedef short INT16;
typedef unsigned int UINT32;
typedef short SINT16;
typedef int SINT32;
typedef char CHAR;
typedef unsigned char UCHAR;

/// Types of slices
typedef enum
{
        SLICE_P = 0,
        SLICE_B,
        SLICE_I,
        SLICE_SP,
        SLICE_SI,
        NUMBER_SLICETYPES
} slicetype_e;

/// YUV pixel domain image arrays for uncompressed video frame
typedef struct yuv_frame_t
{
        int     width;          // frame (allocated) buffer width in bytes, always dividible by 16;
        int     height;         // frame (allocated) buffer height in bytes;
        int     image_width;    // actual image width in pixels, less or equal to buffer width;
        int     image_height;   // actual image height in pixels, less or equal to buffer height;
        unsigned char *y;       // pointer to Y component data
        unsigned char *u;       // pointer to U component data
        unsigned char *v;       // pointer to V component data
        int     buffer_owner;   // whether the y, u, v buffers are owned by this frame
} yuv_frame_t;

#endif 
