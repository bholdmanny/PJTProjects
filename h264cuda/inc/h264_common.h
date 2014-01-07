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


#ifndef __GPU_H264_COMMON_H__
#define __GPU_H264_COMMON_H__

#undef min
#define min(x,y) ((x<y)?(x):(y))
#undef max
#define max(x,y) ((x>y)?(x):(y))


typedef enum E_ERR {
    ERR_SUCCESS = 0,
    ERR_MEM,
    ERR_ARG,
    ERR_OVERFLOW,
    ERR_FAILURE
} E_ERR;

#define BYTES_IN_WORD 4
#define  ROUNDUP_STREAM_PROCESSORS(x) (-( (-(int)(x)) & -((int)16) ) )

#endif  
