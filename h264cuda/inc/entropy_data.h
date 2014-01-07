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

#ifndef _ENTROPY_DATA_H
#define _ENTROPY_DATA_H

#include "h264_types.h"

typedef struct bitstream_t
{
	unsigned char *p_buffer;
	unsigned char *p_buffer_curr;
	unsigned int buffer_size;
	unsigned int bitbuf;
	int zero_count;
	int bitpos;
} bitstream_t;

#endif
