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


#ifndef __NAL_UNIT_H__
#define __NAL_UNIT_H__

#define MAX_NAL_TYPE_ID 31

typedef enum
{
	NALU_TYPE_SLICE = 1,
	NALU_TYPE_DPA   = 2,
	NALU_TYPE_DPB   = 3,
	NALU_TYPE_DPC   = 4,
	NALU_TYPE_IDR   = 5,
	NALU_TYPE_SEI   = 6,
	NALU_TYPE_SPS   = 7,
	NALU_TYPE_PPS   = 8,
	NALU_TYPE_PD    = 9,
	NALU_TYPE_ESEQ  = 10,
	NALU_TYPE_ESTRM = 11,
	NALU_TYPE_FILL  = 12

} NAL_TYPE_E;


#endif 
