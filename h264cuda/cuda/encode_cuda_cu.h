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


#include "../inc/encoder_context.h"

void encode_cuda(encoder_context_t *p_enc );
//void cudaCalcBoundaryStrength(S_BLK_MB_INFO *pBlkMBInfo,
//							  S_BOUNDARY_STRENGTH_REF *BSRef,
//							  int disable_deblocking_filter_idc,
//						      int MBNumX,
//						      int MBNumY,
//							  int Slice_num,
//							  int Slice_flag
//						      ) ;