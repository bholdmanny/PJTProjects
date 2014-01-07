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


#ifndef _DEBLOCK_H__
#define _DEBLOCK_H__
#include "mb_info.h"
#include "../inc/h264_types.h"
#include "../inc/mb_info.h"
#include "../inc/encoder_context.h"


#define DEBLK_TBLS_SIZE         (NUM_QP + (NUM_QP>>2) + (NUM_QP>>2))
#define DEBLK_IDXA_TBL_OFFSET   0
#define DEBLK_IDXB_TBL_OFFSET   (NUM_QP)
#define DEBLK_QP2CHR_TBL_OFFSET (NUM_QP + (NUM_QP>>2))

#define DEBLK_COMP_MBINFO_SIZE_IN_WORDS 2
typedef unsigned char S_BOUNDARY_STRENGTH_REF[2][4][4];
void CalcBoundaryStrength(S_BLK_MB_INFO *MBInfo,
                          S_BLK_MB_INFO *MBInfoLeft,
                          S_BLK_MB_INFO *MBInfoTop,
                          S_BOUNDARY_STRENGTH_REF *BSRef,
                          int disable_deblocking_filter_idc);

void DeblockMB(S_BOUNDARY_STRENGTH_REF *BSRef,
               int QPyCurr, int QPyLeft, int QPyTop,
               int alpha_c0_offset,
               int beta_offset,
               yuv_frame_t *frame,
               int MBx, int MBy,
               int DontFilterVertEdges, int DontFilterHorzEdges);

void EdgeLoopLuma(unsigned char* SrcPtr,
                  unsigned char Strength[4],
                  int QP,
                  int AlphaC0Offset, int BetaOffset,
                  int PtrInc, int inc);

void EdgeLoopChroma(unsigned char* SrcPtr,
                    unsigned char Strength[4],
                    int QP,
                    int AlphaC0Offset, int BetaOffset,
                    int PtrInc, int inc);
void pad_deblock_out_frame(yuv_frame_t *in_out, int PadAmount);
void deblock_colflat_chroma(
                                        unsigned char *in1,         // input Cb image
                                        unsigned char *in2,         // input Cr image
                                        unsigned char *out,         // output image (inputs combined)
                                        int num_rows,               // input image height
                                        int num_cols               // input image width
                                        );
E_ERR deblock_colflat_luma(
                                       unsigned char *in,          // input image
                                       unsigned char *out,         // output image
                                       int num_rows,               // input/output image height
                                       int num_cols                // input/output image width
                                       );


E_ERR deblock( encoder_context_t *p_enc );

#endif  
