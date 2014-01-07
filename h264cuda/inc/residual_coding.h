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


#ifndef _RESIDUAL_CODING_H
#define _RESIDUAL_CODING_H
#include "encoder_context.h"
#include "h264_types.h"
#include "mb_info.h"

#define USE_ME_SAD_FOR_MODE_DEC
#define CAVLC_LEVEL_LIMIT                   (2063)  

#define MC_RC_STRIP_SIZE                    ( 80 )  
#define IFRAME_RC_STRIP_SIZE                ( 60 )   

#define MERGE_CHROMA        1

#define MV_ZERO_INTER_SAD_BIAS    (256)


#define MAKE_DIV_SIZE(size)         ((((size) + (16 - 1)) / (16)) * (16))

#define DC_COEFS_BYTES_PER_MB     ( NUM_DC_COEFS_PER_MB * sizeof(short) )           // = 16 * 2 = 32
#define DC_COEFS_WORDS_PER_MB     ( DC_COEFS_BYTES_PER_MB / sizeof(int) )       // = 32 / 4 = 8

#define STRIP_BLOCKS(x)           ( x * BLOCKS_PER_MB )
#define STRIP_PIXELS(x)           ( STRIP_BLOCKS(x) * (BLK_SIZE / sizeof(int)) )
#define STRIP_PIXEL_ROW(x)        ( x * (MB_WIDTH / sizeof(int8x4)) )
#define STRIP_COEFS(x)            ( x * MB_TOTAL_SIZE * sizeof(short) / sizeof(int) )
#define STRIP_DC_COEFS(x)         ( MAKE_DIV_SIZE(x * DC_COEFS_WORDS_PER_MB) )



typedef struct S_QP_DATA
{
    int QpVal;
    int QuantAdd;
    int QuantAddDC;
    int QuantShift;
    int DQuantShift;
    int PredPenalty;
} S_QP_DATA;

void InitQPDataAndTablesFromQP (
    S_QP_DATA *pQpData,
    short **pQuantTable,
    short **pDQuantTable,
    int QPValue,
    int IsIDRFrame,
    int IsLuma
    );
#endif

