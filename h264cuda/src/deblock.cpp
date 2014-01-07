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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "../inc/h264_types.h"
#include "../inc/mb_info.h"
#include "../inc/encoder_context.h"

#include "../inc/const_defines.h"
#include "../inc/encoder_tables.h"
#include "../inc/deblock.h"

// Macro used to define streams
#ifndef MAX_STRIP_SIZE
#define MAX_STRIP_SIZE 44
#endif // MAX_STRIP_SIZE

// Maximum frame width deblocking can handle
#define MAX_DEBLOCK_FRM_WIDTH  1920
#define MAX_DEBLOCK_FRM_WIDTHC (MAX_DEBLOCK_FRM_WIDTH/2)

// 2 Extra MBs for padding
#define Y_STRIP_LEN_HDEC        (((MAX_STRIP_SIZE+1+2)*2*8*8)/4)
#define Y_STRIP_LEN_QDEC        ((2*(MAX_STRIP_SIZE+1+2)*2*4*4)/4)

#define Y_STRIP_LEN             (((MAX_STRIP_SIZE+1+2)*2*16*16)/4)
#define UV_STRIP_LEN            (((MAX_STRIP_SIZE+1+2)*2*2*8*8)/4)
#define Y_INTER_STRIP_LEN       (((MAX_STRIP_SIZE+1)+3)*8*16)
#define Y_TOP_INTER_STRIP_LEN   (((MAX_STRIP_SIZE+1)+3)*4*16)
#define Y_BOT_INTER_STRIP_LEN   Y_TOP_INTER_STRIP_LEN
#define UV_INTER_STRIP_LEN      (((MAX_STRIP_SIZE+1)+3)*4*16)
#define UV_TOP_INTER_STRIP_LEN  (((MAX_STRIP_SIZE+1)+3)*1*16)
#define UV_BOT_INTER_STRIP_LEN  (((MAX_STRIP_SIZE+1)+3)*2*16)
#define MV_INFO_STRIP_LEN       ((MAX_STRIP_SIZE+1) * BLOCKS_PER_MB * 2)
#define LF_MB_INFO_STRIP_LEN    ((MAX_STRIP_SIZE+1) * 16)
#define MB_INFO_STRIP_LEN       ((MAX_STRIP_SIZE+1) * BLOCKS_PER_MB * 2)
#define BS_ABTC_STR_LEN         ((MAX_STRIP_SIZE+1+3)*16*12)

#define Y_FRMTOP_STRIP_LEN      (16*(MAX_DEBLOCK_FRM_WIDTH+2*16)/4)
#define U_FRMTOP_STRIP_LEN      (16*(MAX_DEBLOCK_FRM_WIDTHC+2*8)/4)
#define V_FRMTOP_STRIP_LEN      U_FRMTOP_STRIP_LEN
#define Y_FRMTOP_CF_STRIP_LEN   Y_FRMTOP_STRIP_LEN
#define UV_FRMTOP_CF_STRIP_LEN  U_FRMTOP_STRIP_LEN

#define Y_FRMTOP_STRIP_LEN_HLF  (16*(MAX_DEBLOCK_FRM_WIDTH/2+2*8)/4)
#define Y_FRMTOP_STRIP_LEN_QT  (16*(MAX_DEBLOCK_FRM_WIDTH/4+2*4)/4)

#define ROUND_UPTO_NEXT2nX(x, n) (((x) + (1<<(n)) - 1) & ~((1<<(n)) - 1))

int p_colf_topy_idx[ROUND_UPTO_NEXT2nX(8*MAX_STRIP_SIZE+24, 4)];
int p_padd_topy_idx[16];
int p_colf_topuv_idx[16];
int p_padd_topuv_idx[16];

#define  Y_IDX_STR_LENGTH               (2*16)
#define  Y_BOT_IDX_STR_LENGTH           (2*16)
#define  UV_IDX_STR_LENGTH              (2*16)
#define  UV_BOT_IDX_STR_LENGTH          (2*16)
#define  UV_OUTP_IDX_STR_LENGTH         (2*16)
#define  UV_BOT_OUTP_IDX_STR_LENGTH     (2*16)
//#define  MB_INFO_IDX_STR_LENGTH         ((MAX_STRIP_SIZE+1)*2*8)
#define  MB_INFO_IDX_STR_LENGTH         ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4)
//#define  MB_INFO_IDX_STR_BOT_LENGTH     ((MAX_STRIP_SIZE+1)*2*8)
#define  MB_INFO_IDX_STR_BOT_LENGTH     ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4)
//#define  LF_MB_INFO_IDX_STR_LENGTH      (1*STREAM_PROCESSORS)
#define MV_INFO_IDX_STR_LENGTH          (ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4))
#define MV_INFO_IDX_STR_BOT_LENGTH          (ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4))

#define  Y_IDX_STR_OFFSET               0
#define  Y_BOT_IDX_STR_OFFSET           (Y_IDX_STR_OFFSET + Y_IDX_STR_LENGTH)
#define  UV_IDX_STR_OFFSET              (Y_BOT_IDX_STR_OFFSET + Y_BOT_IDX_STR_LENGTH)
#define  UV_BOT_IDX_STR_OFFSET          (UV_IDX_STR_OFFSET + UV_IDX_STR_LENGTH)
#define  UV_OUTP_IDX_STR_OFFSET         (UV_BOT_IDX_STR_OFFSET + UV_BOT_IDX_STR_LENGTH)
#define  UV_BOT_OUTP_IDX_STR_OFFSET     (UV_OUTP_IDX_STR_OFFSET + UV_OUTP_IDX_STR_LENGTH)
#define  MB_INFO_IDX_STR_OFFSET         (UV_BOT_OUTP_IDX_STR_OFFSET + UV_BOT_OUTP_IDX_STR_LENGTH)
#define  MB_INFO_IDX_STR_BOT_OFFSET     (MB_INFO_IDX_STR_OFFSET+MB_INFO_IDX_STR_LENGTH)
#define  MV_INFO_IDX_STR_OFFSET			(MB_INFO_IDX_STR_BOT_OFFSET+MB_INFO_IDX_STR_BOT_LENGTH)
#define  MV_INFO_IDX_STR_BOT_OFFSET		(MV_INFO_IDX_STR_OFFSET+MV_INFO_IDX_STR_LENGTH)

#define IDX_STR_SIZE     				(MV_INFO_IDX_STR_BOT_OFFSET + MV_INFO_IDX_STR_BOT_LENGTH)

//-------------------------------------------------------------------
// Local defines, macros, and constants
//-------------------------------------------------------------------

#define CLIP0_255(x) (IClip(0, 255, (x)))
#define  IClip(Min, Max, Val) (((Val)<(Min)) ? (Min) : (((Val)>(Max)) ? (Max) : (Val)))

// Indexed with [direction][edge][4x4 block]
//   direction: 0 - vertical, 1 - horizontal
//   edge: 0-3, from left to right (dir = 0) or top to bottom (dir = 1)
//   block: 0-3: which block within a particular edge, from top to
//               bottom for dir=0, and left to right for dir=1.
//typedef unsigned char S_BOUNDARY_STRENGTH_REF[2][4][4];

//-------------------------------------------------------------------
// Local protocols
//-------------------------------------------------------------------


// Make BSRef a global var, will be very useful for debugging
//S_BOUNDARY_STRENGTH_REF *BSRef = NULL;


// Function to allocate and initialize deblocking context at the start of a sequence
// encoding. It is assumed that frame height and width would stay the same
// through all frames in a sequence.
E_ERR init_deblock_context(encoder_context_t *p_enc)
{
    yuv_frame_t *frame = p_enc->pRecFrame;
    int image_width = frame->image_width;
    int image_height = frame->image_height;
    int rec_width = frame->width;
    int rec_height = frame->height;
    int mb_numx = image_width>>4;
    int mb_numy = image_height>>4;
    int num_chroma_pix = (rec_width * rec_height)/4;

    // All tables used by deblocking collapsed into single
    // memory block. All tables are sized to in multiple of
    // words.
    int *p_deblk_tbls;
    int *p_y_idx;
    int *p_uv_idx;
    int *p_lf_mb_info_idx;
    int *p_mb_info_idx;
    int *p_mb_info_idx_bot;
    int *p_mv_info_idx;
    int *p_mv_info_idx_bot;

    int *p_colf_y_idx;
    int *p_colf_uv_idx;
    int *p_col_flat_frm_topuv_idx;
	int	 *offsets_array	;

    int i;
    E_ERR err = ERR_SUCCESS;

    p_deblk_tbls = (int *)malloc(DEBLK_TBLS_SIZE * sizeof(int));
	offsets_array = (int *)malloc(sizeof(int) * IDX_STR_SIZE);
	p_y_idx = offsets_array;
	p_uv_idx = &(offsets_array[UV_IDX_STR_OFFSET]);
	p_mb_info_idx = &(offsets_array[MB_INFO_IDX_STR_OFFSET]);
	p_mb_info_idx_bot = &(offsets_array[MB_INFO_IDX_STR_BOT_OFFSET]);
	p_mv_info_idx = &(offsets_array[MV_INFO_IDX_STR_OFFSET]);
    p_mv_info_idx_bot = &(offsets_array[MV_INFO_IDX_STR_BOT_OFFSET]);

    p_lf_mb_info_idx = (int *)malloc(16 * sizeof(int));

    p_colf_y_idx = (int *)malloc(ROUND_UPTO_NEXT2nX(8*MAX_STRIP_SIZE+24, 4) * sizeof(int));
    p_colf_uv_idx =(int *) malloc(ROUND_UPTO_NEXT2nX(8*MAX_STRIP_SIZE+24, 4) * sizeof(int));
    p_col_flat_frm_topuv_idx =(int *) malloc((8*128) * sizeof(int));

    if (! (p_y_idx && p_uv_idx && p_lf_mb_info_idx && p_mb_info_idx &&
           p_mb_info_idx_bot && p_colf_y_idx && p_colf_uv_idx &&
           p_col_flat_frm_topuv_idx && p_mv_info_idx && p_mv_info_idx_bot)){
        printf("Failed to allocate memory for deblock context %s:%d \n", __FILE__, __LINE__);
        err = ERR_MEM;
    }

    if (!err)
    {
        // Copy tables into unified table memory
        memcpy(p_deblk_tbls + DEBLK_IDXA_TBL_OFFSET, IndexATable, sizeof(IndexATable));
        memcpy(p_deblk_tbls + DEBLK_IDXB_TBL_OFFSET, IndexBTable, sizeof(IndexBTable));
        memcpy(p_deblk_tbls + DEBLK_QP2CHR_TBL_OFFSET, QP_TO_CHROMA_MAPPING, sizeof(QP_TO_CHROMA_MAPPING));

        for (i = 0; i < 16; i++){
            p_y_idx[i]     = i*2*rec_width;
            p_y_idx[16+i]  = (i*2+1)*rec_width;
            p_uv_idx[i]    = i*(rec_width/2);
            p_uv_idx[16+i] = i*(rec_width/2) + num_chroma_pix;
        }
        //Populate MB Info indices
        for (i = 0; i < (MAX_STRIP_SIZE+1); i++){
        int rec_size = sizeof(S_BLK_MB_INFO_COMPRESSED);
        int mv_rec_size =sizeof(int);//= sizeof(int16x2);

        p_mb_info_idx[4*i]   = 16*i*rec_size;
        p_mb_info_idx[4*i+1] = 16*i*rec_size + 8*rec_size;
        p_mb_info_idx[4*i+2] = 16*i*rec_size + 16*mb_numx*rec_size;
        p_mb_info_idx[4*i+3] = 16*i*rec_size + 16*mb_numx*rec_size + 8*rec_size;

        p_mv_info_idx[4*i]   = 16*i*mv_rec_size;
        p_mv_info_idx[4*i+1] = 16*i*mv_rec_size + 8*mv_rec_size;
        p_mv_info_idx[4*i+2] = 16*i*mv_rec_size + 16*mb_numx*mv_rec_size;
        p_mv_info_idx[4*i+3] = 16*i*mv_rec_size + 16*mb_numx*mv_rec_size + 8*mv_rec_size;

        }

    // Generate bottom index streams for frames whose height is not
    // a multiple of 32.
        if (mb_numy % 2){
            for (i = 0; i < 16; i++){
                p_y_idx[32+i] = (i >= 8) ? 0 : (mb_numy-1)*rec_width*16 + i*2*rec_width;
                p_y_idx[48+i] = (i >= 8) ? 0 : (mb_numy-1)*rec_width*16 + (i*2+1)*rec_width;
                // If frame height is not a 32 multiple, bottom
                // most row will load inconsequential data from 0th row
                // of the frame. And this load should also 0 as base offset
                // to avoid using negative values in index stream. Note bot
                // stream is used only when height is not multiple of 32!!
                p_uv_idx[32+i] = (i >= 8) ? 0 : ((mb_numy-1)*(rec_width/2)*8 + i*(rec_width/2));
                p_uv_idx[48+i] = p_uv_idx[32+i] + num_chroma_pix;
            }
            for (i = 0; i < (MAX_STRIP_SIZE+1); i++){
                int rec_size = sizeof(S_BLK_MB_INFO_COMPRESSED);
                int mv_rec_size ;//= sizeof(int16x2);
                p_mb_info_idx_bot[4*i]   = 16*i*rec_size;
                p_mb_info_idx_bot[4*i+1] = 16*i*rec_size + 8*rec_size;
                p_mb_info_idx_bot[4*i+2] = 16*i*rec_size;
                p_mb_info_idx_bot[4*i+3] = 16*i*rec_size + 8*rec_size;

                p_mv_info_idx_bot[4*i]   = 16*i*mv_rec_size;
                p_mv_info_idx_bot[4*i+1] = 16*i*mv_rec_size + 8*mv_rec_size;
                p_mv_info_idx_bot[4*i+2] = 16*i*mv_rec_size;
                p_mv_info_idx_bot[4*i+3] = 16*i*mv_rec_size + 8*mv_rec_size;
            }
        }

        for (i = 0; i < (MAX_STRIP_SIZE+3); i++){
            p_colf_y_idx[8*i]   = i*64;
            p_colf_y_idx[8*i+1] = i*64 + 4*rec_width;
            p_colf_y_idx[8*i+2] = i*64 + 8*rec_width;
            p_colf_y_idx[8*i+3] = i*64 + 12*rec_width;
            p_colf_y_idx[8*i+4] = i*64 + 16*rec_width;
            p_colf_y_idx[8*i+5] = i*64 + 20*rec_width;
            p_colf_y_idx[8*i+6] = i*64 + 24*rec_width;
            p_colf_y_idx[8*i+7] = i*64 + 28*rec_width;

            p_colf_uv_idx[8*i]   = i*32;
            p_colf_uv_idx[8*i+1] = i*32 + 2*rec_width;
            p_colf_uv_idx[8*i+2] = i*32 + 4*rec_width;
            p_colf_uv_idx[8*i+3] = i*32 + 6*rec_width;
            p_colf_uv_idx[8*i+4] = i*32 + 8*rec_width;
            p_colf_uv_idx[8*i+5] = i*32 + 10*rec_width;
            p_colf_uv_idx[8*i+6] = i*32 + 12*rec_width;
            p_colf_uv_idx[8*i+7] = i*32 + 14*rec_width;

            p_colf_topy_idx[8*i]   = i*64;
            p_colf_topy_idx[8*i+1] = i*64 + 32;
            p_colf_topy_idx[8*i+2] = i*64 + 4*rec_width;
            p_colf_topy_idx[8*i+3] = i*64 + 4*rec_width + 32;
            p_colf_topy_idx[8*i+4] = i*64 + 8*rec_width;
            p_colf_topy_idx[8*i+5] = i*64 + 8*rec_width + 32;
            p_colf_topy_idx[8*i+6] = i*64 + 12*rec_width;
            p_colf_topy_idx[8*i+7] = i*64 + 12*rec_width+ 32;
        }
        for (i = 0; i < rec_width/16; i++){
            p_col_flat_frm_topuv_idx[8*i]   = i * 16;
            p_col_flat_frm_topuv_idx[8*i+1] = i * 16 + rec_width;
            p_col_flat_frm_topuv_idx[8*i+2] = i * 16 + 2*rec_width;
            p_col_flat_frm_topuv_idx[8*i+3] = i * 16 + 3*rec_width;
            p_col_flat_frm_topuv_idx[8*i+4] = i * 16 + 4*rec_width;
            p_col_flat_frm_topuv_idx[8*i+5] = i * 16 + 5*rec_width;
            p_col_flat_frm_topuv_idx[8*i+6] = i * 16 + 6*rec_width;
            p_col_flat_frm_topuv_idx[8*i+7] = i * 16 + 7*rec_width;
        }
        for (i = 0; i < 16; i++){
            p_padd_topy_idx[i] = i*rec_width;
            p_padd_topuv_idx[i] = (i < 8) ? (i*rec_width/2) : ((i-8)*rec_width/2 + num_chroma_pix);
            p_colf_topuv_idx[i] = (i < 4) ? (i*2*rec_width) : (4*2*rec_width);
        }

        // Update context structure with allocated pointers
        p_enc->deblock_ctxt.p_deblk_tbls        = p_deblk_tbls;
        p_enc->deblock_ctxt.p_y_idx             = p_y_idx;
        p_enc->deblock_ctxt.p_uv_idx            = p_uv_idx;
        p_enc->deblock_ctxt.p_lf_mb_info_idx    = p_lf_mb_info_idx;
        p_enc->deblock_ctxt.p_mb_info_idx       = p_mb_info_idx;
        p_enc->deblock_ctxt.p_mb_info_idx_bot   = p_mb_info_idx_bot;
        p_enc->deblock_ctxt.p_mv_info_idx       = p_mv_info_idx;
        p_enc->deblock_ctxt.p_mv_info_idx_bot   = p_mv_info_idx_bot;
        p_enc->deblock_ctxt.p_colf_y_idx        = p_colf_y_idx;
        p_enc->deblock_ctxt.p_colf_uv_idx       = p_colf_uv_idx;
        p_enc->deblock_ctxt.p_col_flat_frm_topuv_idx = p_col_flat_frm_topuv_idx;

    }
    return err;
}

void free_deblock_context(encoder_context_t *p_enc)
{
    free(p_enc->deblock_ctxt.p_deblk_tbls);
    free(p_enc->deblock_ctxt.p_y_idx);
    free(p_enc->deblock_ctxt.p_lf_mb_info_idx);
    free(p_enc->deblock_ctxt.p_colf_y_idx);
    free(p_enc->deblock_ctxt.p_colf_uv_idx);
    free(p_enc->deblock_ctxt.p_col_flat_frm_topuv_idx);
}





//====================================================================
//   The main MB-filtering function.  The basic strategy is to
//   translate all the information in the VSofts data structure that
//   are passed to the function into a new simplified data structure.
//--------------------------------------------------------------------
E_ERR deblock(
                       encoder_context_t *p_enc
                       )
{
    E_ERR err = ERR_SUCCESS;
    S_BLK_MB_INFO *pBlkMBInfo          = p_enc->pBlkMBInfo;
    int disable_deblocking_filter_idc  = p_enc->loopfilter_params.disable_flag;
    int slice_alpha_c0_offset          = p_enc->loopfilter_params.alpha_c0_offset;
    int slice_beta_offset              = p_enc->loopfilter_params.beta_offset;
    yuv_frame_t *frame                 = p_enc->pRecFrame; // Input & Output
	clock_t start,end;
    
    // Do not access p_enc directly below this line
    int MBNumX = frame->image_width>>4;
    int MBNumY = frame->image_height>>4;
    int MBcurr, MBcurrLeft, MBcurrTop;
    int s, x, y, m, CurrStripSize;
    
    int StripSize = 24;   // = getStripSize();
    // The StripSize variable doesn't account for the extra boundary
    // macroblock that needs to be processed in order to stitch
    // together the filtering of two adjacent strips (See comments
    // below on the "Main stream loop").  So we have to add +1 when we
    // use this value in many places.
    int NumStrips = (MBNumX / StripSize) + ((MBNumX % StripSize) ? 1 : 0);
 
    S_BOUNDARY_STRENGTH_REF *BSRef = (S_BOUNDARY_STRENGTH_REF *)malloc(MBNumX * MBNumY * sizeof(*BSRef));
    if (!BSRef) {
        printf("Error allocating memory for loop filter boundary strength indices!!  Exiting...");
        exit(-1);
    }
	
 //   start = clock();    
	//FILE *output_file = NULL;
	//output_file = fopen("strength.txt", "wb");
	////fwrite(BSRef, 1, MBNumX*MBNumY*32, output_file);
	//fwrite(frame->y + frame->width*(frame->height-frame->image_height)/2 , 1, frame->width*frame->image_height, output_file);
	//fclose(output_file);

	/*cudaCalcBoundaryStrength(pBlkMBInfo,
							BSRef,
							disable_deblocking_filter_idc,
							MBNumX,
							MBNumY
							);*/
    /*cudaDeblockMB(BSRef,
                  pBlkMBInfo,
                  slice_alpha_c0_offset,
                  slice_beta_offset,
                  frame,
				  MBNumX,
				  MBNumY);*/

    //  The computation is strip-mined such that a strip is a
    //  consecutive group of macroblocks all on the same row.  The
    //  CudaC code is written to process strips in vertical order,
    //  rather than in row-major order.  That is, the loop processes
    //  the strips contain the macroblocks at macroblock coordinates
    //  (0,0) to (N,0), then (0,1) to (N,1), etc.  At this point, it
    //  goes back to the top and process MB strips (N,0) to (2N-1,0),
    //  then (2N,1) to (2N-1,1), and so on.  (Notice the overlap of
    //  the Nth macroblock on each row in each pair of strips).
    //
    //  This overlap is needed because of the nature of the deblocking
    //  algorithm described in the H.264 standard.  In order to
    //  conform to the standard, if the last macroblock (call it
    //  LastStripMB) is on the right edge of the image, no special
    //  boundary conditions need to be handled.  If LastStripMB is in
    //  the interior of the image, than only the vertical edges are
    //  filtered for that macroblock.  In this case, when the strip
    //  next to the current one is processed (i.e., the strip that
    //  contains the macroblocks from LastStripMB to
    //  LastStripMB+StripSize), only the horizontal edges will be
    //  filtered for LastStripMB.  But notice that LastStripMB must be
    //  processed by the kernel twice.  This is slightly inefficient,
    //  but if the strips are large enough (say > 25
    //  macroblocks/strip) this has a neglible impact on performance.

    // Loop over row segments (each row segment is StripSize
    // wide, though the last segment may be < StripSize wide)
    for (s = 0; s < NumStrips; s++) {

        // Calculate x macroblock coordinate for this strip
        x = s * (StripSize);
        // The last strip might be smaller than the rest
        CurrStripSize = ((x + StripSize + 1) <= MBNumX) ? (StripSize+1) : (MBNumX-x);

        for (y = 0 ; y < MBNumY ; y++ ) {

            for (m = 0; m < CurrStripSize; m++) {

                MBcurr = y*MBNumX + x + m;
                MBcurrLeft = (m == 0) ? (MBcurr) : (MBcurr - 1);
                MBcurrTop = (y == 0) ? (MBcurr) : (MBcurr - MBNumX);
                
                // Calculate boundary strengths for the macroblock
                CalcBoundaryStrength(&pBlkMBInfo[MBcurr*BLOCKS_PER_MB],
                                     &pBlkMBInfo[MBcurrLeft*BLOCKS_PER_MB],
                                     &pBlkMBInfo[MBcurrTop*BLOCKS_PER_MB],
                                     &BSRef[MBcurr],
                                     disable_deblocking_filter_idc
                                     );
                
                // Deblock the macroblock
                DeblockMB(&BSRef[MBcurr],
                          pBlkMBInfo[MBcurr*BLOCKS_PER_MB].QP,
                          pBlkMBInfo[MBcurrLeft*BLOCKS_PER_MB].QP,
                          pBlkMBInfo[MBcurrTop*BLOCKS_PER_MB].QP,
                          slice_alpha_c0_offset,
                          slice_beta_offset,
                          frame,                        
                          x + m, y,
                          (s > 0) && (m == 0),
                          (s < (NumStrips - 1)) && (m == (CurrStripSize - 1)));
            }
        }
    }


	FILE *output_file = NULL;
	output_file = fopen("strength.txt", "wb");
	//fwrite(BSRef, 1, MBNumX*MBNumY*32, output_file);
	fwrite(frame->y + frame->width*(frame->height-frame->image_height)/2, 1, frame->width*frame->image_height, output_file);
	fclose(output_file);
	
			
    /*cudaCalcBoundaryStrength(pBlkMBInfo,
							BSRef,
							disable_deblocking_filter_idc,
							MBNumX,
							MBNumY
							);*/
   /* cudaDeblockMB(BSRef,
                  pBlkMBInfo,
                  slice_alpha_c0_offset,
                  slice_beta_offset,
                  frame,
				  MBNumX,
				  MBNumY);*/
	/*cudaDeblockMB(BSRef,
                  pBlkMBInfo,
                  slice_alpha_c0_offset,
                  slice_beta_offset,
                  frame,
				  MBNumX,
				  MBNumY,
				  disable_deblocking_filter_idc);*/

	//FILE *output_file = NULL;
	//output_file = fopen("strength.txt", "wb");
	////fwrite(BSRef, 1, MBNumX*MBNumY*32, output_file);
	//fwrite(frame->y + frame->width*(frame->height-frame->image_height)/2 , 1, frame->width*frame->image_height, output_file);
	//fclose(output_file);

    end = clock();
    pad_deblock_out_frame(p_enc->pRecFrame, REFERENCE_FRAME_PAD_AMT); // dec
    
    err = deblock_colflat_luma(p_enc->pRecFrame->y, 
                                    p_enc->reference_frames->reference_frame_list[0].luma_component,
                                    p_enc->pRecFrame->height, p_enc->pRecFrame->width);
    if (!err)
    {
        deblock_colflat_chroma(p_enc->pRecFrame->u, p_enc->pRecFrame->v,
                                    p_enc->reference_frames->reference_frame_list[0].chroma_component,
                                    p_enc->pRecFrame->height>>1, p_enc->pRecFrame->width>>1);
    }

	//p_enc->new_timers.de_block +=(end - start);
	output_file = fopen("strength.txt", "wb");
	//fwrite(BSRef, 1, MBNumX*MBNumY*32, output_file);
	fwrite(frame->y  , 1, frame->width*frame->height, output_file);
	fclose(output_file);
	free(BSRef);
    return err;
}


//====================================================================
void CalcBoundaryStrength
//====================================================================
//   Calculate the boundary strengths for all the edges for one
//   Macroblock
//--------------------------------------------------------------------
(
 // Inputs
 S_BLK_MB_INFO *MBInfo,
 S_BLK_MB_INFO *MBInfoLeft,
 S_BLK_MB_INFO *MBInfoTop,
 // Output
 S_BOUNDARY_STRENGTH_REF *BSRef,
 // Inputs
 int disable_deblocking_filter_idc
 )
//--------------------------------------------------------------------
{
    int d, i, j;

    // Do edges in both directions (d=0: vertical; d=1: horizontal)
    for (d = 0; d < 2; d++) {

        S_BLK_MB_INFO *NeighborBlk = ((d == 0) ? MBInfoLeft : MBInfoTop);

        int BlkIsIntra = ((MBInfo->Type == INTRA_LARGE_BLOCKS_MB_TYPE)
                          || (MBInfo->Type == INTRA_SMALL_BLOCKS_MB_TYPE));
        int NeighborIsIntra = ((NeighborBlk->Type == INTRA_LARGE_BLOCKS_MB_TYPE)
                               || (NeighborBlk->Type == INTRA_SMALL_BLOCKS_MB_TYPE));

        // Loop over four edges in a MB
        for (i = 0; i < 4; i++) {
    
            // loop over four 4x4 block positions along each edge in order
            // to determine strength of filtering for each group of 4 pixel
            // boundaries (all 4 pixel boundaries within a single 4x4 block
            // position all use the same strength)
            for (j = 0; j < 4; j++) {

                // indices into MV[y][x] and CBP[y][x] arrays
                int pXidx = ((d == 0) ? i-1 : j  );
                int pYidx = ((d == 0) ? j   : i-1);
                int qXidx = ((d == 0) ? i   : j  );
                int qYidx = ((d == 0) ? j   : i  );

                // If this is 4x4 block is on the macroblock edge,
                // calculate the appropriate index into the
                // neighboring macroblock's 2-D array of 4x4 blocks.
                int NeighborX = (d == 0) ? 3 : j;
                int NeighborY = (d == 1) ? 3 : j;

                // Get p_Blk and q_Blk
                S_BLK_MB_INFO p_Blk = ((i == 0)
                                       ? NeighborBlk[NeighborY*4 + NeighborX]
                                       : MBInfo[pYidx*4 + pXidx]);
                S_BLK_MB_INFO q_Blk = MBInfo[qYidx*4 + qXidx];

                // Get MVs for each 4x4 block to the left and above
                S_MV pMV = p_Blk.MV;
                S_MV qMV = q_Blk.MV;
                // Get Reference frames for each 4x4 block to the left and above
                int pRef = p_Blk.RefFrameIdx;
                int qRef = q_Blk.RefFrameIdx;
                // Check for coded coefficients in each 4x4 block to the left and above
                int pCodedCoeffs = (p_Blk.TotalCoeffLuma != 0);
                int qCodedCoeffs = (q_Blk.TotalCoeffLuma != 0);

                // Calculate boundary strength (bS)
                if ((i == 0) && (NeighborIsIntra || BlkIsIntra))
                {
                    (*BSRef)[d][i][j] = 4;
                }
                else if (BlkIsIntra)
                {
                    (*BSRef)[d][i][j] = 3;
                }
                else if (pCodedCoeffs || qCodedCoeffs)
                {
                    (*BSRef)[d][i][j] = 2;
                }
                else if ((pRef != qRef)
                         || ((abs(pMV.x - qMV.x) >= 4)
                             || (abs(pMV.y - qMV.y) >= 4)))
                {
                    (*BSRef)[d][i][j] = 1;
                }
                else
                {
                    (*BSRef)[d][i][j] = 0;
                }

                // check boundary conditions
                if (
                    // Check for left picture edge
                    ((d == 0) && (i == 0) && (MBInfo->Loc & LOC_MB_LEFT_PICTURE_EDGE))
                    // Check for top picture edge
                    || ((d == 1) && (i == 0) && (MBInfo->Loc & LOC_MB_TOP_PICTURE_EDGE))
                    // Check for disable_deblocking_filter_idc
                    || (disable_deblocking_filter_idc == 1) 
                    // Check for slice edge
                    || ((disable_deblocking_filter_idc == 2)
                        && (((d == 0) && (i == 0) && (MBInfo->Loc & LOC_MB_LEFT_SLICE_EDGE))
                            || ((d == 1) && (i == 0) && (MBInfo->Loc & LOC_MB_TOP_SLICE_EDGE))))
                    )
                {
                    (*BSRef)[d][i][j] = 0;
                }
            }  // for (j...
        }  // for (i...
    }  // for (d...
}


//====================================================================
void DeblockMB
//====================================================================
//   Filter (deblock) a single macroblock
//--------------------------------------------------------------------
(
 S_BOUNDARY_STRENGTH_REF *BSRef,
 int QPyCurr, int QPyLeft, int QPyTop,
 int alpha_c0_offset,
 int beta_offset,
 yuv_frame_t *frame,
 int MBx, int MBy,
 int DontFilterVertEdges, int DontFilterHorzEdges
 )
//--------------------------------------------------------------------
{
    int d, i;
    int img_width, img_height;
    int buf_width, buf_height;
    unsigned char *y,*u, *v;
    buf_width  = frame->width;
    buf_height = frame->height;
    img_width  = frame->image_width;
    img_height = frame->image_height;
    y = frame->y + buf_width*(buf_height-img_height)/2 + (buf_width-img_width)/2;
    u = frame->u + buf_width*(buf_height-img_height)/8 + (buf_width-img_width)/4;
    v = frame->v + buf_width*(buf_height-img_height)/8 + (buf_width-img_width)/4;  
    
    // Do edges in both directions (d=0: vertical; d=1: horizontal)
    for (d = 0; d < 2; d++) {
        
        int QPyAv = (QPyCurr + ((d == 0) ? QPyLeft : QPyTop) + 1) >> 1;
        int QPcAv = (QP_TO_CHROMA_MAPPING[QPyCurr] + ((d == 0)
                                                      ? QP_TO_CHROMA_MAPPING[QPyLeft]
                                                      : QP_TO_CHROMA_MAPPING[QPyTop]) + 1) >> 1;

        unsigned char *SrcY = y + (MBy << 4)*(buf_width) + (MBx << 4);
        unsigned char *SrcU = u + (MBy << 3)*(buf_width >> 1) + (MBx << 3);
        unsigned char *SrcV = v + (MBy << 3)*(buf_width >> 1) + (MBx << 3);

        int PtrIncY = ((d == 0) ? buf_width : 1);
        int IncY = ((d == 0) ? 1 : buf_width);
        int PtrIncC = ((d == 0) ? (buf_width >> 1) : 1);
        int IncC = ((d == 0) ? 1 : (buf_width >> 1));

        // Loop over four edges in a MB
        for (i = 0; i < 4; i++) {

            int QPy = (i == 0) ? QPyAv : QPyCurr;
            int QPc = (i == 0) ? QPcAv : QP_TO_CHROMA_MAPPING[QPyCurr];

            // Call actual filtering function
            if ( ! (((d == 0) & DontFilterVertEdges)
                    || ((d == 1) & DontFilterHorzEdges))) {
                EdgeLoopLuma(SrcY, (*BSRef)[d][i], QPy,
                             alpha_c0_offset, beta_offset,
                             PtrIncY, IncY);
            }

			/*if ( d==1&&DontFilterVertEdges) {
                EdgeLoopLuma(SrcY, (*BSRef)[d][i], QPy,
                             alpha_c0_offset, beta_offset,
                             PtrIncY, IncY);
            }*/
            SrcY += (IncY << 2);

            if ((i & 1) == 0) {

                if ( ! (((d == 0) & DontFilterVertEdges)
                        || ((d == 1) & DontFilterHorzEdges))) {
                    EdgeLoopChroma(SrcU, (*BSRef)[d][i], QPc,
                                   alpha_c0_offset, beta_offset,
                                   PtrIncC, IncC);
                    EdgeLoopChroma(SrcV, (*BSRef)[d][i], QPc,
                                   alpha_c0_offset, beta_offset,
                                   PtrIncC, IncC);
                }
                SrcU += (IncC << 2);
                SrcV += (IncC << 2);
            }
        }
    }
}


//====================================================================
void EdgeLoopLuma
//====================================================================
// Filters one edge (16 pixels) of luma 
//--------------------------------------------------------------------
(
    // pointer to the first pixel of the edge, (right or bottom).
    unsigned char *SrcPtr,
    // filtering strength, top to bottom for vert. edges and left to
    // right for horizontal ones
    unsigned char Strength[4],
    // quant step
    int QP,
    int AlphaC0Offset, int BetaOffset,
    // pixel offset along the border
    int PtrInc,
    // pixel offset across the border
    int inc
 )
//--------------------------------------------------------------------
{
    // TODO: This low-level function is largely VSofts code---need to
    // comment it.

    int      ap, aq,  Strng ;
    int      inc2, inc3, inc4 ;
    int      C0, c0, Delta, dif, AbsDelta ;
    int      L2, L1, L0, R0, R1, R2, RL0 ;
    const unsigned char *ClipTab;   
    int      Alpha, Beta;
    int      small_gap;
    int      indexA, indexB;
    int      i,j;

    inc2    = inc<<1 ;    
    inc3    = inc + inc2 ;    
    inc4    = inc<<2 ;

    indexA = IClip(0, MAX_QP, QP + AlphaC0Offset);
    indexB = IClip(0, MAX_QP, QP + BetaOffset);

    Alpha=ALPHA_TABLE[indexA];
    Beta=BETA_TABLE[indexB];  
    ClipTab=CLIP_TAB[indexA];

    for( j=0 ; j<4 ;j++  )
    {
        Strng = Strength[j];
        assert(Strng != 0xFF);
        if (!Strng)
        {   
            SrcPtr += PtrInc << 2;
            continue;
        }
        for (i=0; i<4; i++) 
        {
            L0  = SrcPtr [-inc ] ;
            R0  = SrcPtr [    0] ;
            AbsDelta  = abs( Delta = R0 - L0 )  ;
            if( AbsDelta < Alpha )
            {
                C0  = ClipTab[ Strng ] ;
                L1  = SrcPtr[-inc2] ;
                R1  = SrcPtr[ inc ] ;
                if( ((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0  ) 
                {
                    L2  = SrcPtr[-inc3] ;
                    R2  = SrcPtr[ inc2] ;
                    aq  = (abs( R0 - R2) - Beta ) < 0  ;
                    ap  = (abs( L0 - L2) - Beta ) < 0  ;
                    RL0             = L0 + R0 ;

                    if(Strng == 4 )
                    {
                        // INTRA strong filtering
                        small_gap = (AbsDelta < ((Alpha >> 2) + 2));
     
                        aq &= small_gap;
                        ap &= small_gap;
                        if (aq)
                        {
                            SrcPtr[   0 ] = (unsigned char)(( L1 + ((R1 + RL0) << 1) +  SrcPtr[ inc2] + 4) >> 3);
                            SrcPtr[ inc ] = (unsigned char)(( SrcPtr[ inc2] + R0 + R1 + L0 + 2) >> 2);
                            SrcPtr[ inc2] = (unsigned char)((((SrcPtr[ inc3] + SrcPtr[ inc2]) <<1) + SrcPtr[ inc2] + R1 + RL0 + 4) >> 3);
                        }
                        else
                        {
                            SrcPtr[   0 ] = (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2);
                        }

                        if (ap)
                        {
                            SrcPtr[-inc ] = (unsigned char)(( R1 + ((L1 + RL0) << 1) +  SrcPtr[-inc3] + 4) >> 3);
                            SrcPtr[-inc2] = (unsigned char)(( SrcPtr[-inc3] + L1 + L0 + R0 + 2) >> 2);
                            SrcPtr[-inc3] = (unsigned char)((((SrcPtr[-inc4] + SrcPtr[-inc3]) <<1) + SrcPtr[-inc3] + L1 + RL0 + 4) >> 3);
                        }
                        else
                        {
                            SrcPtr[-inc ] = (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2);
                        }
                    }
                    else
                    {
                        // normal filtering
                        c0               = C0 + ap + aq ;
                        dif              = IClip( -c0, c0, ( (Delta << 2) + (L1 - R1) + 4) >> 3 ) ;
                        SrcPtr[  -inc ]  = (unsigned char)(CLIP0_255(L0 + dif));
                        SrcPtr[     0 ]  = (unsigned char)(CLIP0_255(R0 - dif));
                        if( ap )
                            SrcPtr[-inc2] += (unsigned char)(IClip( -C0,  C0, ( L2 + ((RL0+1) >> 1) - (L1<<1)) >> 1 ));
                        if( aq  )
                            SrcPtr[  inc] += (unsigned char)(IClip( -C0,  C0, ( R2 + ((RL0+1) >> 1) - (R1<<1)) >> 1 ));
                    }
                }
            }
            SrcPtr += PtrInc ;      // Increment to next set of pixel
        }
    }
}


//====================================================================
void EdgeLoopChroma
//====================================================================
//   Filters one edge (8 pixels) of chroma 
//--------------------------------------------------------------------
(
    // pointer to the first pixel of the edge, (right or bottom).
    unsigned char* SrcPtr,
    // filtering strength, top to bottom for vert. edges and left to
    // right for horizontal ones
    unsigned char Strength[4],
    // quant step
    int QP,
    int AlphaC0Offset, int BetaOffset,
    // pixel offset along the border
    int PtrInc,
    // pixel offset across the border
    int inc
)
//--------------------------------------------------------------------
{
    // TODO: This low-level function is largely VSofts code---need to
    // comment it.

    int      Strng ;
    int      inc2=inc<<1;
    int      C0, c0, Delta, dif, AbsDelta ;
    int      L1, L0, R0, R1;
    const unsigned char *ClipTab;   
    int      Alpha, Beta;  
    int      indexA, indexB;
    int      i,j;

    indexA = IClip(0, MAX_QP, QP + AlphaC0Offset);
    indexB = IClip(0, MAX_QP, QP + BetaOffset);

    Alpha=ALPHA_TABLE[indexA];
    Beta=BETA_TABLE[indexB];  
    ClipTab=CLIP_TAB[indexA];

    for( j=0 ; j<4 ; j++)
	{
        Strng = Strength[j];
        assert(Strng != 0xFF);
        if (Strng)
        {
            for (i=0; i<2; i++) 
			{
                L0  = SrcPtr [-inc ] ;
                R0  = SrcPtr [    0] ;
                AbsDelta  = abs( Delta = R0 - L0 )  ;
                if( AbsDelta < Alpha )
                {
                    C0  = ClipTab[ Strng ] ;
                    L1  = SrcPtr[-inc2] ;
                    R1  = SrcPtr[ inc ] ;
                    if( ((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0  ) 
                    {
                        if(Strng == 4 )
                        {
                            // INTRA strong filtering
                            SrcPtr[   0 ] = (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2);
                            SrcPtr[-inc ] = (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2);
                        }
                        else
                        {
                            // normal filtering
                            c0               = C0+1;
                            dif              = IClip( -c0, c0, ( (Delta << 2) + (L1 - R1) + 4) >> 3 ) ;
                            SrcPtr[  -inc ]  = (unsigned char)(CLIP0_255(L0 + dif));
                            SrcPtr[     0 ]  = (unsigned char)(CLIP0_255(R0 - dif));

                        }
                    }
                }
                SrcPtr += PtrInc ;      // Increment to next set of pixel
            }
		}
        else
            SrcPtr += PtrInc << 1 ;
    }
}


////////////////////////////////////////////////////////////////
E_ERR deblock_colflat_luma(
                                       unsigned char *in,          // input image
                                       unsigned char *out,         // output image
                                       int num_rows,               // input/output image height
                                       int num_cols                // input/output image width
                                       )
////////////////////////////////////////////////////////////////
//
//
//    Description:  
//
//      This function takes an input image in normal row-major raster order and
//      outputs in Column flattened order.
//      Assumes that the number of rows is a multiple of 4
//                  
//    Returns:      Nothing
//
////////////////////////////////////////////////////////////////
{
    int i, j;
    E_ERR err = ERR_SUCCESS;

    if ( (num_rows % 4) != 0) {
        printf("colflat_luma_cref: ERROR! num_rows %d is not a multiple of 4\n", num_rows);
        err = ERR_FAILURE;
    }

    if (!err)
    {
        // Do this the really simple way to make sure it is a good reference!
        // i and j are input image coordinates
        for(i = 0; i < num_rows/4; i++) { // for every quad of rows
            for(j = 0; j < num_cols; j++) { // for each column output
                *out++ = in[(i*4   )*num_cols + j];
                *out++ = in[(i*4 +1)*num_cols + j];
                *out++ = in[(i*4 +2)*num_cols + j];
                *out++ = in[(i*4 +3)*num_cols + j];             
            }
        }
    }

    return err;
}




////////////////////////////////////////////////////////////////
void deblock_colflat_chroma(
                                        unsigned char *in1,         // input Cb image
                                        unsigned char *in2,         // input Cr image
                                        unsigned char *out,         // output image (inputs combined)
                                        int num_rows,               // input image height
                                        int num_cols               // input image width
                                        )
////////////////////////////////////////////////////////////////
//
//
//    Description:  
//
//      This function takes two input images in normal row-major raster order and
//      outputs in Column flattened order.
//
//      Assumes that the number of rows is a multiple of 2
//                  
//    Returns:      Nothing
//
////////////////////////////////////////////////////////////////
{
    int i, j;
    
    if ( (num_rows % 2) != 0) {
        fprintf(stderr, "colflat_chroma_cref: ERROR! num_rows %d is not a multiple of 2\n", num_rows);
        exit(-1);
    }
    

    // Do this the really simple way to make sure it is a good reference!
    
    // i and j are input image coordinates
    for(i = 0; i < num_rows/2; i++) { // for every pair of input rows
        for(j = 0; j < num_cols; j++) { // for each column output
            *out++ = in1[(i*2   )*num_cols + j];
            *out++ = in1[(i*2 +1)*num_cols + j];
            *out++ = in2[(i*2   )*num_cols + j];
            *out++ = in2[(i*2 +1)*num_cols + j];
        }
    }

    return;
}


void pad_deblock_out_frame(yuv_frame_t *Reference, int PadAmount)
{
    int j;
    int buf_width  = Reference->width;
    int buf_height = Reference->height;
    int img_width  = Reference->image_width;
    int img_height = Reference->image_height;
    char *frame_y = (char *)Reference->y;
    char *frame_u = (char *)Reference->u;
    char *frame_v = (char *)Reference->v;
    
    assert(2*PadAmount == buf_width - img_width);
    assert(2*PadAmount == buf_height - img_height);
    
    // Pad Y plane
    for (j = PadAmount; j < (img_height + PadAmount); j++) {
        memset(frame_y + j * buf_width, frame_y[j*buf_width + PadAmount], PadAmount);
        memset(frame_y + j * buf_width + img_width + PadAmount,
               frame_y[j * buf_width + img_width + PadAmount - 1], PadAmount);
    }
    
    for (j = 0; j < PadAmount; j++) {
        memcpy(frame_y + j * buf_width, frame_y + PadAmount * buf_width, buf_width);
    }
    
    for (j = img_height + PadAmount; j < (img_height + 2*PadAmount); j++) {
        memcpy(frame_y + j * buf_width,
               frame_y + (img_height + PadAmount - 1) * buf_width, buf_width);
    }
    
    
    img_height /= 2; img_width /= 2;
    buf_height /= 2; buf_width /= 2;
    PadAmount /= 2;
    // Pad UV pclusters
    for (j = PadAmount; j < (img_height + PadAmount); j++) {
        memset(frame_u + j * buf_width, frame_u[j*buf_width + PadAmount], PadAmount);
        memset(frame_u + j * buf_width + img_width + PadAmount,
               frame_u[j * buf_width + img_width + PadAmount - 1], PadAmount);
        memset(frame_v + j * buf_width, frame_v[j*buf_width + PadAmount], PadAmount);
        memset(frame_v + j * buf_width + img_width + PadAmount,
               frame_v[j * buf_width + img_width + PadAmount - 1], PadAmount);
        
    }
    
    for (j = 0; j < PadAmount; j++) {
        memcpy(frame_u + j * buf_width, frame_u + PadAmount * buf_width, buf_width);
        memcpy(frame_v + j * buf_width, frame_v + PadAmount * buf_width, buf_width);
    }
    
    for (j = img_height + PadAmount; j < (img_height + 2*PadAmount); j++) {
        memcpy(frame_u + j * buf_width,
               frame_u + (img_height + PadAmount - 1) * buf_width, buf_width);
        memcpy(frame_v + j * buf_width,
               frame_v + (img_height + PadAmount - 1) * buf_width, buf_width);
    }
}
