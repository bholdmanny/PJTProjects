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

#ifndef _ENCODER_CONTEXT_H
#define _ENCODER_CONTEXT_H

#include "h264_common.h"
#include "parset.h"
#include "const_defines.h"
#include "entropy_data.h"
#include "rc.h"
#include "cavlc_data.h"
#include "me.h"
#include"cuve264b.h"
#include <time.h>


#define ENCODER_NAME_SIZE   40
#define SLICE_MAX			70

struct encoder_context_t;

typedef E_ERR STANDARD_ENCODER_FN(struct encoder_context_t **, int);
typedef E_ERR (*P_STANDARD_ENCODER_FN)(struct encoder_context_t **, int);

typedef struct FRAME_ENC_INFO
{
    int     slice_type;     // type of the slice (see slicetype_e)
    int     frame_num;      // logical number of the frame
    int     idr_flag;       // 1=IDR (key) frame, 0=regular frame;
    int     forced_qp;      //
    int     frame_qp;       // qp used for the frame
    int     num_mbs;        // total number of mbs in the frame
    int     num_intra_mb;   // number of intra mbs in encoded frame
    int     num_encoded_mb; // Number of macroblocks that are encoded in the bitstream (i.e., not "skipped")
    int     num_bits;       // number of encoded bits in slice
    int     header_bits;    // bits encoded for headers and motion
    int     texture_bits;   // bits encoded for texture (residuals)
    int     num_me_cands;   // average number of ME candidates
    int     num_big_mvs;    // number of MBs with MVs exceeding a threshold
    int     frame_id;
    int     scene_detected; // last frame shoulda been an IFrame
} FRAME_ENC_INFO;

// Storage for encoder mb.
typedef struct S_TRANSFORM_COEFS
{
    int    total_coef_bytes;
    short *pDctCoefs;
    short *pDctCoefs_u;
    short *pDctCoefs_v;
    short *pDcCoefs;
    short *pDcCoefs_u;
    short *pDcCoefs_v;
    short *p_coef_flat_luma;
    short *p_coef_flat_chroma;
} S_TRANSFORM_COEFS;

typedef struct LOOPFILTER_PARAMS
{
    int disable_flag;
    int disable_cross_slice_flag;
    int alpha_c0_offset;
    int beta_offset;
    int deblock_mode;
} LOOPFILTER_PARAMS;



typedef struct PERF_TIMERS
{
    clock_t prep_encode_frame;
    clock_t encode_frame;

    clock_t me_total;
   // clock_t me_search;
    clock_t pframe_residual_luma;
    clock_t pframe_residual_chroma;
    clock_t pframe_residual_inter;
    clock_t pframe_residual_intra;
    clock_t rc_total;
    clock_t cavlc_timers;
	clock_t iframe_residual;
	clock_t pframe_total;
	clock_t pframe_mc;
	clock_t de_block;

} PERF_TIMERS;

typedef struct SC_RC_BUFFERS
{
    int *p_input_frame_index;
    int *p_recon_frame_index;
    int *p_recon_frame_index_lt_d1;
    int *p_dc_qcoefs_index;
    int *p_input_frame_index_c;
    int *p_recon_frame_index_c;
    int *p_recon_frame_index_c_lt_d1;
    int *p_dct_qcoef_index_c;
    int *p_dc_qcoef_index_c;
    int *p_input_frame_index_pf_c;
    int *p_recon_frame_index_pf_c;
    int *p_dct_coef_index_pf_c;
    int *p_dc_coef_frame_index_pf_c;
    int *p_perm_indx_c;
} SC_RC_BUFFERS;

// Context maintained by deblocking between frames
// Primarily used for index streams etc to avoid index
// streams being inited for every frame.
typedef struct DEBLOCK_CONTEXT_T{
    int *p_deblk_tbls; //3*NUM_QP/4
    int *p_y_idx; //[32*2]
    int *p_uv_idx; //[32*2]
    int *p_lf_mb_info_idx; //[16]
    int *p_mb_info_idx; //[ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4)]
    int *p_mb_info_idx_bot; //[ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4)]
    int *p_mv_info_idx; //[ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4)]
    int *p_mv_info_idx_bot; //[ROUND_UPTO_NEXT2nX(4*MAX_STRIP_SIZE+4, 4)]
    int *p_colf_y_idx; //[ROUND_UPTO_NEXT2nX(8*MAX_STRIP_SIZE+24, 4)]
    int *p_colf_uv_idx; //[ROUND_UPTO_NEXT2nX(MAX_STRIP_SIZE+3, 4)]
    int *p_col_flat_frm_topuv_idx; //[128]
}DEBLOCK_CONTEXT;

typedef struct CAVLC_CONTEXT_T {
    // cavlc_block_context related fields
    int *p_index;
    int *block_context_param_buf;
    int last_consecutive_skip_mbs;
    S_TEXTURE_BLOCK_PAIRS *p_coef_zigzag;
    S_CAVLC_CONTEXT *p_context;

    // fields related to gen_bits and put_bits
    int *p_cavlc_leftover_info;
    int *param_buf;
    int *p_tables_buf;
    unsigned int *p_mb_bits;
} CAVLC_CONTEXT;

///////////////////////////////////////////////////////////////////////////////////////////////////
// STRUCTURE:	COLFLAT_REF_FRM_T
// DESCRIPTION:	This structure consists of pointers to column flattened and padded luma frame and
//				column_flattened, interleaved and padded chroma frame that make up the reference
//				frame for the decoder's motion compensation
typedef struct	
{
	unsigned char		*luma_component;	// Pointer to the Luma Component of the reference frame
	unsigned char		*chroma_component;	// Pointer to the interleaved chroma component
}COLFLAT_REF_FRM_T;


///////////////////////////////////////////////////////////////////////////////////////////////////
// STRUCTURE:	REFERENCE_FRAMES_CREF_T
// DESCRIPTION:	This structure consists of information regarding the reference frame / pictures of
//				the current instance of the decoder with pointer to the base of the reference frames
//				and a reference pictures' indexing list
typedef struct
{
	int					num_of_reference_frames;	// Number of reference frames for the current context
	int					luma_frame_size;			// Size of the Luma reference frame
	int					chroma_frame_size;			// Size of the Chroma reference frame (interleaved)
	int                 *reference_frame_index;		// Pointer to the index of reference frames
	COLFLAT_REF_FRM_T	*reference_frame_list;		// Pointer to the list of reference frames
}REFERENCE_FRAMES_CREF_T;

typedef enum E_ENCODER_TYPE {
    ENCODER_CREF = 0    
} E_ENCODER_TYPE;

typedef struct ME_CONTEXT
{
    CUVME_HANDLE  me_handle;
    CUVME_MV_RESULTS  *p_me_mv_results;
    CUVME_MB_INFO     *p_me_mb_info;
    CUVME_MB_CHARAC   *p_me_mb_characs;   
    int me_level;
    int search_range_x;
    int search_range_y;
} ME_CONTEXT;

typedef struct RC_CONTEXT
{
    CUVRC_HANDLE        rc_handle;
    CUVRC_FRAME_TYPE    curr_frame_type;
    CUVRC_FRAME_TYPE    prev_frame_type;
    CUVRC_ROI           roi_params[MAX_NUM_ROIS];
    int                 num_rois;
    int                 *p_qp_info;
    int                 min_QP;
    int                 max_QP;
    int                 iframe_qp;
    int                 pframe_qp;
    int                 target_framerate;
    int                 target_bitrate;
    int                 rc_mode;
} RC_CONTEXT;

typedef struct{
	// Persistent memory pointer
	int * p_mem;
	//  Memory size
	int size;
	// Flag that indicates whether  memory is allocated by the application.
	int is_app_mem;
	// Flag that indicates whether  memory has been changed by the application.
	int app_mod_flag;
	// Width for which the above sizes are allcated.
	int last_width;
	// Height for which the above sizes are allcated.
	int last_height;
}S_CUVEH264BE_HEAP_MEM;

typedef struct
{
  unsigned int mem_offset_0;
  unsigned int mem_offset_1;
  unsigned int mem_offset_2;
  unsigned int mem_offset_3;
  unsigned int mem_offset_4;
  unsigned int mem_offset_5;
  unsigned int mem_offset_6;
  unsigned int mem_offset_7;
  unsigned int mem_offset_8;
  unsigned int mem_offset_9;
}SCRATCH_MEM_INFO;

//All encoder context data ---
typedef struct encoder_context_t
{
    char name[ENCODER_NAME_SIZE+1];
    //ENCODER_FUNCTIONS fns;
    E_ENCODER_TYPE type;
	//slice var related 
	encoder_context_t *slice[SLICE_MAX];
	int i_slice_num;
	int first_mb;
	int last_mb;
	int width_mb;
	int height_mb;

    int pic_width;      // Desired coding width luma
    int pic_height;     // Desired coding height luma
    int width;          // pic_width padded to multiple of macroblocks
    int height;         // pic_height padded to multiple of macroblocks

    int frames_cnt;     // Number of encoded or decoded frames
    int last_IDR_frame; // Number of the most recent IDR frame
    int slice_in_frame;	// Proceeded slice No in frame
    int last_assigned_id;
    int intra_period;
    
    int desired_bitrate;
    int desired_framerate;
    int force_iframe;
    
    int generate_sps_pps_nalus;       // If set to non zero, encoder has changed size.
    int generate_sei_nalu;
    int entropy_coding_mode;   // 0 for CAVLC and 1 for CABAC
    int constrained_intra_flag;     // If 0, non-constrained intra. If 1, constrained intra
    seq_parameter_set_rbsp_t SequenceParameterSet; // SPS
    pic_parameter_set_rbsp_t PictureParameterSet;  // PPS
    slice_header_params_t slice_params;            // slice header params
    user_data_sei_message_t sei_rbsp;

    LOOPFILTER_PARAMS loopfilter_params;
 
    // Frames
    yuv_frame_t padded_ref_frame;
    yuv_frame_t padded_rec_frame;
    unsigned char *p_recon_flat_luma;
    unsigned char *p_recon_flat_chroma;
    yuv_frame_t inter_pred_frame; // frame for storing prediction
    yuv_frame_t input_frame;     // pointer to input frame
    yuv_frame_t input_frame_source; // this frame maybe allocated if input format is raster. 
    yuv_frame_t *p_source_frame;  // pointer to source input, maybe in raster format or bf format. In case of bf format, this should be the same as p_input_frame
    yuv_frame_t *p_input_frame;   // pointer to input image to encoder
    int input_needs_conversion;    // flag
    int input_format;
    int input_format_changed;

    yuv_frame_t *pRecFrame;       // Pointer to reconstructed frame
    yuv_frame_t *pRefFrame;       // Pointer to reference frame
    int own_input_frame;          // whether the encoder owns it's own buffers for the input frame
    
    // Adaptive MB QP memory 
    int *p_mb_qps;

    // Frames for ref and rec. For now, assuming only one reference frame.
    S_TRANSFORM_COEFS transform_coefs;
    struct S_BLK_MB_INFO *pBlkMBInfo;
    struct S_BLK_MB_INFO_COMPRESSED *p_blk_mb_info;
    struct S_QP_DATA *p_qp_data;

    FRAME_ENC_INFO frame_info;   // info about currently encoded frame.
    //RATE_CONTROL_INFO rc_info;

    // iframe residual coding stream related buffers
    SC_RC_BUFFERS sc_rc_buffers; 

    CAVLC_CONTEXT cavlc_ctxt;

    // Output bitstream buffer
    int enable_bitstream_generation;
    int estimated_output_buffer_size;
    int own_output_buffer;       // whether the encoder owns it's own buffer for output bitstream
    bitstream_t bitstream;

    // for performance measurement
   // S_PERF_TIMERS timers;
	PERF_TIMERS  new_timers;

    // miscellaneous
    int avg_mb_sad;
    int intra_prediction_selection;

    // Decoder MC Module
    REFERENCE_FRAMES_CREF_T		*reference_frames;
    REFERENCE_FRAMES_CREF_T		*reference_frames_0;
    REFERENCE_FRAMES_CREF_T		*reference_frames_1;

    // MC consts
    unsigned int mul_fact;
    unsigned int mul_err;

    int max_ref_frames_num;
    
    ME_CONTEXT me_context;

    CUVME_Y_BUFFER *half_res_1_me_y;
    CUVME_Y_BUFFER *half_res_2_me_y;
    CUVME_Y_BUFFER *quarter_res_1_me_y;
    CUVME_Y_BUFFER *quarter_res_2_me_y;

    // Deblock context between frames
    DEBLOCK_CONTEXT deblock_ctxt;

    // RC need reset 
    int rc_need_reset;
    RC_CONTEXT  rc_context;    
    int *p_qp_list;
    int mb_adapt_qp_on;
    int roi_on;
    int sps_pps_rate; 
    int intra_mb_level;
    int pseudo_i_frame;
    int me_search_range_x;
    int me_search_range_y;
    int non_ref_frame_interval;
    int enc_mem_size;
    int me_mem_size;
    int enable_signature_generation;
    S_CUVEH264BE_HEAP_MEM s_persist_mem;
    S_CUVEH264BE_HEAP_MEM s_nonpersist_individual_mem;
    SCRATCH_MEM_INFO   scratch_mem_info;
    int *ptr_filter_coefs;
} encoder_context_t;

E_ERR init_encoder(encoder_context_t *p_enc, E_ENCODER_TYPE enc_type,int slice_num);



///////////////////////////////////////////////////////////////////////////////
//  other high-level functions
///////////////////////////////////////////////////////////////////////////////
E_ERR encoder_mem_alloc (encoder_context_t *p_enc, int alloc_input, int alloc_me);
void encoder_mem_free (encoder_context_t *p_enc);
void set_encode_dimensions (encoder_context_t *p_enc, int pic_width, int pic_height);
void set_encoder_name(encoder_context_t *p_enc, char *name);
void set_intra_prediction (encoder_context_t *p_enc, int intra_prediction_selection);
void set_loopfilter_params (encoder_context_t *p_enc, int alpha_c0_offset, int beta_offset);

/**
 * free all encoder data
 * @param context  - codec context
 */
void destroy_encoder(encoder_context_t *p_enc);





//---- Internal high level functions ------
void enc_set_frames(encoder_context_t *p_enc);
int set_frame_qp(encoder_context_t *p_enc, int qp);

/* Fill all frame info for newly provided frame */
E_ERR init_frame_info(encoder_context_t *p_enc);
void init_pps(encoder_context_t *p_enc, pic_parameter_set_rbsp_t *p_pps);
void init_sps(encoder_context_t *p_enc, seq_parameter_set_rbsp_t *p_sps);
void init_sei(encoder_context_t *p_enc, user_data_sei_message_t *p_sei);

int output_sps_nalu (seq_parameter_set_rbsp_t *p_sps, bitstream_t *p_bitstream);
int output_pps_nalu(pic_parameter_set_rbsp_t *p_pps, bitstream_t *p_bitstream);
int output_sei_nalu(user_data_sei_message_t *p_sei, bitstream_t *pBitstream);
void output_slice_header(encoder_context_t *p_enc, slice_header_params_t *p_slice);
void init_slice_params (encoder_context_t *p_enc);
E_ERR init_deblock_context(encoder_context_t *p_enc);
void free_deblock_context(encoder_context_t *p_enc);

#endif //__ENCODER_CONTEXT_H__
