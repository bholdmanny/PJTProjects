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

#ifndef __ME_CONTEXT_H__
#define __ME_CONTEXT_H__

#include "me.h"
#include "h264_types.h"
#include "const_defines.h"

#define MAX_NUM_FORMATS 5 // as of now
#define MAX_MULTI_CHANNELS 16
struct ME_context_t;
extern const int QP2QUANT_MELIB[];

typedef struct { short x, y; } coord_t;
typedef enum ME_STATE_MACHINE {
    OPEN = 0, // OPENED
	INIT,     // OPENED and initalised
	REF_SET
} ME_STATE_MACHINE;


typedef struct ZERORES_CONTEXT_T{
    int *ref_index;
    int *input_index;
}ZERORES_CONTEXT;

typedef struct TWOSTEP_PLUS_CONTEXT_T{
	int *MVmapArray;
	int *IndexOrgY;
    int *IndexOrgYBot; //store the indice for the last row of MB
	int *IndexLRMV;
	int *IndexLRMV_lastiteration;
    int *IndexOrgYBot_m32; //store the indice for the last row of MB
	int *IndexLRMV_m32;
	int *IndexLRMV_lastiteration_m32;	
    int *IndexOrgYBot_m20_sp8; //store the indice for the last row of MB
	int *IndexLRMV_m20_sp8;
	int *IndexLRMV_lastiteration_m20_sp8;		
    coord_t *north_mvmap_ping;
    coord_t *north_mvmap_pong;
}TWOSTEP_PLUS_CONTEXT;

typedef struct ME_CANDIDATE_TUNE {
    int cull_integer_candidates;         // FALSE -- Rather than the standard 4 candidate choices, only look at 2
    int integer_unique_range;            // 1 -- anything closer than this is considered the same candidate area and is thrown out
    int integer_vote_max;                // 2 -- maximum number of candidates produced by the integer vote
    int integer_clip_range;              // 1 -- this change to the clip range is only needed if we enable subpel skip
    int integer_unique_limit;            // 2 -- max number of integer candidates before candidates are culled based on distance metric
    int subpel_unique_range;             // 1 -- anything closer than this is considered the same candidate area and is thrown out
    int strip_count_offset;              // 1 -- added to numUnique running average to trigger complexity reduction bailouts sooner (1 = 6 cands high water mark, 3 = 4 cands high water mark)
    int enable_subpel_skip;              // False -- allow the entire subpel pipeline (half and quarter search) to be skipped in some instances
    int subpel_skip_offset;              // doesn't do anything if enable_subpel_skip is not true.  offset between the 4 8x8 SADs and 16x16 SAD below which subpel is tested
    int cull_subpel_candidates;          // False -- Rather than the standard 8 candidate choices, only look at 1
    int subpel_vote_max;                 // 3 -- maximum number of candidates produced by the half-pel vote
    unsigned int subpel_unique_limit;    // 2 -- max number of subpel candidates before candidates are culled based on distance metric
} ME_CANDIDATE_TUNE;

typedef enum E_ME_CANDIDATE_TUNE
{
    ME_CAND_LOW_COMPLEXITY = 0,
    ME_CAND_NORMAL_COMPLEXITY,
    ME_CAND_HIGH_COMPLEXITY,
    ME_CAND_EXPERIMENTAL,
    ME_NUM_CANDIDATE_TUNE
} E_ME_CANDIDATE_TUNE;


typedef struct CUVME_NONPERSISTENT_MEM_OFFSETS
{
	int	idx_north_mvmap_ping ;
	int	idx_north_mvmap_pong ;
	int	idx_p_temp_mvs ;
	int	idx_LowSad ;
	int	idx_nb ;
	int	idx_integer_mvmap ;
	int	idx_ptr_mvs_local_0; 
	int	idx_integer_mb_info ;
	int	idx_CandidateCountInt; 
	int	idx_sum_vars_frame ;
	int	idx_sum_mean_frame ;
	int	idx_quarter_res_src ;
	int	idx_malloced_forw_quarter_res_ref_ptr  ;
	int	idx_malloced_forw_quarter_res_ref ;
	int	idx_malloced_back_quarter_res_ref_ptr  ;
	int	idx_malloced_back_quarter_res_ref  ;
	int	idx_half_res_src  ;
	int	idx_malloced_forw_half_res_ref_ptr  ;
	int	idx_malloced_forw_half_res_ref  ;
	int	idx_malloced_back_half_res_ref_ptr  ;
	int	idx_malloced_back_half_res_ref  ;
	int	idx_block_flat_yuv  ;
	int	idx_block_flat_src  ;
	int	idx_HRresults ;
	int	idx_QRresults ;
	int	idx_HR_MB_Info ;
	int	idx_QR_MB_Info ;
}CUVME_NONPERSISTENT_MEM_OFFSETS;

typedef struct ME_context_t
{
	ME_STATE_MACHINE curr_state;
    //ME_FUNCTIONS fns;
    SINT32 width;
    SINT32 height;
    CUVME_APP_TYPE prime_format;
    SINT32 max_references_forward;
    SINT32 max_references_backward;
    SINT32 num_formats; 
    int me_mode;
    CUVME_Y_BUFFER *ptr_src_picture;
    CUVME_Y_BUFFER *ptr_forw_ref_frame[MAX_FORW_REF_FRAMES];
    CUVME_Y_BUFFER *ptr_back_ref_frame[MAX_BACK_REF_FRAMES]; 
    unsigned int search_range_x;
    unsigned int search_range_y;
    unsigned int min_search_size[MAX_NUM_FORMATS];
	CUVME_MV_RESULTS *ptr_mvs[MAX_NUM_FORMATS];
    CUVME_MV_RESULTS *ptr_mvs_local[MAX_NUM_FORMATS];
    CUVME_MV_RESULTS *ptr_res_out[MAX_NUM_FORMATS];
    CUVME_MB_INFO *ptr_mb_info[MAX_NUM_FORMATS];
	PIXEL *ptr_pred_picture;
    PIXEL *ptr_modified_src;
    // for performance measurement
   // S_ME_PERF_TIMERS timers;
    CUVME_Y_BUFFER quarter_res_src;
    CUVME_Y_BUFFER *malloced_forw_quarter_res_ref[MAX_FORW_REF_FRAMES];
	CUVME_Y_BUFFER *malloced_back_quarter_res_ref[MAX_BACK_REF_FRAMES];
    CUVME_Y_BUFFER *forw_quarter_res_ref[MAX_FORW_REF_FRAMES];
	CUVME_Y_BUFFER *back_quarter_res_ref[MAX_BACK_REF_FRAMES];
    CUVME_Y_BUFFER half_res_src;
    CUVME_Y_BUFFER *malloced_forw_half_res_ref[MAX_FORW_REF_FRAMES];
	CUVME_Y_BUFFER *malloced_back_half_res_ref[MAX_BACK_REF_FRAMES];
    CUVME_Y_BUFFER *forw_half_res_ref[MAX_FORW_REF_FRAMES];
	CUVME_Y_BUFFER *back_half_res_ref[MAX_BACK_REF_FRAMES];
	CUVME_NONPERSISTENT_MEM_OFFSETS  nonpersistent_mem_offsets;
	void  *nonpersistent_mem_baseptr;
	int   nonpersistent_mem_givenby_app_flag;
	int  nonpersistent_mem_size_givenby_app;
	int  meinit_isdone;
    int me_decimate;
    int me_quarter_res;
	int me_twostep;
	int me_twostep_plus;
    int me_half_res;
    int me_integer;
    int me_subpel;
	int *CenterInIdx_load;
	int *CenterInIdx_store;
	int *CenterInIdx_store_qr;
	int *zero_buf;
	int *Idx_load_fr_src;
	int *Idx_load_hr_src;
	int *Idx_load_qr_src;
	int *Idx_store_hr_src;
	int *Idx_store_qr_src;
	int *Idx_load_p_hr_src;
	int *Idx_load_p_qr_src;
	int *Idx_store_p_hr_src;
	int *Idx_store_p_qr_src;
	int *Idx_load_hr_padtopbottom;
	int *Idx_store_hr_padtopbottom;
	int *Idx_load_qr_padtopbottom;
	int *Idx_store_qr_padtopbottom;
	int *Idx_qr_ref_zerorow_zerocolumn;
	int	*p_temp_mvs;
	int *Idx_store_mv;
	int *Idx_load_mvsad;
	int *Idx_store_mvsad_mbchara;
	int *Idx_store_mvreplicate;
	int splitflag;
    int num_forw_references;
    int num_back_references;
    int num_lowres_forw_references;
    int num_lowres_back_references;
	unsigned int NumRowsMB_Sub_Section;
	unsigned int NumColsMB_Sub_Section;
	unsigned int NumRows_Count_SubSec_Srch;
	unsigned int NumCols_Count_SubSec_Srch;
	unsigned int NumColsMB_Section;
	unsigned int NumRowsMB_Section;
	int dec_ref_avail;
    int sad_thresh;
    int AvgHRSAD;
    int LastAvgHRSAD;
    int CRef;
    CUVME_MV_RESULTS *HRresults;
    CUVME_MV_RESULTS *QRresults;
    CUVME_MV_RESULTS **integer_mvmap;
    CUVME_MB_INFO *HR_MB_Info;
    CUVME_MB_INFO *QR_MB_Info;
    CUVME_MB_INFO **integer_mb_info;
    struct subpel_mvmap_t **subpel_mvmap;
    unsigned int *LowSad;
    unsigned int *nb;
    unsigned int *sp;
	int FrameQP;
    unsigned int *QuarterPelSearchCoords;
    unsigned int *CandidateCountInt;
    ME_CANDIDATE_TUNE candidate_tune_values;

    ZERORES_CONTEXT zero_res_ctxt;
	TWOSTEP_PLUS_CONTEXT twostep_plus_ctxt;
    int num_mvs;
    int num_sec_formats;
    int scene_cut_early_exit;
    CUVME_Y_BUFFER *p_pred_picture;
    int flag_do_mc;
    int source_format;
    CUVME_Y_BUFFER *block_flat_src;
    CUVME_Y_BUFFER *p_half_res_src;
    CUVME_Y_BUFFER *p_quarter_res_src;
    int store_dec_src_flag;
	yuv_frame_t block_flat_yuv;
	int lambda_factor;
	int	do_zero_search;
	int	do_low_res;
	int do_int_search;
	int	do_int_and_halfpel_search;
	int	do_decimation_for_low_res;
	CUVME_MB_CHARAC *p_mb_characs;
	int get_mb_characs_flag;
	int average_mb_mean;
	int AvgHRSADHist[3];
	int average_mb_meanHist[3];
	int avg_var;
	int *sum_vars_frame;
	int *sum_mean_frame;
	int prevp_var;
	int futp_var;
	int prevp_mean;
	int futp_mean;
	int curr_mean;
	int curr_var;
	int fwd_ref_frame_num; 
	int bwd_ref_frame_num; 
	int direction_ref; // 0- forward reference frame used; 1 - backward ref used;  2 - both refs used.
	int ref_frame_distance_fwd; // temporal distance between the input frame and the forward reference frame
	int ref_frame_distance_bwd; //temporal distance between the input frame and the backward reference frame
	int only_pic_characs;
	int b_frames_since_last_p;
	int enable_bwd_reference; // to use  backward reference frame this flag must be enabled by the application before calling cuvme_init and CUVME_get_nonpersistent_mem_size
  int *Est_SumMeanDevnTempArray;
} ME_context_t; 

void me_populate_default_context(ME_context_t *, int CRef);
CUVME_ERROR me_init_src_dec_search_context(ME_context_t *);

void me_ConvertRasterToBlockFlattened(unsigned char *raster_input, unsigned char *block_op,
                                   int source_width, int source_height);

CUVME_ERROR me_calc_nonpersistentmem_offsets(ME_context_t * p_ME_context,
										   int *scratch_mem_size,
										   int width,
										   int height);
CUVME_ERROR me_assign_nonpersistentmem_pointers(ME_context_t * p_ME_context);

CUVME_ERROR me_allocate_non_persistent_memory(ME_context_t * p_ME_context);

CUVME_ERROR me_free_non_persistent_memory(ME_context_t * p_ME_context);


#endif 

