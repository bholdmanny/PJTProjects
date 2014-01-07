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


#include "cuve264b.h"
#include "h264_types.h"
#include "encoder_context.h"


typedef struct SESSION_PARAMS {
    int ref;
    int intra4x4_mode;
    int TestMode;
    int PrintStats;
    int PrintTiming;
    int debug_level;
    int AlphaC0Offset;
    int BetaOffset;
    int DisableDeblockingAcrossSlices;
    int DisableLoopFilter;
    int SeiFlags;
    int SliceMode;
    int SliceParam;
    int FrameHeight;
    int FrameWidth;
    int Height;
    int Width;
    int Xoffset;
    int Yoffset;
    int rc_mode;
    int QP;
    int QPI;
    int IFrate;  // If set to 0, only the first frame will be an I-frame.  If set to 1, all frames will be an I-frame.
    int Fps;
    int min_QP;
    int max_QP;
    int target_bitrate;
    int frame_start;
    int frame_end;
    int trace_start;
    int trace_end;
    int me;
    int input_format;
    int mem_leak_test;
	int slice_num;
} SESSION_PARAMS;

#define CHK_ERR(e, s) { if (e) { printf("(%s, line %d) %s: API error code #%d\n", __FILE__, __LINE__, s, e);  exit(e); } }

void copy_and_pad_frame_for_encode(
    CUVE264B_IMAGE_SPECS *p_input_image,
    yuv_frame_t *p_frame
    );

CUVE264B_ERROR init_output_bitstream_buffer(
    encoder_context_t *p_enc,
    CUVE264B_BITSTREAM_BUFFER *p_output_buffer
    );

void calc_psnr(
    yuv_frame_t *frame0,
    yuv_frame_t *frame1,
    double *res_snr_y,
    double *res_snr_u,
    double *res_snr_v
    );
int check_input_buffer_conformance_to_encoder (
    CUVE264B_IMAGE_SPECS  *p_input_image,
    encoder_context_t     *p_enc
               );

void convert_raster_to_block_flattened_sc 
(
    unsigned char *p_raster_input_y,
    unsigned char *p_raster_input_u,
    unsigned char *p_raster_input_v,
    int source_width,
    int source_height,
    int source_offset_x,
    int source_offset_y,
    unsigned char *p_block_flat_y,
    unsigned char *p_block_flat_u,
    int dest_width,
    int dest_height
);

void print_frame_stats(CUVE264B_FRAME_ENCODE_INFO *p_stats, double psnr_y, double psnr_u, double psnr_v, int frame_num, int frame_start);

void print_enc_stats(
    double total_psnr_y, double total_psnr_u, double total_psnr_v,
    int total_bits, int total_me_cands,
    int qp_sum, int min_qp, int max_qp,
    int num_frames);