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

#ifndef _PARSET_H
#define _PARSET_H

#include "h264_types.h"

typedef enum
{
        PROFILE_IDC_BASELINE = 66,
        PROFILE_IDC_MAIN     = 77,
        PROFILE_IDC_EXTENDED = 88
} profile_idc_e;


#define MAX_VALUE_OF_CPB_CNT   32
#define TMP_MAX_NUM_REFS 16

typedef struct vui_info_t
{
    unsigned short sar_width;       ///< aspect ratio width;
    unsigned short sar_height;      ///< aspect ration height;
    unsigned int num_units_in_tick; ///< frame rate divider;
    unsigned int time_scale;        ///< fps = time_scale/num_units_in_tick;
    unsigned char fixed_frame_rate_flag;    ///< 0/1;
    //< To be expanded...
} vui_info_t;

typedef struct
{
	unsigned  cpb_cnt;							  // ue(v)
	unsigned  bit_rate_scale;					  // u(4)
	unsigned  cpb_size_scale;					  // u(4)
												  // ue(v)
	unsigned  bit_rate_value[MAX_VALUE_OF_CPB_CNT];
												  // ue(v)
	unsigned  cpb_size_value[MAX_VALUE_OF_CPB_CNT];
	unsigned  vbr_cbr_flag[MAX_VALUE_OF_CPB_CNT];  // u(1)
												  // u(5)
	unsigned  initial_cpb_removal_delay_length/*_minus1*/;
	unsigned  cpb_removal_delay_length/*_minus1*/;    // u(5)
	unsigned  dpb_output_delay_length/*_minus1*/;     // u(5)
	unsigned  time_offset_length;				  // u(5)
} hrd_parameters_t;

typedef struct
{
	vui_info_t extra_data;
	int      aspect_ratio_info_present_flag;	  // u(1)
	unsigned  aspect_ratio_idc;					  // u(8)
	int      overscan_info_present_flag;		  // u(1)
	int      overscan_appropriate_flag;			  // u(1)
	int      video_signal_type_present_flag;	  // u(1)
	unsigned  video_format;						  // u(3)
	int      video_full_range_flag;				  // u(1)
	int      colour_description_present_flag;	  // u(1)
	unsigned  colour_primaries;					  // u(8)
	unsigned  transfer_characteristics;			  // u(8)
	unsigned  matrix_coefficients;				  // u(8)
	int      chroma_location_info_present_flag;	  // u(1)
	unsigned  chroma_location_frame;			  // ue(v)
	unsigned  chroma_location_field;			  // ue(v)
	int      fixed_frame_rate_flag;				  // u(1)
	int      nal_hrd_parameters_present_flag;	  // u(1)
	hrd_parameters_t nal_hrd_parameters;		  // hrd_paramters_t
	int      vcl_hrd_parameters_present_flag;	  // u(1)
	hrd_parameters_t vcl_hrd_parameters;		  // hrd_paramters_t
	int      low_delay_hrd_flag;				  // u(1)
	int      bitstream_restriction_flag;		  // u(1)
												  // u(1)
	int      motion_vectors_over_pic_boundaries_flag;
	unsigned  max_bytes_per_pic_denom;			  // ue(v)
	unsigned  max_bits_per_mb_denom;			  // ue(v)
	unsigned  log2_max_mv_length_vertical;		  // ue(v)
	unsigned  log2_max_mv_length_horizontal;	  // ue(v)
	unsigned  max_dec_frame_reordering;			  // ue(v)
	unsigned  max_dec_frame_buffering;			  // ue(v)
} vui_seq_parameters_t;

#define MAX_NUM_SLICE_GROUPS/*_MINUS1*/  8

typedef struct pic_parameter_set_rbsp_t
{
	int error_no;									  // indicates the parameter set is valid
	unsigned pic_parameter_set_id;				  // ue(v)
	unsigned seq_parameter_set_id;				  // ue(v)
	int entropy_coding_mode;					  // u(1)
	int pic_order_present_flag;					  // u(1)
	unsigned num_slice_groups;					  // ue(v)
	unsigned slice_group_map_type;				  // ue(v)
												  // ue(v)
	unsigned run_length[MAX_NUM_SLICE_GROUPS];
												  // ue(v)
	unsigned top_left[MAX_NUM_SLICE_GROUPS];
												  // ue(v)
	unsigned bottom_right[MAX_NUM_SLICE_GROUPS];
	int slice_group_change_direction_flag;		  // u(1)
	unsigned slice_group_change_rate;			  // ue(v)
	unsigned num_slice_group_map_units;			  // ue(v)
	unsigned * slice_group_id;					  // complete MBAmap u(v)
	unsigned num_ref_idx_active[2];				  // ue(v)

	int weighted_pred_flag;						  // u(1)   // For baseline profile, this value is always zero
	int weighted_bipred_idc;					  // u(2)   // For baseline profile, this value is always zero
	unsigned int luma_log2_weight_denom;          // ue(v)
	unsigned int chroma_log2_weight_denom;        // ue(v)
	int wp_weight_list0[TMP_MAX_NUM_REFS][3];     // se(v)
	int wp_weight_list1[TMP_MAX_NUM_REFS][3];     // se(v)
	int wp_offset_list0[TMP_MAX_NUM_REFS][3];  // se(v)
	int wp_offset_list1[TMP_MAX_NUM_REFS][3];  // se(v)


	int pic_init_qp;							  // se(v)
	int pic_init_qs;							  // se(v)
	int chroma_qp_index_offset;					  // se(v)
	int deblocking_filter_parameters_present_flag;// u(1)
	int constrained_intra_pred_flag;			  // u(1)
	int redundant_pic_cnt_present_flag;			  // u(1)
	int vui_pic_parameters_flag;				  // u(1)

	int frame_cropping_flag;					  // u(1)
	unsigned frame_cropping_rect_left_offset;	  // ue(v)
	unsigned frame_cropping_rect_right_offset;	  // ue(v)
	unsigned frame_cropping_rect_top_offset;	  // ue(v)
	unsigned frame_cropping_rect_bottom_offset;	  // ue(v)
}pic_parameter_set_rbsp_t;

#define MAX_NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE  256

typedef struct
{
	int error_no;									  // 0 indicates the parameter set is valid 
	unsigned profile_idc;						  // u(8)
	unsigned level_idc;							  // u(8)

	int more_than_one_slice_group_allowed_flag;	  // u(1)
	int arbitrary_slice_order_allowed_flag;		  // u(1)
	int redundant_slices_allowed_flag;			  // u(1)

	unsigned seq_parameter_set_id;				  // ue(v)
	unsigned log2_max_frame_num;				  // ue(v)
	unsigned pic_order_cnt_type;
	unsigned log2_max_pic_order_cnt_lsb;		  // ue(v)
	int delta_pic_order_always_zero_flag;		  // u(1)
	int offset_for_non_ref_pic;					  // se(v)
	int offset_for_top_to_bottom_field;			  // se(v)
												  // ue(v)
	unsigned num_ref_frames_in_pic_order_cnt_cycle;
	int offset_for_ref_frame[MAX_NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE];
	unsigned num_ref_frames;					  // ue(v)
	int required_frame_num_update_behaviour_flag;  // u(1)
	unsigned pic_width_in_mbs;					  // ue(v)
	unsigned frame_height_in_mbs;					  // calculated !!!
	unsigned pic_height_in_map_units;			  // ue(v)
	int frame_mbs_only_flag;					  // u(1)
	int mb_adaptive_frame_field_flag;			  // u(1)
	int direct_8x8_inference_flag;				  // u(1)
	int vui_parameters_present_flag;			  // u(1)
	vui_seq_parameters_t vui_info;	  // vui_seq_parameters_t

	int constrained_set0_flag;
	int constrained_set1_flag;
	int constrained_set2_flag;

	int frame_cropping_flag;                   //u_1 (bs);
	int frame_cropping_rect_left_offset;       //ue_v (bs);
	int frame_cropping_rect_right_offset;      //ue_v (bs);
	int frame_cropping_rect_top_offset;        //ue_v (bs);
	int frame_cropping_rect_bottom_offset;     //ue_v (bs);

}seq_parameter_set_rbsp_t;

typedef struct slice_header_params_t
{
	int first_mb_in_slice;
	slicetype_e slice_type;
	int pic_parameter_set_id;
	int frame_num;
	int field_pic_flag; // 1 - field; 0 - frame
	int bottom_field_flag; 
	int idr_flag;
	int idr_pic_id;

	int pic_order_cnt_lsb;
	int delta_pic_order_cnt_bottom;
	int delta_pic_order_cnt[2];
	int redundant_pic_cnt;
	int direct_spatial_mv_pred_flag;
	int num_ref_pic_active_fwd;
	int num_ref_pic_active_bwd;
    int remapping_of_pic_nums_idc[2][TMP_MAX_NUM_REFS];
	int abs_diff_pic_num[2][TMP_MAX_NUM_REFS];
	int long_term_pic_idx[2][TMP_MAX_NUM_REFS];
	int ref_pic_list_reordering_flag[2];
	int qp;
	int disable_deblocking_filter_idc;
	int slice_alpha_c0_offset;
	int slice_beta_offset;
	int no_output_of_prior_pics_flag;
	int long_term_reference_flag;
	int cabac_init_idc;
	int slice_group_change_cycle;
	int adaptive_ref_pic_buffering_flag;

} slice_header_params_t;

typedef struct user_data_sei_message_t
{
  char *p_bs_info;
  unsigned char min_mv_width;
  unsigned char min_mv_height;
  unsigned char subpel_lvl;
  unsigned char intra_in_inter;
  unsigned char max_ref_frames;
  unsigned char bkwd_refs;
  unsigned char deblock_non_ref;
} user_data_sei_message_t;
#endif

