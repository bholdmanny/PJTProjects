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


#ifndef _RC_INTERNAL_SETTINGS_H_
#define _RC_INTERNAL_SETTINGS_H_

void rc_reset_fr_computer (RC_STATE *rcs);
CUVRC_ERROR rate_control_setup(RC_STATE* handle,
						int enc_width,
						int enc_height);
void set_defaults(RC_STATE* rcs);
CUVRC_ERROR set_basic_rc_parameters(RC_STATE* rcs);
void set_frame_type(RC_STATE *rcs);
void update_previous_frame_statistics(RC_STATE *rcs);
void bitrate_buffer_reset (RC_STATE *rcs);
int set_initial_qp(RC_STATE *rcs);
void reset_rois(RC_STATE *rcs) ;
void calc_iframe_pframe_bits(RC_STATE *rcs, int pframe_bits);
void update_previous_gop_error(RC_STATE *rcs);
void set_abr_limits(ABR_CONTEXT *abr_params);
void compare_errors(RC_STATE *rcs);
void modify_frame_types(RC_STATE *rcs);

void update_forced_iframe_error(RC_STATE *rcs);
void dstribute_scene_change_error(RC_STATE *rcs);
int set_qp(RC_STATE *rcs);
void modify_initial_qp(RC_STATE *rcs);


int calc_next_frame_type(unsigned long long *frame_timestamp,
						 int read_pos,
						 int add_one_frame,
						 int inter_frame_interval_ms,
						 unsigned long long iframe_time,
						 int iframe_period);
void update_buffer(RC_STATE *p_rc_context, int num_bits);
void modify_input_data(RC_STATE *rcs);
void update_scene_change_error(RC_STATE *rcs);
void populate_mb_qps_array(RC_STATE *rcs);
void copy_frame_qp_to_mb_qp(RC_STATE *rcs);
#endif 
