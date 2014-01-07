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

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../inc/rc_common.h"
#include "../inc/rc_internal_settings.h"

void rc_reset_fr_computer (RC_STATE *rcs)
{
    unsigned int i;
//    unsigned int init_frame_time;
    RC_FR_COMPUTER *p_comp = &rcs->fr_computer;
    
    p_comp->buf_size = 0;
    p_comp->buf_mask = RC_FR_COMPUTER_BUF_SIZE - 1; 
    p_comp->cur_pos  = 0;

//    init_frame_time =  (unsigned int)(1000.0 / (float)rcs->target_fps);
    for (i = 0; i < RC_FR_COMPUTER_BUF_SIZE; i++)
    {
        p_comp->frame_time[i] = 0;
        p_comp->frame_timestamp[i] = 0;
    }
    p_comp->num_frames = 0;
    p_comp->time_low = 0;
    p_comp->time_high = 0;
    p_comp->total_time = 0;
    p_comp->prev_time = 0;
    p_comp->cur_time = 0;
    p_comp->iframe_time = 0;
    p_comp->iframe_time_adjustment = 0;
    p_comp->write_pos = 0;
    p_comp->read_pos = -1;
    p_comp->new_timer = 1;    
    
}




CUVRC_ERROR set_basic_rc_parameters(RC_STATE* rcs)
{
	int input_mode = (rcs->bitrate_type << 1) | rcs->iframe_period_type;
	float num_p_frames_per_sec;
	
	switch(input_mode)
	{
	case 0: {
		//bitrate : BPS
		// i frame distance : milliseconds
		rcs->target_bps = rcs->target_bitrate;
		rcs->target_bpf = rcs->target_bitrate/rcs->target_fps;
		rcs->if_persec	= (1/(float)rcs->iframe_period)*1000;
		rcs->ifrate = (rcs->target_fps * rcs->iframe_period)/1000;
		break;
	}
	case 1: {
		//bitrate : BPS
		// i frame distance : frames
		rcs->target_bps = rcs->target_bitrate;
		rcs->target_bpf = rcs->target_bitrate/rcs->target_fps;
		rcs->ifrate = rcs->iframe_period;
		rcs->if_persec	= (rcs->target_fps/(float)rcs->iframe_period);
		break;
	}
	case 2: {
		//bitrate : BPF
		// i frame distance : milliseconds
		rcs->target_bps = rcs->target_bitrate * rcs->target_fps;
		rcs->target_bpf = rcs->target_bitrate;
		rcs->ifrate = (rcs->target_fps * rcs->iframe_period)/1000;
		rcs->if_persec	= (1/(float)rcs->iframe_period)*1000;
		break;
	}
	case 3: {
		//bitrate : BPF
		// i frame distance : frames
		rcs->target_bps = rcs->target_bitrate * rcs->iframe_period;
		rcs->target_bpf = rcs->target_bitrate;
		rcs->ifrate =  rcs->iframe_period;
		if(rcs->target_bps < 0)
		{
			rcs->target_bps = rcs->target_bpf * rcs->target_fps;
			rcs->if_persec = rcs->target_fps / rcs->ifrate;			
		}
		else
		{
			rcs->target_fps = rcs->iframe_period;
			rcs->if_persec	= 1;
		}
		break;
	}
//	default:
//		return CUVRC_ERR_FATAL_INTERNAL;
	}
    rcs->ifrate_uint = rcs->ifrate;
	num_p_frames_per_sec = rcs->target_fps - rcs->if_persec;
	if(rcs->non_ref_frame_interval != 0)
		rcs->ref_p_frames_per_sec = (int)(num_p_frames_per_sec / (rcs->non_ref_frame_interval + 1));
	else
		rcs->ref_p_frames_per_sec = num_p_frames_per_sec;	
		
	rcs->non_ref_frames_per_sec = num_p_frames_per_sec - rcs->ref_p_frames_per_sec;
	rcs->num_prev_gop_bframes = ((rcs->ifrate_uint - 1) % (rcs->bframe_interval + 1));
	rcs->prev_gop_brames_time = (rcs->num_prev_gop_bframes / rcs->target_fps)*1000;
	if(rcs->non_ref_frame_interval > (rcs->ifrate_uint - 1))
	{
		printf("ERROR : interval between two reference frames is larger than GOP size. Returning error!!!\n");
		return CUVRC_ERR_ARG;
	}
	return CUVRC_ERR_SUCCESS;
}

//  set parameters & initialize hypothetical buffer: 
CUVRC_ERROR rate_control_setup(
	RC_STATE* handle, 
    int enc_width,
    int enc_height)
{
    float i2p_ratio, prev_i2p_ratio; 
    int kfrate, bitrate;
    float if_persec, fpsplus;
//    int initial_qp;
    RC_STATE *rcs = (RC_STATE *)handle;
    int pframe_bits, non_ref_frame_bits;
    int avg_p_qp;
    CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
    
    if(rcs->need_reset)
    {
    	err = set_basic_rc_parameters(rcs);
    	rcs->original_values.target_bps = rcs->target_bps;
    }
    
    prev_i2p_ratio		= rcs->iframeratio;
    
    i2p_ratio           = rcs->iframeratio;
    /*kfrate              = rcs->ifrate; */                          // intra frame rate

    bitrate             = rcs->target_bps;    
//    rcs->mode           = rc_info->rc_mode;    
        
    if_persec		= rcs->if_persec;
    
	if( if_persec <= 1 )
	{
		i2p_ratio = 4;  
	}
	else if(if_persec <= 2)
	{
		i2p_ratio = 4;  
	}
	else if(if_persec <= 3)
	{
		i2p_ratio = 4;   
	}
	else if (if_persec <= 4)
	{
		i2p_ratio = 4;   
	}
	else if (if_persec <= 7)
	{
		i2p_ratio = 3;   
	}
	else if(if_persec > 7)
	{
		i2p_ratio = 2;   
	}
	else
	{
		i2p_ratio = 8;
	}
		

    if(rcs->curr_gop_frames_after_last_iframe > 0)
    {
        avg_p_qp = rcs->sum_p_qp_in_gop/(rcs->curr_gop_frames_after_last_iframe);
        rcs->avg_p_qp = avg_p_qp; 
        // if there has been a lot of variation in qp in the previous gop,
        // it indicates that the ratio of I frame to P frames bit allocation is not appropriate 
        // & needs to be changed
        // here the assumption is that, even though there is a scene change, the type of motion
        // will be similar to that before the scene change & will not change drastically
        if(((avg_p_qp - rcs->iframe_stat.qp) >= 10) || ((rcs->last_qp - rcs->iframe_stat.qp) >= 10))
        	prev_i2p_ratio = max(prev_i2p_ratio - 1, 1);
        else if (((avg_p_qp - rcs->iframe_stat.qp) <= -3) || ((rcs->last_qp - rcs->iframe_stat.qp) < -5))
        	prev_i2p_ratio++;      
        
        if(rcs->total_non_ref_num_bits != 0)
        {
        	int ref_pframes_after_last_iframe, non_ref_frames_after_last_iframe, ref_p_num_bits, non_ref_num_bits;
        	ref_pframes_after_last_iframe = rcs->curr_gop_frames_after_last_iframe / (rcs->non_ref_frame_interval + 1);
        	non_ref_frames_after_last_iframe = rcs->curr_gop_frames_after_last_iframe - ref_pframes_after_last_iframe;
        	if ((ref_pframes_after_last_iframe != 0) && (non_ref_frames_after_last_iframe != 0)) 
        	{
				ref_p_num_bits = rcs->total_ref_p_num_bits / ref_pframes_after_last_iframe;
				non_ref_num_bits = rcs->total_non_ref_num_bits / non_ref_frames_after_last_iframe;
				rcs->p_2_non_ref_p_ratio = (float)ref_p_num_bits / non_ref_num_bits;
				rcs->p_2_non_ref_p_ratio = clip_3(rcs->p_2_non_ref_p_ratio, 2, 1);
			}
		}
    }
		
    rcs->iframeratio 	=	min(i2p_ratio, prev_i2p_ratio);
    rcs->iframeratio 	=	max(rcs->iframeratio, 2);
	if(rcs->valid_avg_var)
	{
		// if the variance of the scene cut picture is very high
		// we can't afford to allocate very less bits to the I frame
		if(rcs->avg_var >= 20)
			rcs->iframeratio = max(rcs->iframeratio, 2.5);
		if(rcs->avg_var >= 25)
			rcs->iframeratio = max(rcs->iframeratio, 3);
	}

    fpsplus             = rcs->non_ref_frames_per_sec + rcs->p_2_non_ref_p_ratio * (float)(rcs->ref_p_frames_per_sec + rcs->iframeratio * if_persec);
    non_ref_frame_bits = (int)( bitrate / fpsplus );
    if(rcs->scene_change)
    {
    	rcs->prev.iframe_bits = rcs->original_values.iframe_bits;
    	rcs->prev.pframe_bits = rcs->original_values.pframe_bits;
    	rcs->prev.non_ref_frame_bits = rcs->original_values.non_ref_frame_bits;
    	rcs->prev.scaled_non_ref_frame_bits = rcs->non_ref_frame_bits;
    	rcs->prev.last_qp = rcs->last_qp;  
    	rcs->prev.last_actual_bits = rcs->num_bits;
    }
    calc_iframe_pframe_bits(rcs, non_ref_frame_bits);

    
    if((rcs->need_full_reset) || rcs->scene_change)
    {    	
		// if we should reset the gop at scene change then modify the value of gop size counter
		if((rcs->need_full_reset) || (rcs->gop_reset_flag == CUVRC_SCENE_CHANGE_RESET_GOP))
		{
			rcs->current_gop_size = rcs->ifrate - rcs->num_prev_gop_bframes;
			rcs->fr_computer.iframe_time_adjustment = rcs->iframe_period;
		}
    }
    	
    
    if(rcs->need_full_reset)
    {   	
		rcs->HRB_fullness   = 0;
		rcs->frame_num = 0;
		rcs->iframe_flag    = 1;
		rcs->last_iframe_flag = 0;		
		rcs->scene_change = 0;
		rcs->num_mbs = enc_width * enc_height / 256;
		rcs->large_mv_thresh = rcs->num_mbs / 10;
		rcs->imb_thresh = rcs->num_mbs / 10;		
		rcs->is_bframe_from_prev_gop = FALSE;
		rcs->use_prev_bits = FALSE;
	}       
    return err;
}

void set_defaults(RC_STATE* rcs)
{
	
	rcs->prime_format	= CUVRC_H264;
	rcs->need_reset		=	1;
	rcs->need_full_reset = 1;
	rcs->ifrate			= DEFAULT_IFRAME_RATE;
    rcs->ifrate_uint    = DEFAULT_IFRAME_RATE;
	rcs->iframe_period	=	DEFAULT_IFRAME_RATE;
	rcs->iframe_period_type = DEFAULT_IFRAME_PERIOD_TYPE;
	rcs->gop_num		= 0;
	rcs->iframeratio	=	DEFAULT_IFRAMERATIO;
	rcs->num_mbs		= 	0;
	rcs->iframe_bits	= 0;
	rcs->pframe_bits	= 0;
	rcs->non_ref_frame_bits = 0;
	rcs->sum_p_qp_in_gop = 0;
    rcs->avg_p_qp = DEFAULT_MIN_QP;
	rcs->scale_avg_p_qp  = 0;
	rcs->current_gop_size = rcs->ifrate;
	rcs->curr_gop_frames_after_last_iframe = 0;
	rcs->prev_gop_bit_error = 0;
	rcs->iframeratio	= DEFAULT_IFRAMERATIO;
	rcs->if_persec		= (DEFAULT_FRAMERATE/(float)DEFAULT_IFRAME_RATE);
	rcs->large_mv_thresh = 0;
	rcs->imb_thresh = 0;
	rcs->target_bitrate	= DEFAULT_BITRATE/DEFAULT_FRAMERATE;
	rcs->bitrate_type	= DEFAULT_BITRATE_TYPE;	
	rcs->bitrate_mode	= CUVRC_NORMAL_BITRATE;
	rcs->frame_qp		=	DEFAULT_PFRAME_QP;
	rcs->actualQP		=	DEFAULT_PFRAME_QP;
	rcs->original_qp	= 	DEFAULT_PFRAME_QP;
	rcs->last_qp		= 	DEFAULT_PFRAME_QP;
	rcs->last_frame_type = CUVRC_P_FRAME;
	rcs->last_p_qp		= DEFAULT_PFRAME_QP;
	rcs->first_p_qp_changed = FALSE;
	rcs->im_width		= 0;
	rcs->im_height		= 0;
	rcs->width_in_mbs	= 0;
	rcs->height_in_mbs	= 0;
	rcs->HRB_fullness	= 0;
	rcs->high_motion_error = 0;
	rcs->iframe_flag	= TRUE;
	rcs->force_iframe_flag = TRUE;
	rcs->curr_frame_type = CUVRC_I_FRAME;
	rcs->next_frame_type = CUVRC_I_FRAME;
	rcs->start_of_gop_flag = TRUE;
	rcs->min_qp = DEFAULT_MIN_QP;
	rcs->max_qp = DEFAULT_MAX_QP;
	rcs->quality_mode	= CUVRC_NORMAL_QUALITY;
	rcs->target_bits	= 0;
	rcs->avg_sad		 = 0;
	rcs->num_bits		 = 0;
	rcs->total_p_num_bits	 = 0;
	rcs->num_imbs = 0;
	rcs->num_enc_mbs = 0;
	rcs->num_big_mvs = 0;
	rcs->num_skipped_mbs = 0;
	rcs->last_iframe_flag = 0;
	rcs->last_sad = 0;
	rcs->i_p_frame_diff	 = IFRAMEDIFF_CBR;
	rcs->is_initial_qp	 = FALSE;
	rcs->scene_change	 = FALSE;
	rcs->scene_change_error = 0;
	rcs->gop_reset_flag = CUVRC_SCENE_CHANGE_RESET_GOP;
	rcs->frame_num		 = 0;
	rcs->iframe_stat.qp	=	DEFAULT_IFRAME_QP;
	rcs->iframe_stat.actual_bits = 0;
	rcs->iframe_stat.target_bits = 0;
	rcs->iframe_stat.HRB_fullness_after_iframe = 0;
	rcs->mode	=	RC_MODE_VBR;
	rcs->force_qp_flag	= 0;
	rcs->target_bps		=	DEFAULT_BITRATE;
	rcs->target_bpf		= DEFAULT_BITRATE/DEFAULT_FRAMERATE;
	rcs->target_fps		=	DEFAULT_FRAMERATE;
	rcs->actual_fps		= DEFAULT_FRAMERATE;
	rcs->fsm.current_state = NORMAL_MOTION;
	rcs->fsm.next_state = NORMAL_MOTION;
	rcs->fsm.state_status = CONTINUE;	
	rcs->fsm.wait_state_counter = 0;	
	rcs->fsm.reset_HRB_fullness = 0;	
	rcs->original_values.target_bps = DEFAULT_BITRATE;
	rcs->original_values.pframe_bits = 0;
	rcs->original_values.iframe_bits = 0;
	rcs->original_values.non_ref_frame_bits = 0;
	rcs->original_values.min_qp = DEFAULT_MIN_QP;
	rcs->original_values.max_qp = DEFAULT_MAX_QP;
	rcs->current_frame_motion_type = NORMAL;
	rcs->p_mb_qp = NULL;
	rcs->p_mb_characs = NULL;
	rcs->p_qp_list = NULL;
	reset_rois(rcs);
	rcs->roi_info.extremely_good_quality_rois = NULL;
	rcs->roi_info.good_quality_rois = NULL;
	rcs->roi_info.poor_quality_rois = NULL;
	rcs->roi_info.extremely_poor_quality_rois = NULL;
	rcs->valid_avg_var = FALSE;
	rcs->avg_var = 0;
	rcs->abr_params.scale = DEFAULT_ABR_SCALE;
	set_abr_limits(&rcs->abr_params);
	rcs->scene_change_fsm.current_state = GOP_WITH_NO_SCENE_CHANGE_ERROR;
	rcs->scene_change_fsm.next_state = GOP_WITH_NO_SCENE_CHANGE_ERROR;
	rcs->scene_change_fsm.state_status = CONTINUE;
	rcs->non_ref_pframe_interval = 0;
	rcs->ref_p_frames_per_sec = 0;
	rcs->non_ref_frames_per_sec = 0;
	rcs->non_ref_p_count = 0;
	rcs->scene_change_postponed = FALSE;
	rcs->current_gop_scene_change_error = 0;
	rcs->bframe_interval = 0;
	rcs->bframe_count = 0;
	rcs->num_prev_gop_bframes = 0;
	rcs->prev_gop_brames_time = 0;
	rcs->prev_gop_bframes_count = 0;
	rcs->next_frame_offset = 1;
	rcs->frame_type_fsm.current_state = I_FRAME;
	rcs->frame_type_fsm.next_state = I_FRAME;
	rcs->frame_type_fsm.state_status = CONTINUE;
	rcs->frame_type_fsm.is_prev_ref_frame = TRUE;	
	
	rcs->input_config.prime_format = (CUVRC_APP_TYPE)rcs->prime_format;
	rcs->input_config.im_width = rcs->im_width;
	rcs->input_config.im_height = rcs->im_height;
	rcs->input_config.target_bitrate = rcs->target_bitrate;
	rcs->input_config.bitrate_type = rcs->bitrate_type;
	rcs->input_config.bitrate_mode = rcs->bitrate_mode;
	rcs->input_config.quality_mode = rcs->quality_mode;
	rcs->input_config.iframe_period = rcs->iframe_period;
	rcs->input_config.iframe_period_type = rcs->iframe_period_type;
	rcs->input_config.non_ref_pframe_interval = rcs->non_ref_pframe_interval;
	rcs->input_config.bframe_interval = rcs->bframe_interval;
	rcs->input_config.target_fps = rcs->target_fps;
		
	rcs->reset_postponed = FALSE;
	rcs->prev.iframe_bits = 0;
	rcs->prev.pframe_bits = 0;
	rcs->prev.non_ref_frame_bits = 0;
	rcs->prev.scaled_non_ref_frame_bits = 0;
	rcs->prev.last_qp = 0;	
	rcs->prev.last_actual_bits = 0;
	rcs->is_bframe_from_prev_gop = FALSE;
	rcs->frames_after_last_iframe = 0;
	rcs->non_ref_frame_interval = 0;
	rcs->use_prev_bits = FALSE;
	rcs->non_ref_frame_flag = FALSE;
	rcs->last_non_ref_frame_flag = FALSE;
	rcs->is_backward_prediction = FALSE;
	rcs->p_2_non_ref_p_ratio = (float)DEFAULT_P_2_NON_REF_P_RATIO;
	rcs->total_ref_p_num_bits = 0;
	rcs->total_non_ref_num_bits = 0;
	rcs->last_ref_frame_stat.qp = DEFAULT_PFRAME_QP;
	rcs->last_ref_frame_stat.target_bits = 0;
	rcs->last_ref_frame_stat.actual_bits = 0;
	rcs->last_ref_frame_stat.frame_type = CUVRC_P_FRAME;
	rcs->last_ref_frame_stat.iframe_flag = FALSE;
	rcs->unmodified_HRB_fullness = 0;
	rcs->acc_err_from_last_ref_frame = 0;
	
	rcs->roi_info.extremely_good_quality_max_qp = DEFAULT_EXTREMELY_GOOD_QUALITY_MAX_QP;
	rcs->roi_info.good_quality_max_qp = DEFAULT_GOOD_QUALITY_MAX_QP;
	rcs->roi_info.poor_quality_min_qp = DEFAULT_POOR_QUALITY_MIN_QP;
	rcs->roi_info.extremely_poor_quality_min_qp = DEFAULT_EXTREMELY_POOR_QUALITY_MIN_QP;
	
	rcs->min_frame_qp = rcs->frame_qp;
	rcs->max_frame_qp = rcs->frame_qp;
	
	rcs->force_iframe_postponed = FALSE;
}

void set_frame_type(RC_STATE *rcs)
{
	if ((rcs->force_iframe_flag) && (rcs->gop_reset_flag == CUVRC_SCENE_CHANGE_RESET_GOP)
			&& (rcs->next_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD)) 
	{
		// if i frame is forced then we reset the gop if it is allowed
		rcs->current_gop_size = rcs->ifrate_uint - rcs->num_prev_gop_bframes;		
		rcs->fr_computer.iframe_time_adjustment = rcs->iframe_period;
	}	
	if (rcs->iframe_period_type == CUVRC_PERIOD_FRAMES) 
	{
		if ((rcs->current_gop_size) == (rcs->ifrate_uint - rcs->num_prev_gop_bframes)) 
		{
			rcs->iframe_flag = TRUE;
			rcs->curr_frame_type = CUVRC_I_FRAME;
			rcs->gop_num++;
		} else 
		{
			rcs->iframe_flag = FALSE;
			rcs->curr_frame_type = CUVRC_P_FRAME;
		}

		if((rcs->ifrate_uint - rcs->num_prev_gop_bframes) != 1)
		{
			if ((rcs->current_gop_size + 1) == (rcs->ifrate_uint - rcs->num_prev_gop_bframes))
				rcs->next_frame_type = CUVRC_I_FRAME;
			else
				rcs->next_frame_type = CUVRC_P_FRAME;
		}
		else
		{
			if(rcs->prev_gop_bframes_count <= 1)
				rcs->next_frame_type = CUVRC_I_FRAME;
			else
				rcs->next_frame_type = CUVRC_P_FRAME;
		}
	} else 
	{
		int iframe_period = rcs->iframe_period - rcs->fr_computer.iframe_time_adjustment;
		int inter_frame_interval_ms = (1/rcs->target_fps) * 1000;
		RC_FR_COMPUTER *p_comp = &rcs->fr_computer;
		p_comp->read_pos = (p_comp->read_pos + rcs->next_frame_offset) & p_comp->buf_mask;
		assert(rcs->iframe_period >= rcs->fr_computer.iframe_time_adjustment);
		if((int)(p_comp->frame_timestamp[p_comp->read_pos] - p_comp->iframe_time) > (int)((iframe_period - inter_frame_interval_ms/2) * 1000))
		{
			rcs->iframe_flag = TRUE;
			rcs->curr_frame_type = CUVRC_I_FRAME;
			rcs->gop_num++;
			if(iframe_period == 0)
				// the iframe is forced & doesn't happen becs the gop is complete
				// hence take the current frame time as the i frame time
				p_comp->iframe_time = p_comp->frame_timestamp[p_comp->read_pos];
			else
				p_comp->iframe_time += rcs->iframe_period * 1000;				
			p_comp->iframe_time_adjustment = 0;
		} else 
		{
			rcs->iframe_flag = FALSE;
			rcs->curr_frame_type = CUVRC_P_FRAME;
		}
		
		if ((rcs->next_frame_offset == 1) && (rcs->bframe_count != 1) && (rcs->prev_gop_bframes_count != 1)) 
		{
			// normal cases
			rcs->next_frame_type =(CUVRC_FRAME_TYPE) calc_next_frame_type(p_comp->frame_timestamp, p_comp->read_pos, 1,
					inter_frame_interval_ms, p_comp->iframe_time, rcs->iframe_period);			
		} else if ((rcs->bframe_count == 1) || (rcs->prev_gop_bframes_count == 1)){
			// last b frame in a group of b frames
			// at this point we should scan the timestamps of all the frame after this frame
			// to decide if one of them should be an I frame 
			int loop_cnt = 0;
			int i;
			if (p_comp->write_pos > p_comp->read_pos)
				loop_cnt = p_comp->write_pos - p_comp->read_pos;
			else
				loop_cnt = p_comp->write_pos + (RC_FR_COMPUTER_BUF_SIZE	- p_comp->read_pos);
			for (i = 2; i < loop_cnt; i++) 
			{
				int read_pos = (p_comp->read_pos + i) & p_comp->buf_mask;
				rcs->next_frame_type = (CUVRC_FRAME_TYPE)calc_next_frame_type(p_comp->frame_timestamp, read_pos,0,
									inter_frame_interval_ms, p_comp->iframe_time, rcs->iframe_period);			
				if(rcs->next_frame_type == CUVRC_I_FRAME)
				{
					rcs->num_prev_gop_bframes = i - 2;
					break;
				}				
			}
			if (rcs->next_frame_type != CUVRC_I_FRAME) 
			{
				// check for one frame in the future which might be an I frame
				int read_pos = (p_comp->write_pos - 1) & p_comp->buf_mask;
				rcs->next_frame_type = (CUVRC_FRAME_TYPE)calc_next_frame_type(p_comp->frame_timestamp, read_pos, 1,
						inter_frame_interval_ms, p_comp->iframe_time, rcs->iframe_period);
				if(rcs->next_frame_type == CUVRC_I_FRAME)
					rcs->num_prev_gop_bframes = loop_cnt - 2;
			}			
		} else
			rcs->next_frame_type = CUVRC_P_FRAME;			
	}	
	
	if (rcs->curr_frame_type == CUVRC_I_FRAME)
		rcs->start_of_gop_flag = TRUE;
	else
		rcs->start_of_gop_flag = FALSE;
	
	
	if ((rcs->gop_reset_flag == CUVRC_SCENE_CHANGE_DO_NOT_RESET_GOP)
			&& (rcs->start_of_gop_flag == FALSE)) 
	{
		if (rcs->scene_change) {
			rcs->iframe_flag = TRUE;
			rcs->curr_frame_type = CUVRC_PSEUDO_I_FRAME;
		}		
	}	
	
	if((rcs->non_ref_p_count != rcs->non_ref_pframe_interval) && (rcs->curr_frame_type == CUVRC_P_FRAME))
	{
        rcs->curr_frame_type = CUVRC_NON_REFERENCE_P_FRAME_FORWARD;		
	}

	if((rcs->non_ref_pframe_interval != 0) && (rcs->next_frame_type == CUVRC_P_FRAME))
	{
		if (rcs->curr_frame_type == CUVRC_I_FRAME)
		{
			rcs->next_frame_type = CUVRC_NON_REFERENCE_P_FRAME_FORWARD;
		}
		else
		{
			if((rcs->non_ref_p_count + 1) != rcs->non_ref_pframe_interval)
				rcs->next_frame_type = CUVRC_NON_REFERENCE_P_FRAME_FORWARD;
		}
	}
	
	
	if ((rcs->bframe_interval != 0) || (rcs->next_frame_offset != 1)
			|| (rcs->prev_gop_bframes_count != 0))
	{
		modify_frame_types(rcs);		
	}
	else
		rcs->next_frame_offset = 1;
		
	
	if ((rcs->start_of_gop_flag == FALSE) && (rcs->force_iframe_flag))
	{
		rcs->iframe_flag = TRUE;
		rcs->curr_frame_type = CUVRC_I_FRAME;
	}
	
	rcs->non_ref_frame_flag = ((rcs->curr_frame_type == CUVRC_NON_REFERENCE_P_FRAME_FORWARD) ||
			(rcs->curr_frame_type == CUVRC_NON_REFERENCE_B_FRAME_FORWARD) || 
			(rcs->curr_frame_type == CUVRC_NON_REFERENCE_B_FRAME_BACKWARD));
}
 
// update the error caused in generating bits as comapred to target bits 
void update_buffer(RC_STATE *p_rc_context, int num_bits)
{
	int target_bits;
    p_rc_context->num_bits = num_bits;

    switch(p_rc_context->curr_frame_type)
    {
    case CUVRC_P_FRAME:
        target_bits = p_rc_context->pframe_bits;        
        break;
    case CUVRC_I_FRAME:
    case CUVRC_PSEUDO_I_FRAME:
        target_bits = p_rc_context->iframe_bits;        
        break;
    case CUVRC_NON_REFERENCE_P_FRAME_FORWARD:
    case CUVRC_NON_REFERENCE_B_FRAME_FORWARD:
    case CUVRC_NON_REFERENCE_B_FRAME_BACKWARD:
        target_bits = p_rc_context->non_ref_frame_bits;
        if(p_rc_context->scene_change_postponed == TRUE)
        {
            target_bits = num_bits - (p_rc_context->non_ref_frame_bits / 10);
            p_rc_context->scene_change_error += num_bits - p_rc_context->non_ref_frame_bits
                - (p_rc_context->non_ref_frame_bits / 10);			
        }   
        if(p_rc_context->use_prev_bits)
        	target_bits = p_rc_context->prev.scaled_non_ref_frame_bits;
        break;   	
    default:
        printf("Error : unknown frame type\n");
        exit(-1);
        break;
    }

    if(p_rc_context->current_gop_scene_change_error != 0) 
    {
        int fpsplus  = p_rc_context->iframeratio + p_rc_context->ifrate - 1;
        int remaining_frames = p_rc_context->ifrate - p_rc_context->current_gop_size;
        int add_error = 0;
        if(p_rc_context->start_of_gop_flag == FALSE)
        	// if the remaining gop size is less than 1/3rd of actual gop size,
        	// we clip it to 1/3 rd gop size so that the remaining frames are not overloaded
        	// to compensate for error
            fpsplus  = p_rc_context->iframeratio + p_rc_context->ifrate - min(p_rc_context->current_gop_size, 2 * p_rc_context->ifrate/3);
        switch(p_rc_context->curr_frame_type)
        {
        case CUVRC_P_FRAME:
        	// remaining_frames = 0 condition will occur in case of non integer frame rates        	
        	if(remaining_frames != 0)
        		add_error = p_rc_context->current_gop_scene_change_error / remaining_frames;
            break;
        case CUVRC_I_FRAME:            
            add_error = (p_rc_context->current_gop_scene_change_error * p_rc_context->iframeratio)/fpsplus;
            break;
        case CUVRC_NON_REFERENCE_P_FRAME_FORWARD:            
        	// remaining_frames = 0 condition will occur in case of non integer frame rates
        	if(remaining_frames != 0)
        		add_error = p_rc_context->current_gop_scene_change_error / remaining_frames;
            break;
        case CUVRC_PSEUDO_I_FRAME:            
            add_error = (p_rc_context->current_gop_scene_change_error * p_rc_context->iframeratio)/fpsplus;
            break;
        case CUVRC_NON_REFERENCE_B_FRAME_FORWARD:
        case CUVRC_NON_REFERENCE_B_FRAME_BACKWARD:
        	// (remaining_frames == 0) condition occurs when the configuration is changed from
        	// (i frame rate > 1) to (iframe rate = 1)
            if(remaining_frames != 0)
                add_error = p_rc_context->current_gop_scene_change_error / remaining_frames;
        	break;        	
        default:
            printf("Error : unknown frame type\n");
            exit(-1);
            break;
        }

        p_rc_context->HRB_fullness += add_error;
        p_rc_context->current_gop_scene_change_error -= add_error;

    }
	if ((p_rc_context->HRB_fullness <= 0) && (p_rc_context->high_motion_error <= 0)
			&& (p_rc_context->frame_qp == p_rc_context->original_values.min_qp)
			&& (p_rc_context->num_bits <= target_bits )) 
	{
		p_rc_context->HRB_fullness = 0;
		p_rc_context->high_motion_error = 0;
		p_rc_context->prev_gop_bit_error = 0;
		p_rc_context->acc_err_from_last_ref_frame = 0;
	} else 
	{
		int tmp = p_rc_context->HRB_fullness;
		p_rc_context->HRB_fullness += p_rc_context->num_bits;
		p_rc_context->HRB_fullness -= target_bits;
		if(p_rc_context->non_ref_frame_flag)
			p_rc_context->acc_err_from_last_ref_frame += p_rc_context->HRB_fullness - tmp;
		else
			p_rc_context->acc_err_from_last_ref_frame = p_rc_context->HRB_fullness - tmp;		
	}	

	if (p_rc_context->iframe_flag)
	// if iframe, record qp, target bits, actual bits, etc,
	// to be used as reference for the next iframe
	{			
		p_rc_context->iframe_stat.HRB_fullness_after_iframe = p_rc_context->HRB_fullness;
		p_rc_context->total_p_num_bits = 0;
		p_rc_context->total_ref_p_num_bits = 0;
		p_rc_context->total_non_ref_num_bits = 0;
		p_rc_context->iframe_stat.target_bits = p_rc_context->target_bits;
		p_rc_context->iframe_stat.actual_bits = num_bits;		
        p_rc_context->sum_p_qp_in_gop = 0;		
		p_rc_context->scene_change = FALSE;
		if(!((p_rc_context->gop_reset_flag == CUVRC_SCENE_CHANGE_RESET_GOP) && (p_rc_context->start_of_gop_flag == FALSE)))
		{
			p_rc_context->curr_gop_frames_after_last_iframe = 0;
			p_rc_context->frames_after_last_iframe = 0;
		}
		else
		{
			// the code will enter here when I frame is forced at a B frame position using force_i_frame() 
			// API  & gop reset is allowed
			// in this special case, even though gop reset is allowed we can't reset the gop at this I frame since
			// we have already coded a future frame as reference P
			if(!p_rc_context->is_bframe_from_prev_gop)
				p_rc_context->curr_gop_frames_after_last_iframe++;
			p_rc_context->frames_after_last_iframe++;
		}
			
		if(p_rc_context->start_of_gop_flag == TRUE)
		{
			p_rc_context->current_gop_size = 1;
			p_rc_context->start_of_gop_flag = FALSE;
			p_rc_context->non_ref_p_count = 0;
		}
		else
		{
			if(!p_rc_context->is_bframe_from_prev_gop)
				p_rc_context->current_gop_size++;
			if(p_rc_context->non_ref_p_count == p_rc_context->non_ref_pframe_interval)
				p_rc_context->non_ref_p_count = 0;
			else
				p_rc_context->non_ref_p_count++;
		}
	}
	else
	{
		if (!p_rc_context->is_bframe_from_prev_gop) 
		{
			p_rc_context->current_gop_size++;
			if (!p_rc_context->use_prev_bits) 
			{
				p_rc_context->total_p_num_bits += num_bits;
				p_rc_context->curr_gop_frames_after_last_iframe++;
				if (p_rc_context->curr_frame_type == CUVRC_NON_REFERENCE_P_FRAME_FORWARD)
					p_rc_context->non_ref_p_count++;
				else
					p_rc_context->non_ref_p_count = 0;

				if (p_rc_context->non_ref_frame_flag)
					p_rc_context->total_non_ref_num_bits += num_bits;
				else
					p_rc_context->total_ref_p_num_bits += num_bits;
				p_rc_context->sum_p_qp_in_gop += p_rc_context->frame_qp;
			}
		}
		p_rc_context->frames_after_last_iframe++;			
        assert(p_rc_context->start_of_gop_flag == FALSE);		
	}	
}

void update_previous_frame_statistics(RC_STATE *rcs)
{
	int curr_qp = rcs->frame_qp;

	if (rcs->iframe_flag)
	// if iframe, record qp, target bits, actual bits, etc,
	// to be used as reference for the next iframe
	{
		rcs->iframe_stat.qp = rcs->frame_qp;
	}
	else
	{
		rcs->last_p_qp = rcs->frame_qp;		
	}

	if (!rcs->non_ref_frame_flag) 
	{
		rcs->last_ref_frame_stat.qp = rcs->frame_qp;
		rcs->last_ref_frame_stat.target_bits = rcs->target_bits;
		rcs->last_ref_frame_stat.actual_bits = rcs->num_bits;
		rcs->last_ref_frame_stat.frame_type = rcs->curr_frame_type;
		rcs->last_ref_frame_stat.iframe_flag = ((rcs->curr_frame_type == CUVRC_I_FRAME) ||
				(rcs->curr_frame_type == CUVRC_PSEUDO_I_FRAME)) && (rcs->frame_num != 0);
		
	}

	rcs->last_sad = rcs->avg_sad;
	rcs->last_iframe_flag = rcs->iframe_flag;
	rcs->last_qp = curr_qp;
	rcs->last_frame_type = rcs->curr_frame_type;
	rcs->last_non_ref_frame_flag = rcs->non_ref_frame_flag;
	
}


void bitrate_buffer_reset (RC_STATE *rcs)
{
    RC_BITRATE_STAT *bitrate_stat = &rcs->bitrate_stat;
    unsigned int i;

    bitrate_stat->buf_size = MAX_SIZE_NUM_BITS_BUFFER; // modify when : in the beginning when ifrate changes
    bitrate_stat->gop_size = MAX_SIZE_NUM_BITS_BUFFER; // modify when : in the beginning when ifrate changes
    bitrate_stat->cur_pos  = 0;                         // update when you get num bits
    for(i = 0; i < MAX_SIZE_NUM_BITS_BUFFER; i++)      
        bitrate_stat->bits_error_per_frame[i] = 0;             // update when you get num bits

    bitrate_stat->total_error = 0;                // update when you get num bits
    bitrate_stat->previous_accumulated_error = 0;        // update when you get num bits
}

// the equation used to get the values is delta_qp = log2(avg_var/25)
// from experiments 25 was found to be average variance , & number of bits 
// produced are comparable to the target bits
// also the ratio(avg_var/25) was found to ve very close to (bits_produced/target_bits) ratio
// since (avg_var/25) can be less than 1 we calculate (avg_var* 10/25) to get an
// index pointing into the following table
// we want to limit the qp offsets between -12 & +12 so we'll clip the indices to be in the range of
// 3-40

const int qp_offsets[41] = {-100, -20, -14, -10, -8, -6, -4, -3, 
		-2, -1, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9,
		9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12};


int set_qp(RC_STATE *rcs)
{
	int initial_qp;
	if ((rcs->need_full_reset) || rcs->scene_change) 
	{
		initial_qp = set_initial_qp(rcs);
		initial_qp = max(initial_qp, rcs->min_qp); // this will make sure that we give imp. to bitrate in CBR & qp in VBR

		rcs->original_qp = initial_qp;

		if (!rcs->force_qp_flag)
		{
			rcs->frame_qp = rcs->original_qp;
			rcs->is_initial_qp = 1;
		}
		// reset the difference in qp values of I & P frames
		// to be on safer side : keep the difference high if the qp value is small
		// this will ensure that you don't generate very high number of bits in case 
		// we start with a small qp value & there is high motion in the sequence
		rcs->i_p_frame_diff = IFRAMEDIFF_CBR + 1;
		
			if (rcs->frame_qp < 30)
				rcs->i_p_frame_diff += 2;
			if (rcs->frame_qp < 20)
				rcs->i_p_frame_diff += 2;		
	}
	return 0;
}

int set_initial_qp(RC_STATE *rcs)
{
	int initial_qp;
	switch(rcs->prime_format)
	{
	case CUVRC_H264:
	{
		 	int p0, p1, p2, p3, p4, bits_per_100_pix;
			int prev_gop_bit_error = 0;
			if (rcs->im_width <= 176) {p0 = 5, p1 = 10; p2 = 10; p3 = 30; p4 = 60;}
			else if (rcs->im_height <= 352) {p0 = 5, p1 = 10; p2 = 20; p3 = 60; p4 = 120;}
			else {p0 = 5, p1 = 20; p2 = 60; p3 = 140; p4 = 240;}

			if(rcs->HRB_fullness < -rcs->target_bps/5)
                // if error is negative, we would like to start at a lower qp
                // but to avoid a burst of bits, we choose the clipped version of error
				prev_gop_bit_error = rcs->prev_gop_bit_error / 2;
			else if (rcs->HRB_fullness > rcs->target_bps/5)
                // since a positive error is more devastating we choose the unclipped error
				prev_gop_bit_error = rcs->HRB_fullness/rcs->ifrate;
            else
                prev_gop_bit_error = 0;

			bits_per_100_pix = (rcs->target_bpf - prev_gop_bit_error) * 100 / (rcs->im_width * rcs->im_height);
			if (bits_per_100_pix <= p0)     { initial_qp = 45; }
			else if(bits_per_100_pix <= p1) {initial_qp = 39;}
			else if(bits_per_100_pix <= p1) {initial_qp = 35;}
			else if(bits_per_100_pix <= p2) {initial_qp = 25;}
			else if(bits_per_100_pix <= p3) {initial_qp = 20;}
			else {initial_qp =10;}
			break;
		}
		default:
		{
			printf("Error in codec type\n");
			break;
		}
	}	
	return initial_qp;
}

// m & c for equation y = mx + c where
// x = var & y = bits per macroblock
// the 7 values are for qps : 14, 20, 26, 32, 38, 44, 50
const float h264_slope[7] = {47.72, 38.08, 27.39, 18.07, 10.51, 5.37, 2.16};
const float h264_constant[7] = {182.28, 25.28, -35.89, -46.73, -34.48, -17.16, -3.08};
const int h264_qps[7] = {14, 20, 26, 32, 38, 44, 50};

const float mp4_slope[7] = {39.31, 20.79, 13.72, 10.52, 8.35, 6.47, 5.11};
const float mp4_constant[7] = {-75.02, -72.66, -49.62, -35.69, -25.23, -15.75, -8.45};
const int mp4_qps[7] = {3, 8, 13, 17, 21, 26, 31};

void modify_initial_qp(RC_STATE *rcs)
{
	int initial_qp =  rcs->original_qp;
	int prev_gop_bit_error;
	int target_bits_per_mb;
	int achieved_bits;
	int qp_low;
	int qp_high;
	int bits_low = 0, bits_high = 0;
	float qscale_low, qscale_high, target_qscale;
	int i = 3;
	float m, c;
	float *slope, *constant;
	int *qps;
		
	if (rcs->HRB_fullness <= 0)
		prev_gop_bit_error = rcs->prev_gop_bit_error * rcs->iframeratio / 3;
	else
		prev_gop_bit_error = rcs->prev_gop_bit_error * rcs->iframeratio;	
	
	
	
		slope = (float *)h264_slope;
		constant =  (float *)h264_constant;
		qps = (int *)h264_qps;
	
	qp_low = qps[0];
	qp_high = qps[6];		
	target_bits_per_mb = (rcs->iframe_bits - prev_gop_bit_error)/(rcs->width_in_mbs * rcs->height_in_mbs);
	assert(target_bits_per_mb > 0);
	
	// qp = 32 i.e. i = 3
	achieved_bits = slope[i] * rcs->avg_var + constant[i];
	
	while (1) 
	{
		assert((i >= 0) && (i < 7));
		if (abs(achieved_bits - target_bits_per_mb) > achieved_bits/20) 
		{
			if (target_bits_per_mb > achieved_bits) 
			{
				qp_high = qps[i];
				bits_high = achieved_bits;
				i--;
				if ((i < 0) || (bits_low != 0)) 
					break;
			} else 
			{
				qp_low = qps[i];
				bits_low = achieved_bits;
				i++;
				if ((i >= 7) || (bits_high != 0))
					break;
			}
		} else 
		{
			qp_low = qps[i];
			qp_high = qps[i];
			break;
		}
		achieved_bits = slope[i] * rcs->avg_var + constant[i];
	}
	
	if(qp_low == qp_high)
		initial_qp = qp_low;
	else
	{

			qscale_low = qscale[qp_low];
			qscale_high = qscale[qp_high];
			assert((qp_high - qp_low) == 6);
		
		m = (bits_low - bits_high) / (qscale_low - qscale_high);
		c = bits_low - m * qscale_low;
		target_qscale = (target_bits_per_mb - c)/m;
		assert((target_qscale >= qscale_low) && (target_qscale <= qscale_high));

		
			for (i = (qp_low + 1); i <= qp_high; i++) 
			{
				if ((target_qscale < qscale[i]) || ((target_qscale - qscale[i])	<= qscale[i]/20)) 
				{
					initial_qp = i;
					break;
				}
			}
	
	}
	if(rcs->frame_num != 0)
		initial_qp = max(initial_qp, rcs->last_p_qp - 8);
	
	initial_qp = max(initial_qp, rcs->min_qp); // this will make sure that we give imp. to bitrate in CBR & qp in VBR

	rcs->original_qp = initial_qp;
	if (!rcs->force_qp_flag)
		rcs->frame_qp = rcs->original_qp;

    // reset the difference in qp values of I & P frames
    // to be on safer side : keep the difference high if the qp value is small
    // this will ensure that you don't generate very high number of bits in case 
    // we start with a small qp value & there is high motion in the sequence
    rcs->i_p_frame_diff	 = IFRAMEDIFF_CBR + 1;

        if(rcs->frame_qp < 30)
            rcs->i_p_frame_diff += 2;
        if(rcs->frame_qp < 20)
            rcs->i_p_frame_diff += 2;
        if(rcs->frame_qp >= 40)
        	rcs->i_p_frame_diff -= 2;        	
  
	
}

void reset_rois(RC_STATE *rcs) {	
	rcs->roi_info.rois_present_flag	= FALSE;
	rcs->roi_info.extremely_good_quality_rois_count = 0;
	rcs->roi_info.good_quality_rois_count = 0;
	rcs->roi_info.poor_quality_rois_count = 0;
	rcs->roi_info.extremely_poor_quality_rois_count = 0;
}

void update_scene_change_error(RC_STATE *rcs)
{
	int scene_change_error;
	if(rcs->gop_reset_flag == CUVRC_SCENE_CHANGE_RESET_GOP)
	{
		int partial_error;
		int curr_gop_frames_after_last_iframe = rcs->curr_gop_frames_after_last_iframe + rcs->prev_gop_bframes_count;
		int ref_pframes_after_last_iframe = curr_gop_frames_after_last_iframe / (rcs->non_ref_frame_interval + 1);
		int non_ref_frames_after_last_iframe = curr_gop_frames_after_last_iframe - ref_pframes_after_last_iframe;
				
		scene_change_error = rcs->prev.iframe_bits 
				+ (ref_pframes_after_last_iframe * rcs->prev.pframe_bits) 
				+ (non_ref_frames_after_last_iframe * rcs->prev.non_ref_frame_bits) - 
				(rcs->original_values.target_bps * (curr_gop_frames_after_last_iframe + 1) / rcs->target_fps);
		// distribute the scene_change_error in next two gops
		rcs->scene_change_error += scene_change_error;
		compare_errors(rcs);
		partial_error = rcs->scene_change_error/2;
		rcs->current_gop_scene_change_error += partial_error;
		rcs->scene_change_error -= partial_error;		
		update_previous_gop_error(rcs);
	}
	else
	{
		if (rcs->start_of_gop_flag == FALSE) 
		{
			int current_gop_size = rcs->current_gop_size + rcs->prev_gop_bframes_count;
			int ref_pframes_till_now = (current_gop_size - 1) / (rcs->non_ref_frame_interval + 1);
			int non_ref_frames_till_now = current_gop_size - 1 - ref_pframes_till_now;
			scene_change_error = rcs->prev.iframe_bits
					+ (ref_pframes_till_now * rcs->prev.pframe_bits)
					+ (non_ref_frames_till_now * rcs->prev.non_ref_frame_bits)
					- (ref_pframes_till_now + 1) * rcs->original_values.pframe_bits
					- non_ref_frames_till_now * rcs->original_values.non_ref_frame_bits;
					
			rcs->scene_change_error += scene_change_error;
		}
	}
}

void update_forced_iframe_error(RC_STATE *rcs)
{
	int scene_change_error = 0;
	if((rcs->gop_reset_flag == CUVRC_SCENE_CHANGE_RESET_GOP) && (rcs->start_of_gop_flag == TRUE)) 
	{
		if (rcs->frame_num != 0) {
			int curr_gop_frames_after_last_iframe = rcs->curr_gop_frames_after_last_iframe + rcs->prev_gop_bframes_count;
			int ref_pframes_after_last_iframe = curr_gop_frames_after_last_iframe / (rcs->non_ref_frame_interval + 1);
			int non_ref_frames_after_last_iframe = curr_gop_frames_after_last_iframe - ref_pframes_after_last_iframe;
					
			scene_change_error = rcs->original_values.iframe_bits 
								+ ref_pframes_after_last_iframe * rcs->original_values.pframe_bits 
								+ non_ref_frames_after_last_iframe * rcs->original_values.non_ref_frame_bits
								- (rcs->original_values.target_bps * (curr_gop_frames_after_last_iframe + 1) / rcs->target_fps);

			// distribute the scene_change_error in next two gops
			rcs->scene_change_error += scene_change_error;						
		}
	}
	else
	{
		if (rcs->start_of_gop_flag == FALSE) 
		{
			if((rcs->non_ref_p_count != rcs->non_ref_pframe_interval) || (rcs->bframe_count != rcs->bframe_interval))
				scene_change_error = rcs->original_values.iframe_bits - rcs->original_values.non_ref_frame_bits;
			else
				scene_change_error = rcs->original_values.iframe_bits - rcs->original_values.pframe_bits;
			if(rcs->use_prev_bits)
				scene_change_error = rcs->original_values.iframe_bits - rcs->prev.non_ref_frame_bits;				
			rcs->scene_change_error += scene_change_error;
		}
	}
}

void dstribute_scene_change_error(RC_STATE *rcs)
{
	DISTRIBUTE_ERROR_FSM_PARAMS *fsm = &rcs->scene_change_fsm;
	fsm->state_status = CONTINUE;
	
	   do {
		switch (fsm->current_state) {
		case GOP_WITH_NO_SCENE_CHANGE_ERROR:
			if (rcs->scene_change_error != 0) {
				fsm->next_state = GOP_WITH_SCENE_CHANGE;
			} else {				
				fsm->next_state = GOP_WITH_NO_SCENE_CHANGE_ERROR;
				fsm->state_status = EXIT;
			}
			fsm->current_state = fsm->next_state;
			break;
		case GOP_WITH_SCENE_CHANGE:
		{
			int partial_error;
			compare_errors(rcs);			
			partial_error = rcs->scene_change_error/3;
			rcs->current_gop_scene_change_error += partial_error;
			rcs->scene_change_error -= partial_error;			
			update_previous_gop_error(rcs);			
			fsm->next_state = FIRST_GOP_AFTER_SCENE_CHANGE;
			fsm->state_status = EXIT;
			fsm->current_state = fsm->next_state;
			break;
		}
		case FIRST_GOP_AFTER_SCENE_CHANGE:
			if (rcs->scene_change) {
				fsm->next_state = GOP_WITH_SCENE_CHANGE;
			} else {
				int partial_error;
				compare_errors(rcs);
				partial_error = rcs->scene_change_error/2;
				rcs->current_gop_scene_change_error += partial_error;
				rcs->scene_change_error -= partial_error;
				update_previous_gop_error(rcs);
				fsm->next_state = SECOND_GOP_AFTER_SCENE_CHANGE;
				fsm->state_status = EXIT;
			}
			fsm->current_state = fsm->next_state;
			break;
		case SECOND_GOP_AFTER_SCENE_CHANGE:
		{
			if (rcs->scene_change) {
				fsm->next_state = GOP_WITH_SCENE_CHANGE;
			} else {
				compare_errors(rcs);
				rcs->current_gop_scene_change_error += rcs->scene_change_error;
				rcs->scene_change_error = 0;
				update_previous_gop_error(rcs);
				fsm->next_state = GOP_WITH_NO_SCENE_CHANGE_ERROR;
				fsm->state_status = EXIT;
			}
			fsm->current_state = fsm->next_state;
			break;
		}
		}
	} while (fsm->state_status != EXIT);	   
}

void calc_iframe_pframe_bits(RC_STATE *rcs, int non_ref_frame_bits)
{
	float scale;
	float if_persec = rcs->if_persec;
	int iframe_bits, pframe_bits;
	pframe_bits = (int)((float)non_ref_frame_bits * rcs->p_2_non_ref_p_ratio);
	iframe_bits = (rcs->target_bps - (rcs->ref_p_frames_per_sec * pframe_bits) - ((rcs->target_fps - if_persec - rcs->ref_p_frames_per_sec) * non_ref_frame_bits))/if_persec;
	if ((iframe_bits > 2 * pframe_bits * rcs->iframeratio) || (iframe_bits < 0))
		// if if_persec value is too small, iframe_bits will blow up due to the division
		iframe_bits = pframe_bits * rcs->iframeratio;
	
	
	//rcs->frame_qp = avg_p_qp; 
	 rcs->pframe_bits = pframe_bits;
	 rcs->iframe_bits = iframe_bits;
	 rcs->non_ref_frame_bits = non_ref_frame_bits;
   
    if((rcs->fsm.current_state == HIGH_MOTION) || (rcs->fsm.current_state == HIGH_TO_NORMAL_MOTION_WAIT))
    {
    	scale = rcs->abr_params.higher_br_scale;
    }
    else if(rcs->fsm.current_state == LOWER_BITRATE_AFTER_HIGH_MOTION) 
    {
    	scale = rcs->abr_params.lower_br_scale;    	
    }
    else
    {
    	scale = NORMAL_SCALE;
    }
    	
    non_ref_frame_bits = (float)non_ref_frame_bits/scale;
    pframe_bits = (float)pframe_bits/scale;
    iframe_bits = (rcs->original_values.target_bps - (rcs->ref_p_frames_per_sec * pframe_bits) - ((rcs->target_fps - if_persec - rcs->ref_p_frames_per_sec) * non_ref_frame_bits))/if_persec;
	if (iframe_bits > 2 * (float)rcs->iframe_bits/scale)
		// if if_persec value is too small, iframe_bits will blow up due to the division
		iframe_bits = (float)rcs->iframe_bits/scale;
    
    rcs->original_values.iframe_bits = iframe_bits;    
    rcs->original_values.pframe_bits = pframe_bits;
    rcs->original_values.non_ref_frame_bits = non_ref_frame_bits;

}   

void update_previous_gop_error(RC_STATE *rcs)
{
	int  prev_gop_error;
	int remaining_fpsplus = 1, actual_fpsplus = 1, clipped_remaining_fpsplus = 1;
	actual_fpsplus = rcs->iframeratio + rcs->ifrate - 1;
	
	if(rcs->start_of_gop_flag == FALSE)
	{		
		remaining_fpsplus = rcs->iframeratio + rcs->ifrate - rcs->current_gop_size - 1;
		clipped_remaining_fpsplus = rcs->iframeratio + rcs->ifrate 
							- min((rcs->current_gop_size - 1), 2 * rcs->ifrate/3);
	}

	if(rcs->start_of_gop_flag == TRUE)
		prev_gop_error = (rcs->HRB_fullness + rcs->current_gop_scene_change_error)/actual_fpsplus;
	else
		prev_gop_error = (((rcs->HRB_fullness * remaining_fpsplus)/actual_fpsplus) + rcs->current_gop_scene_change_error)/clipped_remaining_fpsplus;

	rcs->prev_gop_bit_error = prev_gop_error;

}

void set_abr_limits(ABR_CONTEXT *abr_params)
{
	abr_params->lower_br_scale = 1/(1 + 0.1 * abr_params->scale);
	abr_params->higher_br_scale = 1 + 0.5 * abr_params->scale;
}


void compare_errors(RC_STATE *rcs)
{
	int partial_error;
	if((rcs->HRB_fullness <= 0) && (rcs->scene_change_error > 0))
	{
		partial_error = min(-rcs->HRB_fullness, rcs->scene_change_error);
		rcs->scene_change_error = rcs->scene_change_error - partial_error;
		rcs->HRB_fullness = rcs->HRB_fullness + partial_error;
	}		
}



void modify_frame_types(RC_STATE *rcs)
{
	FRAME_TYPE_FSM_PARAMS *fsm = &rcs->frame_type_fsm;
	fsm->state_status = CONTINUE;
   do {
		switch (fsm->current_state) 
		{
		case I_FRAME:
			if((rcs->iframe_period_type == CUVRC_PERIOD_MILLISECONDS) &&
					(rcs->curr_frame_type == CUVRC_P_FRAME))
			{
				// this will happen if, when calculating the next_frame_type, we assumed
				// the timestamp of next frame depending on the frame rate,
				// but the actual timestamp turns out to be less than the assumed one &
				// hence the frame type will be a P frame
				fsm->next_state = P_FRAME;
				fsm->state_status = CONTINUE;
			}
			assert(rcs->curr_frame_type == CUVRC_I_FRAME);
			if(fsm->is_prev_ref_frame)
				rcs->prev_gop_bframes_count = rcs->next_frame_offset - 1;
			else
				rcs->prev_gop_bframes_count = rcs->next_frame_offset - 2;
			if(rcs->next_frame_type == CUVRC_I_FRAME)
			{
				if(rcs->prev_gop_bframes_count > 0)
				{
					rcs->next_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
					rcs->next_frame_offset = -rcs->prev_gop_bframes_count;
					fsm->next_state = PREVIOUS_GOP_B_FRAME;
					fsm->state_status = EXIT;
				}
				else
				{
					assert(rcs->prev_gop_bframes_count == 0);
					rcs->next_frame_type = CUVRC_I_FRAME;
					rcs->next_frame_offset = rcs->num_prev_gop_bframes + 1;
					fsm->is_prev_ref_frame = TRUE;
					fsm->next_state = I_FRAME;
					fsm->state_status = EXIT;
				}								
			}
			else
			{				
				if(rcs->prev_gop_bframes_count > 0)
				{
					rcs->next_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
					rcs->next_frame_offset = -rcs->prev_gop_bframes_count;
					fsm->next_state = PREVIOUS_GOP_B_FRAME;
					fsm->state_status = EXIT;
				}
				else
				{
					assert(rcs->prev_gop_bframes_count == 0);
					rcs->next_frame_type = CUVRC_P_FRAME;
					rcs->next_frame_offset = rcs->bframe_interval + 1;
					fsm->is_prev_ref_frame = TRUE;
					fsm->next_state = P_FRAME;
					fsm->state_status = EXIT;
				}				
			}
			rcs->use_prev_bits = FALSE;
			fsm->current_state = fsm->next_state;
			break;
		case PREVIOUS_GOP_B_FRAME:
		{			
			rcs->curr_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
			rcs->prev_gop_bframes_count--;
			if(rcs->prev_gop_bframes_count > 0)
			{
				rcs->next_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
				rcs->next_frame_offset = 1;
				fsm->next_state = PREVIOUS_GOP_B_FRAME;
				fsm->state_status = EXIT;
			}
			else
			{
				if(rcs->next_frame_type == CUVRC_I_FRAME)
				{
					rcs->next_frame_offset = rcs->num_prev_gop_bframes + 2;
					fsm->is_prev_ref_frame = FALSE; 
					fsm->next_state = I_FRAME;
					fsm->state_status = EXIT;					
				}
				else
				{
					rcs->next_frame_type = CUVRC_P_FRAME;
					rcs->next_frame_offset = rcs->bframe_interval + 2;
					fsm->is_prev_ref_frame = FALSE;
					fsm->next_state = P_FRAME;
					fsm->state_status = EXIT;
				}
			}									
			fsm->current_state = fsm->next_state;
			rcs->use_prev_bits = TRUE;
			break;
		}
		case P_FRAME:
			if(rcs->curr_frame_type == CUVRC_I_FRAME)
			{
				fsm->next_state = I_FRAME;
				fsm->state_status = CONTINUE;
			}
			else if(rcs->curr_frame_type == CUVRC_PSEUDO_I_FRAME)
			{
				fsm->next_state = PSEUDO_I_FRAME;
				fsm->state_status = CONTINUE;
			}
			else
			{
				rcs->curr_frame_type = CUVRC_P_FRAME;
				if(rcs->next_frame_type == CUVRC_I_FRAME)
				{
					rcs->next_frame_offset = rcs->num_prev_gop_bframes + 1;
					fsm->is_prev_ref_frame = TRUE;
					fsm->next_state = I_FRAME;
					fsm->state_status = EXIT;
				}
				else if(rcs->bframe_interval == 0)
				{
					rcs->next_frame_type = CUVRC_P_FRAME;
					rcs->next_frame_offset = 1;
					rcs->bframe_count = 0;
					fsm->is_prev_ref_frame = TRUE;
					fsm->next_state = P_FRAME;
					fsm->state_status = EXIT;					
				}
				else
				{
					rcs->next_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
					if(fsm->is_prev_ref_frame)
						rcs->next_frame_offset = -rcs->next_frame_offset + 1;
					else
						rcs->next_frame_offset = -rcs->next_frame_offset + 2;
					rcs->bframe_count = -rcs->next_frame_offset;
					fsm->next_state = B_FRAME;
					fsm->state_status = EXIT;
				}
			}
			fsm->current_state = fsm->next_state;
			rcs->use_prev_bits = FALSE;
			break;
		case B_FRAME:
		{
			assert(rcs->curr_frame_type == CUVRC_P_FRAME);
			rcs->curr_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
			rcs->bframe_count--;
			if(rcs->bframe_count > 0)
			{
				rcs->next_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
				rcs->next_frame_offset = 1;
				fsm->next_state = B_FRAME;
				fsm->state_status = EXIT;
			}
			else
			{
				if(rcs->next_frame_type == CUVRC_I_FRAME)
				{
					rcs->next_frame_offset = rcs->num_prev_gop_bframes + 2;
					fsm->is_prev_ref_frame = FALSE;
					fsm->next_state = I_FRAME;
					fsm->state_status = EXIT;
				}
				else
				{
					rcs->next_frame_offset = rcs->bframe_interval + 2;
					fsm->is_prev_ref_frame = FALSE;
					fsm->next_state = P_FRAME;
					fsm->state_status = EXIT;
				}
			}
			fsm->current_state = fsm->next_state;
			rcs->use_prev_bits = FALSE;
			break;
		}
		case PSEUDO_I_FRAME:
		{
			assert(rcs->curr_frame_type == CUVRC_PSEUDO_I_FRAME);
			assert(rcs->next_frame_type != CUVRC_I_FRAME);
			if(rcs->bframe_interval == 0)
			{
				rcs->next_frame_type = CUVRC_P_FRAME;
				rcs->next_frame_offset = 1;
				rcs->bframe_count = 0;
				rcs->prev_gop_bframes_count = 0;
				fsm->is_prev_ref_frame = TRUE;
				fsm->next_state = P_FRAME;				
			}
			else
			{
				if (fsm->is_prev_ref_frame)
					rcs->prev_gop_bframes_count = rcs->next_frame_offset - 1;
				else
					rcs->prev_gop_bframes_count = rcs->next_frame_offset - 2;
				assert((rcs->prev_gop_bframes_count > 0));
				rcs->next_frame_type = CUVRC_NON_REFERENCE_B_FRAME_FORWARD;
				rcs->next_frame_offset = -rcs->prev_gop_bframes_count;
				fsm->next_state = PREVIOUS_GOP_B_FRAME;
			}
			fsm->state_status = EXIT;			
			fsm->current_state = fsm->next_state;
			rcs->use_prev_bits = FALSE;
			break;
		}
		}
   }
   while (fsm->state_status != EXIT);
   rcs->is_bframe_from_prev_gop = ((rcs->current_gop_size == 1) & (rcs->curr_frame_type == CUVRC_NON_REFERENCE_B_FRAME_FORWARD));
   if(rcs->is_backward_prediction)
   {
	   assert(rcs->curr_frame_type == CUVRC_NON_REFERENCE_B_FRAME_FORWARD);
	   rcs->curr_frame_type = CUVRC_NON_REFERENCE_B_FRAME_BACKWARD;
   }
}

void modify_input_data(RC_STATE *rcs)
{
	rcs->reset_postponed = FALSE;
	if(rcs->prime_format != rcs->input_config.prime_format)
	{
		rcs->prime_format != rcs->input_config.prime_format;
		
	}
	
	if((rcs->im_width != rcs->input_config.im_width) ||
			(rcs->im_height != rcs->input_config.im_height))
	{
		cuvrc_free(rcs);;
		rcs->im_width = rcs->input_config.im_width;
		rcs->im_height = rcs->input_config.im_height;
		rcs->width_in_mbs = rcs->im_width / MB_WIDTH;
		rcs->height_in_mbs = rcs->im_height / MB_HEIGHT;
		cuvrc_init(rcs);
	}
	
	rcs->target_bitrate = rcs->input_config.target_bitrate;
	rcs->bitrate_type = rcs->input_config.bitrate_type;
	rcs->bitrate_mode = rcs->input_config.bitrate_mode;
	rcs->quality_mode = rcs->input_config.quality_mode;
	if(rcs->quality_mode == CUVRC_CONSTANT_QUALITY)
	{
		rcs->force_qp_flag = 1;
		rcs->frame_qp = rcs->min_qp;
	}
    else
        rcs->force_qp_flag = 0;
	rcs->iframe_period = rcs->input_config.iframe_period;
	rcs->iframe_period_type = rcs->input_config.iframe_period_type;
	rcs->non_ref_pframe_interval = rcs->input_config.non_ref_pframe_interval;
	rcs->bframe_interval = rcs->input_config.bframe_interval;
	rcs->target_fps = rcs->input_config.target_fps;
	if(rcs->non_ref_pframe_interval != 0)
		rcs->non_ref_frame_interval = rcs->non_ref_pframe_interval;
    else if(rcs->bframe_interval != 0)
		rcs->non_ref_frame_interval = rcs->bframe_interval;
    else
        rcs->non_ref_frame_interval = 0;
		
	rcs->need_reset = 1;
	rcs->need_full_reset = 1;
}

int calc_next_frame_type(unsigned long long *frame_timestamp,
						 int read_pos,
						 int add_one_frame,
						 int inter_frame_interval_ms,
						 unsigned long long iframe_time,
						 int iframe_period)
{
	CUVRC_FRAME_TYPE next_frame_type;
	if ((int)((frame_timestamp[read_pos] + add_one_frame * inter_frame_interval_ms * 1000) - iframe_time)
				> (int)((iframe_period - inter_frame_interval_ms/2) * 1000))
		next_frame_type = CUVRC_I_FRAME;
	else
		next_frame_type = CUVRC_P_FRAME;
	return next_frame_type;	
}
void populate_mb_qps_array(RC_STATE *rcs)		
{
	int i, j;
	int width = rcs->width_in_mbs;
	int height = rcs->height_in_mbs;
	int frame_qp = rcs->frame_qp;
	int max_qp_offset = 25;
	int *p_mb_qp = (int *)rcs->p_mb_qp;
	CUVME_MB_CHARAC *p_mb_characs = rcs->p_mb_characs;	
	int *p_qp_list = rcs->p_qp_list;
	int mb_qp, mean_of_mbvar = 0, var_of_mbvar = 0, delta_qp_1, delta_qp_2, qp_0, qp_1, qp_2, qp_3, th0, th1, th2;
	int num_roi_quality_levels = (rcs->roi_info.extremely_good_quality_rois_count != 0)
					+ (rcs->roi_info.good_quality_rois_count != 0)
					+ (rcs->roi_info.poor_quality_rois_count != 0)
					+ (rcs->roi_info.extremely_poor_quality_rois_count != 0);
	// find mean of variance
	for (i = 0; i < width * height; i++) {
		mean_of_mbvar += p_mb_characs[i].variance;
	}
	
	mean_of_mbvar = mean_of_mbvar/ (width * height);	
	
	for (i = 0; i < width * height; i++) {
		var_of_mbvar += abs(p_mb_characs[i].variance - mean_of_mbvar);
	}
	var_of_mbvar = var_of_mbvar/ (width * height);	

	// find variance of variance
	// find delta_qp_2
	delta_qp_2 = (var_of_mbvar + 2)/3;
	delta_qp_2 = clip_3(delta_qp_2, 6, 0);
	// find delta_qp_1
	delta_qp_1 = (delta_qp_2 + 1) / 2;
	// find qps
	qp_0 = rcs->frame_qp -  3 * delta_qp_2/2;
	qp_0 = clip_3(qp_0, rcs->max_qp, rcs->min_qp);
	qp_1 = rcs->frame_qp - 3 * delta_qp_1/2;
	qp_1 = clip_3(qp_1, rcs->max_qp, rcs->min_qp);
	qp_2 = rcs->frame_qp + delta_qp_1;
	qp_2 = clip_3(qp_2, rcs->max_qp, rcs->min_qp);
	qp_3 = rcs->frame_qp + delta_qp_2;
	qp_3 = clip_3(qp_3, rcs->max_qp, rcs->min_qp);
	
	if (num_roi_quality_levels > 2) {
			qp_0 = rcs->frame_qp;
			qp_1 = rcs->frame_qp;
			qp_3 = rcs->frame_qp;
			qp_2 = rcs->frame_qp;
			*(p_qp_list++) = 1;
			*(p_qp_list++) = qp_0;			
	}	
	else if (num_roi_quality_levels > 0) {
		qp_0 = qp_1;
		qp_3 = qp_2;
		*(p_qp_list++) = 2;
		*(p_qp_list++) = qp_0;
		*(p_qp_list++) = qp_2;
	} else {
		// write the qp values to the qp list
		*(p_qp_list++) = 4;
		*(p_qp_list++) = qp_0;
		*(p_qp_list++) = qp_1;
		*(p_qp_list++) = qp_2;
		*(p_qp_list++) = qp_3;
	}	
	
	// get threshholds
	th0 = mean_of_mbvar - var_of_mbvar;
	th0 = clip_3(th0, 255, 0);
	th1 = mean_of_mbvar;
	th1 = clip_3(th1, 255, 0);
	th2 = mean_of_mbvar + var_of_mbvar;
	th2 = clip_3(th2, 255, 0);		
	//int *p_mb_qp = rcs->p_mb_qp;
	for (i = 0; i < width * height; i++) {
		int variance = p_mb_characs[i].variance;

		if (variance <= th0)
			mb_qp = qp_0;
		else if (variance <= th1)
			mb_qp = qp_1;
		else if (variance <= th2)
			mb_qp = qp_2;
		else
			mb_qp = qp_3;
		//*(p_mb_qp++) = mb_qp;
		for (j = 0; j < NUM_BLKS_IN_MB; j++)
			*(p_mb_qp++) = mb_qp;
	}
	
	// update the min & max frame qp
	rcs->min_frame_qp = qp_0;
	rcs->max_frame_qp = qp_3;
}		


void copy_frame_qp_to_mb_qp(RC_STATE *rcs) {
	int i;
	int width = rcs->width_in_mbs;
	int height = rcs->height_in_mbs;
	int frame_qp = rcs->frame_qp;
	int *p_mb_qp = (int *)rcs->p_mb_qp;
	for(i = 0; i < width * height * NUM_BLKS_IN_MB; i++)
		*(p_mb_qp++) = frame_qp;
	rcs->p_qp_list[0] = 1;
	rcs->p_qp_list[1] = rcs->frame_qp;
}