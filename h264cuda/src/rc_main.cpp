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

#include <stdio.h>
#include <assert.h>
#include "../inc/rc_common.h"
#include "../inc/rc_mem_alloc_free.h"
#include "../inc/rc_internal_settings.h"
#include "../inc/rc_process_frame.h"

CUVRC_ERROR cuvrc_open (CUVRC_HANDLE *rc_handle, CUVRC_APP_TYPE app_type)
{
    RC_STATE *p_rc_context;
    CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
    if((rc_handle == NULL) || (app_type < CUVRC_NONE))
        return CUVRC_ERR_ARG;
    p_rc_context = (RC_STATE *)getmem_1d_void(sizeof(RC_STATE));
    if(p_rc_context == NULL)
        return CUVRC_ERR_MEM;
    rc_reset_fr_computer (p_rc_context);
    bitrate_buffer_reset(p_rc_context);
    set_defaults(p_rc_context);
    p_rc_context->prime_format = app_type;
    p_rc_context->input_config.prime_format = app_type;
    //init_perf_timers(p_rc_context);
    *rc_handle = (void *)p_rc_context;  
    return err;
}




CUVRC_ERROR cuvrc_init(CUVRC_HANDLE rc_handle)
{
    RC_STATE *p_rc_context;
    CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    p_rc_context = (RC_STATE *)rc_handle;
    if((err = rc_allocate_memory(p_rc_context)) != CUVRC_ERR_SUCCESS)
            return err;    
    return err;
}

CUVRC_ERROR cuvrc_free (CUVRC_HANDLE rc_handle)
{
    RC_STATE *p_rc_context;
    CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    p_rc_context = (RC_STATE *)rc_handle;
    rc_free_memory(p_rc_context);

    return err;
}

CUVRC_ERROR cuvrc_close (CUVRC_HANDLE rc_handle)
{
    CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    else
        free_1D(rc_handle);
    return err;
}


CUVRC_ERROR cuvrc_configure_dimensions(CUVRC_HANDLE rc_handle, int width, int height, CUVRC_FRAME_FORMAT fmt)
{
    RC_STATE *p_rc_context;
    CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
    if((rc_handle == NULL) || (width <= 0) || (height <= 0))
        return CUVRC_ERR_ARG;
    if((width%MB_WIDTH) || (height%MB_HEIGHT))
        return CUVRC_ERR_ARG;    
    p_rc_context = (RC_STATE *)rc_handle;
    //free and reinit if height and width are not same
    if((width != p_rc_context->im_width) || (height != p_rc_context->im_height))
    {
    	p_rc_context->input_config.im_width = width;
    	p_rc_context->input_config.im_height = height;
    	if(p_rc_context->next_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD)
		{
			cuvrc_free(rc_handle);
			p_rc_context->im_width = width;
			p_rc_context->im_height = height;
			p_rc_context->width_in_mbs = width / MB_WIDTH;
			p_rc_context->height_in_mbs = height / MB_HEIGHT;
			cuvrc_init(rc_handle);
			p_rc_context->need_reset = 1;
			p_rc_context->need_full_reset = 1;
		}
		else
			p_rc_context->reset_postponed = TRUE;    		
    }    
        
    return err;
	
}


CUVRC_ERROR cuvrc_set_previous_frame_actual_bits(CUVRC_HANDLE rc_handle, int num_bits)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    if (num_bits <= 0)
		return CUVRC_ERR_ARG;

	update_buffer(p_rc_context, num_bits);
	return err;
}

CUVRC_ERROR cuvrc_set_previous_frame_intra_mbs(CUVRC_HANDLE rc_handle, int num_intra_mbs)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    if(num_intra_mbs < 0)
            return CUVRC_ERR_ARG;
    p_rc_context->num_imbs = num_intra_mbs;
    
    return err;
}


CUVRC_ERROR cuvrc_set_previous_frame_skipped_mbs(CUVRC_HANDLE rc_handle, int num_skipped_mbs)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    if(num_skipped_mbs < 0)
            return CUVRC_ERR_ARG;
    p_rc_context->num_skipped_mbs = num_skipped_mbs;
    
    return err;
}


CUVRC_ERROR cuvrc_set_previous_frame_num_large_mvs(CUVRC_HANDLE rc_handle, int num_large_mvs)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    if(num_large_mvs < 0)
            return CUVRC_ERR_ARG;
    p_rc_context->num_big_mvs = num_large_mvs;
    
    return err;
}


CUVRC_ERROR cuvrc_set_previous_frame_avg_sad(CUVRC_HANDLE rc_handle, int avg_sad)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *rcs = (RC_STATE *)rc_handle;
		
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    if(avg_sad < 0)
            return CUVRC_ERR_ARG;
    
    rcs->avg_sad = avg_sad;       
    return err;
}


CUVRC_ERROR cuvrc_configure_bitrate(CUVRC_HANDLE rc_handle, int target_bitrate, CUVRC_BITRATE_TYPE bitrate_type, CUVRC_BITRATE_MODE mode)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    if((target_bitrate <= 0) ||  (bitrate_type < CUVRC_BITRATE_BPS) || (bitrate_type > CUVRC_BITRATE_BPF)) 
            return CUVRC_ERR_ARG;
    if((mode < CUVRC_NORMAL_BITRATE) || (mode > CUVRC_PREVIOUS_BITRATE))
        	return CUVRC_ERR_ARG;
    
    p_rc_context->input_config.target_bitrate = target_bitrate;
    p_rc_context->input_config.bitrate_type = bitrate_type;
    if (mode != CUVRC_PREVIOUS_BITRATE)
    	p_rc_context->input_config.bitrate_mode = mode;
    
    if(p_rc_context->next_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD)
    {
    	p_rc_context->target_bitrate = target_bitrate;
    	p_rc_context->bitrate_type = bitrate_type;
    	if (mode != CUVRC_PREVIOUS_BITRATE)
    		p_rc_context->bitrate_mode = mode;
    	p_rc_context->need_reset	= 1;
    	p_rc_context->need_full_reset	= 1;
    }
    else
    	p_rc_context->reset_postponed = TRUE;
        
    return err;
	
}



CUVRC_ERROR cuvrc_configure_iframe_period(CUVRC_HANDLE rc_handle, int iframe_period, CUVRC_IFRAME_PERIOD_TYPE iframe_period_type)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    if((iframe_period < 0) ||  (iframe_period_type < CUVRC_PERIOD_MILLISECONDS) || (iframe_period_type > CUVRC_PERIOD_FRAMES)) 
            return CUVRC_ERR_ARG;
    if(iframe_period == 0)
    	iframe_period = RC_LARGE_NUMBER; 

    // to avoid overflow we'll clip the max iframe period to 2^20
    iframe_period = clip_3(iframe_period, 0x100000, 0);
    p_rc_context->input_config.iframe_period = iframe_period;
    p_rc_context->input_config.iframe_period_type = iframe_period_type;
    if(p_rc_context->next_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD)
    {
		p_rc_context->iframe_period = iframe_period;
		p_rc_context->iframe_period_type = iframe_period_type;
		p_rc_context->need_reset = 1;
		p_rc_context->need_full_reset = 1;
	}
    else
    	p_rc_context->reset_postponed = TRUE;
    return err;
}

CUVRC_ERROR cuvrc_set_mb_level_pointers(CUVRC_HANDLE rc_handle, int *p_mb_qp, CUVME_MB_CHARAC *p_mb_characs, int *p_qp_list)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
	if((rc_handle == NULL) || (p_mb_qp == NULL) || (p_mb_characs == NULL) || (p_qp_list == NULL))
		return CUVRC_ERR_ARG;
	
	p_rc_context->p_mb_qp =  p_mb_qp;
	p_rc_context->p_mb_characs = p_mb_characs;
	p_rc_context->p_qp_list = p_qp_list;
	 
    return err;
}


CUVRC_ERROR cuvrc_set_frame_rate(CUVRC_HANDLE rc_handle, int frame_rate)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
	if((rc_handle == NULL) || (frame_rate <= 0))
		return CUVRC_ERR_ARG;
	
	if(p_rc_context->target_fps != frame_rate)
	{
		p_rc_context->input_config.target_fps = frame_rate;
		if(p_rc_context->next_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD)
		{
			p_rc_context->target_fps = frame_rate;
			p_rc_context->need_reset	= 1;
			p_rc_context->need_full_reset	= 1;
		}
		else
			p_rc_context->reset_postponed = TRUE;			
	}	 
    return err;
}


CUVRC_ERROR cuvrc_set_roi(CUVRC_HANDLE rc_handle, CUVRC_ROI roi) {
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
	ROI_CONTEXT *p_roi_info = &(p_rc_context->roi_info);
	CUVRC_ROI *p_roi;
	if ((rc_handle == NULL) || 
			(roi.x_offset < 0) || (roi.x_offset > p_rc_context->width_in_mbs) ||
			(roi.y_offset < 0) || (roi.y_offset > p_rc_context->height_in_mbs) ||
			(roi.width <= 0) || (roi.height <= 0) ||
			((roi.x_offset + roi.width) > p_rc_context->width_in_mbs) ||
			((roi.y_offset + roi.height) > p_rc_context->height_in_mbs))
		return CUVRC_ERR_ARG;
	
	p_roi_info->rois_present_flag = TRUE;
    switch(roi.quality_level)
	{
	case CUVRC_EXTREMELY_GOOD_QUALITY:
		p_roi = (p_roi_info->extremely_good_quality_rois + p_roi_info->extremely_good_quality_rois_count++);				
		break;
	case CUVRC_GOOD_QUALITY:
		p_roi = (p_roi_info->good_quality_rois + p_roi_info->good_quality_rois_count++);
		break;
	case CUVRC_POOR_QUALITY:
		p_roi = (p_roi_info->poor_quality_rois + p_roi_info->poor_quality_rois_count++);
		break;
	case CUVRC_EXTREMELY_POOR_QUALITY:
		p_roi = (p_roi_info->extremely_poor_quality_rois + p_roi_info->extremely_poor_quality_rois_count++);
		break;
	default:
		return CUVRC_ERR_ARG;
	}
	
	p_roi->x_offset = roi.x_offset;
	p_roi->y_offset = roi.y_offset;
	p_roi->width = roi.width;
	p_roi->height = roi.height;
	p_roi->quality_level = roi.quality_level;
	return err;
}

CUVRC_ERROR cuvrc_set_avg_var(CUVRC_HANDLE rc_handle, int avg_var)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *rcs = (RC_STATE *)rc_handle;

	if (rc_handle == NULL)
		return CUVRC_ERR_ARG;
	if ((avg_var < 0) || (avg_var > 255))
		return CUVRC_ERR_ARG;

	rcs->avg_var = avg_var;
	rcs->valid_avg_var = TRUE;
	return err;
}

CUVRC_ERROR cuvrc_get_current_frame_type(CUVRC_HANDLE rc_handle, CUVRC_FRAME_TYPE *type)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    
    *type = p_rc_context->curr_frame_type;    
    
    return err;
}

CUVRC_ERROR cuvrc_get_next_frame_type(CUVRC_HANDLE rc_handle, CUVRC_FRAME_TYPE *type)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
    if(rc_handle == NULL)
        return CUVRC_ERR_ARG;
    
    *type = p_rc_context->next_frame_type;    
    
    return err;
}

CUVRC_ERROR cuvrc_get_avg_frame_qp(CUVRC_HANDLE rc_handle, int *avg_qp)
{
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)rc_handle;
	if (rc_handle == NULL)
		return CUVRC_ERR_ARG;

	*avg_qp = p_rc_context->frame_qp;

	return err;	
}

CUVRC_ERROR cuvrc_process_frame(CUVRC_HANDLE handle, int scene_cut)
{	
	CUVRC_ERROR err = CUVRC_ERR_SUCCESS;
	RC_STATE *p_rc_context = (RC_STATE *)handle;
	int i;
	if(handle == NULL)
		return CUVRC_ERR_ARG;
	
	
	assert(!((p_rc_context->next_frame_type == CUVRC_NON_REFERENCE_B_FRAME_FORWARD) && scene_cut));
	
	if((p_rc_context->next_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD) &&
			(p_rc_context->force_iframe_postponed == TRUE))
	{
		p_rc_context->force_iframe_flag = TRUE;
		p_rc_context->force_iframe_postponed = FALSE;
	}

	if((p_rc_context->next_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD) && 
		(p_rc_context->reset_postponed == TRUE))
		modify_input_data(p_rc_context);
	
	update_previous_frame_statistics(p_rc_context);
	
	if((p_rc_context->gop_reset_flag == CUVRC_SCENE_CHANGE_DO_NOT_RESET_GOP) && scene_cut
			&& (p_rc_context->next_frame_type == CUVRC_NON_REFERENCE_P_FRAME_FORWARD))
	{
		p_rc_context->scene_change_postponed = TRUE;
		p_rc_context->scene_change = FALSE;		
	}
	else
		p_rc_context->scene_change = scene_cut;
	
	if((p_rc_context->scene_change_postponed == TRUE) &&
			(p_rc_context->next_frame_type != CUVRC_NON_REFERENCE_P_FRAME_FORWARD))
	{
		p_rc_context->scene_change_postponed = FALSE;
		p_rc_context->scene_change = TRUE;		
	}

	
	if((p_rc_context->need_reset) || (p_rc_context->scene_change)) 
	{
		err = rate_control_setup(p_rc_context, p_rc_context->im_width, p_rc_context->im_height);		
	}		
	
	//set current & next frame type
	set_frame_type(p_rc_context);
	
	if(p_rc_context->scene_change)
		update_scene_change_error(p_rc_context);
	
	if(p_rc_context->force_iframe_flag)
	{
		p_rc_context->force_iframe_flag = FALSE;	
		if(!p_rc_context->scene_change)
			update_forced_iframe_error(p_rc_context);
	}
	
	if (p_rc_context->gop_reset_flag == CUVRC_SCENE_CHANGE_RESET_GOP) 
	{
	if((p_rc_context->curr_frame_type == CUVRC_I_FRAME) && (!p_rc_context->scene_change))
		{
			// since this is not a scene change error, move all the error stored in "scene_change_error"
			// in "HRB_fullness"
			compare_errors(p_rc_context);
			p_rc_context->current_gop_scene_change_error += p_rc_context->scene_change_error;
			p_rc_context->scene_change_error = 0;
		}						
	} else 
	{		
		if ((p_rc_context->curr_frame_type == CUVRC_I_FRAME) || (p_rc_context->scene_change)) 
		{
			dstribute_scene_change_error(p_rc_context);
		}
	}
	
	if((p_rc_context->need_reset) || (p_rc_context->scene_change)) 
	{
		set_qp(p_rc_context);
		p_rc_context->need_reset = 0;
		p_rc_context->need_full_reset = 0;
	}
	
	if((p_rc_context->is_initial_qp) && (p_rc_context->valid_avg_var))
		modify_initial_qp(p_rc_context);
	

	if (p_rc_context->force_qp_flag != TRUE)
		rate_control_frame_level(p_rc_context);
	p_rc_context->min_frame_qp = p_rc_context->frame_qp;
	p_rc_context->max_frame_qp = p_rc_context->frame_qp;

	if (p_rc_context->quality_mode == CUVRC_TEXTURE_ADAPTIVE_QUALITY) {
		if ((p_rc_context->p_mb_qp == NULL)
				|| (p_rc_context->p_mb_characs == NULL)
				|| (p_rc_context->p_qp_list == NULL))
			return CUVRC_ERR_ARG;
		else
			// populate the array of qps with the frame qp
			populate_mb_qps_array(p_rc_context);
	} else if (p_rc_context->roi_info.rois_present_flag == TRUE) {
		if ((p_rc_context->p_mb_qp == NULL) || (p_rc_context->p_qp_list
				== NULL))
			return CUVRC_ERR_ARG;
		else
			copy_frame_qp_to_mb_qp(p_rc_context);
	}

    if((p_rc_context->roi_info.rois_present_flag == TRUE) || 
        (p_rc_context->quality_mode == CUVRC_TEXTURE_ADAPTIVE_QUALITY))
    {

    }
	p_rc_context->frame_num++;
	p_rc_context->valid_avg_var = FALSE;
	p_rc_context->is_backward_prediction = FALSE;
	   
	return err;
}


