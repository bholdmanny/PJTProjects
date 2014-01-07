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
#include "../inc/rc_process_frame.h"
#include "../inc/rc_internal_settings.h"

//  call this function prior to encoding each frame
//  returns: QP to use for this frame
int rate_control_before_frame_cbr (
    RC_STATE* rcs,
    int iframe_flag,               ///  indicates whether this frame is an Iframe
    int force_qp
    )
{
    int QP;
    QP = rcs->frame_qp;
    
    if(rcs->force_qp_flag == TRUE) {              /// forget rate control, force QP to this value
        QP = force_qp;
    }
    
    else if(iframe_flag) {
    	int prev_gop_bit_error;
    	int last_target_bits = rcs->target_bits;
    	    	
    	if (rcs->HRB_fullness <= 0)
    		prev_gop_bit_error = rcs->prev_gop_bit_error * rcs->iframeratio / 3;
    	else
    		prev_gop_bit_error = rcs->prev_gop_bit_error * rcs->iframeratio;	
        
        rcs->target_bits = (rcs->iframe_bits - prev_gop_bit_error);

        if (((rcs->last_ref_frame_stat.iframe_flag) && !(rcs->scene_change)) || 
        		((rcs->start_of_gop_flag == FALSE) && (!rcs->is_initial_qp))){
            if (rcs->iframe_stat.actual_bits > 2 * rcs->iframe_stat.target_bits) {
                QP = rcs->iframe_stat.qp + 2;
            }
            else if (rcs->iframe_stat.actual_bits > 1.5 * rcs->iframe_stat.target_bits) {
                QP = rcs->iframe_stat.qp + 1;
            }
            else if (rcs->iframe_stat.actual_bits < 0.5 * rcs->iframe_stat.target_bits) {
                QP = rcs->iframe_stat.qp - 2;
            }
            else if (rcs->iframe_stat.actual_bits < 0.8 * rcs->iframe_stat.target_bits ) {
                QP = rcs->iframe_stat.qp - 1;
            }
            else
            {
                QP = rcs->iframe_stat.qp;
            }
            accumulated_error_to_qp(rcs, rcs->iframe_stat.qp, QP, rcs->iframe_stat.actual_bits);
        }
        else {
            if (rcs->is_initial_qp)
            {
                QP = rcs->frame_qp;
                rcs->is_initial_qp = 0;
                accumulated_error_to_qp(rcs, QP, QP, (15 * rcs->iframe_bits/10));
            }
            else
            {
                QP = rcs->frame_qp - rcs->i_p_frame_diff;
                QP = (QP > rcs->max_qp) ? rcs->max_qp : QP;
                QP = (QP < rcs->min_qp) ? rcs->min_qp : QP;
                rcs->frame_qp = QP;
                if(rcs->curr_gop_frames_after_last_iframe <= 1)
                    accumulated_error_to_qp(rcs, QP, QP, (15 * rcs->iframe_bits/10));
            }           
        }
    }
    
    rcs->iframe_flag = iframe_flag;
    
    return QP;
}

#define NUM_ADD_BITS_PER_IMB        32
//// This function is called after ME and before RC.
//// rcs->avg_sad shall be initialized by ME of the current frame
//// This function also determin whether there has been a scene change and return flag indicating this frame should be an Iframe 
int rate_control_before_res_frame_cbr(
    RC_STATE* rcs
    )
{
    int base_qp;
    int last_target_bits, actual_bits, new_target_bits, d_bits;
    int new_target_qp;
    int pframe_bits = rcs->pframe_bits;
    int HRB_fullness = rcs->HRB_fullness;
    int pframe_bits_low, pframe_bits_high, part_pframe_bits;
    
    if (rcs->iframe_flag)
    {
        return 1;
    }

    if (!(rcs->last_non_ref_frame_flag)) 
    {
		base_qp = rcs->last_qp;
		last_target_bits = rcs->target_bits;
		actual_bits = rcs->num_bits;
	} else 
	{
		base_qp = rcs->last_ref_frame_stat.qp;
		last_target_bits = rcs->last_ref_frame_stat.target_bits;
		actual_bits = rcs->last_ref_frame_stat.actual_bits;
//		d1_bits = rcs->target_bits - rcs->num_bits; 
//		base_qp = (base_qp + (rcs->last_qp - IFRAMEDIFF_CBR)) >> 1;
		if(((rcs->HRB_fullness > (rcs->target_bps/20)) && (actual_bits <= last_target_bits) && 
				(rcs->num_bits > (120 * rcs->non_ref_frame_bits / 100)) && (rcs->acc_err_from_last_ref_frame > 0))
				|| ((rcs->HRB_fullness < -(rcs->target_bps/20))	&& (actual_bits >= last_target_bits) 
						&& (rcs->num_bits < (80 * rcs->non_ref_frame_bits / 100)) 
						&& (rcs->acc_err_from_last_ref_frame < 0)))
		{
			last_target_bits =  rcs->non_ref_frame_bits;
			actual_bits = rcs->num_bits;
		}
		if(rcs->last_ref_frame_stat.iframe_flag)
		{
			base_qp = rcs->iframe_stat.qp + rcs->i_p_frame_diff;			
		}
	}
    
    if(rcs->non_ref_frame_flag)
    {
    	pframe_bits = rcs->non_ref_frame_bits;
    	base_qp = base_qp + IFRAMEDIFF_CBR;    	
    }
    if((rcs->non_ref_frame_interval != 0) && (rcs->last_ref_frame_stat.iframe_flag))
	{
    	// since we have non reference frames, the reference frame should start with a lower qp
    	// than average qp value
    	base_qp = base_qp - 1;
	}
    
    if (rcs->use_prev_bits) 
    {
    	assert(rcs->non_ref_frame_flag == TRUE);
		pframe_bits = rcs->prev.non_ref_frame_bits;
		if (!rcs->is_backward_prediction)
		{			
			if (!rcs->last_non_ref_frame_flag) 
			{
				base_qp = rcs->prev.last_qp;	
				last_target_bits = rcs->prev.non_ref_frame_bits;
				actual_bits = rcs->prev.last_actual_bits;
			}
			else
			{
				base_qp = rcs->last_qp;
				last_target_bits = rcs->target_bits;
				actual_bits = rcs->num_bits;				
			}
		}
	}

    if(rcs->first_p_qp_changed)
    {
    	rcs->first_p_qp_changed = FALSE;
    	base_qp++;
    }    

    d_bits = last_target_bits - actual_bits;
    if(rcs->last_iframe_flag)
    {
    	if((!rcs->use_prev_bits) || (rcs->is_backward_prediction))
    		base_qp += rcs->i_p_frame_diff;
    	// if the last frame was encoded as I frame
    	// then distribute the error caused in the I frame
    	// over all the remaining frames in the gop
    	assert(rcs->ifrate >= 1);
    	rcs->prev_gop_bit_error += -d_bits/rcs->ifrate;
    	d_bits = 0;
    }
    if(rcs->last_ref_frame_stat.iframe_flag)
    	d_bits = 0;
    
    if(((HRB_fullness >= 0) && (rcs->prev_gop_bit_error > 0)) 
    		|| ((HRB_fullness <= 0) && (rcs->prev_gop_bit_error < 0)))
    	pframe_bits  = pframe_bits - rcs->prev_gop_bit_error;
   
    if(((HRB_fullness > 0) && (d_bits < 0)) || ((HRB_fullness < 0) && (d_bits > 0)))
    	new_target_bits  = pframe_bits + d_bits;
    else
    	new_target_bits  = pframe_bits;
    	
    // This is not required since we considerHRB fullness in another logic below
    //new_target_bits  = min (buffer_space, new_target_bits);

    part_pframe_bits = pframe_bits/5;
    pframe_bits_low = pframe_bits - part_pframe_bits;
    pframe_bits_high = pframe_bits + part_pframe_bits;    
    new_target_bits  = max ((int) pframe_bits_low, new_target_bits);
    new_target_bits  = min ((int) pframe_bits_high, new_target_bits);
    //assert(new_target_bits > 0);
    rcs->target_bits = new_target_bits;
    
    if (rcs->non_ref_frame_interval != 0) 
    {
    	int scale = (rcs->non_ref_frame_interval - 1) << (rcs->non_ref_frame_interval - 1); 
    	// since the actual bits to target bits ratio of a reference frame is used
    	// for subsequent non refrence frames & one  reference frame
    	// the ratio needs to be scaled proportional to the number of non reference frames
		if (actual_bits > last_target_bits)
			last_target_bits = (last_target_bits * (87 - scale)/100);
		else
			last_target_bits = last_target_bits * (112 + scale) / 100;
	}
    // this logic makes QP sensitive to the error in bit generation in previous frame
    if(actual_bits > last_target_bits)
    {
        if(actual_bits <= (15 * last_target_bits/10))
            new_target_qp = base_qp;
        else if(actual_bits <= 2 * last_target_bits)
            new_target_qp = base_qp + 1;
        else if(actual_bits <= (25 * last_target_bits/ 10))
            new_target_qp = base_qp + 2;
        else if(actual_bits <= 3 * last_target_bits) 
            new_target_qp = base_qp + 3;
        else if(actual_bits <= (35 * last_target_bits/10))
            new_target_qp = base_qp + 4;
        else if(actual_bits <= 4 * last_target_bits)
            new_target_qp = base_qp + 5;
        else
            new_target_qp = base_qp + 6;
    }
    else
    {
        if(actual_bits >= (7 * last_target_bits/ 10))
            new_target_qp = base_qp;
        else if(actual_bits >= (4 * last_target_bits/ 10))
             new_target_qp = base_qp - 1;
        else if (actual_bits >= (2 * last_target_bits/10))
            new_target_qp = base_qp - 1;
        else if(actual_bits >= (1 * last_target_bits/10))
            new_target_qp = base_qp - 2;
        else 
            new_target_qp = base_qp - 3;
    }

    if ((rcs->last_iframe_flag) && (rcs->ifrate >= 3))
    {
    	if((rcs->HRB_fullness < 0) && (new_target_qp >= base_qp) && (!rcs->use_prev_bits))
    	{
    		// if this is the first P frame in the gop
    		// & the accumulated error is negative
    		// then decrement the qp by 1 only for this frame
    		// to generate higher number of bits
    		new_target_qp = new_target_qp - 1;
    		rcs->first_p_qp_changed = TRUE;
    	}    	
        new_target_qp = (new_target_qp > rcs->max_qp) ? rcs->max_qp : new_target_qp;
        new_target_qp = (new_target_qp < rcs->min_qp) ? rcs->min_qp : new_target_qp;

        rcs->frame_qp = new_target_qp;	
        if(rcs->HRB_fullness > (rcs->target_bps/5))
            accumulated_error_to_qp(rcs, base_qp, new_target_qp, actual_bits);
    }
    else
    {
        accumulated_error_to_qp(rcs, base_qp, new_target_qp, actual_bits);
    }
    
    return 0;
}

void accumulated_error_to_qp(RC_STATE* rcs, int base_qp, int new_target_qp, int actual_bits)
{
    int target_frame_bits;

	//    if ((rcs->last_frame_type != CUVRC_NON_REFERENCE_B_FRAME_FORWARD) && 
	//    		(rcs->last_frame_type != CUVRC_NON_REFERENCE_B_FRAME_BACKWARD)) 
	if (rcs->last_ref_frame_stat.iframe_flag)
		target_frame_bits = rcs->iframe_bits;
	else
		target_frame_bits = rcs->pframe_bits - rcs->prev_gop_bit_error;
	
	if(actual_bits != rcs->last_ref_frame_stat.actual_bits)
	{
		if(!rcs->use_prev_bits)
			target_frame_bits = rcs->non_ref_frame_bits;
		else
			target_frame_bits = rcs->prev.non_ref_frame_bits;
	}
    	
    //the following logic is to make QP sensitive to the cumulative error in the target bitrate generation
    if (((rcs->HRB_fullness > (rcs->target_bps/10)) && (new_target_qp <= base_qp)) || ((actual_bits - target_frame_bits) > (rcs->target_bps/20)))
    {
    	// in case the error is greater than 10% set the target bits to the lowest possible
        if(rcs->HRB_fullness > (rcs->target_bps/4))
            // in case the error > 25% & the rate of increase of qp is not much for past few frames
            new_target_qp = new_target_qp + 4;
    	else if (rcs->HRB_fullness > (rcs->target_bps/5))
			// in case the error > 25%
			new_target_qp = new_target_qp + 3;
		else if ((rcs->HRB_fullness > (rcs->target_bps/6)) && (actual_bits > target_frame_bits))
			// in case the error > 20% but not high error in previous frame
			new_target_qp = new_target_qp + 2;
		else if ((rcs->HRB_fullness > (rcs->target_bps/10)) && (actual_bits > (12*target_frame_bits/10)))
			// in case the error > 10% & high error in previous frame too
			new_target_qp = new_target_qp + 1;
    	
        new_target_qp = min(new_target_qp, base_qp + 6);
		
    }
    
    if (((rcs->HRB_fullness < -(rcs->target_bps/10)) && (new_target_qp >= base_qp)) 
    		|| ((actual_bits - target_frame_bits) < -(rcs->target_bps/20)))
    {
    	// in case the error is less than -10% set the target bits to the highest possible
        if((rcs->HRB_fullness < -(rcs->target_bps/4)))
            // in case the error > 25% & the rate of decrease of qp is not much for past few frames
            new_target_qp = new_target_qp - 3;
    	else if (rcs->HRB_fullness < -(rcs->target_bps/5))
    		// in case the error < -25%
    		new_target_qp = new_target_qp - 2;
    	else if((rcs->HRB_fullness < -(rcs->target_bps/6)) && (actual_bits < target_frame_bits))
    		// in case the error < -20% & high error in previous frame too
    		new_target_qp = new_target_qp - 2;
    	else if ((rcs->HRB_fullness < -(rcs->target_bps/10)) && (actual_bits < (8 * target_frame_bits/10)))
    		// in case the error < -10% & high error in previous frame too
    		new_target_qp = new_target_qp - 1;

    }
    new_target_qp = (new_target_qp > rcs->max_qp) ? rcs->max_qp : new_target_qp;
    new_target_qp = (new_target_qp < rcs->min_qp) ? rcs->min_qp : new_target_qp;

    rcs->frame_qp = new_target_qp;	
}

void rate_control_frame_level (RC_STATE* rcs)
{
	FSM_PARAMS *fsm = &rcs->fsm;
	MOTION_TYPE current_frame_motion_type;
	unsigned int gop_size = rcs->ifrate;
	ABR_CONTEXT *abr_params = &rcs->abr_params;
	fsm->state_status = CONTINUE;
	
	if ((rcs->HRB_fullness <= 0) && (rcs->high_motion_error <= 0)
			&& (rcs->frame_qp == rcs->original_values.min_qp)
            && (rcs->avg_p_qp == rcs->original_values.min_qp)) {
		if(rcs->iframe_flag)
			rcs->target_bits = rcs->iframe_bits;
		else
			rcs->target_bits = rcs->pframe_bits;			
		return;
	}	    		   
	
	if (rcs->bitrate_mode == CUVRC_MOTION_ADAPTIVE_BITRATE) {
		if((!rcs->last_iframe_flag) && (!rcs->scene_change))
		{
			if (((rcs->num_imbs + rcs->num_big_mvs) >= rcs->imb_thresh)
					&& (rcs->avg_sad >= DEFAULT_AVG_SAD_THRESHOLD))
				rcs->current_frame_motion_type = HIGH;
			else
				rcs->current_frame_motion_type = NORMAL;
		}
	}
	current_frame_motion_type = rcs->current_frame_motion_type;
	rcs->unmodified_HRB_fullness = rcs->HRB_fullness;

    do {
		switch (fsm->current_state) {
		case NORMAL_MOTION:
			if (current_frame_motion_type == HIGH) {
				int avg_p_qp = rcs->avg_p_qp;
				fsm->next_state = NORMAL_TO_HIGH_MOTION_WAIT;
				if(rcs->curr_gop_frames_after_last_iframe >= max((rcs->ifrate_uint/3), 5))
					avg_p_qp = rcs->sum_p_qp_in_gop / rcs->curr_gop_frames_after_last_iframe;
				if(rcs->HRB_fullness < 0)
					avg_p_qp = avg_p_qp + (rcs->HRB_fullness /(rcs->target_bps/20));
				rcs->min_qp = max(min(avg_p_qp, rcs->iframe_stat.qp), rcs->original_values.min_qp);					
				fsm->wait_state_counter = 0;
			} else {
				fsm->next_state = NORMAL_MOTION;
				fsm->state_status = EXIT;
			}
			fsm->current_state = fsm->next_state;
			break;
		case NORMAL_TO_HIGH_MOTION_WAIT:
			fsm->wait_state_counter++;
			if (current_frame_motion_type == HIGH) {
				if ((fsm->wait_state_counter >= DEFAULT_WAIT_STATE_COUNT) && (rcs->start_of_gop_flag)) {
					fsm->next_state = HIGH_MOTION;
					fsm->wait_state_counter = 0;
					rcs->imb_thresh = rcs->num_mbs / 10;
					set_targets(rcs, abr_params->higher_br_scale);
					scale_qp(rcs, NORMAL_MOTION_TO_HIGH_MOTION);
				} else {
					fsm->next_state = NORMAL_TO_HIGH_MOTION_WAIT;
					if(fsm->wait_state_counter == (DEFAULT_WAIT_STATE_COUNT >> 1))
						rcs->imb_thresh = rcs->num_mbs / 15;	
					fsm->state_status = EXIT;
				}
			} else {
				fsm->next_state = NORMAL_MOTION;
				rcs->min_qp = rcs->original_values.min_qp;
				fsm->wait_state_counter = 0;
				rcs->imb_thresh = rcs->num_mbs / 10;
			}
			fsm->current_state = fsm->next_state;
			break;
		case HIGH_MOTION:
			if (current_frame_motion_type == HIGH) {
				//rcs->high_motion_error = update_high_motion_error(rcs);
				if(rcs->start_of_gop_flag)
				{
					// this is to prevent the underflow of HRB_fullness
					if(rcs->HRB_fullness < -(rcs->target_bps/2))
					{
						int shift_error = rcs->HRB_fullness + (rcs->target_bps/5);
						rcs->HRB_fullness = rcs->HRB_fullness - shift_error;
						rcs->high_motion_error = rcs->high_motion_error + shift_error;
					}
				}
				fsm->next_state = HIGH_MOTION;
				fsm->state_status = EXIT;
			} else {
				fsm->next_state = HIGH_TO_NORMAL_MOTION_WAIT;
				fsm->wait_state_counter = 0;
			}
			fsm->current_state = fsm->next_state;
			break;
		case HIGH_TO_NORMAL_MOTION_WAIT:
			fsm->wait_state_counter++;
			if (current_frame_motion_type != HIGH) {
				if ((fsm->wait_state_counter >= DEFAULT_WAIT_STATE_COUNT)  && (rcs->start_of_gop_flag)) {
					fsm->next_state = LOWER_BITRATE_AFTER_HIGH_MOTION;
					fsm->wait_state_counter = 0;
					if(rcs->HRB_fullness < 0)
						fsm->reset_HRB_fullness = TRUE;
					set_targets(rcs, abr_params->lower_br_scale);
					scale_qp(rcs, HIGH_MOTION_TO_LOW_BITRATE);
				} else {
					fsm->next_state = HIGH_TO_NORMAL_MOTION_WAIT;
					//rcs->high_motion_error = update_high_motion_error(rcs);
					fsm->state_status = EXIT;
				}
			} else {
				fsm->next_state = HIGH_MOTION;
				fsm->wait_state_counter = 0;
			}
			if ((fsm->next_state == LOWER_BITRATE_AFTER_HIGH_MOTION) && (rcs->HRB_fullness > (rcs->target_bps/10))) 
			{
				// if the error was positive in high bitrate state
				// it will be very high in low bitrate state & rate control
				// will go in panic mode. Hence we distribute it over the gop 
				// when we move to lower bitrate state
				rcs->current_gop_scene_change_error += rcs->HRB_fullness;
				rcs->HRB_fullness = 0;
			}       
			fsm->current_state = fsm->next_state;
			break;
		case LOWER_BITRATE_AFTER_HIGH_MOTION:
            if (current_frame_motion_type != HIGH) {
                signed long long tmp1;
                signed long long tmp = rcs->high_motion_error; //update_high_motion_error(rcs);
                if (tmp < rcs->target_bps/20) {
                    if (rcs->start_of_gop_flag) {
                        rcs->HRB_fullness += rcs->high_motion_error;
                        rcs->high_motion_error = 0;
                        fsm->next_state = NORMAL_MOTION;
                        set_targets(rcs, NORMAL_SCALE);
                        rcs->min_qp = rcs->original_values.min_qp;
                        scale_qp(rcs, LOW_BITRATE_TO_NORMAL_MOTION);
                    } else {                    	
                        rcs->HRB_fullness += tmp;
                        rcs->high_motion_error = 0;
                        fsm->next_state = LOWER_BITRATE_AFTER_HIGH_MOTION;
                        fsm->state_status = EXIT;
                    }
                } else if ((rcs->HRB_fullness < 0)  && (rcs->start_of_gop_flag))
                {
                    tmp1 = rcs->high_motion_error + rcs->HRB_fullness;
                    if (tmp1 < rcs->target_bps/20) 
                    {                    	
                        rcs->HRB_fullness = tmp1;
                        rcs->high_motion_error = 0;
                        fsm->next_state = NORMAL_MOTION;
                        set_targets(rcs, NORMAL_SCALE);
                        rcs->min_qp = rcs->original_values.min_qp;
                        if(fsm->reset_HRB_fullness)
                           	fsm->reset_HRB_fullness = 0;
                        else
                        	scale_qp(rcs, LOW_BITRATE_TO_NORMAL_MOTION);                        
                    } else 
                    {
                    	if(fsm->reset_HRB_fullness)
                    	{
                    		rcs->high_motion_error = tmp1;
                    		rcs->HRB_fullness = 0;
                    		fsm->reset_HRB_fullness = FALSE;                    		
                    	}
                    	else
                    	{
                    		rcs->high_motion_error = tmp;
                    	}
                    	fsm->next_state = LOWER_BITRATE_AFTER_HIGH_MOTION;
						fsm->state_status = EXIT;                    	
                    }
                }
                else {
                    rcs->high_motion_error = tmp;
                    fsm->next_state = LOWER_BITRATE_AFTER_HIGH_MOTION;
                    fsm->state_status = EXIT;
                }
            } else {
                fsm->next_state = LOWER_BITRATE_TO_HIGH_MOTION_WAIT;
                set_targets(rcs, NORMAL_SCALE);
                fsm->wait_state_counter = 0;
            }
            if((fsm->next_state == LOWER_BITRATE_AFTER_HIGH_MOTION) &&
            		(rcs->scene_change) && (rcs->scale_avg_p_qp >= 0))
            {
            	// when there is a scene change in the lower bitrate state,
            	// the quality of the I frame should not be worst than in
            	// the case of normal bitrate
            	// hence, only for the I frame, reducing the qp to a value
            	// corresponding to normal bitrate
            	int tmp = rcs->scale_avg_p_qp;
            	scale_qp(rcs, LOW_BITRATE_TO_NORMAL_MOTION);
            	rcs->frame_qp += rcs->scale_avg_p_qp;
            	rcs->scale_avg_p_qp = tmp;            	
            }           
            if((fsm->next_state == NORMAL_MOTION) && (rcs->HRB_fullness < -(rcs->target_bps/5)))
            {
            	// since lot of negative error was accumulated in low bitrate case
            	// we distribute it over the gop when we move to normal state
            	rcs->current_gop_scene_change_error += rcs->HRB_fullness;
            	rcs->HRB_fullness = 0;
            }            	
            fsm->current_state = fsm->next_state;
            break;
		case LOWER_BITRATE_TO_HIGH_MOTION_WAIT:
			fsm->wait_state_counter++;
			if (current_frame_motion_type == HIGH) {
				if ((fsm->wait_state_counter >= DEFAULT_WAIT_STATE_COUNT) && (rcs->start_of_gop_flag)){
					fsm->next_state = HIGH_MOTION;
					fsm->wait_state_counter = 0;
					rcs->imb_thresh = rcs->num_mbs / 10;
					set_targets(rcs, abr_params->higher_br_scale);
					scale_qp(rcs, LOW_BITRATE_TO_HIGH_MOTION);
				} else {
					fsm->next_state = LOWER_BITRATE_TO_HIGH_MOTION_WAIT;
					if(fsm->wait_state_counter == (DEFAULT_WAIT_STATE_COUNT >> 1))
						rcs->imb_thresh = rcs->num_mbs / 15;	
					fsm->state_status = EXIT;
				}
			} else {
				fsm->next_state = LOWER_BITRATE_AFTER_HIGH_MOTION;
				fsm->wait_state_counter = 0;
				rcs->imb_thresh = rcs->num_mbs / 10;
				set_targets(rcs, abr_params->lower_br_scale);
			}
			fsm->current_state = fsm->next_state;
			break;
		}
	} while (fsm->state_status != EXIT);
    
    if(rcs->start_of_gop_flag == TRUE)
    {
    	update_previous_gop_error(rcs);
    	if(!rcs->scene_change)
    	{
    		// update previous gop statistics
    		rcs->prev.iframe_bits = rcs->original_values.iframe_bits;
    		rcs->prev.pframe_bits = rcs->original_values.pframe_bits;
    		rcs->prev.non_ref_frame_bits = rcs->original_values.non_ref_frame_bits;
    		rcs->prev.scaled_non_ref_frame_bits = rcs->non_ref_frame_bits;
    		rcs->prev.last_qp = rcs->last_qp;  
    		rcs->prev.last_actual_bits = rcs->num_bits;
    		if(rcs->curr_gop_frames_after_last_iframe <= 1)
    		{
    			rcs->frame_qp += rcs->scale_avg_p_qp;
    			rcs->scale_avg_p_qp = 0; 
    		}
    	}
    	else
    	{
    		rcs->frame_qp += rcs->scale_avg_p_qp;
    		rcs->scale_avg_p_qp = 0;    		
    	}
    }
    
    if((rcs->curr_frame_type == CUVRC_I_FRAME) && (rcs->frame_num != 0) &&
    		(rcs->curr_gop_frames_after_last_iframe > 1) && (!rcs->scene_change)
			&& (rcs->force_qp_flag != TRUE) && (rcs->start_of_gop_flag == TRUE))
    		modify_iframeratio(rcs);
          
    rate_control_before_frame_cbr(rcs, rcs->iframe_flag, rcs->frame_qp);
    rate_control_before_res_frame_cbr(rcs);
    rcs->high_motion_error = update_high_motion_error(rcs);

}

signed long long update_high_motion_error(RC_STATE *rcs)
{
	signed long long tmp;
	switch (rcs->curr_frame_type) {
    case CUVRC_P_FRAME:
		tmp = rcs->pframe_bits - rcs->original_values.pframe_bits;
		break;
	case CUVRC_I_FRAME:
	case CUVRC_PSEUDO_I_FRAME:
		tmp =  rcs->iframe_bits - rcs->original_values.iframe_bits;
		break;	
	case CUVRC_NON_REFERENCE_P_FRAME_FORWARD:
	case CUVRC_NON_REFERENCE_B_FRAME_FORWARD:
	case CUVRC_NON_REFERENCE_B_FRAME_BACKWARD:
	{
		tmp = rcs->non_ref_frame_bits - rcs->original_values.non_ref_frame_bits;
		if(rcs->use_prev_bits)
			tmp = rcs->prev.scaled_non_ref_frame_bits - rcs->prev.non_ref_frame_bits;
		break;
	}
	default:
		printf("Error : unknown frame type\n");
		exit(-1);
		break;
	}
	tmp = tmp + rcs->high_motion_error; 
	return tmp;
	
}
//
void set_targets(RC_STATE* rcs, float scale)
{
	//assert((scale >= 0.5) && (scale <= 2));
	rcs->target_bps = scale * rcs->original_values.target_bps;
	rcs->non_ref_frame_bits = scale * rcs->original_values.non_ref_frame_bits;
	rcs->pframe_bits = scale * rcs->original_values.pframe_bits;
	rcs->iframe_bits = scale * rcs->original_values.iframe_bits;

}

// This function calculates the delta value by which the qp for next gop
// should be scaled due to the change in motion state of the sequence
void scale_qp(RC_STATE* rcs, MOTION_TRANSITION_PHASE transition_phase)
{
	int ratio;
	int delta_qp;
	// ratio = bitrate of current phase/bitrate of next phase
	// since this migh be < 1, multiply it by 10 to get the index into qp_offsets array
	switch (transition_phase)
	{
	case NORMAL_MOTION_TO_HIGH_MOTION:
		ratio = 10/rcs->abr_params.higher_br_scale;
		break;
	case HIGH_MOTION_TO_LOW_BITRATE:
		ratio = rcs->abr_params.higher_br_scale*10/rcs->abr_params.lower_br_scale;
		break;
	case LOW_BITRATE_TO_HIGH_MOTION:
		ratio = rcs->abr_params.lower_br_scale*10/rcs->abr_params.higher_br_scale;
		break;
	case LOW_BITRATE_TO_NORMAL_MOTION:
		ratio = rcs->abr_params.lower_br_scale*10;
		break;
	default:
		printf("Error in transition_phase : %d\n", transition_phase);		
	}
	ratio = clip_3(ratio, 40, 3);
	delta_qp = qp_offsets[ratio];
	rcs->scale_avg_p_qp = delta_qp; 
}

// mult_factor = 2^(delta_iframe_qp/6)
static const float mult_factor[101] = {0.003100393, 0.003480073, 0.00390625, 0.004384617, 0.004921567,
		0.005524272, 0.006200785, 0.006960146, 0.0078125, 0.008769235, 0.009843133, 0.011048543,
		0.012401571, 0.013920292, 0.015625, 0.01753847, 0.019686266, 0.022097087, 0.024803141, 0.027840585,
		0.03125, 0.035076939, 0.039372533, 0.044194174, 0.049606283, 0.05568117, 0.0625, 0.070153878, 
		0.078745066, 0.088388348, 0.099212566, 0.11136234, 0.125, 0.140307756, 0.157490131, 0.176776695,
		0.198425131, 0.22272468, 0.25, 0.280615512, 0.314980262, 0.353553391, 0.396850263, 0.445449359,
		0.5, 0.561231024, 0.629960525, 0.707106781, 0.793700526, 0.890898718, 1, 1.122462048, 1.25992105,
		1.414213562, 1.587401052, 1.781797436, 2, 2.244924097, 2.5198421, 2.828427125, 3.174802104,
		3.563594873, 4, 4.489848193, 5.0396842, 5.656854249, 6.349604208, 7.127189745, 8, 8.979696386,
		10.0793684, 11.3137085, 12.69920842, 14.25437949, 16, 17.95939277, 20.1587368, 22.627417,
		25.39841683, 28.50875898, 32, 35.91878555, 40.3174736, 45.254834, 50.79683366, 57.01751796,
		64, 71.83757109, 80.63494719, 90.50966799, 101.5936673, 114.0350359, 128, 143.6751422, 
		161.2698944, 181.019336, 203.1873347, 228.0700718, 256, 287.3502844, 322.5397888};

const float qscale[52] = { 0.63, 0.707, 0.7937, 0.891, 1, 1.122, 1.26,
		1.414, 1.587, 1.782, 2, 2.245, 2.52, 2.828,3.175, 3.564, 4, 4.4898, 
		5.04, 5.657, 6.3496, 7.127, 8, 8.9797, 10.079, 11.314,
		12.699, 14.254, 16, 17.959, 20.1587, 22.627, 25.398, 28.5088, 32,
		35.9188, 40.317, 45.255, 50.797, 57.0175, 64, 71.8376, 80.6349,
		90.50967, 101.594, 114.035, 128, 143.675, 161.26989, 181.019, 203.187,
		228.07 };


static const int log_2[41]= {0, 0, 1, 2, 2, 2, 3, 3, 3, 3,
							3, 3, 4, 4, 5, 6, 6, 7, 7, 7,
							8, 8, 9, 9, 9, 10, 10, 10, 11, 11,
							12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 16};
void modify_iframeratio(RC_STATE *rcs)
{
	//int avg_p_qp = (0.7 * (rcs->sum_p_qp_in_gop + (rcs->current_gop_size >> 1))/rcs->current_gop_size) + (0.3 * (rcs->sum_p_qp_in_gop_1 + (rcs->current_gop_size_1 >> 1))/rcs->current_gop_size_1) + 0.5;
    int avg_p_qp = rcs->sum_p_qp_in_gop/(rcs->curr_gop_frames_after_last_iframe);
    float avg_p_qscale = qscale[avg_p_qp];
    float i_qscale = qscale[rcs->iframe_stat.qp];
    int avg_p_num_bits = rcs->total_p_num_bits/(rcs->curr_gop_frames_after_last_iframe); 
    int avg_p_comlexity, i_complexity;
	int next_iframe_qp;
	int delta_iframe_qp;
	int iframe_bits;
	float if_persec = rcs->if_persec;
//	float fpsplus;
	int bitrate = rcs->target_bps;
//	float scale;
	int pframe_bits, non_ref_frame_bits, total_p_frame_bits;
//	int prev_gop_error;
	int iframe_prev_gop_bit_error;
	int HRB_fullness_deviation = rcs->unmodified_HRB_fullness - rcs->iframe_stat.HRB_fullness_after_iframe;
	rcs->avg_p_qp = avg_p_qp;
	
	
	// get the difference in qp between I & P frame by using the complexity 
	// of I & P frames in previous GOP  
    avg_p_comlexity = avg_p_qscale * avg_p_num_bits;
    i_complexity = i_qscale * rcs->iframe_stat.actual_bits;
    assert(avg_p_comlexity > 0);
  
    	rcs->i_p_frame_diff =  2 + clip_3((i_complexity/avg_p_comlexity)/3, 4, 0);      

	// get the estimate of p qp for next gop using average p qp of previous gop
	// & the error statistics
	if (((rcs->HRB_fullness > rcs->target_bps/10) && (rcs->iframe_stat.HRB_fullness_after_iframe > rcs->target_bps/10))
			|| ((rcs->HRB_fullness < -rcs->target_bps/10) && (rcs->iframe_stat.HRB_fullness_after_iframe
							< -rcs->target_bps/10))) {
		if (rcs->HRB_fullness > rcs->target_bps/5)
			avg_p_qp = avg_p_qp + 3;
		else if (rcs->HRB_fullness > rcs->target_bps/6)
			avg_p_qp = avg_p_qp + 2;
		else if (rcs->HRB_fullness > rcs->target_bps/10)
			avg_p_qp = avg_p_qp + 1;
		else if (rcs->HRB_fullness < -rcs->target_bps/2)
			avg_p_qp = avg_p_qp - 5;
		else if (rcs->HRB_fullness < -rcs->target_bps/3)
			avg_p_qp = avg_p_qp - 4;
		else if (rcs->HRB_fullness < -rcs->target_bps/4)
			avg_p_qp = avg_p_qp - 3;
		else if (rcs->HRB_fullness < -rcs->target_bps/5)
			avg_p_qp = avg_p_qp - 2;
		else if (rcs->HRB_fullness < -rcs->target_bps/10)
			avg_p_qp = avg_p_qp - 1;
	}
	else
	{
		int qp_scale = HRB_fullness_deviation/(rcs->target_bps/30);		
            if (qp_scale < 0)
            {
                qp_scale = clip_3(-qp_scale, 40, 0);
                avg_p_qp = avg_p_qp - log_2[qp_scale];
                if((log_2[qp_scale] >= 6) && (rcs->scale_avg_p_qp <= -6))
                	// the cumulative effect of both, HRB deviation & bitrate change
                	// will be very high
                	// hence zeroing one of them
                	rcs->scale_avg_p_qp = 0;
            }
            else
            {
                qp_scale = clip_3(qp_scale, 40, 0);
                avg_p_qp = avg_p_qp + log_2[qp_scale];
                if((log_2[qp_scale] >= 6) && (rcs->scale_avg_p_qp >= 6))
                	// the cumulative effect of both, HRB deviation & bitrate change
                	// will be very high
                	// hence zeroing one of them
                	rcs->scale_avg_p_qp = 0;
            }
    }

	// if the motion state got changed for this I frame then we need to scale qp
	avg_p_qp = avg_p_qp + rcs->scale_avg_p_qp;
	rcs->scale_avg_p_qp = 0;
	   
    // get the I frame qp for next gop using p frame qp & i_p_frame_diff
	next_iframe_qp = avg_p_qp - rcs->i_p_frame_diff;
    next_iframe_qp =  clip_3(next_iframe_qp, rcs->max_qp, rcs->min_qp);
    
    // get the estimate of i frame bits that will be generated in next gop
    // using the statistics of previous i frame
    delta_iframe_qp = rcs->iframe_stat.qp - next_iframe_qp;    

		iframe_bits = rcs->iframe_stat.actual_bits * mult_factor[delta_iframe_qp + 50];		 
    
    if (rcs->HRB_fullness <= 0)
    	iframe_prev_gop_bit_error = max(rcs->prev_gop_bit_error * rcs->iframeratio / 3, -iframe_bits/5);
    else
    	iframe_prev_gop_bit_error = rcs->prev_gop_bit_error * rcs->iframeratio;
    
    iframe_bits = iframe_bits + iframe_prev_gop_bit_error;
        
    // now distribute the remaining bits among p frames as pframe_bits
    total_p_frame_bits = (bitrate - (if_persec * iframe_bits));
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
    non_ref_frame_bits = total_p_frame_bits / (rcs->ref_p_frames_per_sec * rcs->p_2_non_ref_p_ratio + rcs->non_ref_frames_per_sec);
    pframe_bits = non_ref_frame_bits * rcs->p_2_non_ref_p_ratio;
    // if allocated p frame bits are very less then reduce the I frame bits 
    while((pframe_bits < (rcs->width_in_mbs * rcs->height_in_mbs)) && (next_iframe_qp != rcs->max_qp)) 
    {
    	rcs->i_p_frame_diff--;
		next_iframe_qp = avg_p_qp - rcs->i_p_frame_diff;
		next_iframe_qp =  clip_3(next_iframe_qp, rcs->max_qp, rcs->min_qp);
		delta_iframe_qp = rcs->iframe_stat.qp - next_iframe_qp;
		
			iframe_bits = rcs->iframe_stat.actual_bits * mult_factor[delta_iframe_qp + 50];		
		iframe_bits = iframe_bits + iframe_prev_gop_bit_error;
		total_p_frame_bits = (bitrate - (if_persec * iframe_bits));
		non_ref_frame_bits = total_p_frame_bits / (rcs->ref_p_frames_per_sec * rcs->p_2_non_ref_p_ratio + rcs->non_ref_frames_per_sec);
		pframe_bits = non_ref_frame_bits * rcs->p_2_non_ref_p_ratio;
    }
    
    if(pframe_bits < (rcs->width_in_mbs * rcs->height_in_mbs))
    {
    	// Couldn't allocate i & p frame bits for the qp range
    	// hence using the same bits allocated in the previous gop
    	iframe_bits = rcs->iframe_bits;
    	pframe_bits = rcs->pframe_bits;
    	non_ref_frame_bits = rcs->non_ref_frame_bits;
    	avg_p_qp = rcs->max_qp;
    }
    
    // now storethe calculated avg_p_qp as frame qp after clipping
    rcs->frame_qp = clip_3(avg_p_qp, rcs->max_qp, rcs->min_qp);
    // if avg_p_qp got clipped then recalculate i_p_frame_diff
    if((rcs->frame_qp == rcs->min_qp) || (rcs->frame_qp == rcs->max_qp))
    	rcs->i_p_frame_diff = next_iframe_qp - rcs->frame_qp;    	
    
    // find iframe bits/p frame bits ratio
    rcs->iframeratio = (float)iframe_bits / pframe_bits;
    if(rcs->iframeratio < 1)
    	rcs->iframeratio = 1;
    	
    calc_iframe_pframe_bits(rcs, non_ref_frame_bits);
    update_previous_gop_error(rcs);   

}
