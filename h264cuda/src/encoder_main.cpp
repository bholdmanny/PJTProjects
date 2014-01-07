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
#include <assert.h>

#include "../inc/h264_common.h"

#include "../inc/encoder_context.h"
#include "../inc/output.h"
#include "../inc/mem_alloc_free.h"
#include "../inc/nal_unit.h"
#include "../inc/const_defines.h"
//This function set all frame_info data for the last submitted frame
E_ERR init_frame_info(encoder_context_t *p_enc)
{
    int IDR_flag = -1;
    FRAME_ENC_INFO *p_frame_info;
    RC_CONTEXT    *p_rc = &p_enc->rc_context;
    E_ERR err = ERR_SUCCESS;

    p_frame_info = &p_enc->frame_info;

    if (p_rc->curr_frame_type == CUVRC_I_FRAME)
    {
        p_enc->force_iframe = 1;
        p_enc->generate_sps_pps_nalus = 1;  // enable sps/pps header for all Iframes
        if(p_enc->enable_signature_generation == 1)
	  {
            p_enc->generate_sei_nalu = 1;  
          } 
    }

    if ((p_enc->frames_cnt == 0) || (p_enc->force_iframe))
    {
        IDR_flag = 1;
    }

    if(p_rc->curr_frame_type == CUVRC_PSEUDO_I_FRAME)
    {
      p_enc->pseudo_i_frame = 1;
    }

    if (IDR_flag < 0)
    {
        IDR_flag = (p_enc->intra_period == 0) ?  0 : ((p_enc->frames_cnt - p_enc->last_IDR_frame) > p_enc->intra_period);
    }

    if (IDR_flag) 
    {
        p_enc->last_assigned_id = 0;
        p_enc->last_IDR_frame = p_enc->frames_cnt;
        p_frame_info->slice_type = SLICE_I;
        p_frame_info->frame_num = 0;
        p_frame_info->frame_id  = 0;
    }
    else
    {
        p_enc->last_assigned_id += 2;
        p_frame_info->slice_type = SLICE_P;
        p_frame_info->frame_id  = p_enc->last_assigned_id;
    }

    p_frame_info->idr_flag   = IDR_flag;

    return err;
}

void enc_set_frames (encoder_context_t *p_enc)
{
    yuv_frame_t *pSwap;
    REFERENCE_FRAMES_CREF_T *p_mc_swap;

    // Swap rec and ref frames
    pSwap = p_enc->pRefFrame;
    p_enc->pRefFrame = p_enc->pRecFrame;
    p_enc->pRecFrame = pSwap;
   
    // swap chroma mc buffers
    p_mc_swap = p_enc->reference_frames_0;
    p_enc->reference_frames_0 = p_enc->reference_frames_1;
    p_enc->reference_frames_1 = p_mc_swap;
 
}
    


void init_slice_params( encoder_context_t *p_enc)
{
    int nRefs;
    slice_header_params_t *p_slice;
    seq_parameter_set_rbsp_t *p_sps;
    pic_parameter_set_rbsp_t *p_pps;
    
    p_slice = &p_enc->slice_params;
    p_sps = &p_enc->SequenceParameterSet;
    p_pps = &p_enc->PictureParameterSet;

    p_slice->pic_parameter_set_id = 0;
    
    if (p_sps->frame_mbs_only_flag) 
    {
        p_slice->field_pic_flag = 0;
        p_slice->bottom_field_flag = 0;
    }
    else 
    { 
        //TODO - Interlace material
    }

    p_slice->qp = p_enc->frame_info.frame_qp;
    p_slice->slice_type = (slicetype_e) p_enc->frame_info.slice_type;

    if(p_slice->slice_type == SLICE_I)
    {
        nRefs = 0;
    }
    else
    {
        nRefs = 1;
    }

    if (nRefs > (int)p_pps->num_ref_idx_active[0])
    {
        nRefs = p_pps->num_ref_idx_active[0];
    }

    p_slice->num_ref_pic_active_fwd = nRefs;
    p_slice->first_mb_in_slice = 0;
    p_slice->idr_flag = p_enc->frame_info.idr_flag;
    
    if (p_slice->idr_flag) 
    {        
        if (p_enc->frames_cnt == 0)
        {
            p_slice->idr_pic_id = 0;
        }
        else
        {
            p_slice->idr_pic_id = ((p_slice->idr_pic_id + 1) & 0xFF);
        }
    }

    p_slice->frame_num = p_enc->frame_info.frame_num;
    p_slice->pic_order_cnt_lsb = p_enc->frame_info.frame_id;
    p_slice->pic_order_cnt_lsb &= (1 << p_sps->log2_max_pic_order_cnt_lsb) - 1;
    p_slice->ref_pic_list_reordering_flag[0] = 0;
    p_slice->ref_pic_list_reordering_flag[1] = 0;
    p_slice->long_term_reference_flag = 0;
 
    if (p_slice->idr_flag) 
    {
        p_slice->no_output_of_prior_pics_flag = 0;
    }
    else  
    {
        p_slice->adaptive_ref_pic_buffering_flag = 0;
    }
    
    p_slice->cabac_init_idc = 0; // cabac_init_idc

    // loopfilter
    p_slice->disable_deblocking_filter_idc = p_enc->loopfilter_params.disable_flag;
    p_slice->slice_alpha_c0_offset = p_enc->loopfilter_params.alpha_c0_offset;
    p_slice->slice_beta_offset     = p_enc->loopfilter_params.beta_offset;
}

void encode_vui(bitstream_t *bitstream, vui_info_t *vui_info)
{
	int timing_info_present_flag = 1;
	if (vui_info->sar_height == 0 || vui_info->sar_width == 0)
		put_bits(bitstream, 1, 0);
	else { //aspect_ratio info present
		put_bits(bitstream, 1, 0);
		put_bits(bitstream, 8, 255);
		put_bits(bitstream, 16, vui_info->sar_width);
		put_bits(bitstream, 16, vui_info->sar_height);
	}
	put_bits(bitstream, 1, 0);//(overscan_info_present_flag = 0
	put_bits(bitstream, 1, 0);// video_signal_type_present_flag  = 0) 
	put_bits(bitstream, 1, 0);// chroma_loc_info_present_flag = 0 ) 
	put_bits(bitstream, 1, timing_info_present_flag);
	if (timing_info_present_flag) { // timing_info_present_flag ) 
		put_bits(bitstream, 32, vui_info->num_units_in_tick);
		put_bits(bitstream, 32, vui_info->time_scale);
		put_bits(bitstream, 1, vui_info->fixed_frame_rate_flag);
	}
	put_bits(bitstream, 1, 0); //nal_hrd_parameters_present_flag
	put_bits(bitstream, 1, 0); //vcl_hrd_parameters_present_flag
	put_bits(bitstream, 1, 1); //pic_struct_present_flag 
	put_bits(bitstream, 1, 0); //bitstream_restriction_flag
}


int output_sps_nalu (seq_parameter_set_rbsp_t *pSPS, bitstream_t *pBitstream)
{
	unsigned int i;
	int reserved_zero5 = 0;

    // start code
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 1);

    // NAL type
	put_bits(pBitstream, 8, ((NALU_TYPE_SPS)|32)); //32 here means (nal_ref_idc==1)
	put_bits(pBitstream, 8, pSPS->profile_idc);

	put_bits(pBitstream, 1, pSPS->constrained_set0_flag);
	put_bits(pBitstream, 1, pSPS->constrained_set1_flag);
	put_bits(pBitstream, 1, pSPS->constrained_set2_flag);
	put_bits(pBitstream, 5, reserved_zero5);
	assert (reserved_zero5 == 0);

	put_bits(pBitstream, 8, pSPS->level_idc);

	write_unsigned_uvlc(pBitstream, pSPS->seq_parameter_set_id);
	write_unsigned_uvlc(pBitstream, pSPS->log2_max_frame_num - 4);
	write_unsigned_uvlc(pBitstream, pSPS->pic_order_cnt_type);
	// POC200301
	if (pSPS->pic_order_cnt_type == 0)
	{
		write_unsigned_uvlc(pBitstream, pSPS->log2_max_pic_order_cnt_lsb - 4);
	}
	else if (pSPS->pic_order_cnt_type == 1)
	{
		put_bits(pBitstream, 1, pSPS->delta_pic_order_always_zero_flag);
		write_signed_uvlc(pBitstream, pSPS->offset_for_non_ref_pic);
		write_signed_uvlc(pBitstream, pSPS->offset_for_top_to_bottom_field);
		write_unsigned_uvlc(pBitstream, pSPS->num_ref_frames_in_pic_order_cnt_cycle);

		for(i=0; i<pSPS->num_ref_frames_in_pic_order_cnt_cycle; i++)
		{
			write_signed_uvlc(pBitstream, pSPS->offset_for_ref_frame[i]);
		}
	}

	write_unsigned_uvlc(pBitstream, pSPS->num_ref_frames);
	put_bits(pBitstream, 1, pSPS->required_frame_num_update_behaviour_flag);
	write_unsigned_uvlc(pBitstream, pSPS->pic_width_in_mbs - 1);
	write_unsigned_uvlc(pBitstream, pSPS->pic_height_in_map_units - 1);
	put_bits(pBitstream, 1, pSPS->frame_mbs_only_flag);
	if (!pSPS->frame_mbs_only_flag)
	{
		put_bits(pBitstream, 1, pSPS->mb_adaptive_frame_field_flag);
	}
	put_bits(pBitstream, 1, pSPS->direct_8x8_inference_flag);

	put_bits(pBitstream, 1, pSPS->frame_cropping_flag);
	if (pSPS->frame_cropping_flag) 
	{
		write_unsigned_uvlc(pBitstream, pSPS->frame_cropping_rect_left_offset);
		write_unsigned_uvlc(pBitstream, pSPS->frame_cropping_rect_right_offset);
		write_unsigned_uvlc(pBitstream, pSPS->frame_cropping_rect_top_offset);
		write_unsigned_uvlc(pBitstream, pSPS->frame_cropping_rect_bottom_offset);
	}

	put_bits(pBitstream, 1, pSPS->vui_parameters_present_flag);
	if (pSPS->vui_parameters_present_flag)
    {
		encode_vui(pBitstream, &pSPS->vui_info.extra_data);
    }
	put_bits(pBitstream, 1, 1); //traling 1
	byte_align_bitstream (pBitstream);

	return ERR_SUCCESS;
}

/**
 * Encode Picture Parameter Set (PPS) NAL unit.
 * Buffer must be large enough to receive encoded bits.
 * @param handle  - handle of the encoder instance (returned by vssh_enc_open);
 * @param spps_data - pointer to structure filled in with SPS_ID and output buffer
 *		specification;
 * @return VSSH_OK or error code;
 */
int output_pps_nalu (pic_parameter_set_rbsp_t *pPPS, bitstream_t *pBitstream)
{
	unsigned int i;
	int NumberBitsPerSliceGroupId;

    // start code
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 1);

    put_bits(pBitstream, 8, ((NALU_TYPE_PPS)|32));//32 here means (nal_ref_idc==1)

	write_unsigned_uvlc(pBitstream, pPPS->pic_parameter_set_id);
	write_unsigned_uvlc(pBitstream, pPPS->seq_parameter_set_id);
	put_bits(pBitstream, 1, pPPS->entropy_coding_mode);

	//! Note: as per JVT-F078 the following bit is unconditional.  If F078 is not accepted, then
	//! one has to fetch the correct SPS to check whether the bit is present (hopefully there is
	//! no consistency problem :-(
	//! The current encoder code handles this in the same way.  When you change this, don't forget
	//! the encoder!  StW, 12/8/02
	put_bits(pBitstream, 1, pPPS->pic_order_present_flag);
	write_unsigned_uvlc(pBitstream, pPPS->num_slice_groups-1);

	// FMO stuff begins here
	if (pPPS->num_slice_groups > 1)
    {
		write_unsigned_uvlc(pBitstream, pPPS->slice_group_map_type);
		if (pPPS->slice_group_map_type == 0) {
			for (i=0; i<pPPS->num_slice_groups; i++)
			{
				write_unsigned_uvlc(pBitstream, pPPS->run_length[i]-1); 
			}
		}
		else if (pPPS->slice_group_map_type == 2) {
			for (i=0; i<pPPS->num_slice_groups - 1/* ??? */; i++)
			{
				//! JVT-F078: avoid reference of SPS by using ue(v) instead of u(v)
				write_unsigned_uvlc(pBitstream, pPPS->top_left[i]);
				write_unsigned_uvlc(pBitstream, pPPS->bottom_right[i]);
			}
		}
		else if (pPPS->slice_group_map_type == 3 || pPPS->slice_group_map_type == 4 ||
			       pPPS->slice_group_map_type == 5) 	{
			put_bits(pBitstream, 1, pPPS->slice_group_change_direction_flag);
			write_unsigned_uvlc(pBitstream, pPPS->slice_group_change_rate - 1);
		}
		else if (pPPS->slice_group_map_type == 6) {
			NumberBitsPerSliceGroupId = 0;
			if (pPPS->num_slice_groups >= 4)
				NumberBitsPerSliceGroupId = 3;
			else if (pPPS->num_slice_groups >= 2)
				NumberBitsPerSliceGroupId = 2;
			else if (pPPS->num_slice_groups >= 1)
				NumberBitsPerSliceGroupId = 1;

					//! JVT-F078, exlicitly signal number of MBs in the map
			write_unsigned_uvlc(pBitstream, pPPS->num_slice_groups-1);
			for (i=0; i<pPPS->num_slice_groups; i++)
            {
				put_bits(pBitstream, NumberBitsPerSliceGroupId, pPPS->slice_group_id[i]);
            }
		}
	}
	// End of FMO stuff

	write_unsigned_uvlc(pBitstream, pPPS->num_ref_idx_active[0]-1);
	write_unsigned_uvlc(pBitstream, pPPS->num_ref_idx_active[1]-1);
	put_bits(pBitstream, 1, pPPS->weighted_pred_flag);
	put_bits(pBitstream, 2, pPPS->weighted_bipred_idc);
	write_signed_uvlc(pBitstream, pPPS->pic_init_qp - 26);
	write_signed_uvlc(pBitstream, pPPS->pic_init_qs - 26);
	write_signed_uvlc(pBitstream, pPPS->chroma_qp_index_offset);
	put_bits(pBitstream, 1, pPPS->deblocking_filter_parameters_present_flag);
	put_bits(pBitstream, 1, pPPS->constrained_intra_pred_flag);
	put_bits(pBitstream, 1, pPPS->redundant_pic_cnt_present_flag);

	put_bits(pBitstream, 1, 1); //traling 1
    byte_align_bitstream (pBitstream);  // byte align with zeros

	return ERR_SUCCESS;
}

int output_sei_nalu(user_data_sei_message_t *p_sei, bitstream_t *pBitstream)
{
  unsigned int i;

    // start code
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 0);
    put_raw_bits(pBitstream, 8, 1);

    put_bits(pBitstream, 8, NALU_TYPE_SEI);//(nal_ref_idc==0)
    put_bits(pBitstream, 8, 5);   // payload type is user data unregistered
    put_bits(pBitstream, 8, 36);  // payload size

    for(i = 0; i < 16; i++)
       put_bits(pBitstream, 8, 0);  //uuid_iso_iec_11578

    for(i = 0; i < 13; i++)
      put_bits(pBitstream, 8, p_sei->p_bs_info[i]);

      put_bits(pBitstream, 8, p_sei->min_mv_width);
      put_bits(pBitstream, 8, p_sei->min_mv_height);
      put_bits(pBitstream, 8, p_sei->subpel_lvl);
      put_bits(pBitstream, 8, p_sei->intra_in_inter);
      put_bits(pBitstream, 8, p_sei->max_ref_frames);
      put_bits(pBitstream, 8, p_sei->bkwd_refs);
      put_bits(pBitstream, 8, p_sei->deblock_non_ref);

      put_bits(pBitstream, 1, 1); //traling 1
      byte_align_bitstream (pBitstream);  // byte align with zeros

      return ERR_SUCCESS;
}

void init_sps (encoder_context_t *p_enc, seq_parameter_set_rbsp_t *p_sps)
{
	// (MBs/second, level). assuming 60 fps. Taken from Table A.1
	int MBs_level_array[NUM_LEVELS][2] = {{1485,1}, {1485,1}, {3000,11}, {6000,12}, {11880,13}, {11880,20},
	{19800,21}, {20250,22}, {40500,30}, {108000,31},{216000,32},{245760,40},{245760,41},
	{522240,42},{589824,50},{983040,51}};
	int pic_width_in_mbs;
	int frame_height_in_mbs;
	//TODO - more details and more sps from settings
	p_sps->profile_idc = PROFILE_IDC_BASELINE;
	pic_width_in_mbs = p_enc->width / MB_WIDTH;
	frame_height_in_mbs = p_enc->height / MB_HEIGHT;    
	{
		int num_mbs_in_frame, i;
		num_mbs_in_frame = pic_width_in_mbs*frame_height_in_mbs;
		// Its assumed that max fps support needed will be 60 fps.
		// Level is selected based on picture size and fps
		for(i = 0; i < NUM_LEVELS; i++)
		{
			if(MBs_level_array[i][0] > num_mbs_in_frame*MAX_FPS)
				break;
		}
		// If its beyond the highest profile, set it to highest profile. Not sure
		// if reporting error is better but its academic
		if(i == NUM_LEVELS)
			i = NUM_LEVELS - 1;
		p_sps->level_idc = MBs_level_array[i][1];
	}
	p_sps->constrained_set0_flag = (p_sps->profile_idc == PROFILE_IDC_BASELINE); // Subclause A.2.1 - OFF
	p_sps->constrained_set1_flag = (p_sps->profile_idc == PROFILE_IDC_MAIN);     // Subclause A.2.2 - OFF
	p_sps->constrained_set2_flag = (p_sps->profile_idc == PROFILE_IDC_EXTENDED); // Subclause A.2.3 - OFF

	p_sps->seq_parameter_set_id                   = 0;
	p_sps->log2_max_frame_num                     = 8;
	p_sps->pic_order_cnt_type                     = 0;
	
    // POC200301
	if (p_sps->pic_order_cnt_type == 0)
	{
		p_sps->log2_max_pic_order_cnt_lsb         = 8;
	}
	else if (p_sps->pic_order_cnt_type == 1)
	{
		p_sps->delta_pic_order_always_zero_flag      = 0;//u_1  (bs);
		p_sps->offset_for_non_ref_pic                = 0;//se_v (bs);
		p_sps->offset_for_top_to_bottom_field        = 0;//se_v (bs);
		p_sps->num_ref_frames_in_pic_order_cnt_cycle = 0;//ue_v (bs);
	}
    p_sps->num_ref_frames = p_enc->max_ref_frames_num;

	p_sps->required_frame_num_update_behaviour_flag = 0;
    p_sps->pic_width_in_mbs    = pic_width_in_mbs;
	p_sps->frame_mbs_only_flag = 1;
    p_sps->frame_height_in_mbs = frame_height_in_mbs;
	p_sps->pic_height_in_map_units  = p_sps->frame_height_in_mbs;
	if (p_sps->frame_mbs_only_flag == 0)
    { // this only applies with field coding
		p_sps->pic_height_in_map_units /= 2;
    }
	p_sps->mb_adaptive_frame_field_flag = 0;
	p_sps->direct_8x8_inference_flag = 1;       // YL - is this right? probably does not matter if we don't do b-frames

    // cropping
    p_sps->frame_cropping_flag = (p_enc->width != p_enc->pic_width ) || (p_enc->height != p_enc->pic_height);    // disabled
    p_sps->frame_cropping_rect_left_offset   = 0;
    p_sps->frame_cropping_rect_right_offset  = (p_enc->width - p_enc->pic_width) >> 1; 
	p_sps->frame_cropping_rect_top_offset    = 0; 
    p_sps->frame_cropping_rect_bottom_offset = (p_enc->height - p_enc->pic_height) >> 1;

    p_sps->vui_parameters_present_flag = 1;
	p_sps->vui_info.extra_data.fixed_frame_rate_flag = 1; //TODO if no
	p_sps->vui_info.extra_data.sar_height = 0; //unspecified aspect ratio
	p_sps->vui_info.extra_data.sar_width = 0;
    p_sps->vui_info.extra_data.time_scale = p_enc->desired_framerate; 
	p_sps->vui_info.extra_data.num_units_in_tick = 10000;
}

void init_pps(encoder_context_t *p_enc, pic_parameter_set_rbsp_t *p_pps)
{
	p_pps->slice_group_id = NULL;
	p_pps->pic_parameter_set_id  = 0;
	p_pps->seq_parameter_set_id  = 0;
	p_pps->entropy_coding_mode   = 0;

	p_pps->pic_order_present_flag  = 0;
	p_pps->num_slice_groups  = 1;

	p_pps->num_ref_idx_active[0] = p_enc->max_ref_frames_num;
	p_pps->num_ref_idx_active[1] = 1; // B-frames
	p_pps->weighted_pred_flag = 0;
	p_pps->weighted_bipred_idc = 0;
    p_pps->pic_init_qp =  p_enc->rc_context.pframe_qp;
	p_pps->pic_init_qs = p_pps->pic_init_qp;
	p_pps->chroma_qp_index_offset = 0;
	p_pps->deblocking_filter_parameters_present_flag = 1;
    p_pps->constrained_intra_pred_flag = p_enc->constrained_intra_flag;
	p_pps->redundant_pic_cnt_present_flag = 0;
}

char strm_info[] =  "GPU BITSTREAM";

void init_sei(encoder_context_t *p_enc, user_data_sei_message_t *p_sei)
{

  p_sei->p_bs_info       =  strm_info;
  p_sei->min_mv_width    =  MB_WIDTH;
  p_sei->min_mv_height   =  MB_HEIGHT;
  p_sei->subpel_lvl      =  0;
  p_sei->intra_in_inter  =  0;
  p_sei->max_ref_frames  =  1;
  p_sei->bkwd_refs       =  0;
}

void output_slice_header(encoder_context_t *p_enc, slice_header_params_t *p_slice)
{
    bitstream_t *p_bitstream = &p_enc->bitstream;
    seq_parameter_set_rbsp_t *p_sps = &p_enc->SequenceParameterSet;
    pic_parameter_set_rbsp_t *p_pps = &p_enc->PictureParameterSet;
    int qp = p_enc->frame_info.frame_qp;
    RC_CONTEXT *p_rc;    

    int nal_type  = p_slice->idr_flag ? NALU_TYPE_IDR : NALU_TYPE_SLICE;
    int nal_ref_idc = (p_slice->slice_type == SLICE_B) ? 0 : 32;
    unsigned int first_mb_in_slice;

        p_rc    =   &p_enc->rc_context;

        if(p_rc->curr_frame_type == CUVRC_NON_REFERENCE_P_FRAME_FORWARD) 
            nal_ref_idc = 0;

    // start code
    put_raw_bits(p_bitstream, 8, 0);
    put_raw_bits(p_bitstream, 8, 0);
    put_raw_bits(p_bitstream, 8, 0);
    put_raw_bits(p_bitstream, 8, 1);

    //For now we hard-code B-frames as not referenced. May be need to change this.
    put_bits(p_bitstream, 8, (nal_type|nal_ref_idc)); // 32 here means (nal_ref_idc==1) 

    // First Part of slice header...
    // call BIG UVLC because mb number could exceed 12 bits for HD video
    first_mb_in_slice = p_slice->first_mb_in_slice;

    if (first_mb_in_slice > (1 << 11))
    {
        write_unsigned_uvlc_big(p_bitstream, first_mb_in_slice);
    }
    else
    {
        write_unsigned_uvlc(p_bitstream, first_mb_in_slice);
    }
    write_unsigned_uvlc(p_bitstream, p_slice->slice_type);
    write_unsigned_uvlc(p_bitstream, p_slice->pic_parameter_set_id);

    // Rest of slice header...
    put_bits(p_bitstream, p_sps->log2_max_frame_num, p_slice->frame_num);

    if (!p_sps->frame_mbs_only_flag)
    {
        put_bits(p_bitstream, 1, p_slice->field_pic_flag);
    }
    if (p_slice->field_pic_flag)
    {
        put_bits(p_bitstream, 1, p_slice->bottom_field_flag);
    }

    if (p_slice->idr_flag)
    {
        write_unsigned_uvlc(p_bitstream, p_slice->idr_pic_id);
    }

    // POC200301
    if (p_sps->pic_order_cnt_type == 0)
    {
        put_bits(p_bitstream, p_sps->log2_max_pic_order_cnt_lsb, p_slice->pic_order_cnt_lsb);
        if (p_pps->pic_order_present_flag  ==  1 )
        {
            write_signed_uvlc (p_bitstream, p_slice->delta_pic_order_cnt_bottom);
        }
    }
    if (p_sps->pic_order_cnt_type == 1 && !p_sps->delta_pic_order_always_zero_flag)
    {
        write_signed_uvlc(p_bitstream, p_slice->delta_pic_order_cnt[0]);
        if (p_pps->pic_order_present_flag  ==  1 )
        {
            write_signed_uvlc(p_bitstream, p_slice->delta_pic_order_cnt[1]);
        }
    }
    //! redundant_pic_cnt is missing here
    if (p_pps->redundant_pic_cnt_present_flag)
    {
        put_bits(p_bitstream, 1, p_slice->redundant_pic_cnt);
    }

    if (p_slice->slice_type == SLICE_B)
    {
        put_bits(p_bitstream, 1, p_slice->direct_spatial_mv_pred_flag);
    }

    if ((p_slice->slice_type == SLICE_P) || (p_slice->slice_type == SLICE_SP) || (p_slice->slice_type == SLICE_B)) 
    {
        int send_active = (p_slice->num_ref_pic_active_fwd != (int)p_pps->num_ref_idx_active[0]);
        if (!send_active && p_slice->slice_type == SLICE_B)
        {
            send_active = (p_slice->num_ref_pic_active_bwd != (int)p_pps->num_ref_idx_active[1]);
        }
        put_bits(p_bitstream, 1, send_active);
        if (send_active)
        {
            write_unsigned_uvlc(p_bitstream, p_slice->num_ref_pic_active_fwd - 1);
            if (p_slice->slice_type == SLICE_B)
            {
                write_unsigned_uvlc(p_bitstream, p_slice->num_ref_pic_active_bwd - 1);
            }
        }
    }

    //refs reordering stuff: TODO more complicated
    if (p_slice->slice_type != SLICE_I && p_slice->slice_type != SLICE_SI)
    {
        assert(p_slice->ref_pic_list_reordering_flag[0] == 0);
        put_bits(p_bitstream, 1, p_slice->ref_pic_list_reordering_flag[0]);
        if (p_slice->slice_type==SLICE_B)
        {
            assert (p_slice->ref_pic_list_reordering_flag[1] == 0);
            put_bits(p_bitstream, 1, p_slice->ref_pic_list_reordering_flag[1]);
        }
    }

    if (nal_ref_idc)
    { 
        if (p_slice->idr_flag)
        {
            put_bits(p_bitstream, 1, p_slice->no_output_of_prior_pics_flag);
            put_bits(p_bitstream, 1, p_slice->long_term_reference_flag);
        }
        else
        {
            put_bits(p_bitstream, 1, p_slice->adaptive_ref_pic_buffering_flag);
        }
    }

    if (p_pps->entropy_coding_mode && p_slice->slice_type!=SLICE_I && p_slice->slice_type!=SLICE_SI)
    {
        write_unsigned_uvlc(p_bitstream, p_slice->cabac_init_idc);
    }

    write_signed_uvlc(p_bitstream, (qp - p_pps->pic_init_qp));

    if (p_pps->deblocking_filter_parameters_present_flag)
    {
        write_unsigned_uvlc(p_bitstream, p_slice->disable_deblocking_filter_idc);
        if (p_slice->disable_deblocking_filter_idc != 1)
        {
            write_signed_uvlc(p_bitstream, p_slice->slice_alpha_c0_offset/2);
            write_signed_uvlc(p_bitstream, p_slice->slice_beta_offset/2);
        }
    }
}
