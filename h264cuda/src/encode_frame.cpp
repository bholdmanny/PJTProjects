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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../inc/cavlc.h"
//#include "../inc/cavlc_ref.h"
#include "../inc/h264_common.h"
#include "../inc/encoder_context.h"
#include "../inc/output.h"
#include "../inc/residual_coding.h"
#include "../inc/entropy_data.h"
#include "../inc/cavlc.h"
#include "../inc/deblock.h"
#include "../cuda/encode_cuda_cu.h"


int enc_begin_frame_encode(encoder_context_t *p_enc)
{
    E_ERR err = ERR_SUCCESS;

    RC_CONTEXT *p_rc;    
    int num_mb_hor, num_mb_ver;
    int new_qp;
    CUVRC_ERROR rc_err = CUVRC_ERR_SUCCESS;
    int scene_cut = 0;
    int curr_frame_type;
    int num_roi = 0;

    err = init_frame_info(p_enc);
   return err;
 }

int enc_end_frame_encode (encoder_context_t *p_enc)
{
    E_ERR err = ERR_SUCCESS;

    p_enc->frames_cnt++;
    //p_enc->frame_info.forced_qp = 0;
    p_enc->force_iframe = 0;
    p_enc->pseudo_i_frame = 0;

    return err;
}
void increment_frame_num(encoder_context_t *p_enc, int max_bits)
{
    p_enc->frame_info.frame_num++;
    p_enc->frame_info.frame_num &= (1 << max_bits) - 1;
}

//====================================================================
E_ERR encode_frame_cuda
(
    encoder_context_t *p_enc 
 )
{
    int bytes_in_pict;
    int bytes_in_buffer;
    RC_CONTEXT *p_rc;    
    int num_mb_hor, num_mb_ver;
    E_ERR err = ERR_SUCCESS;
    CUVRC_ERROR rc_err = CUVRC_ERR_SUCCESS;
	int num_roi,new_qp,curr_frame_type;
	clock_t start,end;
	
	bytes_in_buffer = 0;
    p_rc    =   &p_enc->rc_context;
	num_mb_hor    = p_enc->width / MB_WIDTH;
	num_mb_ver    = p_enc->height / MB_HEIGHT;
	p_rc    =   &p_enc->rc_context;

    if (p_enc->generate_sps_pps_nalus)
    {
          init_sps (p_enc, &p_enc->SequenceParameterSet);
          init_pps (p_enc, &p_enc->PictureParameterSet);
    }

    if(p_enc->generate_sei_nalu)
    {
		init_sei(p_enc, &p_enc->sei_rbsp);
	}

    init_slice_params (p_enc);

    /* set rc ptrs */
    rc_err = cuvrc_set_mb_level_pointers(p_rc->rc_handle, p_enc->p_mb_qps, p_enc->me_context.p_me_mb_characs, p_enc->p_qp_list);

	if(!rc_err)
	{
        if(p_enc->roi_on == 1)
		{
			if(p_rc->num_rois)
			{
				  for(num_roi = 0; num_roi < p_rc->num_rois; num_roi++) 
				  {
						if(!rc_err)
						rc_err = cuvrc_set_roi(p_enc->rc_context.rc_handle, p_enc->rc_context.roi_params[num_roi]);
						if (rc_err)
						{
								printf ("RC returned error code %d", rc_err);
								err = ERR_FAILURE;
						   return err;
						}      
					}//num_roi
			}
			else
			{  
				 printf("Error: ROI params not set \n");
				 err = ERR_FAILURE;
				 return err;
			}  
		}
	}

    if(!rc_err)
		rc_err = cuvrc_get_avg_frame_qp(p_rc->rc_handle, &new_qp);

    p_enc->frame_info.scene_detected = 0; 
    set_frame_qp(p_enc, new_qp);

	if((p_enc->frame_info.forced_qp == 1) && (p_enc->roi_on == 0))
	{
		p_enc->frame_info.frame_qp =  (p_enc->frame_info.slice_type == SLICE_I) ? p_enc->rc_context.iframe_qp : p_enc->rc_context.pframe_qp;
	}

	if(!rc_err)
	  rc_err = cuvrc_process_frame(p_rc->rc_handle, p_enc->frame_info.scene_detected);
	if(!rc_err)
	  rc_err = cuvrc_get_avg_frame_qp(p_rc->rc_handle, &p_enc->frame_info.frame_qp);
	/* get current frame type */
	if(!rc_err)
	{
		curr_frame_type = p_rc->curr_frame_type;
		rc_err = cuvrc_get_current_frame_type(p_rc->rc_handle, &p_rc->curr_frame_type);  

		if(curr_frame_type != p_rc->curr_frame_type)
		{
			/* Handle scene cut */
			p_enc->force_iframe = 0;
			p_enc->pseudo_i_frame = 0;
			err = init_frame_info(p_enc);
			init_slice_params (p_enc);
		}    
	}

	if (!err)
	{
		p_enc->frame_info.scene_detected=0;  

		if((p_enc->frame_info.forced_qp == 1) && (p_enc->roi_on == 0))
		{
			p_enc->frame_info.frame_qp =  (p_enc->frame_info.slice_type == SLICE_I) ? p_enc->rc_context.iframe_qp : p_enc->rc_context.pframe_qp;
		}
  
		output_stream_init(&p_enc->bitstream);

	 }
	// copy slice infomation 
	int i_bs_size = p_enc->bitstream.buffer_size / p_enc->i_slice_num;
	for(int i = 0; i < p_enc->i_slice_num; i++ )
	{
		encoder_context_t *penc = p_enc->slice[i];
		if( i > 0 )
		{
			memcpy( penc, p_enc, sizeof(encoder_context_t) );
			penc->bitstream.p_buffer += i*i_bs_size;
			output_stream_init(&penc->bitstream);
			penc->slice_in_frame = i;
		}
		penc->slice_params.first_mb_in_slice = (i *p_enc->height_mb/ p_enc->i_slice_num) * p_enc->width_mb;
		penc->first_mb = (i *p_enc->height_mb/ p_enc->i_slice_num) * p_enc->width_mb;
		penc->last_mb = ((i+1) * p_enc->height_mb / p_enc->i_slice_num) * p_enc->width_mb;
	}

	//Write sps pps and slice nal nuit header
	if (p_enc->enable_bitstream_generation) 
	{
			if (p_enc->generate_sps_pps_nalus)
			{
				output_sps_nalu (&p_enc->SequenceParameterSet, &p_enc->bitstream);
				output_pps_nalu (&p_enc->PictureParameterSet, &p_enc->bitstream);
				p_enc->generate_sps_pps_nalus = 0;
			 }
			if (p_enc->generate_sei_nalu)
			{
				output_sei_nalu (&p_enc->sei_rbsp, &p_enc->bitstream);
				p_enc->generate_sei_nalu = 0;
			}

			bytes_in_buffer = num_bytes_in_bitstream (&p_enc->bitstream);

			for(int i = 0; i < p_enc->i_slice_num; i++ )
			{
				encoder_context_t *penc = p_enc->slice[i];
				output_slice_header (
				   penc,
				   &penc->slice_params
				   );
			}
	}
	
	if ((p_enc->slice_params.slice_type != SLICE_I) || (p_enc->mb_adapt_qp_on == 1))
	{
		//start = clock();
		if (p_enc->me_context.me_level > 0)
		{
			cuvme_set_qp(p_enc->me_context.me_handle, p_enc->frame_info.frame_qp);
		}
	}
	
		

		start = clock();
		encode_cuda(p_enc);
		end = clock();
		p_enc->new_timers.encode_frame +=(end - start);

		start = clock();
		for(int i = 0; i < p_enc->i_slice_num; i++ )
		{
			encoder_context_t *penc = p_enc->slice[i];
			put_bits(&penc->bitstream, 1, 1);
			byte_align_bitstream (&penc->bitstream);
		}
		end = clock();
		p_enc->new_timers.encode_frame +=(end - start);


		start = clock();
		//deblock(p_enc);
		end = clock();
		p_enc->new_timers.de_block +=(end - start);

		if (!err)
		{    	
			int num_mbs    = num_mb_hor * num_mb_ver;
			bytes_in_pict  = output_stream_done(&p_enc->bitstream) - bytes_in_buffer;
			for(int i = 1; i < p_enc->i_slice_num; i++ )
			{
				encoder_context_t *penc = p_enc->slice[i];
				bytes_in_pict  += output_stream_done(&penc->bitstream);
			}

			 p_enc->frame_info.num_bits = 8 * bytes_in_pict;
			 /* set actual bits consumed by this frame */
			 cuvrc_set_previous_frame_actual_bits(p_rc->rc_handle, p_enc->frame_info.num_bits);
			 cuvrc_set_previous_frame_avg_sad(p_rc->rc_handle, p_enc->avg_mb_sad);
			 cuvrc_set_previous_frame_intra_mbs(p_rc->rc_handle, p_enc->frame_info.num_intra_mb);
			 cuvrc_set_previous_frame_num_large_mvs(p_rc->rc_handle, p_enc->frame_info.num_big_mvs);
			 cuvrc_set_previous_frame_skipped_mbs(p_rc->rc_handle, (num_mbs - p_enc->frame_info.num_encoded_mb));

		}
		if(!err)
		{

			if(p_rc->curr_frame_type != CUVRC_NON_REFERENCE_P_FRAME_FORWARD) 
			increment_frame_num(p_enc, p_enc->SequenceParameterSet.log2_max_frame_num);

			p_rc->prev_frame_type       =  p_rc->curr_frame_type;
			/* get next frame type */
			if(!rc_err)
			{
				 rc_err = cuvrc_get_next_frame_type(p_rc->rc_handle, &p_rc->curr_frame_type);
			}
		}

		if (rc_err)
		{
			printf ("RC returned error code %d", rc_err);
			err = ERR_FAILURE;
		}
		return err;
  }

