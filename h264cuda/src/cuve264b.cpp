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


#include <assert.h>
#include <stdlib.h>
#include<stdio.h>
#include <string.h>
#include <time.h>

#include "../inc//cuve264b.h"
#include "../inc/encoder_context.h"
#include "../inc/output.h"
#include "../inc/h264_common.h"
#include "../inc/const_defines.h"
#include "../inc/encode_frame.h"
#include "../inc/rc.h"
#include "../inc/mem_alloc_free.h"
#include "../inc/cuve264b_utils.h"

// prototypes

static CUVE264B_ERROR convert_error(E_ERR err);

CUVE264B_ERROR cuve264b_open (CUVE264B_HANDLE *p_handle,int slice_num) 
{
    encoder_context_t *p_enc;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;
    E_ERR init_err;


    if ((p_handle == NULL))
    {
        printf ("NULL pointer passed to cuve264b_open()\n");
        err = CUVE264B_ERR_ARG;
        return err;
    }

    if (!err)
    {
        p_enc = (encoder_context_t *) malloc (sizeof(encoder_context_t));
        if (p_enc == NULL)
        {
            err = CUVE264B_ERR_MEM;
        }
    }

    if (!err)
    {
        init_err = init_encoder(p_enc, ENCODER_CREF, slice_num);
        err = convert_error(init_err);
    }

    if (!err)
    {
        *p_handle = (CUVE264B_HANDLE)p_enc;
    }
    else
    {
        *p_handle = NULL;
    }
	p_enc->slice[0] = p_enc;
	for( int i = 1; i <p_enc->i_slice_num; i++ )
        p_enc->slice[i] = (encoder_context_t *) malloc (sizeof(encoder_context_t));

    return err;
}

CUVE264B_ERROR cuve264b_close(
                              CUVE264B_HANDLE handle
                              //(pointer to) context of encoder instance
                              )
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if (handle == NULL) 
    {
        printf ("NULL pointer passed to cuve264b_close()\n");
        err = CUVE264B_ERR_ARG;
    }
    
    if (!err)
    {
        destroy_encoder(p_enc);
        free(p_enc);
        p_enc = NULL;
    }
    
    return err;
}

CUVE264B_ERROR cuve264b_set_input_image_format (CUVE264B_HANDLE handle, CUVE264B_E_IMG_FORMAT image_format)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;

    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if (handle == NULL) 
    {
        printf ("NULL pointer passed to cuve264b_close()\n");
        err = CUVE264B_ERR_ARG;
        return err;
    }

    if(!err)
    {
    
        if (p_enc->input_format != image_format)
        {
            p_enc->input_format_changed = 1;
        }    

        if (image_format == CUVE264B_IMG_FMT_YUV_RASTER)
        {
            p_enc->input_format = 0;
        }
        else if (image_format == CUVE264B_IMG_FMT_YUV_GPUBF)
        {
            p_enc->input_format = 1;
        }
        else
        {
            printf ("Unsupported format %d passed to cuve264b_set_input_image_format()\n", image_format);
            err = CUVE264B_ERR_ARG;
        }
    }
    
    return err;
}


CUVE264B_ERROR cuve264b_prep_encode_frame (
    CUVE264B_HANDLE handle,
    CUVE264B_IMAGE_SPECS  *p_input_image	//Input
)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    RC_CONTEXT *p_rc = &p_enc->rc_context;    
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;
	CUVRC_ERROR rc_err = CUVRC_ERR_SUCCESS;
    E_ERR alloc_err;
	int num_roi = 0;


    if ((p_input_image == NULL) || (handle == NULL))
    {
        printf ("NULL pointer passed to cuve264b_prep_encode_frame()\n");
        err = CUVE264B_ERR_ARG;
        return err;
    }


    if(!err)
    {
        if((p_input_image->width <= 0 ) || (p_input_image->height <= 0) || (p_input_image->width < 128) || (p_input_image->height < 96))
        {
         printf("Invalid width height params:\n");
         err = CUVE264B_ERR_ARG;
        }
    }

        // Do not proceed if memory for any of the scratch/persistent sections is NULL and is_app_mem = 1
        // Indicates that the Application wants to set memory but is setting the wrong size.
	
        if( (p_enc->s_nonpersist_individual_mem.p_mem == NULL && p_enc->s_nonpersist_individual_mem.is_app_mem == 1)){
			printf("App Memory not set correctly. Exiting from encode\n");
			err = CUVE264B_ERR_MEM;
		}

    if (!err)
    {

        // Figure out the encoding dimension and whether we need to re-allocate memory
        if ((p_input_image->width != p_enc->pic_width) || (p_input_image->height != p_enc->pic_height) || (p_enc->input_format_changed != 0))
        {
            int need_alloc_input;

            // Coding dimension has changed. Now memory need to be re-allocated
            if (p_enc->pic_width != 0)
            {
                // free old memory 
                encoder_mem_free (p_enc);
            }

            // set new encode dimensions
            set_encode_dimensions (p_enc, p_input_image->width, p_input_image->height);

            
            if (p_enc->input_needs_conversion)
            {
                p_enc->p_source_frame = &p_enc->input_frame_source;
                p_enc->p_input_frame  = &p_enc->input_frame;
            }
            else
            {
                p_enc->p_source_frame = &p_enc->input_frame;
                p_enc->p_input_frame  = p_enc->p_source_frame;
            }

            // decide whether the given input buffer can be used directly without a copy
            need_alloc_input = !check_input_buffer_conformance_to_encoder(p_input_image, p_enc);

            if (!need_alloc_input)
            {
                alloc_empty_yuv_frame (p_enc->p_source_frame, p_input_image->buffer.buf_width, p_input_image->buffer.buf_height, p_input_image->width, p_input_image->height);
                p_enc->p_source_frame->y = p_input_image->buffer.p_y_buffer;
                p_enc->p_source_frame->u = p_input_image->buffer.p_u_buffer;
                p_enc->p_source_frame->v = p_input_image->buffer.p_v_buffer;
            }

            p_enc->rc_need_reset = 1;

            // allocate memory for encode
            alloc_err = encoder_mem_alloc (p_enc, need_alloc_input, 1);
            err = convert_error(alloc_err);

            p_enc->input_format_changed = 0;
        }
    }

    if (!err)
    {
        if (p_enc->p_source_frame->buffer_owner)
        {
            copy_and_pad_frame_for_encode (p_input_image, p_enc->p_source_frame);
        }
        else
        {
            p_enc->p_source_frame->y = p_input_image->buffer.p_y_buffer;
            p_enc->p_source_frame->u = p_input_image->buffer.p_u_buffer;
            p_enc->p_source_frame->v = p_input_image->buffer.p_v_buffer;
        }
        
        if(p_rc->prev_frame_type != CUVRC_NON_REFERENCE_P_FRAME_FORWARD) 
             enc_set_frames (p_enc);
    }
    return err;
}








//----------------------------------------------------------------------
CUVE264B_ERROR cuve264b_encode_frame
// this function actually encode the frame specified in the last call to
// cuve264b_prep_encode_frame()
(
    CUVE264B_HANDLE handle,         	                // (pointer to) context of encoder instance
    CUVE264B_BITSTREAM_BUFFER *p_output_buffer	// Input and output
)
//----------------------------------------------------------------------
{
//    int overflow;

     encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_FRAME_ENCODE_INFO inf;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;
    E_ERR enc_err, init_err;
	clock_t start,end;
    
    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_encode_frame()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {

      if (!err)
      {
        init_err =(E_ERR)init_output_bitstream_buffer(p_enc, p_output_buffer);
        err = convert_error(init_err);
      }

      if (!err)
      {                
        enc_err = (E_ERR)enc_begin_frame_encode (p_enc);
        err = convert_error(enc_err);
      }

	  start = clock();
      if (!err)
      {  
        //enc_err = encode_frame_multi_channel (&p_enc, 1);   //¶àÍ¨µÀ
		enc_err = encode_frame_cuda (p_enc);
        err = convert_error(enc_err);
      }
	  end = clock();
	  /*p_enc->new_timers.prep_encode_frame += (end - start);*/

      if (!err)
      {
        enc_err = (E_ERR) enc_end_frame_encode (p_enc);
        err = convert_error(enc_err);
      }

      if (!err)
      {
        cuve264b_get_frame_encode_info(handle, &inf);
      }

    }

    return err;
}


CUVE264B_ERROR cuve264b_set_name
(
 CUVE264B_HANDLE handle,
 char *name
)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;

    if ((handle == NULL) || (name == NULL))
    {
        printf ("NULL pointer passed to cuve264b_set_name()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
        set_encoder_name(p_enc, name);
    }
    
    return err;
}




CUVE264B_ERROR cuve264b_set_rate_control
(
    CUVE264B_HANDLE handle,
    CUVE264B_E_RC_MODE rc_mode
)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    RC_CONTEXT    *p_rc      = &p_enc->rc_context;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;

    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_rate_control()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
        p_rc->rc_mode = rc_mode;
        switch (rc_mode)
        {
        case CUVE264B_RC_VBR:
             cuvrc_configure_bitrate(p_rc->rc_handle, p_rc->target_bitrate, CUVRC_BITRATE_BPS, CUVRC_NORMAL_BITRATE);
             p_enc->frame_info.forced_qp = 1;
             break;
        case CUVE264B_RC_CBR:
             cuvrc_configure_bitrate(p_rc->rc_handle, p_rc->target_bitrate, CUVRC_BITRATE_BPS, CUVRC_NORMAL_BITRATE);
             p_enc->frame_info.forced_qp = 0;
             break;
        case CUVE264B_RC_CVBR:
            cuvrc_configure_bitrate(p_rc->rc_handle, p_rc->target_bitrate, CUVRC_BITRATE_BPS, CUVRC_MOTION_ADAPTIVE_BITRATE);
            p_enc->frame_info.forced_qp = 0;
            break;
        default:
            printf("Invalid rate control mode (%d) passed to cuve264b_set_rate_control()\n", rc_mode);
            err = CUVE264B_ERR_ARG;
        }
    }

    return err;
}

CUVE264B_ERROR cuve264b_set_target_bitrate (CUVE264B_HANDLE handle, int bitrate)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err       = CUVE264B_ERR_SUCCESS;
    RC_CONTEXT *p_rc         = &p_enc->rc_context;
    int frame_rate           = p_enc->rc_context.target_framerate;
    CUVE264B_E_RC_MODE rc_mode;


    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_target_bitrate()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err && (bitrate < 0))
    {
        printf ("Invalid bitrate (%d) passed to cuve264b_set_target_bitrate() (should be > 0)\n", bitrate);
        err = CUVE264B_ERR_ARG;
    }

     if(!err)
     {
        p_enc->rc_context.target_bitrate  = bitrate;
        rc_mode = (CUVE264B_E_RC_MODE) p_rc->rc_mode;
        switch (rc_mode)
        {
        case CUVE264B_RC_VBR:
             cuvrc_configure_bitrate(p_rc->rc_handle, p_rc->target_bitrate, CUVRC_BITRATE_BPS, CUVRC_NORMAL_BITRATE);
             break;
        case CUVE264B_RC_CBR:
             cuvrc_configure_bitrate(p_rc->rc_handle, p_rc->target_bitrate, CUVRC_BITRATE_BPS, CUVRC_NORMAL_BITRATE);
             break;
        case CUVE264B_RC_CVBR:
            cuvrc_configure_bitrate(p_rc->rc_handle, p_rc->target_bitrate, CUVRC_BITRATE_BPS, CUVRC_MOTION_ADAPTIVE_BITRATE);
            break;
        default:
            printf("Invalid rate control mode (%d) passed to cuve264b_set_target_bitrate()\n", rc_mode);
            err = CUVE264B_ERR_ARG;
        }
     }
    return err;
}


CUVE264B_ERROR cuve264b_set_target_framerate (CUVE264B_HANDLE handle, int framerate)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_target_framerate()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err && (framerate < 0))
    {
        printf ("Invalid framerate (%d) passed to cuve264b_set_target_framerate() (should be > 0)\n", framerate);
        err = CUVE264B_ERR_ARG;
    }

    if(!err)
    {
        p_enc->rc_context.target_framerate = framerate;      

        cuvrc_set_frame_rate(p_enc->rc_context.rc_handle, framerate);

    }

    return CUVE264B_ERR_SUCCESS;
}

CUVE264B_ERROR cuve264b_set_qp_values (CUVE264B_HANDLE handle, int min_qp, int max_qp, int iframe_qp, int pframe_qp)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;
    
    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_qp_values()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
        // min qp
        if (min_qp == CUVE264B_QP_NO_CHANGE)
        {
            min_qp = p_enc->rc_context.min_QP;
        }

        if (min_qp < MIN_QP || min_qp > MAX_QP )
        {
            min_qp = DEFAULT_MIN_QP;
        }
    
        if (p_enc->rc_context.min_QP != min_qp)
        {
            p_enc->rc_context.min_QP = min_qp;
            p_enc->rc_need_reset = 1;
        }

        // max qp
        if (max_qp == CUVE264B_QP_NO_CHANGE)
        {
            max_qp = p_enc->rc_context.max_QP;
        }

        if (max_qp < MIN_QP || max_qp > MAX_QP )
        {
            max_qp = DEFAULT_MAX_QP;
        }
    
        if (p_enc->rc_context.max_QP != max_qp)
        {
            p_enc->rc_context.max_QP = max_qp;
            p_enc->rc_need_reset = 1;
        }
    
        // iframe qp
        if (iframe_qp == CUVE264B_QP_NO_CHANGE)
        {
            iframe_qp = p_enc->rc_context.iframe_qp;
        }

        if (iframe_qp < p_enc->rc_context.min_QP)
        {
            iframe_qp = p_enc->rc_context.min_QP;
        }
        else if (iframe_qp > p_enc->rc_context.max_QP)
        {
            iframe_qp = p_enc->rc_context.max_QP;
        }

        if (p_enc->rc_context.iframe_qp != iframe_qp)
        {
            p_enc->rc_context.iframe_qp = iframe_qp;
            p_enc->rc_need_reset = 1;
        }

        // pframe qp
        if (pframe_qp == CUVE264B_QP_NO_CHANGE)
        {
            pframe_qp = p_enc->rc_context.pframe_qp;
        }

        if (pframe_qp < p_enc->rc_context.min_QP)
        {
            pframe_qp = p_enc->rc_context.min_QP;
        }
        else if (pframe_qp > p_enc->rc_context.max_QP)
        {
            pframe_qp = p_enc->rc_context.max_QP;
        }
    
        if (p_enc->rc_context.pframe_qp != pframe_qp)
        {
            p_enc->rc_context.pframe_qp = pframe_qp;
            p_enc->rc_need_reset = 1;
        }
    }

    return err;
}

CUVE264B_ERROR cuve264b_set_iframe_interval (CUVE264B_HANDLE handle, int interval)
{
    encoder_context_t   *p_enc  = (encoder_context_t *) handle;
    RC_CONTEXT          *p_rc   = &p_enc->rc_context;
    CUVE264B_ERROR err          = CUVE264B_ERR_SUCCESS;
    CUVRC_ERROR rc_err          = CUVRC_ERR_SUCCESS;


    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_iframe_interval()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err && (interval < 0))
    {
        printf ("Invalid value %d passed to cuve264b_set_iframe_interval(). Should be > 0\n", interval);
        err = CUVE264B_ERR_ARG;
    }

    if(!err)
    {
        rc_err = cuvrc_configure_iframe_period(p_rc->rc_handle, interval, CUVRC_PERIOD_FRAMES);

       if (rc_err)
       {
          printf ("RC returned error code %d", rc_err);
          err =  (CUVE264B_ERROR)ERR_FAILURE;
       }
    }

    return err;
}


CUVE264B_ERROR cuve264b_set_intra_prediction_level (CUVE264B_HANDLE handle, int intra_prediction_level)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_intra_prediction_level()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
        set_intra_prediction (p_enc, intra_prediction_level);
    }

    return err;
}

CUVE264B_ERROR cuve264b_set_motion_estimation_level  (CUVE264B_HANDLE handle, int me_level)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;

    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_motion_estimation_level()\n");
        err = CUVE264B_ERR_ARG;
    }


    if (!err)
    {
        if ((me_level > 32) || (me_level < 20))
        {
            printf ("Undefined ME level (%d) passed to cuve264b_set_motion_estimation_level()\n", me_level);
            err = CUVE264B_ERR_ARG;
        }
        else
        {
            p_enc->me_context.me_level = me_level;
            if (me_level >= 0)
            {
            	err = (CUVE264B_ERROR)cuvme_set_me_mode (p_enc->me_context.me_handle, me_level);
            }
        }
    }

    return err;
}

CUVE264B_ERROR cuve264b_set_loopfilter_parameters (CUVE264B_HANDLE handle, int alpha_c0_offset, int beta_offset)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;

    if (handle == NULL)
    {
        printf ("NULL pointer passed to cuve264b_set_loopfilter_parameters()\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
       if((alpha_c0_offset > 12) || (alpha_c0_offset < -12) || (beta_offset > 12) || (beta_offset < -12))
       {
            printf ("Invalid params (%d, %d) passed to cuve264b_set_loopfilter_parameters()\n", alpha_c0_offset, beta_offset);
            err = CUVE264B_ERR_ARG;        
       }
       else
       {
         alpha_c0_offset = (alpha_c0_offset >> 1) << 1;
         beta_offset     = (beta_offset >> 1) << 1;
         set_loopfilter_params (p_enc, alpha_c0_offset, beta_offset);
       }
    }

    return err;
}


CUVE264B_ERROR cuve264b_get_output_buffer
(
    CUVE264B_HANDLE handle,
    CUVE264B_BITSTREAM_BUFFER *p_output_buffer,
	int slice_num
)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if ((handle == NULL) || (p_output_buffer == NULL))
    {
        printf ("cuve264b_get_output_buffer() error: pointer to buffer is NULL\n");
        err = CUVE264B_ERR_FAILURE;
    }

    if (!err)
    {
        p_output_buffer->p_buffer = p_enc->slice[slice_num]->bitstream.p_buffer;
        p_output_buffer->total_num_bytes = p_enc->slice[slice_num]->bitstream.buffer_size;
        p_output_buffer->used_num_bytes = p_enc->slice[slice_num]->bitstream.p_buffer_curr - p_enc->slice[slice_num]->bitstream.p_buffer;
    }

    return err;
}



CUVE264B_ERROR cuve264b_get_psnr (CUVE264B_HANDLE handle, double *p_psnr_y, double *p_psnr_u, double *p_psnr_v)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if ((handle == NULL) || (p_psnr_y == NULL) || (p_psnr_u == NULL) || (p_psnr_v == NULL))
    {
        printf ("cuve264b_get_psnr() error: pointer to buffer is NULL\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
            calc_psnr (&p_enc->input_frame, p_enc->pRecFrame, p_psnr_y, p_psnr_u, p_psnr_v);
    }

    return err;
}

CUVE264B_ERROR cuve264b_get_frame_encode_info (CUVE264B_HANDLE handle, CUVE264B_FRAME_ENCODE_INFO *p_info)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if ((handle == NULL) || (p_info == NULL))
    {
        printf ("cuve264b_get_frame_encode_info() error: pointer to buffer is NULL\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
        p_info->frame_number = p_enc->frame_info.frame_num - 1;
        p_info->frame_type = (CUVE264B_E_FRAME_TYPE)p_enc->rc_context.prev_frame_type; // translate_slice_type (p_enc->frame_info.slice_type);
        p_info->num_bits   = p_enc->frame_info.num_bits;
        p_info->header_bits = p_enc->frame_info.header_bits;
        p_info->texture_bits = p_enc->frame_info.texture_bits;
        p_info->num_intra_mbs = p_enc->frame_info.num_intra_mb;
        p_info->num_encoded_mbs = p_enc->frame_info.num_encoded_mb;
        p_info->num_me_cands = p_enc->frame_info.num_me_cands;
        p_info->qp_used = p_enc->frame_info.frame_qp;
        p_info->num_slices = p_enc->i_slice_num;
        p_info->scene_detected = p_enc->frame_info.scene_detected;
    }

    return err;
}

CUVE264B_ERROR cuve264b_enable_deblocking_loopfilter (CUVE264B_HANDLE handle, int enable)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if (handle == NULL)
    {
        printf ("cuve264b_enable_deblocking_loopfilter() error: pointer to buffer is NULL\n");
        err = CUVE264B_ERR_ARG;
    }

    if (!err)
    {
        if (enable)
        {
            p_enc->loopfilter_params.disable_flag = 0;
            p_enc->loopfilter_params.deblock_mode = enable; 
        }
        else
        {
            p_enc->loopfilter_params.disable_flag = 1;
        }
    }

    return err;
}





CUVE264B_ERROR cuve264b_reset_timing (CUVE264B_HANDLE handle)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if (handle == NULL)
    {
        printf ("cuve264b_reset_timing() error: pointer to buffer is NULL\n");
        err = CUVE264B_ERR_ARG;
    }
	p_enc->new_timers.me_total = 0;
	p_enc->new_timers.rc_total = 0;
	p_enc->new_timers.iframe_residual = 0;
	p_enc->new_timers.cavlc_timers = 0;
	p_enc->new_timers.pframe_mc = 0;
	p_enc->new_timers.de_block = 0;
	p_enc->new_timers.pframe_total = 0;
	p_enc->new_timers.pframe_residual_luma = 0;
	p_enc->new_timers.pframe_residual_chroma = 0;
	p_enc->new_timers.pframe_residual_inter = 0;
	p_enc->new_timers.pframe_residual_intra = 0;
	p_enc->new_timers.encode_frame = 0;
	p_enc->new_timers.prep_encode_frame = 0;
    return err;

}

CUVE264B_ERROR cuve264b_print_timing (CUVE264B_HANDLE handle, int level)
{
    encoder_context_t *p_enc = (encoder_context_t *) handle;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;


    if (handle == NULL)
    {
        printf ("cuve264b_print_timing() error: pointer to buffer is NULL\n");
        err = CUVE264B_ERR_ARG;
    }

	printf("me_total %d\n",p_enc->new_timers.me_total);
	printf("rc_total %d\n",p_enc->new_timers.rc_total);
	printf("pframe_MC %d\n",p_enc->new_timers.pframe_mc);
	printf("pframe_total %d\n",p_enc->new_timers.pframe_total);
	printf("pframe_inter %d\n",p_enc->new_timers.pframe_residual_inter);
	printf("pframe_intra %d\n",p_enc->new_timers.pframe_residual_intra);
	printf("pframe_residual_luma %d\n",p_enc->new_timers.pframe_residual_luma);
	printf("pframe_residual_chroma %d\n",p_enc->new_timers.pframe_residual_chroma);
	printf("iframe_residual %d\n",p_enc->new_timers.iframe_residual);
	printf("cavlc_timers %d\n",p_enc->new_timers.cavlc_timers);
	printf("de_block %d\n",p_enc->new_timers.de_block);
	printf("encode_frame %d\n",p_enc->new_timers.encode_frame);
	printf("prep_encode_frame %d\n",p_enc->new_timers.prep_encode_frame);

    return err;
}


static CUVE264B_ERROR convert_error(E_ERR err)
{
    CUVE264B_ERROR api_errcode = (CUVE264B_ERROR)ERR_SUCCESS;

    switch (err)
    {
    case ERR_SUCCESS:
        break;
    case ERR_MEM:
        api_errcode = CUVE264B_ERR_MEM;
        break;
    case ERR_OVERFLOW:
        api_errcode = CUVE264B_ERR_OVERFLOW;
        break;
    default:
        api_errcode = CUVE264B_ERR_FATAL_INTERNAL;
    }

    return api_errcode;
}


