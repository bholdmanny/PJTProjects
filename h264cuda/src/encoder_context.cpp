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
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../inc/cuve264b.h"
#include "../inc/encoder_context.h"
#include "../inc/mb_info.h"
#include "../inc/encoder_tables.h"
#include "../inc/const_defines.h"
#include "../inc/mem_alloc_free.h"
#include "../inc/residual_coding.h"

#include "../inc/cavlc_data.h"
#include "../inc/cavlc.h"

//-------------------------------------------------------------------

E_ERR create_ref_frames (encoder_context_t *p_enc)
{
    E_ERR err = ERR_SUCCESS;

    if (!err)
    {
        err = alloc_yuv_frame(
                              &p_enc->padded_ref_frame,
                              (p_enc->width + 2 * REFERENCE_FRAME_PAD_AMT),
                              (p_enc->height + 2 * REFERENCE_FRAME_PAD_AMT),
                              p_enc->width,
                              p_enc->height
                              );
    }


    if (!err)
    {
        err = alloc_yuv_frame(
                              &p_enc->padded_rec_frame,
                              (p_enc->width + 2 * REFERENCE_FRAME_PAD_AMT),
                              (p_enc->height + 2 * REFERENCE_FRAME_PAD_AMT),
                              p_enc->width,
                              p_enc->height
                              );
    }

    if (!err)
    {
      if(p_enc->s_nonpersist_individual_mem.is_app_mem == 1)
	{            
		p_enc->p_recon_flat_luma   = (unsigned char *)p_enc->s_nonpersist_individual_mem.p_mem;
        p_enc->p_recon_flat_chroma = (unsigned char *)((unsigned long)p_enc->s_nonpersist_individual_mem.p_mem + p_enc->scratch_mem_info.mem_offset_1);
	}
      else
	{
	  p_enc->p_recon_flat_luma   = (unsigned char *)getmem_1d_void((p_enc->width + 16) * p_enc->height);
          p_enc->p_recon_flat_chroma = (unsigned char *)getmem_1d_void((p_enc->width + 16) * p_enc->height / 2);

	    /* Set scratch mem ptr to indicate encoder lib has malloced internally */
            p_enc->s_nonpersist_individual_mem.p_mem = (int *)p_enc->p_recon_flat_luma;
	}

        if ((p_enc->p_recon_flat_luma == NULL) || (p_enc->p_recon_flat_chroma == NULL))
        {
            printf ("Failed to allocate p_recon_flat_luma and p_recon_flat_chroma\n");
            err = ERR_MEM;
        }
    }

    return err;
}

void destroy_ref_frames (encoder_context_t *p_enc)
{
    free_yuv_frame(&p_enc->padded_ref_frame);
    free_yuv_frame(&p_enc->padded_rec_frame);
    if(p_enc->s_nonpersist_individual_mem.is_app_mem == 0)
      {
        free_1D(p_enc->p_recon_flat_luma);
        free_1D(p_enc->p_recon_flat_chroma);

      }
}



int create_storage_for_me (ME_CONTEXT *p_me_context, int num_mb_x, int num_mb_y)
{
    E_ERR err = ERR_SUCCESS;
    int num_mbs_roundup = ROUNDUP_STREAM_PROCESSORS (num_mb_y) * num_mb_x + 16;

    p_me_context->p_me_mv_results = (CUVME_MV_RESULTS *) getmem_1d_void(sizeof (CUVME_MV_RESULTS) * (num_mbs_roundup + 2) * BLOCKS_PER_MB);

    if (p_me_context->p_me_mv_results == NULL)
    {
        err = ERR_MEM;
    }
    else
    {
        memset (p_me_context->p_me_mv_results, 0, sizeof (CUVME_MV_RESULTS) * (num_mbs_roundup + 2) * BLOCKS_PER_MB);
        p_me_context->p_me_mv_results += BLOCKS_PER_MB;
    }

    p_me_context->p_me_mb_info = (CUVME_MB_INFO *) getmem_1d_void(sizeof (CUVME_MB_INFO) * num_mbs_roundup);

    if (p_me_context->p_me_mb_info == NULL)
    {
        err = ERR_MEM;
    }
    else
    {
        memset (p_me_context->p_me_mb_info, 0, sizeof (CUVME_MB_INFO) * num_mbs_roundup);
    }

    p_me_context->p_me_mb_characs  =  (CUVME_MB_CHARAC *)getmem_1d_void(sizeof(CUVME_MB_CHARAC) * num_mbs_roundup);

     if (p_me_context->p_me_mb_characs == NULL){
         printf("Failed to allocate memory for MB characs \n");
         err = ERR_MEM;
     }
    else
    {
      memset (p_me_context->p_me_mb_characs, 0, sizeof (CUVME_MB_CHARAC) * num_mbs_roundup);
    }


    return err;
}

#define COEFS_PER_BLOCK_FLAT    (16 + 4)       // 16 dct coeffs + 4 dc (all blocks duplicated)
E_ERR create_storage_for_coefficients (encoder_context_t *p_enc, S_TRANSFORM_COEFS  *p_transform_coefs, int width, int height, E_ENCODER_TYPE type)
{
    int NumMBHor;
    int NumMBVer;
    int NumMBs, num_mbs_padded;
    short *p_mem;
    int offset;
    int luma_dc_padded, chroma_dc_padded;
    int chroma_dct_padded;
    int num_mbs_chroma_padded;
    E_ERR err = ERR_SUCCESS;

    NumMBHor = width / MB_WIDTH;
    NumMBVer = height / MB_HEIGHT;
    NumMBs = NumMBHor * NumMBVer;

    // pDcCoeffs are padded avoid memcpy in residual coding
    // luma is padded to multiple of 2 macroblocks
    // chroma is padded to multiple of 4 macroblocks
    luma_dc_padded =  (NumMBHor) * (NumMBVer - 1)+ ((NumMBHor + 3) / 4) * 4;    // only need 2, but better over pad than under
    chroma_dc_padded = (NumMBHor) * (NumMBVer - 1)+ ((NumMBHor + 7) / 8) * 8;   // only need 4, but better over pad than under
    chroma_dct_padded = (NumMBHor) * (NumMBVer - 1)+ ((NumMBHor + 3) / 4) * 4;
      
//Currently removed the temporary coefficients storage for I frame by calling cavlc block context preparation kernel in the I frame intra pipeline itself	  
	if(type == ENCODER_CREF)
	{
	    p_transform_coefs->total_coef_bytes = (
	                                           width * height +                        // luma ac coefs 
	                                           2 * chroma_dct_padded * MB_TOTAL_SIZE_C +       // chroma ac coefs
	                                           luma_dc_padded * NUM_DC_COEFS_PER_MB +          // luma dc coefs
	                                           2 * chroma_dc_padded * NUM_DC_COEFS_PER_MB_C      // chroma dc coefs
	                                           ) * sizeof (short);
	    p_mem = (short *) malloc (p_transform_coefs->total_coef_bytes);
	        
	    if (p_mem == NULL)
	    {
	        printf("Couldn't allocate memory for coefficients!");
	        err = ERR_MEM;
	    }
	}
    // For residual coding kernels to work
    // 1. pDctCoefs_u has lower addrest than DctCoefs_v
    // 2. pDcCoefs_u has lower address than DcCoefs_v

    // For cavlc index streams to work, the following need to be true
    // 1. pDcCoefs has the lowest address 

    if (!err)
    {
//Currently removed the temporary coefficients storage for I frame by calling cavlc block context preparation kernel in the I frame intra pipeline itself	  	
		if(type == ENCODER_CREF)
		{
	        p_transform_coefs->pDcCoefs    = p_mem;             offset = luma_dc_padded * NUM_DC_COEFS_PER_MB;    
	        p_transform_coefs->pDctCoefs   = p_mem + offset;    offset += width * height; 
	        p_transform_coefs->pDctCoefs_u = p_mem + offset;    offset += chroma_dct_padded * MB_TOTAL_SIZE_C;
	        p_transform_coefs->pDctCoefs_v = p_mem + offset;    offset += chroma_dct_padded * MB_TOTAL_SIZE_C;
	        p_transform_coefs->pDcCoefs_u  = p_mem + offset;    offset += chroma_dc_padded * NUM_DC_COEFS_PER_MB_C;
	        p_transform_coefs->pDcCoefs_v  = p_mem + offset;
		}
        num_mbs_padded = (NumMBHor) * (NumMBVer - 1) + ((NumMBHor + 3) / 4) * 4;
	num_mbs_chroma_padded = (((NumMBHor * NumMBVer) + 3) / 4) * 4; 

        if(p_enc->s_nonpersist_individual_mem.is_app_mem == 1)
	  {
            p_transform_coefs->p_coef_flat_luma   = (short *)((unsigned long)p_enc->s_nonpersist_individual_mem.p_mem + p_enc->scratch_mem_info.mem_offset_3);
            p_transform_coefs->p_coef_flat_chroma = (short *)((unsigned long)p_enc->s_nonpersist_individual_mem.p_mem + p_enc->scratch_mem_info.mem_offset_4);
	  }
	else
	  {
	     p_transform_coefs->p_coef_flat_luma   = (short *)getmem_1d_void(sizeof(short) * COEFS_PER_BLOCK_FLAT * 16 * num_mbs_padded);
             p_transform_coefs->p_coef_flat_chroma = (short *)getmem_1d_void(sizeof(short) * COEFS_PER_BLOCK_FLAT * BLOCKS_PER_MB_C * 2 * num_mbs_chroma_padded);
	  }

        if (p_transform_coefs->p_coef_flat_luma == NULL || p_transform_coefs->p_coef_flat_chroma == NULL)
        {
            printf ("Failed to allocat p_coef_flat_luma and p_coef_flat_chroma\n");
            err = ERR_MEM;
        }
    }

    return err;
}

void destroy_storage_for_me (ME_CONTEXT *p_me_context)
{
    if (p_me_context->p_me_mv_results != NULL)
    {
      free_1D(p_me_context->p_me_mv_results - BLOCKS_PER_MB);
    }
    free_1D(p_me_context->p_me_mb_info);
    free_1D(p_me_context->p_me_mb_characs);
}

E_ERR init_encoder(encoder_context_t * p_enc, E_ENCODER_TYPE enc_type,int slice_num)
{
    E_ERR err = ERR_SUCCESS;
    user_data_sei_message_t *p_sei = &p_enc->sei_rbsp;

    set_encoder_name(p_enc, "undefined");

    switch (enc_type)
    {
    case ENCODER_CREF:
        //p_enc->fns = g_cref_fns;
        err = (E_ERR)cuvme_open (&p_enc->me_context.me_handle);
        if (err)
        {
            printf ("init_encoder: error in cuvme_open()");
        }
        err = (E_ERR)cuvrc_open (&p_enc->rc_context.rc_handle, CUVRC_H264);
        if (err)
        {
            printf ("init_encoder: error in cuvrc_open()");
        }
        break;
    default:
        printf("init_encoder(): Invalid encoder type specified!");
        err = ERR_ARG;
    }

    // intitialize me
    {
        ME_CONTEXT *p_me = &p_enc->me_context;
        
        p_me->me_level = ME_MODE_TWENTY;

        if (!err)
        {

            err = (E_ERR)cuvme_set_app_type (p_me->me_handle, H264);        
        }
        if (!err)
        {
            err = (E_ERR)cuvme_set_me_mode (p_me->me_handle, ME_MODE_TWENTY);        
        }
        if (!err)
        {
            err = (E_ERR)cuvme_set_search_range(p_me->me_handle, 32, 32);
        }
        if (!err)
        {
            err = (E_ERR)cuvme_set_num_mvs(p_me->me_handle, BLOCKS_PER_MB);
        }

    }

    //intialize rc
    {
        RC_CONTEXT *p_rc = &p_enc->rc_context;

        if (!err)
        {
            err = (E_ERR)cuvrc_get_current_frame_type (p_rc->rc_handle, &p_rc->curr_frame_type);
            p_rc->prev_frame_type       =   p_rc->curr_frame_type;            
        }

    }

    if (!err)
    {
		//ÉèÖÃsliceÊýÁ¿
		p_enc->i_slice_num = slice_num;
        // reset encoder
        p_enc->type = enc_type;

        p_enc->frames_cnt = 0;
        p_enc->frame_info.forced_qp = 0;

        p_enc->intra_period = 0;
        p_enc->last_IDR_frame = 0;
        p_enc->slice_in_frame = 0;
        p_enc->constrained_intra_flag = 0;
        p_enc->generate_sps_pps_nalus = 0;
    
        // set default values for loopfilter
        p_enc->loopfilter_params.disable_flag = 0;
        p_enc->loopfilter_params.disable_cross_slice_flag = 0;
        p_enc->loopfilter_params.alpha_c0_offset = 0;
        p_enc->loopfilter_params.beta_offset     = 0;

        // set default values for me
        p_enc->max_ref_frames_num        = 1;
        // set default rc info
        p_enc->frame_info.frame_qp          = DEFAULT_MIN_QP;
        p_enc->rc_context.iframe_qp         = DEFAULT_IFRAME_QP;
        p_enc->rc_context.pframe_qp         = DEFAULT_PFRAME_QP;
        p_enc->rc_context.min_QP            = DEFAULT_MIN_QP;
        p_enc->rc_context.max_QP            = DEFAULT_MAX_QP;
        p_enc->rc_context.rc_mode           = 0;
        p_enc->rc_context.target_bitrate    = DEFAULT_BITRATE;
        p_enc->rc_context.target_framerate  = DEFAULT_FRAMERATE;
        p_enc->sps_pps_rate          =   DEFAULT_IFRAME_RATE;

        p_enc->rc_need_reset = 1;

        p_enc->pic_width  = 0;
        p_enc->pic_height = 0;
        p_enc->width  = 0;
        p_enc->height = 0;
        p_enc->pRecFrame   = NULL;
        p_enc->pRefFrame   = NULL;
        
        p_enc->p_source_frame = NULL;
        p_enc->p_input_frame = NULL;
        p_enc->input_format = 0;
        p_enc->input_format_changed = 0;
        p_enc->input_needs_conversion = 1;
        if(p_enc->type == ENCODER_CREF)
        {
            p_enc->input_needs_conversion = 0;
        }
        
        p_enc->enable_bitstream_generation = 1;
        p_enc->own_input_frame = 0;
        p_enc->own_output_buffer = 0;

        // set some default values
        p_enc->desired_bitrate = 300000;
        p_enc->desired_framerate = 60*10000;
        p_enc->estimated_output_buffer_size = 1200000;
        p_enc->intra_period = 0;
        p_enc->intra_prediction_selection = 0;

        // This turns on Intra4x4 for P-frames for all MBs (unless someone sets this later on for each frame)
        p_enc->avg_mb_sad             = 0;    
        p_enc->mb_adapt_qp_on         = 0;
        p_enc->intra_mb_level         = 5;
        p_enc->pseudo_i_frame         = 0;
        p_enc->me_search_range_x      = 32;
        p_enc->me_search_range_y      = 32;
        p_enc->non_ref_frame_interval = 0;
        p_enc->roi_on                 = 0;

        p_enc->s_nonpersist_individual_mem.p_mem         = NULL;
        p_enc->s_nonpersist_individual_mem.size          = 0;
        p_enc->s_nonpersist_individual_mem.is_app_mem    = 0;
        p_enc->s_nonpersist_individual_mem.app_mod_flag  = 0;
        p_enc->s_nonpersist_individual_mem.last_width    = 0;
        p_enc->s_nonpersist_individual_mem.last_height   = 0;

		p_enc->loopfilter_params.deblock_mode             = 0;
		p_enc->generate_sei_nalu                          = 0;
        p_enc->enable_signature_generation                = 0;
        p_sei->deblock_non_ref                            =  1;
    }
    return err;
}

void set_encoder_name
(
 encoder_context_t *p_enc,
 char *name
)
{
    if (strlen(name) > ENCODER_NAME_SIZE) {
        printf("set_encoder_name: warning: length of name (%s) must be less than %d characters\n", name, ENCODER_NAME_SIZE);
        memcpy(p_enc->name, name, ENCODER_NAME_SIZE);
        p_enc->name[ENCODER_NAME_SIZE] = '\0';
        printf("set_encoder_name: warning: truncating length name to %d characters, from %s to %s\n", ENCODER_NAME_SIZE, name, p_enc->name);
    } else {
        strcpy(p_enc->name, name);
    }
}

void set_encode_dimensions (encoder_context_t *p_enc, int pic_width, int pic_height)
{
    p_enc->pic_width  = pic_width;
    p_enc->pic_height = pic_height;

    p_enc->width = (pic_width + 15) & (~15);
    p_enc->height = (pic_height + 15) & (~15);
}

void set_intra_prediction (encoder_context_t *p_enc, int intra_prediction_selection)
{
    if(intra_prediction_selection >= 0 && intra_prediction_selection <= 1)
    {
        p_enc->intra_prediction_selection = intra_prediction_selection;
    }
    else
    {
        printf ("WARNING - Unsupported intra prediction mode %d. Set to default 0\n", intra_prediction_selection);
        p_enc->intra_prediction_selection = 0;
    }
}

int set_frame_qp
// Force the next frame to be coded using a fixed qp
(
   encoder_context_t *p_enc, int qp
)
{
    if (qp < p_enc->rc_context.min_QP)
    {
        qp = p_enc->rc_context.min_QP;
    }
    if (qp > p_enc->rc_context.max_QP)
    {
        qp = p_enc->rc_context.max_QP;
    }

    p_enc->frame_info.frame_qp = qp;
    
    return 1;
}

void set_loopfilter_params (encoder_context_t *p_enc, int alpha_c0_offset, int beta_offset)
{
    p_enc->loopfilter_params.alpha_c0_offset = alpha_c0_offset;
    p_enc->loopfilter_params.beta_offset     = beta_offset;
}

// all memory allocation goes here, including all memory required for streams
E_ERR encoder_mem_alloc (encoder_context_t *p_enc, int alloc_input, int alloc_me)
{
    int width;
    int height;
    int num_mb_x;
    int num_mb_y;
    int num_mbs;

    int vframe_offset_in;
    int vframe_offset_rec;
    int vframe_dct_offset;
    int vframe_dc_offset;

    int i, j;
    unsigned int factor, error;	
    float       fraction, x, y;
    int edged_width, edged_height;


    E_ERR err = ERR_SUCCESS;

    width  = p_enc->width;
    height = p_enc->height;

    num_mb_x = width / MB_WIDTH;
    num_mb_y = height / MB_HEIGHT;
    num_mbs  = num_mb_x * num_mb_y;
	p_enc->width_mb = num_mb_x;
	p_enc->height_mb = num_mb_y;

    edged_width =  width + 2 * MB_WIDTH;
    edged_height = height + 2 * MB_HEIGHT; 

    p_enc->frame_info.num_mbs = num_mbs;

    // reference frames
    err = create_ref_frames (p_enc);

    if (!err)
    {
        //assign reconstruction frame and reference frames
        p_enc->pRecFrame = &p_enc->padded_rec_frame;
        p_enc->pRefFrame = &p_enc->padded_ref_frame;

	if(p_enc->s_nonpersist_individual_mem.is_app_mem == 1) /* replace with app memory */
	  {
            alloc_empty_yuv_frame(&p_enc->inter_pred_frame, width, height, width, height); 
            p_enc->inter_pred_frame.y = (unsigned char *)((unsigned long)p_enc->s_nonpersist_individual_mem.p_mem + p_enc->scratch_mem_info.mem_offset_2);
            p_enc->inter_pred_frame.u = (unsigned char *)((unsigned long)p_enc->inter_pred_frame.y + (width * height));
            p_enc->inter_pred_frame.v = (unsigned char *)((unsigned long)p_enc->inter_pred_frame.u + (width/2 * height/2));
	  }
	else
	  {
             err = alloc_yuv_frame(&p_enc->inter_pred_frame, width, height, width, height);
	  }
    }

    if(!err)
    {
        
      p_enc->p_mb_qps =  (int*)malloc(((num_mbs + 3)/4 * 4) * BLOCKS_PER_MB * sizeof(unsigned int) );

       if (p_enc->p_mb_qps == NULL){
           printf("Failed to allocate memory for Adaptive MB Qps\n");
           err = ERR_MEM;
       }
       else
       {
         memset(p_enc->p_mb_qps, 0, ((num_mbs + 3)/4 * 4) * BLOCKS_PER_MB);
       } 


       p_enc->p_qp_list	 = (int *) malloc(10 * sizeof(int));

       if (p_enc->p_qp_list == NULL){
           printf("Failed to allocate memory for QP list \n");
           err = ERR_MEM;
       }

    }

    if (!err)
    {
        if (alloc_input)
        {
            if (p_enc->input_needs_conversion)
            {
                err = alloc_yuv_frame (&p_enc->input_frame_source, width, height, width, height);
            }
            else
            {
                err = alloc_yuv_frame (&p_enc->input_frame, width, height, width, height);
            }
            p_enc->own_input_frame = 1;
        }
        else
        {
            p_enc->own_input_frame = 0;
        }
    }

    if (!err && p_enc->input_needs_conversion)
    {
        unsigned char *p_yuv;
        
        alloc_empty_yuv_frame (&p_enc->input_frame, width, height, width, height);
        
        p_yuv = (unsigned char *) getmem_1d_void(width*height + width * (height / 2 + 16)); // imglib conversion function needs 8 lines of padding for chroma
        
        if (p_yuv == NULL)
        {
            printf("alloc_yuv_frame(): error allocating memory\n");
            err = ERR_MEM;
        }

        if (!err)
        {
            p_enc->input_frame.y = p_yuv;
            p_enc->input_frame.u = p_yuv + width * height;
            p_enc->input_frame.v = p_yuv + width * height + width * height / 4;
        
            p_enc->input_frame.buffer_owner = 1;
        }
    }

    if (!err)
    {
        //allocations for col-flat MC - no protections for small stuff for time being..
        p_enc->reference_frames_0 = (REFERENCE_FRAMES_CREF_T *) malloc(sizeof(REFERENCE_FRAMES_CREF_T));
        if (p_enc->reference_frames_0 == NULL)
        {
            err = ERR_MEM;
        }
      
        p_enc->reference_frames = p_enc->reference_frames_0;

        p_enc->reference_frames_1 = (REFERENCE_FRAMES_CREF_T *) malloc(sizeof(REFERENCE_FRAMES_CREF_T));
        if (p_enc->reference_frames_1 == NULL)
        {
            err = ERR_MEM;
        }
     
   

    }

    if (!err)
    {
        REFERENCE_FRAMES_CREF_T *p_mc_stor;
        int padded_img_size;

        p_mc_stor = p_enc->reference_frames_0;
        p_mc_stor->num_of_reference_frames = 1;
        p_mc_stor->reference_frame_list = (COLFLAT_REF_FRM_T *) 
            malloc(sizeof(COLFLAT_REF_FRM_T) * p_mc_stor->num_of_reference_frames);

        if (p_mc_stor->reference_frame_list == NULL)
        {
            err = ERR_MEM;
        }
        else
        {
            padded_img_size = (width + (2 * REFERENCE_FRAME_PAD_AMT)) * 
                              (height + (2 * REFERENCE_FRAME_PAD_AMT)) * 
                              p_mc_stor->num_of_reference_frames;
		// COLFLTRM           
			if(p_enc->type == ENCODER_CREF)
			{
	            p_mc_stor->reference_frame_list[0].luma_component = 
		      (unsigned char *) getmem_1d_void(padded_img_size);
	            
	            // chroma (both components)
	            p_mc_stor->reference_frame_list[0].chroma_component = 
		      (unsigned char *) getmem_1d_void(padded_img_size / 2);
	            
	            if ( (p_mc_stor->reference_frame_list[0].luma_component == NULL) ||
	                 (p_mc_stor->reference_frame_list[0].chroma_component == NULL) )
	            {
	                err = ERR_MEM;
	            }
	            else
	            {
	                memset ( p_mc_stor->reference_frame_list[0].luma_component,   0, padded_img_size);
	                memset ( p_mc_stor->reference_frame_list[0].chroma_component, 0, padded_img_size/2);                
	            }
			}
			else
			{
	            p_mc_stor->reference_frame_list[0].chroma_component = 
		      (unsigned char *) getmem_1d_void(padded_img_size / 2);
	            
	            if ((p_mc_stor->reference_frame_list[0].chroma_component == NULL) )
	            {
	                err = ERR_MEM;
	            }
	            else
	            {
	                memset ( p_mc_stor->reference_frame_list[0].chroma_component, 0, padded_img_size/2);                
	            }
              }
        }

	/* Alloc second ref frame buffer */


        p_mc_stor = p_enc->reference_frames_1;
        p_mc_stor->num_of_reference_frames = 1;
        p_mc_stor->reference_frame_list = (COLFLAT_REF_FRM_T *) malloc(sizeof(COLFLAT_REF_FRM_T) * p_mc_stor->num_of_reference_frames);
        if (p_mc_stor->reference_frame_list == NULL)
        {
            err = ERR_MEM;
        }
        else
        {
            padded_img_size = (width + (2 * REFERENCE_FRAME_PAD_AMT)) * 
                              (height + (2 * REFERENCE_FRAME_PAD_AMT)) * 
                              p_mc_stor->num_of_reference_frames;
		// COLFLTRM           
            if(p_enc->type == ENCODER_CREF)
            {
	      p_mc_stor->reference_frame_list[0].luma_component = //(unsigned char *) malloc (padded_img_size);
		(unsigned char *) getmem_1d_void(padded_img_size);
	            
	            // chroma (both components)
	      p_mc_stor->reference_frame_list[0].chroma_component = //(unsigned char *) malloc (padded_img_size / 2);
		(unsigned char *) getmem_1d_void(padded_img_size / 2);
	            
	            if ( (p_mc_stor->reference_frame_list[0].luma_component == NULL) ||
	                 (p_mc_stor->reference_frame_list[0].chroma_component == NULL) )
	            {
	                err = ERR_MEM;
	            }
	            else
	            {
	                memset ( p_mc_stor->reference_frame_list[0].luma_component,   0, padded_img_size);
	                memset ( p_mc_stor->reference_frame_list[0].chroma_component, 0, padded_img_size/2);                
	            }
            }
            else
            {
               p_mc_stor->reference_frame_list[0].chroma_component = 
		 (unsigned char *) getmem_1d_void(padded_img_size / 2);
	            
               if ((p_mc_stor->reference_frame_list[0].chroma_component == NULL) )
               {
                  err = ERR_MEM;
               }
               else
               {
                 memset ( p_mc_stor->reference_frame_list[0].chroma_component, 0, padded_img_size/2);                
               }
            }
	}
    }

    if (!err)
    {
        err = (E_ERR)create_storage_for_me (&p_enc->me_context, num_mb_x, num_mb_y);
    }
	 if (!err)
    {
        // memory for storing coefficients
      err = create_storage_for_coefficients (p_enc, &p_enc->transform_coefs, width, height, p_enc->type);
    }

    if (!err)
    {
        if (p_enc->type == ENCODER_CREF)
        {
            // allocate memory for MBInfo -- one extra MB at the beginning and one at the end for CAVLC
            p_enc->pBlkMBInfo  = (S_BLK_MB_INFO *) malloc ((num_mbs + 2) * BLOCKS_PER_MB * sizeof(S_BLK_MB_INFO));
         
            if (p_enc->pBlkMBInfo == NULL) {
                printf("Couldn't allocate memory for macroblock info data structure storage!\n");
                err = ERR_MEM;
            } else {
                memset(p_enc->pBlkMBInfo, 0, (num_mbs + 2) * BLOCKS_PER_MB * sizeof(S_BLK_MB_INFO));
                p_enc->pBlkMBInfo += BLOCKS_PER_MB;
            }
        }
    }

    if (!err)
    {
        p_enc->p_blk_mb_info = (S_BLK_MB_INFO_COMPRESSED *) getmem_1d_void((num_mbs + 2) * BLOCKS_PER_MB * sizeof(S_BLK_MB_INFO_COMPRESSED));
        if (p_enc->p_blk_mb_info == NULL) {
            printf("Couldn't allocate memory for macroblock info data structure storage!\n");
            err = ERR_MEM;
        } else {
            memset(p_enc->p_blk_mb_info, 0, (num_mbs + 2) * BLOCKS_PER_MB * sizeof(S_BLK_MB_INFO_COMPRESSED));
            p_enc->p_blk_mb_info += BLOCKS_PER_MB;
        }
    }

    if (!err)
    {
		for(int k =0 ; k < p_enc->i_slice_num;k++)
		{
			// initialize mb info
			for (j = 0; j < num_mb_y/ p_enc->i_slice_num; j++)
			{
				for (i = 0; i < num_mb_x; i++)
				{
					if (p_enc->type == ENCODER_CREF)
					{
						InitMBInfo(&p_enc->pBlkMBInfo[(j * num_mb_x + i + (num_mb_y*num_mb_x*k)/p_enc->i_slice_num) * BLOCKS_PER_MB], i, j, num_mb_x, num_mb_y,k);
					}
					InitMBInfoCompressed(&p_enc->p_blk_mb_info[(j * num_mb_x + i+ (num_mb_y*num_mb_x*k)/p_enc->i_slice_num) * BLOCKS_PER_MB], i, j, num_mb_x, num_mb_y);
				}
			}
		}

        // allocate and initialize residual coding related buffers
        vframe_offset_in  = p_enc->input_frame.v - p_enc->input_frame.u;
        vframe_offset_rec = p_enc->pRecFrame->v - p_enc->pRecFrame->u;
        vframe_dct_offset = (p_enc->transform_coefs.pDctCoefs_v - p_enc->transform_coefs.pDctCoefs_u) * sizeof(short);
        vframe_dc_offset  = (p_enc->transform_coefs.pDcCoefs_v - p_enc->transform_coefs.pDcCoefs_u) * sizeof(short);

        /* MC  factor init */
   	    /* Calculation of FACTOR and ERROR for MC Optimization */

	    fraction    =   (1/(float)num_mb_x); 
	    factor      =   (unsigned int)(fraction * FRAC_MUL); //magic number is 1 << 20;
	    x           =   (num_mbs * factor)/ (float)FRAC_MUL;
	    y           =   ((num_mb_y - x) );
	    error       =   (unsigned int)(y * FRAC_MUL);

        /* Store the calculated values into the encoder context */
        p_enc->mul_fact  =  factor;
        p_enc->mul_err   =  error;

    }

    if (!err)
      {
	p_enc->ptr_filter_coefs = (int *) malloc(MAX_FILTER_COEFFS  * MAX_NUM_FILTER_SETS * sizeof(int));
        if (p_enc->ptr_filter_coefs == NULL)
	  {
	  err = ERR_MEM;
	  }
	else
	  {
	    /* Filter Taps 1 */
            p_enc->ptr_filter_coefs[0] = 0;
            p_enc->ptr_filter_coefs[1] = 0;
            p_enc->ptr_filter_coefs[2] = 128;
            p_enc->ptr_filter_coefs[3] = 0;
            p_enc->ptr_filter_coefs[4] = 0;
	    /* Filter Taps 2 */
            p_enc->ptr_filter_coefs[5] = -1;
            p_enc->ptr_filter_coefs[6] = 7;
            p_enc->ptr_filter_coefs[7] = 116;
            p_enc->ptr_filter_coefs[8] = 7;
            p_enc->ptr_filter_coefs[9] = -1;
	    /* Filter Taps 3 */
            p_enc->ptr_filter_coefs[10] = -2;
            p_enc->ptr_filter_coefs[11] = 16;
            p_enc->ptr_filter_coefs[12] = 100;
	    p_enc->ptr_filter_coefs[13] = 16;
            p_enc->ptr_filter_coefs[14] = -2;
	    /* Filter Taps 4 */
            p_enc->ptr_filter_coefs[15] = 0;
            p_enc->ptr_filter_coefs[16] = 26;
            p_enc->ptr_filter_coefs[17] = 76;
            p_enc->ptr_filter_coefs[18] = 26;
            p_enc->ptr_filter_coefs[19] = 0;
	    /* Filter Taps 5 */
            p_enc->ptr_filter_coefs[20] = 3;
            p_enc->ptr_filter_coefs[21] = 30;
            p_enc->ptr_filter_coefs[22] = 62;
            p_enc->ptr_filter_coefs[23] = 30;
            p_enc->ptr_filter_coefs[24] = 3;
	  }
      }

    if (!err)
    {
        err = init_deblock_context(p_enc);
    }

    if (!err)
    {
        err = init_cavlc_context(p_enc);
    }

    if (!err)
    {
        err = (E_ERR)cuvme_init(p_enc->me_context.me_handle, p_enc->width, p_enc->height, 0);
    }

    if (!err)
    {
         err = (E_ERR)cuvme_set_search_range(p_enc->me_context.me_handle, p_enc->me_search_range_x, p_enc->me_search_range_y); 
    }

    if (!err)
    {
        err = (E_ERR)cuvrc_init(p_enc->rc_context.rc_handle);
    }

    if (!err)
    {
      err =  (E_ERR)cuvrc_configure_dimensions(p_enc->rc_context.rc_handle, p_enc->width, p_enc->height, (CUVRC_FRAME_FORMAT)0);
    }

    p_enc->generate_sps_pps_nalus = 1;    // this should be set back to zero after new sps and pps are encoded

    return err;
}


void encoder_mem_free (encoder_context_t *p_enc)
{
    destroy_ref_frames (p_enc);

    if(p_enc->s_nonpersist_individual_mem.is_app_mem == 0)
      {
        free_yuv_frame(&p_enc->inter_pred_frame);
      }


    if (p_enc->own_input_frame)
    {
        free_yuv_frame(p_enc->p_input_frame);
        p_enc->p_input_frame = NULL;
        if (p_enc->p_source_frame != NULL)
        {
            free_yuv_frame(p_enc->p_source_frame);
            p_enc->p_source_frame = NULL;
        }
        p_enc->own_input_frame = 0;
    }
    else
    {
        if (p_enc->input_needs_conversion)
        {
            if (p_enc->p_input_frame != NULL)
            {
                free_yuv_frame(p_enc->p_input_frame);
            }
            p_enc->p_input_frame = NULL;
        }
    }

    if (p_enc->own_output_buffer)
    {
        free(p_enc->bitstream.p_buffer);
        p_enc->bitstream.p_buffer = NULL;
        p_enc->own_output_buffer = 0;
    }

    // mbinfo
    if ((p_enc->type == ENCODER_CREF) && (p_enc->pBlkMBInfo != NULL))
    {
        free(p_enc->pBlkMBInfo - BLOCKS_PER_MB);
    }
    if (p_enc->p_blk_mb_info != NULL)
    {
      free_1D(p_enc->p_blk_mb_info - BLOCKS_PER_MB);
    }

    // me
    destroy_storage_for_me(&p_enc->me_context);

    if (p_enc->reference_frames != NULL)
    {
     if(p_enc->type == ENCODER_CREF)
       {
	 free_1D(p_enc->reference_frames_0->reference_frame_list[0].luma_component);
	 free_1D(p_enc->reference_frames_1->reference_frame_list[0].luma_component);
       }
        free_1D(p_enc->reference_frames_0->reference_frame_list[0].chroma_component);
        free (p_enc->reference_frames_0->reference_frame_list);
        free (p_enc->reference_frames_0);
        free_1D(p_enc->reference_frames_1->reference_frame_list[0].chroma_component);
        free (p_enc->reference_frames_1->reference_frame_list);
        free (p_enc->reference_frames_1);
    }

    // deblock
    free_deblock_context(p_enc);

    // cavlc
    free_cavlc_context(p_enc);

    cuvme_free(p_enc->me_context.me_handle);

    // RC
    free(p_enc->p_mb_qps);
    free(p_enc->p_qp_list);
    free(p_enc->ptr_filter_coefs);
    
    cuvrc_free(p_enc->rc_context.rc_handle);
}

void destroy_encoder(encoder_context_t *p_enc)
{
    encoder_mem_free(p_enc);
    cuvme_close(p_enc->me_context.me_handle);
    cuvrc_close(p_enc->rc_context.rc_handle);
    p_enc->me_context.me_handle = NULL;    
    p_enc->rc_context.rc_handle = NULL;
    p_enc->pRecFrame = NULL;
    p_enc->pRefFrame = NULL;
    p_enc->p_source_frame = NULL;
    p_enc->p_input_frame = NULL;
}
