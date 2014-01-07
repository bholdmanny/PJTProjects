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
#include <string.h>
#include <stdio.h>

#include "../inc/me_context.h"
#include "../inc/mem_alloc_free.h"
#include "../inc/me.h"
#include "../inc/const_defines.h"
#include "../inc/h264_types.h"

const int QP2QUANT_MELIB[40]=
{
   1, 1, 1, 1, 2, 2, 2, 2,
   3, 3, 3, 4, 4, 4, 5, 6,
   6, 7, 8, 9,10,11,13,14,
  16,18,20,23,25,29,32,36,
  40,45,51,57,64,72,81,91
};

//====================================================================
CUVME_ERROR cuvme_close (CUVME_HANDLE me_handle)
//====================================================================
//This function frees the memory associated with the corresponding instance of ME
//--------------------------------------------------------------------
{
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if(me_handle == NULL)
        return CUVME_ERR_ARG;
    else
        free_1D(me_handle);

    return err;
}

//====================================================================
CUVME_ERROR cuvme_free (CUVME_HANDLE me_handle)
//====================================================================
//pvme_free frees all the memory allocated by the ME library.
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if(me_handle == NULL)
        return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t *)me_handle;
    if(p_ME_context->curr_state < INIT)
        return CUVME_ERR_FAILURE;

	//me_free_persistent_memory(p_ME_context);
	// If non persistent is allocated by ME lib then free it.
	if(p_ME_context->nonpersistent_mem_givenby_app_flag == 0)
	{
		err = me_free_non_persistent_memory(p_ME_context);
	}
	
    p_ME_context->curr_state = OPEN;
    return err;
}



//====================================================================
CUVME_ERROR cuvme_get_avg_var(CUVME_HANDLE handle,int *avg_var)
//====================================================================
// cuvme_get_avg_var gets the average variance for the MBs of a frame. 
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if((handle == NULL) || (avg_var == NULL))
        return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t *)handle;
	*avg_var = p_ME_context->avg_var;
	return err;
}


//====================================================================
CUVME_ERROR cuvme_init(CUVME_HANDLE me_handle, int width, int height, int dec_ref_avail)
// cuvme_init allocates memory required internally by the library
// (to store indices etc) and for the decimated frames.
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
	int nonpersistent_mem_size;
	
    if((me_handle == NULL) || (width == 0) || (height == 0))
        return CUVME_ERR_ARG;
    if((width%MB_WIDTH) || (height%MB_HEIGHT))
        return CUVME_ERR_ARG;
	if((width < 128) || (height < 96))
		return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t *)me_handle;
    if(p_ME_context->curr_state != OPEN)
        return CUVME_ERR_ARG;
	
	p_ME_context->meinit_isdone = 1;
    p_ME_context->width = width;
    p_ME_context->height = height;
    p_ME_context->dec_ref_avail = dec_ref_avail;
    p_ME_context->search_range_x = DEFAULT_SEARCH_RANGE_X;
    p_ME_context->search_range_y = DEFAULT_SEARCH_RANGE_Y;
	p_ME_context->AvgHRSAD = 0;
    p_ME_context->LastAvgHRSAD = 0;
	p_ME_context->direction_ref = CUVME_FORWARD_REF;
	p_ME_context->fwd_ref_frame_num = 0;
	p_ME_context->bwd_ref_frame_num = -1;
	p_ME_context->prevp_var = 0;
	p_ME_context->futp_var = 0;
	p_ME_context->prevp_mean = 0;
	p_ME_context->futp_mean = 0;
	p_ME_context->curr_mean = 0;
	p_ME_context->curr_var = 0;
	p_ME_context->ref_frame_distance_fwd = 0;
	p_ME_context->ref_frame_distance_bwd = 0;

   // Check if Application has allocated the non persistent memory. If not allocate it.
	if(p_ME_context->nonpersistent_mem_givenby_app_flag == 0)
	{
	    if((err = me_allocate_non_persistent_memory(p_ME_context)) != CUVME_ERR_SUCCESS)
		{
			printf("Error in Allocating memory for ME lib's non-persistent memory. Exiting.\n");
	        return err;
		}
	}

	// Assign the non persistent memory buffer pointers.
	// firsr calculate the offsets.
	err = me_calc_nonpersistentmem_offsets(p_ME_context, 
										 &nonpersistent_mem_size,
										 p_ME_context->width,
										 p_ME_context->height);	
	if(err != CUVME_ERR_SUCCESS)
		return err;
	// check if the application has allocated the required non persistent memory.
	if(p_ME_context->nonpersistent_mem_givenby_app_flag == 1)
	{
		if(nonpersistent_mem_size > p_ME_context->nonpersistent_mem_size_givenby_app)
		{
			printf("The non persistent memory allocated by Application is insufficient\n");
			return CUVME_ERR_MEM;
		}
	}
	// now assign the pointers
	err = me_assign_nonpersistentmem_pointers(p_ME_context);
	if(err != CUVME_ERR_SUCCESS)
		return err;
		
		
    if(err == CUVME_ERR_SUCCESS)
        p_ME_context->curr_state = INIT;

    p_ME_context->forw_quarter_res_ref[0] =
        p_ME_context->malloced_forw_quarter_res_ref[0];
    p_ME_context->forw_half_res_ref[0] = 
        p_ME_context->malloced_forw_half_res_ref[0];
    p_ME_context->back_quarter_res_ref[0] =
        p_ME_context->malloced_back_quarter_res_ref[0];
    p_ME_context->back_half_res_ref[0] = 
        p_ME_context->malloced_back_half_res_ref[0];
    return err;
}

//====================================================================
CUVME_ERROR cuvme_open (CUVME_HANDLE *me_handle)
//====================================================================
// cuvme_open opens an instance of a reference ME. The reference ME is a C version
// of the ME which runs entirely on PC.
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if(me_handle == NULL)
        return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t*)getmem_1d_void(sizeof(ME_context_t));
    if(p_ME_context == NULL)
        return CUVME_ERR_MEM;
    *me_handle = (void *)p_ME_context;
    memset(p_ME_context, 0, sizeof(ME_context_t));
    me_populate_default_context(p_ME_context, 1);
    p_ME_context->curr_state = OPEN;
    return err;
}

//====================================================================
CUVME_ERROR cuvme_set_app_type (CUVME_HANDLE me_handle, CUVME_APP_TYPE app_type)
//====================================================================
// CUVME_set_codec_ref sets the codec type for the ME. It is mainly used to determine
// the kind of motion compensation needed. Default is NONE.
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    p_ME_context = (ME_context_t *)me_handle;
    if((me_handle == NULL) || (app_type > H264) || (app_type < NONE))
        return CUVME_ERR_ARG;

    p_ME_context->prime_format = app_type;
    return err;
}


//====================================================================
CUVME_ERROR cuvme_set_qp(CUVME_HANDLE me_handle, int qp)
//====================================================================
// cuvme_set_qp sets the QP to be used for MVCost calculation. Default mode is 32
//--------------------------------------------------------------------
{
	//This function may be extended as necessary
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
	ME_context_t *p_ME_context = (ME_context_t *)me_handle;
    if(me_handle == NULL)
        return CUVME_ERR_ARG;
	p_ME_context->FrameQP = qp;
    return err;
}

//====================================================================
CUVME_ERROR cuvme_set_me_mode (CUVME_HANDLE me_handle, int me_mode)
//====================================================================
// cuvme_set_me_mode sets the mode of the ME. Each mode corresponds to set of different
// tools. Default mode is ZEROMVS.
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if((me_handle == NULL))
        return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t *)me_handle;
    p_ME_context->me_mode = me_mode;

    p_ME_context->me_quarter_res = 0;
    p_ME_context->me_half_res = 0;
	p_ME_context->me_twostep = 0;
	p_ME_context->me_twostep_plus = 0;
    p_ME_context->me_integer = 0;
    p_ME_context->me_subpel = 0;
    // Turn down the mode to previous valid me mode
    // CURRENTLY SUPPORTING 0,5,10,20,22,30,32
    if(p_ME_context->me_mode > 32)
            p_ME_context->me_mode = 32;    
    else if((p_ME_context->me_mode > 30) && (p_ME_context->me_mode < 32))
        p_ME_context->me_mode = 30;
    else if(p_ME_context->me_mode < 0)
        p_ME_context->me_mode = 0;
	if((p_ME_context->me_mode != 22) && (p_ME_context->me_mode != 32))
	{
		p_ME_context->me_mode = (p_ME_context->me_mode/5)*5;
		if(p_ME_context->me_mode != 5)
			p_ME_context->me_mode = (p_ME_context->me_mode/10)*10;
	}
    switch(p_ME_context->me_mode)
    {
    case 0:
        break;
    case 5:
        p_ME_context->me_half_res = 1;
        break;
    case 10:
        p_ME_context->me_quarter_res = 1;
        p_ME_context->me_half_res = 1;
        break;
    case 20:
	case 22:
        p_ME_context->me_quarter_res = 1;
        p_ME_context->me_half_res = 1;
        p_ME_context->me_twostep = 1;
        break;
    case 30:
    case 32:
        p_ME_context->me_quarter_res = 1;
        p_ME_context->me_half_res = 1;
        p_ME_context->me_twostep_plus = 1;
        break;
    default:
        err = CUVME_ERR_ARG;
        break;
    }
    return err;
}
//====================================================================
CUVME_ERROR cuvme_set_num_mvs (CUVME_HANDLE me_handle, int num_mvs)
// cuvme_open opens an instance of a reference ME. The reference ME is a C version
// of the ME which runs entirely on DSP MIPS.
//====================================================================
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if(me_handle == NULL)
        return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t *)me_handle;
    p_ME_context->num_mvs = num_mvs;
    return err;
}

//====================================================================
CUVME_ERROR cuvme_set_predicted_picture(CUVME_HANDLE handle, 
                                        CUVME_Y_BUFFER *p_pred_picture)
//====================================================================
// cuvme_set_predicted_picture passes the predicted picture pointer where ME library
// stores the predicted picture. This API should be called once 
// before every call to CUVME_search. p_pred_picture picture buffer should be of size
// (num_mbs_in_picture + STREAM_PROCESSORS)*LUMA_MB_SIZE bytes where LUMA_MB_SIZE is 256. Additional 
// memory of STREAM_PROCESSORS*LUMA_MB_SIZE bytes is needed to have an optimal implementation
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if((handle == NULL) || (p_pred_picture == NULL))
        return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t *)handle;
    p_ME_context->p_pred_picture = p_pred_picture;
    p_ME_context->flag_do_mc = 1;
    return err;
}

//====================================================================
CUVME_ERROR cuvme_set_reference_frame (CUVME_HANDLE me_handle, CUVME_Y_BUFFER *p_ref,
                                       char ref_frame_num, char direction)
//====================================================================
// cuvme_set_reference_frame passes the references frames to be used by CUVME_search. 
// This API should be called at once before every call to CUVME_search. 
// ref_frame_num indicates the frame number in multiple reference frames. direction 
// indicates whether the frame is forward or a backward reference. 0 indicates 
// forward and 1 indicates backward (used only in B frames). The p_ref
// buffer should be padded 16 bytes on all the sides. 

//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    p_ME_context = (ME_context_t *)me_handle;

    if((me_handle == NULL) || (p_ref == NULL))
        return CUVME_ERR_ARG;
    if(((direction != 0) && (direction != 1)) || (ref_frame_num != 0))
    {
        return CUVME_ERR_ARG;
    }
    if(p_ME_context->curr_state < INIT)
        return CUVME_ERR_FAILURE;
    if(direction == 0)
    {
        p_ME_context->num_forw_references += 1;
        if(p_ME_context->num_forw_references > MAX_FORW_REF_FRAMES)
            err = CUVME_ERR_ARG;
        p_ME_context->ptr_forw_ref_frame[ref_frame_num] =
            p_ref;
    }
    else
    {
		if(p_ME_context->enable_bwd_reference == 0)
			return CUVME_ERR_FAILURE;
        p_ME_context->num_back_references += 1;
        if(p_ME_context->num_back_references > MAX_BACK_REF_FRAMES)
            err = CUVME_ERR_ARG;
        p_ME_context->ptr_back_ref_frame[ref_frame_num] =
            p_ref;			
    }
    if(err == CUVME_ERR_SUCCESS)
        p_ME_context->curr_state = REF_SET;
    return err;

}

//====================================================================
CUVME_ERROR cuvme_set_return_mb_characteristics(CUVME_HANDLE handle, 
    CUVME_MB_CHARAC *p_mb_characs)
//====================================================================
// cuvme_set_return_mb_characteristics passes pointer to return the MB level
// characteristics like mean and median which can used for in encoder
// and for video analytics
//====================================================================
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if(handle == NULL)
        return CUVME_ERR_ARG;
    p_ME_context = (ME_context_t *)handle;
	p_ME_context->p_mb_characs = p_mb_characs;
	p_ME_context->get_mb_characs_flag = 1;
	return err;
}

//====================================================================
CUVME_ERROR cuvme_set_search_range (CUVME_HANDLE me_handle, int search_range_x,
                                    int search_range_y)
//====================================================================
// cuvme_set_search_range sets the search range of the ME. search_range_x is
// horizontal search range on both left and right and search_range_y is vertical
// search range on both top and bottom.
//--------------------------------------------------------------------
{
    ME_context_t *p_ME_context;
    CUVME_ERROR err = CUVME_ERR_SUCCESS;
    if((me_handle == NULL))
        return CUVME_ERR_ARG;
    if((search_range_x % MB_WIDTH) || (search_range_y % MB_WIDTH))
        return CUVME_ERR_ARG;
    if((search_range_x > MAX_SEARCH_RANGE_X) || (search_range_y > MAX_SEARCH_RANGE_Y))
        return CUVME_ERR_ARG;
    search_range_x = (search_range_x/16)*16; // multiple of 16
    search_range_y = (search_range_y/16)*16; // multiple of 16
    if(search_range_x < MIN_SEARCH_RANGE_X)
        search_range_x = MIN_SEARCH_RANGE_X;
    if(search_range_y < MIN_SEARCH_RANGE_Y)
        search_range_y = MIN_SEARCH_RANGE_Y;
    p_ME_context = (ME_context_t *)me_handle;
	p_ME_context->search_range_x = search_range_x;
	p_ME_context->search_range_y = search_range_y;
    return err;
}
