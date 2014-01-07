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

#include "../inc/me_context.h"
#include "../inc/me_common.h"
#include "../inc/mem_alloc_free.h"
//-------------------------------------------------------------------


ME_CANDIDATE_TUNE g_candidate_tune_values[] = {
    //CUL, IU,IV,IC,IL,SU,SC,  SKIP,  SO, CUL_S,SV,SU) 
    {TRUE, 13, 1, 4, 2, 7, 3,  TRUE, 344,  TRUE, 1, 2}, // Low Complexity
    {FALSE, 1, 2, 1, 2, 1, 1, FALSE, 344, FALSE, 3, 2}, // Normal Complexity
    {FALSE, 1, 4, 1, 2, 1,-2, FALSE, 344, FALSE, 4, 2}, // High Complexity
    {FALSE,13, 2, 4, 1, 7, 3, FALSE,   0, FALSE, 3, 2}, // Experimental 
};
//-------------------------------------------------------------------
// Local protocols
//-------------------------------------------------------------------

CUVME_ERROR me_init_src_dec_search_context(ME_context_t *p_ME_context)
{
    return CUVME_ERR_SUCCESS;
}

void me_populate_default_context(ME_context_t * p_ME_context, int CRef)
{
    p_ME_context->num_forw_references = 0;
    p_ME_context->num_back_references = 0;
    p_ME_context->num_lowres_forw_references = 0;
    p_ME_context->num_lowres_back_references = 0;
	p_ME_context->candidate_tune_values = g_candidate_tune_values[ME_CAND_NORMAL_COMPLEXITY];  // only normal_complexity will match with GPU builds.
    p_ME_context->num_mvs = 1;
    p_ME_context->num_sec_formats = 0;
    p_ME_context->flag_do_mc = 0;
    p_ME_context->store_dec_src_flag = 0;
    p_ME_context->FrameQP = 24; // default value, compromise between H264 
    p_ME_context->source_format = RASTER_ORDER;
	p_ME_context->scene_cut_early_exit = 0;
	p_ME_context->me_mode = 20;
	p_ME_context->get_mb_characs_flag = 0;
	p_ME_context->only_pic_characs = 0;
	p_ME_context->b_frames_since_last_p = 0;
	p_ME_context->enable_bwd_reference = 0; // By default ME lib assumes that backward reference frame will not be used.
    switch (CRef)
    {
    case 1:
        p_ME_context->CRef = 1;
        break;
    default:
        printf("cuve264b_set_encoder: Invalid encoder type specified!");
    }

}
//// This function will be called only when Application does not allocate the  non persistent memory.
CUVME_ERROR me_allocate_non_persistent_memory(ME_context_t * p_ME_context)
{

	CUVME_ERROR  err = CUVME_ERR_SUCCESS;
	int			 non_persistent_mem_size;
	
	if(p_ME_context->nonpersistent_mem_givenby_app_flag == 1)
	{
		printf("ME library trying to allocate non persistent memory. Application has already done it. Exiting\n");
		return CUVME_ERR_FAILURE;
	}
	err = me_calc_nonpersistentmem_offsets(p_ME_context,
											   &non_persistent_mem_size,
												p_ME_context->width,
												p_ME_context->height);
												
	p_ME_context->nonpersistent_mem_baseptr = (void *)getmem_1d_void(non_persistent_mem_size);
	if(p_ME_context->nonpersistent_mem_baseptr == NULL)
		return CUVME_ERR_MEM;
	
	return err;
}
CUVME_ERROR me_free_non_persistent_memory(ME_context_t * p_ME_context)
{
	if(p_ME_context->nonpersistent_mem_givenby_app_flag == 1)
	{
		printf("ME library trying to free non persistent memory, which was allocated by Application. Exiting\n");
		return CUVME_ERR_FAILURE;
	}

	free_1D(p_ME_context->nonpersistent_mem_baseptr);
	
	return CUVME_ERR_SUCCESS;
	
}
CUVME_ERROR me_calc_nonpersistentmem_offsets(ME_context_t * p_ME_context,
										   int *scratch_mem_size,
										   int width,
										   int height)
{
	int mem_size = 256; // neglecting the first 256 bytes
	CUVME_NONPERSISTENT_MEM_OFFSETS		*p_offsets;
	int length, numMBsFrame, numRoundedMBs;
	int pad_height, pad_width;
	int quarter_pad_width, quarter_pad_height;
	int half_pad_width, half_pad_height;
	int mbheight, mbwidth;

    if(height < 16*MB_HEIGHT)
        height = 16*MB_HEIGHT;
    numMBsFrame = (width/MB_WIDTH) * (height/MB_WIDTH);
    numRoundedMBs = ROUNDUP_STREAM_PROCESSORS((height/MB_WIDTH)) * (width/MB_WIDTH);	
	mbheight = height/MB_HEIGHT;
	mbwidth = width/MB_WIDTH;
    pad_width = width + 2 * MAX_SEARCH_RANGE_X; 
    pad_height = height + 2 * MAX_SEARCH_RANGE_Y;
    // This is done to allocate extra memory for the decimation in which
    // all clusters work on ROW_STRIP_DECIMATE rows
    pad_height = ((pad_height + 16*ROW_STRIP_DECIMATE + MAX_SEARCH_RANGE_Y)/
                (16*ROW_STRIP_DECIMATE))*(16*ROW_STRIP_DECIMATE);
    quarter_pad_width = pad_width/4;
    quarter_pad_height = pad_height/4;
    half_pad_width = pad_width/2;
    half_pad_height = pad_height/2;
	
	p_offsets = &p_ME_context->nonpersistent_mem_offsets;
    
	p_offsets->idx_north_mvmap_ping = (mem_size);
	mem_size += ALIGN_MEM_32BYTES((TARGET_INTEGER_STRIP_SIZE+2)*sizeof(coord_t));

	p_offsets->idx_north_mvmap_pong = (mem_size);
	mem_size += ALIGN_MEM_32BYTES((TARGET_INTEGER_STRIP_SIZE+2)*sizeof(coord_t));
	
	p_offsets->idx_p_temp_mvs = (mem_size);
	mem_size += ALIGN_MEM_32BYTES((( (MAX_FRAME_WIDTH/MB_WIDTH)*(MAX_FRAME_HEIGHT/MB_HEIGHT)*SIZE_OF_MVINFO )*sizeof(int)));
	
	p_offsets->idx_LowSad = (mem_size);
	mem_size += ALIGN_MEM_32BYTES(numRoundedMBs * sizeof(unsigned int));
	
	p_offsets->idx_nb = (mem_size);
	mem_size += ALIGN_MEM_32BYTES(numMBsFrame * NUM_SEARCH_LOCATIONS * sizeof(unsigned int));
	
	p_offsets->idx_integer_mvmap = mem_size;
	length = sizeof(CUVME_MV_RESULTS *)*(mbheight+2)+  sizeof(CUVME_MV_RESULTS)*(mbwidth+2)*(mbheight+2);
	mem_size += ALIGN_MEM_32BYTES(length);
	
	p_offsets->idx_ptr_mvs_local_0 = mem_size;
	mem_size += ALIGN_MEM_32BYTES((((width/MB_WIDTH) * (height/MB_HEIGHT)) + 16)* sizeof(CUVME_MV_RESULTS));

	p_offsets->idx_integer_mb_info = mem_size;
	length = sizeof(CUVME_MB_INFO*)*(mbheight+2) + sizeof(CUVME_MB_INFO)*(mbwidth+2)*(mbheight+2);
	mem_size += ALIGN_MEM_32BYTES(length);

	p_offsets->idx_CandidateCountInt = mem_size;
	mem_size += ALIGN_MEM_32BYTES(numMBsFrame * sizeof(unsigned int));
	
	p_offsets->idx_sum_vars_frame = mem_size;
	mem_size += ALIGN_MEM_32BYTES(16 * sizeof(unsigned int));
	
	p_offsets->idx_sum_mean_frame = mem_size;
	mem_size += ALIGN_MEM_32BYTES(16 * sizeof(unsigned int));
	
	p_offsets->idx_quarter_res_src = mem_size;
	mem_size += ALIGN_MEM_32BYTES(((width+2*REFERENCE_FRAME_PAD_AMT)/4) *  ((height+2*REFERENCE_FRAME_PAD_AMT)/4));
	
	p_offsets->idx_malloced_forw_quarter_res_ref_ptr  = mem_size;
	mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_Y_BUFFER));
	
	p_offsets->idx_malloced_forw_quarter_res_ref = mem_size;
	mem_size += ALIGN_MEM_32BYTES(quarter_pad_width * quarter_pad_height);
	
	p_offsets->idx_half_res_src  = mem_size;
	mem_size += ALIGN_MEM_32BYTES(((width+2*REFERENCE_FRAME_PAD_AMT)/2) *   ((height+2*REFERENCE_FRAME_PAD_AMT)/2));
	
	p_offsets->idx_malloced_forw_half_res_ref_ptr  = mem_size;
	mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_Y_BUFFER));
	
	p_offsets->idx_malloced_forw_half_res_ref  = mem_size;
	mem_size += ALIGN_MEM_32BYTES(half_pad_width * half_pad_height);
	
	p_offsets->idx_block_flat_yuv  = mem_size;
	mem_size += ALIGN_MEM_32BYTES(width*height + width * (height / 2 + 16));
	
	p_offsets->idx_block_flat_src  = mem_size;
	mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_Y_BUFFER));

//#ifndef NEW_LOW_RES  // the following set of memories are not required for NEW low res.
	p_offsets->idx_HRresults = mem_size;
	mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_MV_RESULTS)*numMBsFrame);
	
	p_offsets->idx_QRresults = mem_size;
	mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_MV_RESULTS)*numMBsFrame);

	p_offsets->idx_HR_MB_Info = mem_size;
	mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_MB_INFO)*numMBsFrame);

	p_offsets->idx_QR_MB_Info = mem_size;
	mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_MB_INFO)*numMBsFrame);	
//#endif
    if(p_ME_context->enable_bwd_reference)
	{
		p_offsets->idx_malloced_back_quarter_res_ref_ptr  = mem_size;
		mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_Y_BUFFER));
		
		p_offsets->idx_malloced_back_quarter_res_ref  = mem_size;
		mem_size += ALIGN_MEM_32BYTES(quarter_pad_width * quarter_pad_height);	
		
		p_offsets->idx_malloced_back_half_res_ref_ptr  = mem_size;
		mem_size += ALIGN_MEM_32BYTES(sizeof(CUVME_Y_BUFFER));
		
		p_offsets->idx_malloced_back_half_res_ref  = mem_size;
		mem_size += ALIGN_MEM_32BYTES(half_pad_width * half_pad_height);	
	}
	*scratch_mem_size  = mem_size;
	
	return CUVME_ERR_SUCCESS;
}
CUVME_ERROR assign_y_frame_me_ptr(
    CUVME_Y_BUFFER *p_frame,
    int buffer_width,
    int buffer_height,
    int active_width,
    int active_height,
    int offset_x,
    int offset_y,
	unsigned char *p_yuv
    )
{
    p_frame->y = p_yuv;
    p_frame->buffer_width = buffer_width;
    p_frame->buffer_height = buffer_height;
    p_frame->active_width = active_width;
    p_frame->active_height = active_height;
    p_frame->offset_x = offset_x;
    p_frame->offset_y = offset_y;
    return CUVME_ERR_SUCCESS;
}
CUVME_ERROR me_assign_nonpersistentmem_pointers(ME_context_t * p_ME_context)
{
	CUVME_NONPERSISTENT_MEM_OFFSETS		*p_offsets;
	CUVME_ERROR 		err;
	unsigned char 	*base_mem;
    int quarter_pad_width, quarter_pad_height;
    int half_pad_width, half_pad_height;
    int pad_width, pad_height, width, height;
	unsigned char *p_yuv;
	int i, y;
	CUVME_MB_INFO **mbinfo;
	int mbwidth, mbheight;
	CUVME_MV_RESULTS **mvmap;
	
    width = p_ME_context->width;
    height = p_ME_context->height;
    if(height < 16*MB_HEIGHT)
        height = 16*MB_HEIGHT;
    pad_width = width + 2 * MAX_SEARCH_RANGE_X; 
    pad_height = height + 2 * MAX_SEARCH_RANGE_Y;
    // This is done to allocate extra memory for the decimation in which
    // all clusters work on ROW_STRIP_DECIMATE rows
    pad_height = ((pad_height + 16*ROW_STRIP_DECIMATE + MAX_SEARCH_RANGE_Y)/
                (16*ROW_STRIP_DECIMATE))*(16*ROW_STRIP_DECIMATE);
    quarter_pad_width = pad_width/4;
    quarter_pad_height = pad_height/4;
    half_pad_width = pad_width/2;
    half_pad_height = pad_height/2;	
	mbwidth = width/MB_WIDTH;
	mbheight =height/MB_HEIGHT;
	
	// Align the base pointer to 32 bytes.
	base_mem = (unsigned char 	*)ALIGN_MEM_32BYTES((unsigned long)p_ME_context->nonpersistent_mem_baseptr);
	p_offsets = &p_ME_context->nonpersistent_mem_offsets;
    
	p_ME_context->twostep_plus_ctxt.north_mvmap_ping =
		(coord_t *)(base_mem + p_offsets->idx_north_mvmap_ping);

	p_ME_context->twostep_plus_ctxt.north_mvmap_pong =
		(coord_t *)(base_mem + p_offsets->idx_north_mvmap_pong);
	
	p_ME_context->p_temp_mvs = 	(int *)(base_mem + p_offsets->idx_p_temp_mvs );
	
	p_ME_context->LowSad = (unsigned int *)(base_mem + p_offsets->idx_LowSad);

	p_ME_context->nb = (unsigned int *)(base_mem + p_offsets->idx_nb);	

	mvmap = (CUVME_MV_RESULTS **)(base_mem + p_offsets->idx_integer_mvmap);

	if (mvmap != NULL)
    {
        //now setup 1st row pointer to point to mvmap[0][0];
        mvmap[0] = (CUVME_MV_RESULTS *)((unsigned char *)mvmap + sizeof(CUVME_MV_RESULTS *)*(mbheight+2));
        //bump up padding amount (1 entry padding on either side!)
        mvmap[0]++;
        //setup remaining rows based or prior row
        for(y=1;y<mbheight+2;y++) mvmap[y] = mvmap[y-1] + (mbwidth+2);
        //bump up default mvmap pointer to start at [0][0] after padding (which is mvmap[1][1]!)
        mvmap++; 
        //now mvmap points to [0][0] inside the halo!
    }
	p_ME_context->integer_mvmap = mvmap;	
	p_ME_context->ptr_mvs_local[0] = (CUVME_MV_RESULTS *)(base_mem + p_offsets->idx_ptr_mvs_local_0);	

	mbinfo = (CUVME_MB_INFO **)(base_mem + p_offsets->idx_integer_mb_info);

   if (mbinfo != NULL)
    {
        //now setup 1st row pointer to point to mbinfo[0][0];
        mbinfo[0] = (CUVME_MB_INFO *)((unsigned char *)mbinfo + sizeof(CUVME_MB_INFO *)*(mbheight+2));
        //bump up padding amount (1 entry padding on either side!)
        mbinfo[0]++;
        //setup remaining rows based or prior row
        for(y=1;y<mbheight+2;y++) mbinfo[y] = mbinfo[y-1] + (mbwidth+2);
        //bump up default mbinfo pointer to start at [0][0] after padding (which is mbinfo[1][1]!)
        mbinfo++; 
        //now mbinfo points to [0][0] inside the halo!
    }	
	p_ME_context->integer_mb_info = mbinfo;
	p_ME_context->CandidateCountInt = (unsigned int *)(base_mem + p_offsets->idx_CandidateCountInt );

	p_ME_context->sum_vars_frame = (int *)(base_mem + p_offsets->idx_sum_vars_frame);

	p_ME_context->sum_mean_frame = (int *)	(base_mem + p_offsets->idx_sum_mean_frame);
	

    if((err = assign_y_frame_me_ptr(&p_ME_context->quarter_res_src, 
										(width+2*REFERENCE_FRAME_PAD_AMT)/4, 
										(height+2*REFERENCE_FRAME_PAD_AMT)/4,
										(width+2*REFERENCE_FRAME_PAD_AMT)/4, 
										(height+2*REFERENCE_FRAME_PAD_AMT)/4, 
										0, 
										0,
										(unsigned char*)(base_mem + p_offsets->idx_quarter_res_src))) != CUVME_ERR_SUCCESS)
        return err;	
	
    if(!p_ME_context->dec_ref_avail)
    {
        i = 0;
	
		p_ME_context->malloced_forw_quarter_res_ref[i] = (CUVME_Y_BUFFER *)(base_mem + p_offsets->idx_malloced_forw_quarter_res_ref_ptr);
		if((err = assign_y_frame_me_ptr(p_ME_context->malloced_forw_quarter_res_ref[i], quarter_pad_width, 
			quarter_pad_height, width/4, height/4, 
			REFERENCE_FRAME_PAD_AMT/4, REFERENCE_FRAME_PAD_AMT/4,
			(unsigned char*)(base_mem + p_offsets->idx_malloced_forw_quarter_res_ref))) != CUVME_ERR_SUCCESS)
			return err;
		p_ME_context->malloced_forw_quarter_res_ref[i]->y += MAX_SEARCH_RANGE_X/4 +
			((width + 2 * MAX_SEARCH_RANGE_X)/4)*MAX_SEARCH_RANGE_Y/4; 
        
		if(p_ME_context->enable_bwd_reference)
		{
			p_ME_context->malloced_back_quarter_res_ref[i] = (CUVME_Y_BUFFER *)(base_mem + p_offsets->idx_malloced_back_quarter_res_ref_ptr);
			if((err = assign_y_frame_me_ptr(p_ME_context->malloced_back_quarter_res_ref[i], quarter_pad_width, 
				quarter_pad_height, width/4, height/4, 
				REFERENCE_FRAME_PAD_AMT/4, REFERENCE_FRAME_PAD_AMT/4,
				(unsigned char*)(base_mem + p_offsets->idx_malloced_back_quarter_res_ref))) != CUVME_ERR_SUCCESS)
				return err;
			p_ME_context->malloced_back_quarter_res_ref[i]->y += MAX_SEARCH_RANGE_X/4 +
				((width + 2 * MAX_SEARCH_RANGE_X)/4)*MAX_SEARCH_RANGE_Y/4; 
		}
    }	
	
    if((err = assign_y_frame_me_ptr(&p_ME_context->half_res_src,
										(width+2*REFERENCE_FRAME_PAD_AMT)/2,  
										(height+2*REFERENCE_FRAME_PAD_AMT)/2,
										(width+2*REFERENCE_FRAME_PAD_AMT)/2,  
										(height+2*REFERENCE_FRAME_PAD_AMT)/2, 
										0, 
										0,
										(unsigned char*)(base_mem + p_offsets->idx_half_res_src))) != CUVME_ERR_SUCCESS)
        return err;
		
   if(!p_ME_context->dec_ref_avail)
    {
		i = 0;
		p_ME_context->malloced_forw_half_res_ref[i] = (CUVME_Y_BUFFER *)(base_mem + p_offsets->idx_malloced_forw_half_res_ref_ptr);
		if((err = assign_y_frame_me_ptr(p_ME_context->malloced_forw_half_res_ref[i], half_pad_width, 
			half_pad_height, width/2, height/2, 
			REFERENCE_FRAME_PAD_AMT/2, REFERENCE_FRAME_PAD_AMT/2,
			(unsigned char*)(base_mem + p_offsets->idx_malloced_forw_half_res_ref))) != CUVME_ERR_SUCCESS)
			return err;

		p_ME_context->malloced_forw_half_res_ref[i]->y += MAX_SEARCH_RANGE_X/2 +
			((width + 2 * MAX_SEARCH_RANGE_X)/2)*MAX_SEARCH_RANGE_Y/2; 
	
		if(p_ME_context->enable_bwd_reference)
		{
			p_ME_context->malloced_back_half_res_ref[i] = (CUVME_Y_BUFFER *)(base_mem + p_offsets->idx_malloced_back_half_res_ref_ptr);
			if((err = assign_y_frame_me_ptr(p_ME_context->malloced_back_half_res_ref[i], half_pad_width, 
				half_pad_height, width/2, height/2, 
				REFERENCE_FRAME_PAD_AMT/2, REFERENCE_FRAME_PAD_AMT/2,
				(unsigned char*)(base_mem + p_offsets->idx_malloced_back_half_res_ref))) != CUVME_ERR_SUCCESS)
				return err;

			p_ME_context->malloced_back_half_res_ref[i]->y += MAX_SEARCH_RANGE_X/2 +
				((width + 2 * MAX_SEARCH_RANGE_X)/2)*MAX_SEARCH_RANGE_Y/2; 
		}

    }	
	
	p_ME_context->block_flat_yuv.width = width;
	p_ME_context->block_flat_yuv.height = height;
	p_ME_context->block_flat_yuv.image_width = width;
	p_ME_context->block_flat_yuv.image_height = width;
	
	
	p_yuv = (base_mem + p_offsets->idx_block_flat_yuv);
	p_ME_context->block_flat_yuv.y = p_yuv;
	p_ME_context->block_flat_yuv.u = p_yuv + width * height;
	p_ME_context->block_flat_yuv.v = p_yuv + width * height + width * height / 4;	
	
    p_ME_context->block_flat_src = (CUVME_Y_BUFFER *)(base_mem + p_offsets->idx_block_flat_src);

    p_ME_context->block_flat_src->y = p_yuv;
    p_ME_context->block_flat_src->buffer_width = width;
    p_ME_context->block_flat_src->buffer_height = height;
    p_ME_context->block_flat_src->active_width = width;
    p_ME_context->block_flat_src->active_height = height;
    p_ME_context->block_flat_src->offset_x = 0;
    p_ME_context->block_flat_src->offset_y = 0;
	
	p_ME_context->HRresults = (CUVME_MV_RESULTS *)(base_mem + p_offsets->idx_HRresults); 

	p_ME_context->QRresults = (CUVME_MV_RESULTS *)(base_mem + p_offsets->idx_QRresults);  

    p_ME_context->HR_MB_Info = (CUVME_MB_INFO *) (base_mem + p_offsets->idx_HR_MB_Info);  
    p_ME_context->QR_MB_Info = (CUVME_MB_INFO *) (base_mem + p_offsets->idx_QR_MB_Info);  
	return CUVME_ERR_SUCCESS;
}

void me_ConvertRasterToBlockFlattened(unsigned char *raster_input, unsigned char *block_op,
                                   int source_width, int source_height)
{
    int NumMbsX = source_width/MB_WIDTH;
    int NumMbsY = source_height/MB_HEIGHT;
    int i,j,k,l;
    unsigned char *local_raster_input;

    // loop runs from left to right and top to bottom for MBs
    for(i = 0; i < NumMbsY; i++)
    {
        for(j = 0; j < NumMbsX; j++)
        {
            // 8x8s in encode order
            for(k = 0; k < 4; k++)
            {
                // 4x4s in encode order
                for(l = 0; l < 4; l++)
                {
                    // the output is written sequentially but the input is accessed by
                    // doing appropriate pointer calculations
                    local_raster_input = raster_input + (i*NumMbsX*MB_TOTAL_SIZE) + j * MB_WIDTH +  
                        + (k/2)*8*source_width + (k%2)*8 + (l/2)*4*source_width + (l%2)*4;
                    memcpy(block_op, local_raster_input, 4);
                    block_op += 4;
                    local_raster_input += source_width;
                    memcpy(block_op, local_raster_input, 4);
                    block_op += 4;
                    local_raster_input += source_width;
                    memcpy(block_op, local_raster_input, 4);
                    block_op += 4;
                    local_raster_input += source_width;
                    memcpy(block_op, local_raster_input, 4);
                    block_op += 4;
                    local_raster_input += source_width;
                }
            }
        }
    }
}
