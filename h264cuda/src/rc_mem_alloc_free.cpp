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
#include <stdio.h>
#include "../inc/rc_common.h"
#include "../inc/rc_mem_alloc_free.h"
#include "../inc/mem_alloc_free.h"
#include <string.h>


CUVRC_ERROR rc_allocate_memory(RC_STATE *rcs)
{
	int num_mbs = rcs->width_in_mbs * rcs->height_in_mbs;
	if((rcs->roi_info.extremely_good_quality_rois = (CUVRC_ROI *)getmem_1d_void(sizeof(CUVRC_ROI) * num_mbs)) == NULL)
		return CUVRC_ERR_MEM;
	if((rcs->roi_info.good_quality_rois = (CUVRC_ROI *)getmem_1d_void(sizeof(CUVRC_ROI) * num_mbs)) == NULL)
		return CUVRC_ERR_MEM;
	if((rcs->roi_info.poor_quality_rois = (CUVRC_ROI *)getmem_1d_void(sizeof(CUVRC_ROI) * num_mbs)) == NULL)
		return CUVRC_ERR_MEM;
	if((rcs->roi_info.extremely_poor_quality_rois = (CUVRC_ROI *)getmem_1d_void(sizeof(CUVRC_ROI) * num_mbs)) == NULL)
		return CUVRC_ERR_MEM;
	return CUVRC_ERR_SUCCESS;
}

void rc_free_memory(RC_STATE *rcs)
{
	free_1D(rcs->roi_info.extremely_good_quality_rois);
	free_1D(rcs->roi_info.good_quality_rois);
	free_1D(rcs->roi_info.poor_quality_rois);
	free_1D(rcs->roi_info.extremely_poor_quality_rois);
	
	rcs->roi_info.extremely_good_quality_rois = NULL;
	rcs->roi_info.good_quality_rois = NULL;
	rcs->roi_info.poor_quality_rois = NULL;
	rcs->roi_info.extremely_poor_quality_rois = NULL;
}

