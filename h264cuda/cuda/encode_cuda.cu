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
#include <string.h>
#include <time.h>
#include <cutil_inline.h>
#include <cutil_inline_runtime.h>
#include <cutil.h>
#include "../inc/me_common.h"
#include "../inc/me_context.h"
#include "../inc/residual_coding.h"
#include "../inc/encoder_tables.h"
#include "../inc/cavlc_data.h"
#include "../inc/cavlc.h"
#include "../inc/deblock.h"
#include "../inc/h264_common.h"

#include "../inc/const_defines.h"
#include "../inc/entropy_data.h"
#include "../inc/encoder_context.h"
#include "../inc/output.h"

//include projec
#include "me_LR_search_kernel.cu"
#include "me_refinement_kernel.cu"
#include "iframe_residual_coding_kernel.cu"
#include "iframe_residual_chroma_kernel.cu"

#include "pframe_inter_residual_coding_kernel.cu"
#include "calc_cbp_and_total_coef.cu"

#include "cavlc_block_context_kernel.cu"
#include "cavlc_texture_symbols_kernel.cu"
#include "cavlc_texture_codes_kernel.cu"
#include "cavlc_header_code_kernel.cu"
#include "cavlc_bit_pack_kernel.cu"
#include "deblock_kernel.cu"

///新修改的函数及kernel
#include "intra_coding_kernel.cu"
#include "intra_coding_kernel_chroma.cu"
#include "pframe_LR_serach_kernel.cu"

//每一帧的编码入口函数
void encode_cuda(encoder_context_t *p_enc )
{

	clock_t start,end,start1,end1;
	int enc_width,enc_height,enc_width_c,enc_height_c;
	int width_ref,height_ref,width_ref_c,height_ref_c;
	int num_mb_ver, num_mb_hor;
	int num_mbs;
	int  i, j;
	int  frame_sad_sum;

	S_QP_DATA     QpData;
	short         *pQuantTable;
	short         *pDQuantTable;
	S_QP_DATA     QpData_chroma;
	short         *pQuantTable_chroma;
	short         *pDQuantTable_chroma;
	
	start1 = clock();
	unsigned char *p_input          = p_enc->input_frame.y;
	unsigned char *p_recon          = p_enc->pRecFrame->y;
	short         *p_dct_coefs      = p_enc->transform_coefs.pDctCoefs;
	short         *p_dc_coefs       = p_enc->transform_coefs.pDcCoefs;
	int           QP                = p_enc->frame_info.frame_qp;
	S_BLK_MB_INFO *p_blk_mb_info    = p_enc->pBlkMBInfo;
	int           *p_mb_qps         = p_enc->p_mb_qps;
	int           constrained_intra = p_enc->PictureParameterSet.constrained_intra_pred_flag;
	int           intra_pred_select = p_enc->intra_prediction_selection;

	unsigned char *p_input_u        = p_enc->input_frame.u;
	unsigned char *p_input_v        = p_enc->input_frame.v;
	unsigned char *p_recon_u        = p_enc->pRecFrame->u;
	unsigned char *p_recon_v        = p_enc->pRecFrame->v;
	short         *p_dct_coefs_u    = p_enc->transform_coefs.pDctCoefs_u;
	short         *p_dct_coefs_v    = p_enc->transform_coefs.pDctCoefs_v;
	short         *p_dc_coefs_u     = p_enc->transform_coefs.pDcCoefs_u;
	short         *p_dc_coefs_v     = p_enc->transform_coefs.pDcCoefs_v;

	unsigned char  *p_rec_ptr;

	unsigned char *p_rec_u_ptr, *p_rec_v_ptr;
	
	enc_width         = p_enc->width;
	enc_height        = p_enc->height;
	enc_width_c       = p_enc->width/2;
	enc_height_c      = p_enc->height/2;
	width_ref		  = enc_width + 2*REFERENCE_FRAME_PAD_AMT;
	height_ref		  = enc_height + 2*REFERENCE_FRAME_PAD_AMT;
	width_ref_c       = width_ref>>1;
	height_ref_c	  = height_ref>>1;
	num_mb_hor = enc_width / MB_WIDTH;
	num_mb_ver = enc_height / MB_HEIGHT;
	num_mbs	   = num_mb_hor*num_mb_ver;

	p_rec_ptr = p_recon + RECON_FRAME_Y_OFFSET* width_ref + RECON_FRAME_X_OFFSET;
	p_rec_u_ptr  = p_recon_u + RECON_FRAME_Y_OFFSET_C * width_ref_c + RECON_FRAME_X_OFFSET_C;
	p_rec_v_ptr  = p_recon_v + RECON_FRAME_Y_OFFSET_C * width_ref_c + RECON_FRAME_X_OFFSET_C;

	unsigned char *dev_input;
	unsigned char *dev_recon;
	S_BLK_MB_INFO *dev_blk_mb_info;
	S_QP_DATA	*dev_QpData;
	short		*dev_dct_coefs;
	short		*dev_dc_coefs;
	short		*dev_Quant_tab;
	short		*dev_Dquant_tab;

	unsigned char *dev_input_uv;
	unsigned char *dev_recon_uv;
	S_QP_DATA	*dev_QpData_uv;
	short		*dev_dct_coefs_uv;
	short		*dev_dc_coefs_uv;
	short		*Quant_tab_uv;
	short		*Dquant_tab_uv;
	int *dev_ZigZag;

	cutilSafeCall(cudaMalloc((void**) &dev_QpData,sizeof(S_QP_DATA))); 
	cutilSafeCall(cudaMalloc((void**) &dev_QpData_uv,sizeof(S_QP_DATA))); //为重建数据分配显存空间
	
	cutilSafeCall(cudaMalloc((void**) &Quant_tab_uv,BLOCKS_PER_MB*sizeof(short))); 
	cutilSafeCall(cudaMalloc((void**) &Dquant_tab_uv,BLOCKS_PER_MB*sizeof(short))); 
	cutilSafeCall(cudaMalloc((void**) &dev_ZigZag,BLOCKS_PER_MB*sizeof(int)));
	cutilSafeCall(cudaMemcpy(dev_ZigZag,ZigZagScan,16*sizeof(int),cudaMemcpyHostToDevice));
	

	cutilSafeCall(cudaMalloc((void**) &dev_input,MB_WIDTH*MB_HEIGHT*num_mbs)); //为输入的数据分配显存空间
	cutilSafeCall(cudaMalloc((void**) &dev_recon,height_ref*width_ref*sizeof(char))); //为重建数据分配显存空间(1088*1952B)
	cutilSafeCall(cudaMalloc((void**) &dev_recon_uv,height_ref_c*width_ref_c*2)); //为重建数据分配显存空间
	cutilSafeCall(cudaMalloc((void**) &dev_input_uv,MB_TOTAL_SIZE_C*num_mbs*2)); //为输入的数据分配显存空间

	cutilSafeCall(cudaMalloc((void**) &dev_blk_mb_info,BLOCKS_PER_MB*sizeof(S_BLK_MB_INFO)*num_mbs));

	cutilSafeCall(cudaMalloc((void**) &dev_dct_coefs,MB_TOTAL_SIZE*num_mbs*sizeof(short))); //为变换编码后数据分配显存空间
	cutilSafeCall(cudaMalloc((void**) &dev_dc_coefs,BLOCKS_PER_MB*num_mbs*sizeof(short))); //为变换编码后直流数据分配显存空间
	cutilSafeCall(cudaMalloc((void**) &dev_dct_coefs_uv,  MB_TOTAL_SIZE_C*num_mbs*2*sizeof(short))); 
	cutilSafeCall(cudaMalloc((void**) &dev_dc_coefs_uv,  BLOCKS_PER_MB_C*num_mbs*2*sizeof(short))); 

	//将量化表载入常量存储器中
	cutilSafeCall(cudaMalloc((void**) &dev_Quant_tab,BLOCKS_PER_MB*sizeof(short))); //为变换编码后数据分配显存空间
	cutilSafeCall(cudaMalloc((void**) &dev_Dquant_tab,BLOCKS_PER_MB*sizeof(short))); //为变换编码后数据分配显存空间
	end1 = clock();
	p_enc->new_timers.prep_encode_frame += (end1 - start1);
	
	if(p_enc->slice_params.slice_type == SLICE_I)
	{
		start = clock();

		   for (i = 0; i < num_mbs * BLOCKS_PER_MB; i++)
		   {
			 p_blk_mb_info[i].QP = QP;
		   }
			InitQPDataAndTablesFromQP(&QpData, &pQuantTable, &pDQuantTable, QP, 1, 1);
			InitQPDataAndTablesFromQP(&QpData_chroma, &pQuantTable_chroma, &pDQuantTable_chroma, p_blk_mb_info->QP, 1, 0);

			cutilSafeCall(cudaMemcpy(dev_QpData,&QpData,sizeof(S_QP_DATA),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(dev_Quant_tab,pQuantTable,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 
			cutilSafeCall(cudaMemcpy(dev_Dquant_tab,pDQuantTable,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 
			cutilSafeCall(cudaMemcpy(dev_input,p_input,MB_TOTAL_SIZE*num_mbs,cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(dev_blk_mb_info,p_blk_mb_info,BLOCKS_PER_MB*sizeof(S_BLK_MB_INFO)*num_mbs,cudaMemcpyHostToDevice));

			dim3 grid(p_enc->i_slice_num, 1, 1); //grid的设置
			dim3 threads(4, 4, 1); //每一个block中threads的设置,受限于帧内编码的数据依赖限制，能够并行的threads为16.
			
			dim3 block(4, 4, 16);
			iframe_luma_residual_coding <<<grid,block>>>(dev_input,
															 enc_width,
															 num_mb_hor,
															 num_mb_ver,
															 dev_recon + RECON_FRAME_Y_OFFSET* width_ref + RECON_FRAME_X_OFFSET,
															 width_ref,
															 dev_dct_coefs,
															 dev_dc_coefs,
															 dev_blk_mb_info,
															 dev_Quant_tab,
															 dev_Dquant_tab,
															 dev_QpData,
															 constrained_intra,
															 intra_pred_select,
															 p_enc->i_slice_num
															);

			cutilSafeCall(cudaMemcpy(dev_input_uv,p_input_u,MB_TOTAL_SIZE_C*num_mbs,cudaMemcpyHostToDevice)); //加载U分量	
			cutilSafeCall(cudaMemcpy(dev_input_uv+MB_TOTAL_SIZE_C*num_mbs,p_input_v,MB_TOTAL_SIZE_C*num_mbs,cudaMemcpyHostToDevice));//加载V分量
			cutilSafeCall(cudaMemcpy(dev_QpData_uv,&QpData_chroma,sizeof(S_QP_DATA),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(Quant_tab_uv,pQuantTable_chroma,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 
			cutilSafeCall(cudaMemcpy(Dquant_tab_uv,pDQuantTable_chroma,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 

			dim3 grid_chroma(p_enc->i_slice_num, 1, 1);
			dim3 threads_chroma(2, 4, 1); 
			dim3 block_chroma(4, 2, 16); 


			iframe_residual_coding_chroam<<<grid_chroma,block_chroma>>>(dev_input_uv,
																				dev_blk_mb_info,
																				dev_QpData_uv,
																				Quant_tab_uv,
																				Dquant_tab_uv,
																				dev_recon_uv+RECON_FRAME_Y_OFFSET_C * width_ref_c + RECON_FRAME_X_OFFSET_C,
																				dev_dct_coefs_uv,
																				dev_dc_coefs_uv,
																				enc_width_c,
																				enc_height_c,
																				width_ref_c,
																				height_ref_c,
																				num_mb_hor,
																				num_mb_ver,
																				p_enc->i_slice_num
																			);
			dim3 grid_cbp_luma(num_mb_hor/8, num_mb_ver, 1);
			dim3 threads_cbp_luma(BLK_WIDTH, BLK_HEIGHT, 8); 

			CalcCBP_and_TotalCoeff_Luma_cuda<<<grid_cbp_luma,threads_cbp_luma>>>(dev_dct_coefs,
																			  dev_blk_mb_info);
			dim3 grid_cbp_chroma(num_mb_hor/8, num_mb_ver, 1);
			dim3 threads_cbp_chroma(BLOCKS_PER_MB_C, 8, 2);
				
			CalcCBP_and_TotalCoeff_Chroma_cuda<<<grid_cbp_chroma,threads_cbp_chroma>>>(dev_dct_coefs_uv,
																						dev_dc_coefs_uv,
																						dev_blk_mb_info,
																						enc_width_c,
																						enc_height_c);
			cutilSafeCall(cudaMemcpy(p_blk_mb_info,dev_blk_mb_info,BLOCKS_PER_MB*num_mbs*sizeof(S_BLK_MB_INFO),cudaMemcpyDeviceToHost));

			frame_sad_sum = 0;
			for(j = 0; j < num_mb_ver; j++)
			{
				for(i = 0; i < num_mb_hor; i++)
				{   
					frame_sad_sum += (p_enc->pBlkMBInfo + (j * num_mb_hor + i) * BLOCKS_PER_MB)->MinSAD;
				}
			} 
			p_enc->avg_mb_sad = frame_sad_sum / (num_mb_hor * num_mb_ver);
			p_enc->frame_info.num_intra_mb = num_mb_hor * num_mb_ver;
			end = clock();
			p_enc->new_timers.iframe_residual += (end-start);

	}

	else
	{
			start = clock();
			int num_mb_hor_ref,num_mb_ver_ref;
			int RefStride2Begin;
			int RefStride2BeginUV;
			int decimate_ratio = 2;
			int M = decimate_ratio;
			RC_CONTEXT        *p_rc; 
			ME_CONTEXT        *p_me;
			CUVME_MV_RESULTS  *p_me_mv_results;
			CUVME_MB_INFO     *p_me_mb_info;
			CUVME_Y_BUFFER     me_src;
			CUVME_Y_BUFFER    *ptr_me_src;
			CUVME_MB_CHARAC   *p_me_mb_characs;
			ME_CONTEXT        *p_sc_me;

			ME_context_t *p_ME_context;
			int	do_zero_search = 0;
			int	do_low_res = 0;
			int do_int_search = 0;
			int	do_int_and_halfpel_search = 0;
			int	do_decimation_for_low_res = 0;

			unsigned char *i_InputLuma; 
			SINT32 skipbias_factor;
		 
			int            avg_var; 
			CUVME_Y_BUFFER me_ref;
			CUVME_Y_BUFFER me_pred;

			unsigned int  AvgMbSAD          = p_enc->avg_mb_sad;

			unsigned char *dev_input_ref;
			unsigned char *dev_out_HR_ref;
			unsigned char *dev_out_QR_ref;
			unsigned char *dev_input_src;
			unsigned char *dev_out_HR_src;
			unsigned char *dev_out_QR_src;
			CUVME_MV_RESULTS *dev_mvsLocal;
			CUVME_MB_INFO	*dev_mb_info;
			unsigned int     *IntegerPelCenterVecs;
			unsigned char *dev_out_pred;
			CUVME_MV_RESULTS *integer_mvmap;
			CUVME_MB_INFO   *mb_info;
			int *dev_CoeffCosts;

			unsigned int lambda_factor_rc;

			 lambda_factor_rc = QP2QUANT_NEW[p_enc->frame_info.frame_qp];

			if(p_enc->intra_mb_level == 80)
				lambda_factor_rc <<=  8;
			else if(p_enc->intra_mb_level == 90)
				 lambda_factor_rc <<=  7;
			else if(p_enc->intra_mb_level == 100)
				lambda_factor_rc <<= 5;
			else
			   lambda_factor_rc <<= p_enc->intra_mb_level; 

			const int ZigZag[16] = {0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
			const int CoeffCosts[16] = {3,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0};

			S_BLK_MB_INFO *pBlkMBInfo;
			E_ERR err  = ERR_SUCCESS;
			CUVME_ERROR me_err = CUVME_ERR_SUCCESS;
			
			num_mb_hor_ref		  = width_ref/MB_WIDTH;
			num_mb_ver_ref		  = height_ref/MB_HEIGHT;
			RefStride2Begin		  = ((REFERENCE_FRAME_PAD_AMT * width_ref) + REFERENCE_FRAME_PAD_AMT);
			RefStride2BeginUV = RECON_FRAME_Y_OFFSET_C * width_ref_c + RECON_FRAME_X_OFFSET_C;/*((REFERENCE_FRAME_PAD_AMT/2 * p_enc->padded_ref_frame.width/2) + REFERENCE_FRAME_PAD_AMT/2);*/

			p_me				= &p_enc->me_context;
			p_rc                = &p_enc->rc_context;
			p_me_mv_results		= p_me->p_me_mv_results;
			p_me_mb_info		= p_me->p_me_mb_info;
			p_me_mb_characs		= p_me->p_me_mb_characs;
			err                 = ERR_SUCCESS;
			me_err              = CUVME_ERR_SUCCESS;
			ptr_me_src		    = &me_src;

			ptr_me_src->y             = p_enc->input_frame.y;
			ptr_me_src->buffer_width  = p_enc->input_frame.width;
			ptr_me_src->buffer_height = p_enc->input_frame.height;
			ptr_me_src->active_width  = p_enc->width;
			ptr_me_src->active_height = p_enc->height;
			ptr_me_src->offset_x      = 0;
			ptr_me_src->offset_y      = 0;

			me_pred.y             = p_enc->inter_pred_frame.y;
			me_pred.buffer_width  = p_enc->inter_pred_frame.width;
			me_pred.buffer_height = p_enc->inter_pred_frame.height;
			me_pred.active_width  = p_enc->width;
			me_pred.active_height = p_enc->height;
			me_pred.offset_x      = 0;
			me_pred.offset_y      = 0;

			me_ref.y             = p_enc->pRefFrame->y;
			me_ref.buffer_width  = p_enc->pRefFrame->width;
			me_ref.buffer_height = p_enc->pRefFrame->height;
			me_ref.active_width  = p_enc->width;
			me_ref.active_height = p_enc->height;
			me_ref.offset_x      = REFERENCE_FRAME_PAD_AMT;
			me_ref.offset_y      = REFERENCE_FRAME_PAD_AMT;
			
			// Transform and reconstruct
			InitQPDataAndTablesFromQP(&QpData_chroma, &pQuantTable_chroma, &pDQuantTable_chroma, p_enc->frame_info.frame_qp, 0, 0);
			InitQPDataAndTablesFromQP (&QpData, &pQuantTable, &pDQuantTable, p_enc->frame_info.frame_qp, 0, 1);

			p_ME_context = (ME_context_t *)p_me->me_handle;
			SINT32 lambda_factor;
			/********** decide between forward or backward reference frame to be used *******************/
			// Setting Lambda //TODO: confirm this
			lambda_factor = QP2QUANT_MELIB[p_ME_context->FrameQP - 12];
			skipbias_factor = 1;
			if(p_ME_context->FrameQP > 42)
			{
				skipbias_factor = 2;
			}

			me_err = cuvme_set_reference_frame(p_me->me_handle, &me_ref, 0, 0);
			if (!me_err)
			{
				me_err = cuvme_set_predicted_picture(p_me->me_handle, &me_pred);
			}
		    
			if (!me_err)
			{
				if((p_enc->mb_adapt_qp_on == 1) || (p_enc->intra_mb_level == 0)) 
				{ 
				  me_err = cuvme_set_return_mb_characteristics(p_me->me_handle, p_me_mb_characs); 
				}    
			}
			//ME 处理
			if(!me_err)
			{  
				p_ME_context->ptr_src_picture = ptr_me_src;
					
				if(p_ME_context->nonpersistent_mem_givenby_app_flag)
				{
						me_assign_nonpersistentmem_pointers(p_ME_context);
				}
				if(p_ME_context->num_mvs == 1)
				{
						p_ME_context->ptr_mvs[0] = p_me_mv_results;
				}
				else
				{
						p_ME_context->ptr_mvs[0] = p_ME_context->ptr_mvs_local[0];
						p_ME_context->ptr_res_out[0] = p_me_mv_results;
				}
				p_ME_context->ptr_mb_info[0] = p_me_mb_info;
					// We need to set all the MVs to zero only for zero mode
				if((p_ME_context->me_mode < 5) || (p_ME_context->CRef))
				{
						unsigned int NumMbs = (p_ME_context->width / MB_WIDTH) *
							(p_ME_context->height / MB_HEIGHT);
						memset(p_ME_context->ptr_mvs[0], 0, NumMbs*sizeof(CUVME_MV_RESULTS));
						memset(p_ME_context->ptr_mb_info[0], 0, NumMbs*sizeof(CUVME_MB_INFO));
				}
				if(!p_ME_context->num_lowres_forw_references)        
				{
						p_ME_context->forw_quarter_res_ref[0] =
							p_ME_context->malloced_forw_quarter_res_ref[0];
						p_ME_context->forw_half_res_ref[0] = 
							p_ME_context->malloced_forw_half_res_ref[0];
				}
				if(!p_ME_context->num_lowres_back_references)        
				{
						p_ME_context->back_quarter_res_ref[0] =
							p_ME_context->malloced_back_quarter_res_ref[0];
						p_ME_context->back_half_res_ref[0] = 
							p_ME_context->malloced_back_half_res_ref[0];
				}

				if(p_ME_context->source_format == RASTER_ORDER)
				{

					{
							me_ConvertRasterToBlockFlattened(ptr_me_src->y, p_ME_context->block_flat_src->y, 
								p_ME_context->width,p_ME_context->height);
					}
						p_ME_context->block_flat_src->buffer_width = ptr_me_src->buffer_width;
						p_ME_context->block_flat_src->buffer_height = ptr_me_src->buffer_height;
						p_ME_context->block_flat_src->active_width = ptr_me_src->active_width;
						p_ME_context->block_flat_src->active_height = ptr_me_src->active_height;
						p_ME_context->block_flat_src->offset_x = ptr_me_src->offset_x;
						p_ME_context->block_flat_src->offset_y = ptr_me_src->offset_y;

						p_ME_context->ptr_src_picture = p_ME_context->block_flat_src;
				}
				p_ME_context->do_zero_search = 0;
				p_ME_context->do_low_res = 0;
				p_ME_context->do_int_search = 0;
				p_ME_context->do_int_and_halfpel_search = 0;
				p_ME_context->do_decimation_for_low_res = 0;
				switch(p_ME_context->me_mode)
				{
					case 0:
						p_ME_context->do_zero_search = 1;
						do_zero_search = 1;
						break;
					case 5:
					case 10:
						p_ME_context->do_low_res = 1;
						do_low_res = 1;
						break;
					case 20:
					case 22:
						p_ME_context->do_low_res = 1;
						do_low_res = 1;
						p_ME_context->do_int_search = 1;
						do_int_search = 1;
						break;
					case 30:
					case 32:
						p_ME_context->do_low_res = 1;
						do_low_res = 1;
						p_ME_context->do_int_and_halfpel_search = 1;
						do_int_and_halfpel_search = 1;
						break;
					default:
						break;
				}
				
				if(!p_ME_context->num_lowres_forw_references)
				{
					p_ME_context->do_decimation_for_low_res = 1;
					do_decimation_for_low_res = 1;
				}

				if((p_ME_context->FrameQP - 12) > 0)
					p_ME_context->lambda_factor = QP2QUANT_MELIB[p_ME_context->FrameQP - 12];//
				else 
					p_ME_context->lambda_factor = QP2QUANT_MELIB[0];//

					i_InputLuma	= p_ME_context->ptr_src_picture->y;
				end = clock();
				p_enc->new_timers.pframe_total +=(end -start);
				start = clock();
				if(do_decimation_for_low_res)
				{
					
					cutilSafeCall(cudaMalloc((void**) &dev_out_HR_ref,(height_ref/M)*(width_ref/M))); //为切分后1/2像素数据分配空间，为输入数据的1/4大小
					cutilSafeCall(cudaMalloc((void**) &dev_out_QR_ref,(height_ref/(2*M))*(width_ref/(2*M))));

					cutilSafeCall(cudaMalloc((void**) &dev_input_ref,width_ref*height_ref)); //为输入的数据分配显存空间
					cutilSafeCall(cudaMemcpy(dev_input_ref,p_ME_context->ptr_forw_ref_frame[0]->y,width_ref*height_ref,cudaMemcpyHostToDevice));

					dim3 grid_ref(num_mb_hor_ref>>1,num_mb_ver_ref, 1); //grid的设置
					dim3 threads_ref((MB_WIDTH*2)>>1, MB_HEIGHT>>1, 1); 

					me_Decimate_kernel<<<grid_ref,threads_ref>>>(dev_input_ref,dev_out_HR_ref,dev_out_QR_ref,height_ref,width_ref);
				}
				if(do_low_res)
				{
					cutilSafeCall(cudaMemcpy(dev_input,p_enc->input_frame.y,MB_TOTAL_SIZE*num_mbs*sizeof(unsigned char),cudaMemcpyHostToDevice));

					cutilSafeCall(cudaMalloc((void**) &dev_out_HR_src,(enc_width/M)*(enc_height/M))); //为切分后1/2像素数据分配空间，为输入数据的1/4大小
					cutilSafeCall(cudaMalloc((void**) &dev_out_QR_src,(enc_width/(2*M))*(enc_height/(2*M))));
					cutilSafeCall(cudaMalloc((void**) &dev_mvsLocal,num_mb_hor*num_mb_ver*sizeof(CUVME_MV_RESULTS)));
					cutilSafeCall(cudaMalloc((void**) &dev_mb_info,num_mbs*sizeof(CUVME_MB_INFO))); //为输入的数据分配显存空间
					
					cutilSafeCall(cudaMemcpy(dev_blk_mb_info,p_enc->pBlkMBInfo,BLOCKS_PER_MB*sizeof(S_BLK_MB_INFO)*num_mbs,cudaMemcpyHostToDevice));

					cutilSafeCall(cudaMalloc((void**) &dev_input_src,enc_width*enc_height*sizeof(char))); //为输入的数据分配显存空间
					cutilSafeCall(cudaMemcpy(dev_input_src,i_InputLuma,enc_width*enc_height*sizeof(char),cudaMemcpyHostToDevice));

					unsigned int *HR_SAD_dev;
					cutilSafeCall(cudaMalloc((void**) &HR_SAD_dev,num_mb_hor*num_mb_ver*32*sizeof(unsigned int)));

					dim3 grid_src(num_mb_hor>>1,num_mb_ver, 1); //grid的设置
					dim3 threads_src((MB_WIDTH*2)>>1, MB_HEIGHT>>1, 1); 

					dim3 grid_QR(num_mb_hor/6,num_mb_ver/4,1);
					dim3 threads_QR(16,16,1);

					dim3 grid_QR_new(num_mb_hor/6,(num_mb_ver+2)/3,1);

					dim3 grid_HR_SAD(num_mb_hor,num_mb_ver,1);
					dim3 threads_HR_SAD(8,4,4);

					dim3 grid_HR(num_mb_hor/6,num_mb_ver/4,1);
					dim3 threads_HR(8,4,1);

					me_Decimate_kernel<<<grid_src,threads_src>>>(dev_input,dev_out_HR_src,dev_out_QR_src,enc_height,enc_width);
					

					me_QR_LowresSearch<<<grid_QR_new,threads_QR>>>(dev_out_QR_src,
																		dev_out_QR_ref,
																		dev_mvsLocal,
																		enc_width/LOWRES_DEC_RATIO,
																		enc_height/LOWRES_DEC_RATIO,
																		width_ref/LOWRES_DEC_RATIO,
																		height_ref/LOWRES_DEC_RATIO,
																		num_mb_hor,
																		num_mb_ver,
																		2*QR_WEIGHT,
																		QR_SEARCH_SIZE,
																		QR_ZERO_BIAS, 
																		lambda_factor,
																		skipbias_factor
																);

					me_HR_Cal_Candidate_SAD_kernel<<<grid_HR_SAD,threads_HR_SAD>>>(dev_out_HR_src, 
																				  dev_out_HR_ref,          																													
																				  dev_mvsLocal,																																																										
																				  enc_width/HLFRES_DEC_RATIO, 
																				  enc_height/HLFRES_DEC_RATIO,
																				  width_ref/HLFRES_DEC_RATIO, 
																				  height_ref/HLFRES_DEC_RATIO, 
																				  num_mb_hor,
																				  num_mb_ver,
																				  HR_SEARCH_SIZE,
																				  HR_SAD_dev
																				  );

					dim3 grid_HR_new(num_mb_hor/6,(num_mb_ver+2)/3,1); 
					me_HR_Candidate_Vote<<<grid_HR_new,threads_HR>>>(  HR_SAD_dev,
																			  dev_mvsLocal,																													
																			  dev_mb_info,
																			  enc_width/HLFRES_DEC_RATIO, 
																			  enc_height/HLFRES_DEC_RATIO, 
																			  width_ref/HLFRES_DEC_RATIO, 
																			  height_ref/HLFRES_DEC_RATIO, 
																			  num_mb_hor,
																			  num_mb_ver,
																			  4*HR_WEIGHT, 
																			  HR_SEARCH_SIZE,
																			  HR_ZERO_BIAS, 
																			  lambda_factor,
																			  (skipbias_factor*3)
																		  );
					cutilSafeCall(cudaFree(HR_SAD_dev));

				}
					
				if(do_int_and_halfpel_search||do_int_search)
				{
					cutilSafeCall(cudaMalloc((void**) &IntegerPelCenterVecs,num_mb_hor*num_mb_ver*sizeof(int)));
					cutilSafeCall(cudaMalloc((void**)&dev_out_pred,enc_width*enc_height*sizeof(unsigned char)));
					cutilSafeCall(cudaMalloc((void**)&integer_mvmap,num_mb_hor*num_mb_ver*sizeof (CUVME_MV_RESULTS)));
					cutilSafeCall(cudaMalloc((void**)&mb_info,num_mb_hor*num_mb_ver*sizeof(CUVME_MB_INFO)));
				 
					dim3 grid_MV(1,num_mb_ver,1);
					dim3 threads_MV(num_mb_hor,1,1);

					dim3 grid_Int(num_mb_hor,1,1);
					dim3 threads_Int(16,3,3);

					me_ClipVec_ForFrame<<<grid_MV,threads_MV>>>(dev_mvsLocal,IntegerPelCenterVecs,p_ME_context->search_range_x,
											p_ME_context->search_range_y, p_ME_context->candidate_tune_values.integer_clip_range,num_mb_hor,num_mb_ver/*,dev_ref_index,dev_MV*/);

					//循环处理每一行宏块，因为相邻行宏块之间有数据相关性

						me_IntegerSimulsadVote_kernel<<<grid_Int,threads_Int>>>(dev_input_src,dev_input_ref,IntegerPelCenterVecs,integer_mvmap,mb_info,dev_out_pred,
																		num_mb_hor,num_mb_ver,RefStride2Begin,lambda_factor,enc_width,width_ref,
																		dev_blk_mb_info);
					cutilSafeCall(cudaMemcpy(p_ME_context->p_pred_picture->y,dev_out_pred,enc_width*enc_height*sizeof(unsigned char),cudaMemcpyDeviceToHost));
				}



				p_ME_context->num_lowres_forw_references = 0;
				p_ME_context->num_lowres_back_references = 0;
				p_ME_context->LastAvgHRSAD = p_ME_context->AvgHRSAD;
			
				p_ME_context->num_sec_formats = 0;
				if(err == CUVME_ERR_SUCCESS)
					p_ME_context->curr_state = INIT;
				p_ME_context->flag_do_mc = 0;
				p_ME_context->store_dec_src_flag = 0;
				p_ME_context->get_mb_characs_flag = 0;
				p_ME_context->num_forw_references = 0;
				p_ME_context->num_back_references = 0;		
				p_ME_context->ref_frame_distance_fwd = 0;
				p_ME_context->ref_frame_distance_bwd = 0;
			}
			p_sc_me             = &p_enc->me_context;
			pBlkMBInfo          = p_enc->pBlkMBInfo;
			if (!err)
			{

			 me_err = cuvme_get_avg_var(p_sc_me->me_handle ,&avg_var);
			 if (me_err)
			 {
			 printf ("ME returned error code %d", me_err);
			 err = ERR_FAILURE;
			 }

			 if(!err)
				err    = (E_ERR)cuvrc_set_avg_var(p_rc->rc_handle ,avg_var);

			 if (err)
			 {
			  printf ("RC returned error code %d", err);
			  err = ERR_FAILURE;
			 }
			}

				end = clock();
				p_enc->new_timers.me_total += (end-start);
				p_enc->new_timers.pframe_total +=(end -start);
				start = clock();
				//P帧亮度分量的编码
				cutilSafeCall(cudaMalloc((void**) &dev_CoeffCosts,BLOCKS_PER_MB*sizeof(int)));

				cutilSafeCall(cudaMemcpy(dev_recon,(p_rec_ptr),enc_height*width_ref*sizeof(char)-RECON_FRAME_X_OFFSET,cudaMemcpyHostToDevice));

				cutilSafeCall(cudaMemcpy(dev_CoeffCosts,CoeffCosts,sizeof(int)*16,cudaMemcpyHostToDevice));
				//将量化表载入常量存储器中
				cutilSafeCall(cudaMemcpy(dev_QpData,&QpData,sizeof(S_QP_DATA),cudaMemcpyHostToDevice));
				cutilSafeCall(cudaMemcpy(dev_Quant_tab,pQuantTable,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 
				cutilSafeCall(cudaMemcpy(dev_Dquant_tab,pDQuantTable,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 
				
				dim3 grid_inter((num_mb_hor>>2),num_mb_ver,1); //每一个block处理4个宏块，第一维为每一行宏块应该包含的block数量，第二维是图像的高度（以宏块为单位）
				dim3 threads_inter(4,4,4);					 //每一个block有64个线程，第一维对宏块内各个block索引，第二维对宏块进行索引
				// cuda kernel
				dim3 grid_intra(p_enc->i_slice_num,1,1); 
				dim3 threads_intra(4,4,1);
				//帧间编码
				pframe_inter_resudial_coding_luma_kernel<<<grid_inter,threads_inter>>>(dev_input,
																			 dev_out_pred,
																			 enc_width,
																			 dev_recon+RefStride2Begin,
																			 width_ref,
																			 dev_dct_coefs,
																			 dev_Quant_tab,
																			 dev_Dquant_tab,
																			 dev_QpData,
																			 dev_ZigZag,
																			 dev_CoeffCosts
																			 );
				
				end = clock();
				p_enc->new_timers.pframe_residual_inter += (end-start);
				p_enc->new_timers.pframe_residual_luma += (end-start);
				p_enc->new_timers.pframe_total += (end -start);
				start = clock();
				//帧内预测编码				
				dim3 block_intra(4,4,16);
				pframe_intra_resudial_coding_luma<<<grid_intra,block_intra>>>(dev_input,
																			 dev_out_pred,
																			 enc_width,
																			 dev_recon+RefStride2Begin,
																			 width_ref,
																			 dev_blk_mb_info,
																			 dev_dct_coefs,
																			 dev_dc_coefs,
																			 dev_Quant_tab,
																			 dev_Dquant_tab,
																			 dev_QpData,
																			 AvgMbSAD,
																			 lambda_factor_rc,
																			 num_mb_hor,
																			 num_mb_ver,
																			 p_enc->i_slice_num
																			 );

				//色度分量的编码过程
				end = clock();
				p_enc->new_timers.pframe_residual_intra += (end-start);
				p_enc->new_timers.pframe_residual_luma += (end-start);
				p_enc->new_timers.pframe_total += (end -start);
				start = clock();
		 
				unsigned char *dev_pred_uv;
				unsigned char *dev_ref_uv;

				cutilSafeCall(cudaMalloc((void**) &dev_ref_uv, (width_ref_c)*(height_ref_c)*2*sizeof(unsigned char))); 
				cutilSafeCall(cudaMalloc((void**) &dev_pred_uv, (enc_width_c)*(enc_height_c)*2*sizeof(unsigned char)));

				cutilSafeCall(cudaMemcpy(dev_ref_uv, p_enc->pRefFrame->u,(width_ref_c)*(height_ref_c)*sizeof(unsigned char),cudaMemcpyHostToDevice)); 
				cutilSafeCall(cudaMemcpy(dev_ref_uv+(width_ref_c)*(height_ref_c), p_enc->pRefFrame->v,(width_ref_c)*(height_ref_c)*sizeof(unsigned char),cudaMemcpyHostToDevice)); 
				
				dim3 grid_mcc(num_mb_hor,num_mb_ver,1);
				dim3 threads_mcc(8,8,2);
				MotionCompensateChroma_kernel<<<grid_mcc,threads_mcc>>> ( dev_ref_uv,
																	dev_pred_uv,
																	dev_blk_mb_info,
																	enc_width_c,
																	enc_height_c,
																	width_ref_c,
																	height_ref_c,
																	RefStride2BeginUV
																	);
				cutilSafeCall(cudaMemcpy(p_enc->inter_pred_frame.u,dev_pred_uv,(enc_width_c)*(enc_height_c)*sizeof(char),cudaMemcpyDeviceToHost));
				cutilSafeCall(cudaMemcpy(p_enc->inter_pred_frame.v, dev_pred_uv+(enc_width_c)*(enc_height_c),(enc_width_c)*(enc_height_c)*sizeof(char),cudaMemcpyDeviceToHost));
				
				end = clock();
				p_enc->new_timers.pframe_residual_chroma += (end-start);
				p_enc->new_timers.pframe_mc += (end-start);
				p_enc->new_timers.pframe_total += (end -start);
				start = clock();	
				
				cutilSafeCall(cudaMemcpy(dev_input_uv,p_enc->input_frame.u,(enc_width_c)*(enc_height_c),cudaMemcpyHostToDevice)); //加载U分量	
				cutilSafeCall(cudaMemcpy(dev_input_uv+(enc_width_c)*(enc_height_c),p_enc->input_frame.v,(enc_width_c)*(enc_height_c),cudaMemcpyHostToDevice));//加载V分量

				cutilSafeCall(cudaMemcpy(dev_QpData_uv,&QpData_chroma,sizeof(S_QP_DATA),cudaMemcpyHostToDevice));
				cutilSafeCall(cudaMemcpy(Quant_tab_uv,pQuantTable_chroma,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 
				cutilSafeCall(cudaMemcpy(Dquant_tab_uv,pDQuantTable_chroma,BLOCKS_PER_MB*sizeof(short),cudaMemcpyHostToDevice)); 
				
				dim3 grid_intre_c((num_mb_hor>>1),num_mb_ver,1); //一个block处理两种分量的2个8*8宏块，也就是共同4个块
				dim3 threads_intre_c(8,4,2); //每一个block有64个线程，前32个线程处理一种分量，每一个宏块需要16个线程

				//帧间编码的kernel配置参数
				dim3 grid_intra_c(p_enc->i_slice_num,1,1); 
				dim3 threads_intra_c(2,4,1); 
				
				ChromaPFrameInterResidualCoding_kernel<<<grid_intre_c,threads_intre_c>>> (dev_input_uv,
																		  dev_pred_uv,
																		  dev_recon_uv+RefStride2BeginUV,
																		  dev_dct_coefs_uv,
																		  dev_dc_coefs_uv,
																		  Quant_tab_uv,
																		  Dquant_tab_uv,
																		  dev_QpData_uv,
																		  enc_width_c,
																		  enc_height_c,
																		  width_ref_c,
																		  height_ref_c,
																		  num_mb_hor,
																		  num_mb_ver
																		);
				
				end = clock();
				p_enc->new_timers.pframe_residual_chroma += (end-start);
				p_enc->new_timers.pframe_residual_inter += (end-start);
				p_enc->new_timers.pframe_total += (end -start);
				start = clock();

				Chroma_PFrame_Intra_ResidualCoding_kernel<<<grid_intra_c,threads_intra_c>>> (dev_input_uv,
																			  dev_recon_uv+RefStride2BeginUV,
																			  dev_blk_mb_info,
																			  dev_dct_coefs_uv,
																			  dev_dc_coefs_uv,
																			  Quant_tab_uv,
																			  Dquant_tab_uv,
																			  dev_QpData_uv,
																			  enc_width_c,
																			  enc_height_c,
																			  width_ref_c,
																			  height_ref_c,
																			  num_mb_hor,
																			  num_mb_ver,
																			  p_enc->i_slice_num
																		);
				end = clock();
				p_enc->new_timers.pframe_residual_chroma += (end-start);
				p_enc->new_timers.pframe_residual_intra += (end-start);
				p_enc->new_timers.pframe_total += (end -start);
				start = clock();

				dim3 grid_cbp_luma(num_mb_hor/8, num_mb_ver, 1);
				dim3 threads_cbp_luma(BLK_WIDTH, BLK_HEIGHT, 8);
				CalcCBP_and_TotalCoeff_Luma_cuda<<<grid_cbp_luma,threads_cbp_luma>>>(dev_dct_coefs,
																				  dev_blk_mb_info);

				dim3 grid_cbp_chroma(num_mb_hor/8, num_mb_ver, 1);
				dim3 threads_cbp_chroma(BLOCKS_PER_MB_C, 8, 2);
				
				CalcCBP_and_TotalCoeff_Chroma_cuda<<<grid_cbp_chroma,threads_cbp_chroma>>>(dev_dct_coefs_uv,
																						dev_dc_coefs_uv,
																						dev_blk_mb_info,enc_width_c,
																						enc_height_c);


				cutilSafeCall(cudaMemcpy(p_enc->pBlkMBInfo,dev_blk_mb_info,BLOCKS_PER_MB*num_mbs*sizeof(S_BLK_MB_INFO),cudaMemcpyDeviceToHost));

				cutilSafeCall(cudaFree(dev_pred_uv));
				cutilSafeCall(cudaFree(dev_ref_uv));	

				cutilSafeCall(cudaFree(dev_input_ref));
				cutilSafeCall(cudaFree(dev_input_src));
				cutilSafeCall(cudaFree(dev_out_HR_src));
				cutilSafeCall(cudaFree(dev_out_HR_ref));
				cutilSafeCall(cudaFree(dev_out_QR_src));
				cutilSafeCall(cudaFree(dev_out_QR_ref));
				cutilSafeCall(cudaFree(dev_mvsLocal));
				cutilSafeCall(cudaFree(dev_mb_info));
				cutilSafeCall(cudaFree(IntegerPelCenterVecs));
				cutilSafeCall(cudaFree(integer_mvmap));
				cutilSafeCall(cudaFree(mb_info));  
				cutilSafeCall(cudaFree(dev_out_pred)); 
				cutilSafeCall(cudaFree(dev_CoeffCosts));
				
				
				end = clock();
				p_enc->new_timers.pframe_residual_chroma += (end-start);
				p_enc->new_timers.pframe_residual_intra += (end-start);
				p_enc->new_timers.pframe_total += (end -start);

				start = clock();
				frame_sad_sum = 0;
				for(j = 0; j < num_mb_ver; j++)
				{
					for(i = 0; i < num_mb_hor; i++)
					{
						frame_sad_sum += (pBlkMBInfo + (j * num_mb_hor + i) * BLOCKS_PER_MB)->MinSAD;
						
					}
				}
				p_enc->avg_mb_sad = frame_sad_sum / (num_mb_hor * num_mb_ver);
				end = clock();
				p_enc->new_timers.pframe_total += (end -start);
		}
		

			//CAVLC implementation based cuda
			int I_Slice, FrameQP;
			bitstream_t *pBitstream;
			int MBx, MBy;
			int PrevQP;
			unsigned int PackedCount;

			int pPackedSize;
			unsigned int     *pPacked;
			unsigned int     *pPackedCurr;
			int dummy;
			int num_encoded_mbs;
			int MBNum;
			int *PrevSkipMB = (int *)malloc(p_enc->i_slice_num*sizeof(int)); 
			int *header_bits =(int *)malloc(sizeof(int)) ;
			int *texture_bits= (int *)malloc(p_enc->i_slice_num*sizeof(int)); 
			/////////////////////////////////////////////////////////////////
			// Declare temporary buffers and pointers
			/////////////////////////////////////////////////////////////////
			//int               leftover_numbits;
			unsigned int      leftover_value;

			// Read necessary information from encoder context struct (p_enc)
			num_encoded_mbs = 0;
			// Bitstream buffer, before inserting into pBitstream
			pPackedSize = p_enc->bitstream.buffer_size / 4;
			pPacked = (unsigned int *)malloc(sizeof(unsigned int) * pPackedSize);
			pPackedCurr = pPacked;
			I_Slice =  p_enc->frame_info.idr_flag;
			int Slice_num = p_enc->i_slice_num; 
			*header_bits=0;
			*texture_bits=0;
			//leftover_numbits = 0;
			leftover_value = 0;
			start = clock();
			
			//short  *pDcCoefs_ChromaDC;
			int  *ZigZag_tab;
			S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_LumaDC_dev;
			S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_ChromaDC_dev;
			int *SkipBlock;
			int *PrevSkipMB_dev;
			S_CAVLC_CONTEXT_BLOCK *pMBContextOut_LumaAC_dev;

			//short  *pDctCoefs_ChromaAC;
			short  *pDctCoefs_ZigZag_ChromaAC;
			S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_ChromaAC_dev;

			S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_LumaDC_dev;
			S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_LumaDC_dev;
			S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_LumaDC_dev;
			S_RUN_SYMBOLS_BLOCK     *pRunSymbols_LumaDC_dev;

			S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_LumaAC_dev;
			S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_LumaAC_dev;
			S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_LumaAC_dev;
			S_RUN_SYMBOLS_BLOCK     *pRunSymbols_LumaAC_dev;

			S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_ChromaDC_dev;
			S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_ChromaDC_dev;
			S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_ChromaDC_dev;
			S_RUN_SYMBOLS_BLOCK     *pRunSymbols_ChromaDC_dev;

			S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_ChromaAC_dev;
			S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_ChromaAC_dev;
			S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_ChromaAC_dev;
			S_RUN_SYMBOLS_BLOCK     *pRunSymbols_ChromaAC_dev;

			SINGLE_CODE *pCodes_LumaDC_dev;

			SINGLE_CODE *pCodes_LumaAC_dev;

			SINGLE_CODE *pCodes_ChromaAC_dev;
			unsigned char *CoeffTokenTable_dev;
			unsigned char *TotalZerosTable_dev;
			unsigned int *RunIndexTable_dev;
			unsigned char *RunTable_dev;

			cutilSafeCall(cudaMalloc((void**) &pCodes_LumaDC_dev,CODE_PAIRS_PER_LANE*num_mbs*sizeof(SINGLE_CODE))); 

			cutilSafeCall(cudaMalloc((void**) &pCodes_LumaAC_dev,BLOCKS_PER_MB*CODE_PAIRS_PER_LANE*num_mbs*sizeof(SINGLE_CODE))); 

			cutilSafeCall(cudaMalloc((void**) &pCodes_ChromaAC_dev,8*CODE_PAIRS_PER_LANE*num_mbs*sizeof(SINGLE_CODE))); 

			cutilSafeCall(cudaMalloc((void**) &CoeffTokenTable_dev,3*4*17*2*sizeof(unsigned char))); 
			cutilSafeCall(cudaMalloc((void**) &TotalZerosTable_dev,15*16*2*sizeof(unsigned char)));
			cutilSafeCall(cudaMalloc((void**) &RunIndexTable_dev,7*sizeof(unsigned int))); 
			cutilSafeCall(cudaMalloc((void**) &RunTable_dev,44*2*sizeof(unsigned char)));

			cutilSafeCall(cudaMemcpy(CoeffTokenTable_dev,CoeffTokenTable,3*4*17*2*sizeof(unsigned char),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(TotalZerosTable_dev,TotalZerosTable,15*16*2*sizeof(unsigned char),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(RunIndexTable_dev,RunIndexTable,7*sizeof(unsigned int),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(RunTable_dev,RunTable,44*2*sizeof(unsigned char),cudaMemcpyHostToDevice));

			cutilSafeCall(cudaMemset(pCodes_LumaDC_dev,0,CODE_PAIRS_PER_LANE*num_mbs*sizeof(SINGLE_CODE))); 
			cutilSafeCall(cudaMemset(pCodes_LumaAC_dev,0,16*CODE_PAIRS_PER_LANE*num_mbs*sizeof(SINGLE_CODE))); 
			cutilSafeCall(cudaMemset(pCodes_ChromaAC_dev,0,8*CODE_PAIRS_PER_LANE*num_mbs*sizeof(SINGLE_CODE))); 

			cutilSafeCall(cudaMalloc((void**) &pMBContextOut_LumaDC_dev,num_mbs*sizeof(S_CAVLC_CONTEXT_DC_CHROMA))); 
			cutilSafeCall(cudaMalloc((void**) &pMBContextOut_ChromaDC_dev,num_mbs*sizeof(S_CAVLC_CONTEXT_DC_CHROMA)));

			//block texture for DC
			dim3 threads_blk_dc(16, 5, 1); 
			dim3 grid_blk_dc(num_mbs/80, 1, 1);
			cavlc_block_context_DC_kernel <<<grid_blk_dc,threads_blk_dc>>>(dev_dc_coefs,
																			 dev_blk_mb_info,
																			 dev_ZigZag/*ZigZag_tab*/,
																			 dev_dc_coefs,
																			 pMBContextOut_LumaDC_dev,
																			 pMBContextOut_ChromaDC_dev,
																			 num_mb_hor
																			);
			//block contexture for Luma AC
			cutilSafeCall(cudaMalloc((void**) &pMBContextOut_LumaAC_dev,BLOCKS_PER_MB*num_mbs*sizeof(S_CAVLC_CONTEXT_BLOCK))); 
			cutilSafeCall(cudaMalloc((void**) &SkipBlock,sizeof(int)*num_mbs));
			cutilSafeCall(cudaMalloc((void**) &PrevSkipMB_dev,sizeof(int)*Slice_num));

			dim3 threads_luma_ac(16, 8, 1); 
			dim3 grid_luma_ac(num_mb_hor/8, num_mb_ver, 1);
			if(I_Slice )
			{
				cutilSafeCall(cudaMemset(SkipBlock,0,sizeof(int)*num_mbs));
				cutilSafeCall(cudaMemset(PrevSkipMB_dev,0,sizeof(int)*Slice_num));
				cavlc_block_context_iframe_LumaAC_kernel<<<grid_luma_ac,threads_luma_ac>>> (  dev_dct_coefs,
																							  dev_blk_mb_info,
																							  dev_ZigZag,
																							  dev_dct_coefs,
																							  pMBContextOut_LumaAC_dev,\
																							  num_mb_hor
																							);
			}
			else
			{
				cavlc_block_context_iframe_LumaAC_kernel<<<grid_luma_ac,threads_luma_ac>>> (dev_dct_coefs,
																			dev_blk_mb_info,
																			dev_ZigZag,
																			dev_dct_coefs,
																			pMBContextOut_LumaAC_dev,
																			num_mb_hor
																			);
				dim3 threads_mv(80, 1, 1); 
				dim3 grid_mv(num_mbs/80, 1, 1);
				CalcPredictedMVRef_16x16_kernel<<<grid_mv,threads_mv>>>( dev_blk_mb_info,
																			pMBContextOut_LumaAC_dev,
																			SkipBlock,
																			num_mb_hor
																			);
				
				
				dim3 threads_skip(16, 1, 1); 
				dim3 grid_skip(Slice_num, 1, 1);
				cavlc_block_context_PrevSkipMB_kernel<<<grid_skip,threads_skip>>> (
																					SkipBlock,
																					PrevSkipMB_dev,
																					pMBContextOut_LumaAC_dev,
																					num_mbs
																					);	
			}

			//block contexture for chroma AC
			//cutilSafeCall(cudaMalloc((void**) &pDctCoefs_ChromaAC,MB_TOTAL_SIZE_C*2*num_mbs*sizeof(short))); 
			cutilSafeCall(cudaMalloc((void**) &pDctCoefs_ZigZag_ChromaAC,MB_TOTAL_SIZE_C*2*num_mbs*sizeof(short))); 
			cutilSafeCall(cudaMalloc((void**) &pMBContextOut_ChromaAC_dev,BLOCKS_PER_MB_C*2*num_mbs*sizeof(S_CAVLC_CONTEXT_DC_CHROMA)));

			dim3 threads_blk_chrac(16, 4, 2); 
			dim3 grid_blk_chrac(num_mbs/16, 1, 1);
			cavlc_block_context_ChromaAC_kernel <<<grid_blk_chrac,threads_blk_chrac>>>(dev_dct_coefs_uv,
																						 dev_blk_mb_info,
																						 dev_ZigZag,
																						 pDctCoefs_ZigZag_ChromaAC,
																						 pMBContextOut_ChromaAC_dev,
																						 num_mb_hor,
																						 num_mb_ver
																						);

			//texture symbols for luma DC
			cutilSafeCall(cudaMalloc((void**) &pTextureSymbols_LumaDC_dev,num_mbs*sizeof(S_TEXTURE_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbolSuffixLength0_LumaDC_dev,num_mbs*sizeof(S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK))); 
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbols_LumaDC_dev,BLOCKS_PER_MB*num_mbs*sizeof(S_LEVEL_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pRunSymbols_LumaDC_dev,BLOCKS_PER_MB*num_mbs*sizeof(S_RUN_SYMBOLS_BLOCK)));

			dim3 threads_sym_luma_dc(16, 5, 1); 
			dim3 grid_sym_luma_dc(num_mbs/80, 1, 1);
			cavlc_texture_symbols_luma_DC_kernel <<<grid_sym_luma_dc,threads_sym_luma_dc>>>(dev_dc_coefs,
																							 pMBContextOut_LumaDC_dev,
																							 SkipBlock,
																							 pTextureSymbols_LumaDC_dev,
																							 pLevelSymbolSuffixLength0_LumaDC_dev,
																							 pLevelSymbols_LumaDC_dev,
																							 pRunSymbols_LumaDC_dev
																							);

			//texture symbols for luma ac
			cutilSafeCall(cudaMalloc((void**) &pTextureSymbols_LumaAC_dev,num_mbs*BLK_SIZE*sizeof(S_TEXTURE_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbolSuffixLength0_LumaAC_dev,num_mbs*BLK_SIZE*sizeof(S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK))); 
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbols_LumaAC_dev,BLOCKS_PER_MB*num_mbs*BLK_SIZE*sizeof(S_LEVEL_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pRunSymbols_LumaAC_dev,BLOCKS_PER_MB*num_mbs*BLK_SIZE*sizeof(S_RUN_SYMBOLS_BLOCK)));

			dim3 threads_sym_luma_ac(16, 8, 1); 
			dim3 grid_sym_luma_ac(num_mb_hor/8, num_mb_ver, 1);
			cavlc_texture_symbols_luma_AC_kernel <<<grid_sym_luma_ac,threads_sym_luma_ac>>>(dev_dct_coefs,
																							 pMBContextOut_LumaAC_dev,
																							 SkipBlock,
																							 pTextureSymbols_LumaAC_dev,
																							 pLevelSymbolSuffixLength0_LumaAC_dev,
																							 pLevelSymbols_LumaAC_dev,
																							 pRunSymbols_LumaAC_dev
																							);

			cutilSafeCall(cudaMalloc((void**) &pTextureSymbols_ChromaDC_dev,num_mbs*2*sizeof(S_TEXTURE_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbolSuffixLength0_ChromaDC_dev,num_mbs*2*sizeof(S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK))); 
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbols_ChromaDC_dev,BLOCKS_PER_MB_C*2*num_mbs*sizeof(S_LEVEL_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pRunSymbols_ChromaDC_dev,BLOCKS_PER_MB_C*2*num_mbs*sizeof(S_RUN_SYMBOLS_BLOCK)));

			dim3 threads_sym_chroma_dc(16, 5, 2); 
			dim3 grid_sym_chroma_dc(num_mbs/80, 1, 1);
			cavlc_texture_symbols_chroma_DC_kernel <<<grid_sym_chroma_dc,threads_sym_chroma_dc>>>(dev_dc_coefs_uv,
																								 pMBContextOut_ChromaDC_dev,
																								 SkipBlock,
																								 pTextureSymbols_ChromaDC_dev,
																								 pLevelSymbolSuffixLength0_ChromaDC_dev,
																								 pLevelSymbols_ChromaDC_dev,
																								 pRunSymbols_ChromaDC_dev,
																								 num_mbs
																								);
			//texture symbols for chroma ac
			cutilSafeCall(cudaMalloc((void**) &pTextureSymbols_ChromaAC_dev,BLOCKS_PER_MB_C*num_mbs*2*sizeof(S_TEXTURE_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbolSuffixLength0_ChromaAC_dev,BLOCKS_PER_MB_C*num_mbs*2*sizeof(S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK))); 
			cutilSafeCall(cudaMalloc((void**) &pLevelSymbols_ChromaAC_dev,MB_TOTAL_SIZE_C*2*num_mbs*sizeof(S_LEVEL_SYMBOLS_BLOCK)));
			cutilSafeCall(cudaMalloc((void**) &pRunSymbols_ChromaAC_dev,MB_TOTAL_SIZE_C*2*num_mbs*sizeof(S_RUN_SYMBOLS_BLOCK)));
			dim3 threads_sym_chroma_ac(16, 5, 1); 
			dim3 grid_sym_chroma_ac(num_mb_hor/10, num_mb_ver, 1);
			cavlc_texture_symbols_chroma_AC_kernel <<<grid_sym_chroma_ac,threads_sym_chroma_ac>>>(pDctCoefs_ZigZag_ChromaAC,
																									 pMBContextOut_ChromaAC_dev,
																									 SkipBlock,
																									 pTextureSymbols_ChromaAC_dev,
																									 pLevelSymbolSuffixLength0_ChromaAC_dev,
																									 pLevelSymbols_ChromaAC_dev,
																									 pRunSymbols_ChromaAC_dev
																									);

			dim3 threads_code_luma_dc(16, 5, 1); 
			dim3 grid_code_luma_dc(num_mbs/80, 1, 1);
			cavlc_texture_codes_luma_DC_kernel <<<grid_code_luma_dc,threads_code_luma_dc>>>(
																							 pTextureSymbols_LumaDC_dev,
																							 pLevelSymbolSuffixLength0_LumaDC_dev,
																							 pLevelSymbols_LumaDC_dev,
																							 pRunSymbols_LumaDC_dev,
																							 CoeffTokenTable_dev,
																							 TotalZerosTable_dev,
																							 RunIndexTable_dev,
																							 RunTable_dev,
																							 SkipBlock,
																							 pCodes_LumaDC_dev,
																							 texture_bits,
																							 num_mbs
																							);
			dim3 threads_code_luma_ac(16, 5, 1); 
			dim3 grid_code_luma_ac(num_mb_hor/5, num_mb_ver, 1);
			cavlc_texture_codes_luma_DC_kernel <<<grid_code_luma_ac,threads_code_luma_ac>>>(
																							 pTextureSymbols_LumaAC_dev,
																							 pLevelSymbolSuffixLength0_LumaAC_dev,
																							 pLevelSymbols_LumaAC_dev,
																							 pRunSymbols_LumaAC_dev,
																							 CoeffTokenTable_dev,
																							 TotalZerosTable_dev,
																							 RunIndexTable_dev,
																							 RunTable_dev,
																							 SkipBlock,
																							 pCodes_LumaAC_dev,
																							 texture_bits,
																							 num_mbs
																							);

			dim3 threads_chr(16, 5, 1); 
			dim3 grid_chr(num_mb_hor/10, num_mb_ver, 1);
			cavlc_texture_codes_luma_DC_kernel <<<grid_chr,threads_chr>>>(
																	 pTextureSymbols_ChromaAC_dev,
																	 pLevelSymbolSuffixLength0_ChromaAC_dev,
																	 pLevelSymbols_ChromaAC_dev,
																	 pRunSymbols_ChromaAC_dev,
																	 CoeffTokenTable_dev,
																	 TotalZerosTable_dev,
																	 RunIndexTable_dev,
																	 RunTable_dev,
																	 SkipBlock,
																	 pCodes_ChromaAC_dev,
																	 texture_bits,
																	 num_mbs
																	);

			SINGLE_CODE *pCodes_ChromaDC_dev;

			unsigned char *CoeffTokenChromaDCTable_dev;
			unsigned char *TotalZerosChromaDCTable_dev;

			cutilSafeCall(cudaMalloc((void**) &pCodes_ChromaDC_dev,8*2*num_mbs*sizeof(SINGLE_CODE))); 

			cutilSafeCall(cudaMalloc((void**) &CoeffTokenChromaDCTable_dev,4*5*2*sizeof(unsigned char))); 
			cutilSafeCall(cudaMalloc((void**) &TotalZerosChromaDCTable_dev,3*4*2*sizeof(unsigned char)));

			cutilSafeCall(cudaMemcpy(CoeffTokenChromaDCTable_dev,CoeffTokenChromaDCTable,4*5*2*sizeof(unsigned char),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(TotalZerosChromaDCTable_dev,TotalZerosChromaDCTable,3*4*2*sizeof(unsigned char),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemset(pCodes_ChromaDC_dev,0,8*2*num_mbs*sizeof(SINGLE_CODE))); 

			dim3 threads_code_chroma_dc(16, 10, 1); 
			dim3 grid_code_chroma_dc(num_mbs/80, 1, 1);
			cavlc_texture_codes_chroam_DC_kernel<<<grid_code_chroma_dc,threads_code_chroma_dc>>>(
																								 pTextureSymbols_ChromaDC_dev,
																								 pLevelSymbolSuffixLength0_ChromaDC_dev,
																								 pLevelSymbols_ChromaDC_dev,
																								 pRunSymbols_ChromaDC_dev,
																								 CoeffTokenChromaDCTable_dev,
																								 TotalZerosChromaDCTable_dev,
																								 RunIndexTable_dev,
																								 RunTable_dev,
																								 SkipBlock,
																								 pCodes_ChromaDC_dev,
																								 texture_bits
																							   );

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			cutilSafeCall(cudaMemcpy(PrevSkipMB,PrevSkipMB_dev,Slice_num*sizeof(int),cudaMemcpyDeviceToHost));
			
			//对宏块头进行编码
			unsigned char *CBPTable_dev;
			int *HeaderCodeBits_dev;
			SINGLE_CODE *pCodes_Header_MB_dev;
			//SINGLE_CODE *pCodes_Header_MB_dev;
			unsigned int        *packed_words_head_dev;                   
			unsigned int        *word_count_head_dev;
			int                 *leftover_numbits_head_dev;
			unsigned int        *leftover_value_head_dev;

			unsigned int        *packed_words_LDC_dev;                   
			unsigned int        *word_count_LDC_dev;
			int                 *leftover_numbits_LDC_dev;
			unsigned int        *leftover_value_LDC_dev;

			unsigned int        *packed_words_LAC_dev;                   
			unsigned int        *word_count_LAC_dev;
			int                 *leftover_numbits_LAC_dev;
			unsigned int        *leftover_value_LAC_dev;

			unsigned int        *packed_words_CDC_dev;                   
			unsigned int        *word_count_CDC_dev;
			int                 *leftover_numbits_CDC_dev;
			unsigned int        *leftover_value_CDC_dev;

			unsigned int        *packed_words_CAC_dev;                   
			unsigned int        *word_count_CAC_dev;
			int                 *leftover_numbits_CAC_dev;
			unsigned int        *leftover_value_CAC_dev;

			unsigned int        *total_packet_word_mb;
			unsigned int        *total_word_count_mb;
			int					*total_leftover_numbits_mb;
			unsigned int        *total_leftover_value_mb;

			int					*shift_bits_dev;
			unsigned int		*out_index_dev;
			unsigned int		*total_packet_word;
			int					*leftover_numbits_slice;
			unsigned int		*leftover_value_slice;
			unsigned int		*word_num_slice;

			
			cutilSafeCall(cudaMalloc((void**) &CBPTable_dev,CBP_TABLE_SIZE*sizeof(unsigned char))); 
			cutilSafeCall(cudaMalloc((void**) &HeaderCodeBits_dev,num_mbs*sizeof( int)));

			cutilSafeCall(cudaMemcpy(CBPTable_dev,CBPTable,CBP_TABLE_SIZE*sizeof(unsigned char),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(dev_ZigZag,BlockScan,16*sizeof(unsigned int),cudaMemcpyHostToDevice));

			cutilSafeCall(cudaMalloc((void**) &pCodes_Header_MB_dev,11*num_mbs*sizeof(SINGLE_CODE))); 
			cutilSafeCall(cudaMalloc((void**) &packed_words_head_dev,6*num_mbs*sizeof(unsigned int))); 
			cutilSafeCall(cudaMalloc((void**) &word_count_head_dev,num_mbs*sizeof(unsigned int))); 
			cutilSafeCall(cudaMalloc((void**) &leftover_numbits_head_dev,num_mbs*sizeof( int))); 
			cutilSafeCall(cudaMalloc((void**) &leftover_value_head_dev,num_mbs*sizeof(unsigned int))); 

			cutilSafeCall(cudaMalloc((void**) &leftover_numbits_slice,(p_enc->i_slice_num+1)*sizeof (int))); 
			cutilSafeCall(cudaMalloc((void**) &leftover_value_slice,p_enc->i_slice_num*sizeof(unsigned int))); 
			cutilSafeCall(cudaMalloc((void**) &word_num_slice,p_enc->i_slice_num*sizeof(unsigned int))); 

			cutilSafeCall(cudaMemset(pCodes_Header_MB_dev,0,11*num_mbs*sizeof(SINGLE_CODE)));

			dim3 block_header(num_mb_hor,1,1);
			dim3 grid_header(num_mb_ver,1,1);
			int	Max_size_head = 11;//P frame mb
			int shared_mem_size = ((Max_size_head+1)>>1) * num_mb_hor*4;
			if (I_Slice!=0) //I frame
			{
				cavlc_header_codes_Iframe <<<grid_header,block_header>>>
																			(   pMBContextOut_LumaAC_dev,
																				dev_ZigZag,
																				CBPTable_dev,
																				8,
																				pCodes_Header_MB_dev, //8 element for a I MB 1+4+1+1+1(mbtype,subtype(4*4),CHROMAMODE,CBP,delta_quant) 
																				HeaderCodeBits_dev
																			);
				
			}
			else //p frame
			{
				cavlc_header_codes_Pframe <<<grid_header,block_header>>>
																			(   pMBContextOut_LumaAC_dev,
																				SkipBlock,
																				dev_ZigZag,
																				CBPTable_dev,
																				11,
																				pCodes_Header_MB_dev, //8 element for a I MB 1+4+1+1+1(mbtype,subtype(4*4),CHROMAMODE,CBP,delta_quant) 
																				HeaderCodeBits_dev
																			);
				
				
			}
				
				cavlc_bitpack_block_cu<<<grid_header,block_header,shared_mem_size>>>
																				 (
																					 pCodes_Header_MB_dev,
																					 ((I_Slice) ? 8 : 11),
																					 packed_words_head_dev,
																					                 
																					 word_count_head_dev,
																					 leftover_numbits_head_dev,
																					 leftover_value_head_dev
																				 );

				int shared_mem_size_ldc = 13*4*num_mb_hor;
				int shared_mem_size_cdc = 8*4*num_mb_hor;
				int shared_mem_size_ac = 13*4*128;

				cutilSafeCall(cudaMalloc((void**) &packed_words_LDC_dev,13*num_mbs*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &word_count_LDC_dev,num_mbs*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_numbits_LDC_dev,num_mbs*sizeof( int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_value_LDC_dev,num_mbs*sizeof(unsigned int)));

				cutilSafeCall(cudaMalloc((void**) &packed_words_LAC_dev,13*num_mbs*BLOCKS_PER_MB*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &word_count_LAC_dev,num_mbs*BLOCKS_PER_MB*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_numbits_LAC_dev,num_mbs*BLOCKS_PER_MB*sizeof( int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_value_LAC_dev,num_mbs*BLOCKS_PER_MB*sizeof(unsigned int)));

				cutilSafeCall(cudaMalloc((void**) &packed_words_CDC_dev,8*num_mbs*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &word_count_CDC_dev,num_mbs*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_numbits_CDC_dev,num_mbs*sizeof( int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_value_CDC_dev,num_mbs*sizeof(unsigned int)));

				cutilSafeCall(cudaMalloc((void**) &packed_words_CAC_dev,13*num_mbs*BLOCKS_PER_MB_C*2*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &word_count_CAC_dev,num_mbs*BLOCKS_PER_MB_C*2*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_numbits_CAC_dev,num_mbs*BLOCKS_PER_MB_C*2*sizeof( int))); 
				cutilSafeCall(cudaMalloc((void**) &leftover_value_CAC_dev,num_mbs*BLOCKS_PER_MB_C*2*sizeof(unsigned int)));

				cutilSafeCall(cudaMalloc((void**) &total_packet_word_mb,num_mbs*64*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &total_word_count_mb,num_mbs*sizeof(unsigned int))); 
				cutilSafeCall(cudaMalloc((void**) &total_leftover_numbits_mb,num_mbs*sizeof( int))); 
				cutilSafeCall(cudaMalloc((void**) &total_leftover_value_mb,num_mbs*sizeof(unsigned int)));

				cutilSafeCall(cudaMalloc((void**) &shift_bits_dev,num_mbs*sizeof(int)));
				cutilSafeCall(cudaMalloc((void**) &out_index_dev,num_mbs*sizeof(unsigned int)));
				cutilSafeCall(cudaMalloc((void**) &total_packet_word,num_mbs*64*sizeof(unsigned int)));

				cutilSafeCall(cudaMemset(total_packet_word_mb,0,num_mbs*64*sizeof(unsigned int)));
				
				//直流分量的kernel配置
				dim3 block_ldc(num_mb_hor,1,1);
				dim3 grid_ldc(num_mb_ver,1,1);
				//交流Luma分量的kernel配置
				dim3 block_lac(128,1,1);
				dim3 grid_lac((num_mbs*BLOCKS_PER_MB/128),1,1);
				//交流Chroma分量的kernel配置
				dim3 block_cac(128,1,1);
				dim3 grid_cac((num_mbs*BLOCKS_PER_MB_C/64),1,1);
				
				//Luma Dc
				cavlc_bitpack_block_cu<<<grid_ldc,block_ldc,shared_mem_size_ldc>>>
																		 (
																			 pCodes_LumaDC_dev,
																			 26,
																			 packed_words_LDC_dev,             
																			 word_count_LDC_dev,
																			 leftover_numbits_LDC_dev,
																			 leftover_value_LDC_dev
																		 );
				//Chroma Dc
				cavlc_bitpack_block_cu<<<grid_ldc,block_ldc,shared_mem_size_cdc>>>
																		 (
																			 pCodes_ChromaDC_dev,
																			 16,
																			 packed_words_CDC_dev,             
																			 word_count_CDC_dev,
																			 leftover_numbits_CDC_dev,
																			 leftover_value_CDC_dev
																		 );
				//Luma Ac
				cavlc_bitpack_block_cu<<<grid_lac,block_lac,shared_mem_size_ac>>>
																		 (
																			 pCodes_LumaAC_dev,
																			 26,
																			 packed_words_LAC_dev,
																			 word_count_LAC_dev,
																			 leftover_numbits_LAC_dev,
																			 leftover_value_LAC_dev
																		 );
				//Chroma Ac
				cavlc_bitpack_block_cu<<<grid_cac,block_cac,shared_mem_size_ac>>>
																		 (
																			 pCodes_ChromaAC_dev,
																			 26,
																			 packed_words_CAC_dev,              
																			 word_count_CAC_dev,
																			 leftover_numbits_CAC_dev,
																			 leftover_value_CAC_dev
																		 );
				cavlc_bitpack_MB_cu<<<grid_ldc,block_ldc>>>(
															 //intput packet codes of head,lumadc,lumaac,chromadc...
															packed_words_head_dev,
															packed_words_LDC_dev,
															packed_words_LAC_dev,
															packed_words_CDC_dev,
															packed_words_CAC_dev,

															 word_count_head_dev,							 
															 word_count_LDC_dev,
															 word_count_LAC_dev,
															 word_count_CDC_dev,
															 word_count_CAC_dev,

															 leftover_numbits_head_dev,
															 leftover_numbits_LDC_dev,
															 leftover_numbits_LAC_dev,
															 leftover_numbits_CDC_dev,
															 leftover_numbits_CAC_dev,
															 leftover_value_head_dev,
															 leftover_value_LDC_dev,
															 leftover_value_LAC_dev,
															 leftover_value_CDC_dev,
															 leftover_value_CAC_dev,

															 dev_ZigZag,
															 64,
															 ((I_Slice) ? 4 : 6),
															 SkipBlock,
															//ouput packet words for mb
															 total_packet_word_mb,
															 total_word_count_mb,
															 total_leftover_numbits_mb,
															 total_leftover_value_mb
														 );
			dim3 block(num_mbs/p_enc->i_slice_num,1,1);
			dim3 grid(p_enc->i_slice_num,1,1);
			compute_out_position<<<grid,block>>>(
											//input: word of mb and leftover_numbits
											 total_word_count_mb,
											 total_leftover_numbits_mb,
											 //output: out position for mb and shift bits
											 out_index_dev,
											 shift_bits_dev
										);
			parallel_write<<<grid,block>>>(
											total_packet_word_mb,
											total_word_count_mb,
											SkipBlock,

											total_leftover_numbits_mb,
											total_leftover_value_mb,
											out_index_dev,
											shift_bits_dev,
											num_mbs/p_enc->i_slice_num,
											//out_put packet word for slice
											total_packet_word,
											word_num_slice,
											leftover_numbits_slice,
											leftover_value_slice
										);

			unsigned int *pCodes_packed = (unsigned int*) malloc(BLOCKS_PER_MB*13*num_mbs*sizeof(unsigned int));
			unsigned int *word_count = (unsigned int*) malloc(BLOCKS_PER_MB*num_mbs*sizeof(unsigned int));
			int *leftover_numbits = ( int*) malloc(BLOCKS_PER_MB*num_mbs*sizeof( int));
			unsigned int *left_value = (unsigned int*) malloc(BLOCKS_PER_MB*num_mbs*sizeof(unsigned int));

			cutilSafeCall(cudaMemcpy(word_count,word_num_slice,(p_enc->i_slice_num)*sizeof(unsigned int),cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(pCodes_packed,total_packet_word,word_count[p_enc->i_slice_num-1]*sizeof(unsigned int),cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(leftover_numbits,leftover_numbits_slice,(p_enc->i_slice_num)*sizeof( int),cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(left_value,leftover_value_slice,p_enc->i_slice_num*sizeof(unsigned int),cudaMemcpyDeviceToHost));

			pPackedCurr = pCodes_packed;
			int num_word = 0;
			for( i = 0;i<p_enc->i_slice_num;i++)
			{
				encoder_context_t *penc = p_enc->slice[i];
				pBitstream  = &penc->bitstream;
				num_word = word_count[i]-num_word;
				cavlc_put_bits(pPackedCurr, num_word, pBitstream);

				if (leftover_numbits[i])
					put_bits(pBitstream, leftover_numbits[i], left_value[i] >> (32 - leftover_numbits[i]));
				pPackedCurr +=num_word;
				num_word = word_count[i];
				header_bits += write_last_skip_count(PrevSkipMB[i], pBitstream);
			}
			start1 = clock();

			cutilSafeCall(cudaFree(pCodes_Header_MB_dev));
			cutilSafeCall(cudaFree(packed_words_head_dev));
			cutilSafeCall(cudaFree(word_count_head_dev));
			cutilSafeCall(cudaFree(leftover_numbits_head_dev));
			cutilSafeCall(cudaFree(leftover_value_head_dev));


			cutilSafeCall(cudaFree(packed_words_LDC_dev));                   
			cutilSafeCall(cudaFree(word_count_LDC_dev));
			cutilSafeCall(cudaFree(leftover_numbits_LDC_dev));
			cutilSafeCall(cudaFree(leftover_value_LDC_dev));

			cutilSafeCall(cudaFree(packed_words_LAC_dev));                  
			cutilSafeCall(cudaFree(word_count_LAC_dev));
			cutilSafeCall(cudaFree(leftover_numbits_LAC_dev));
			cutilSafeCall(cudaFree(leftover_value_LAC_dev));

			cutilSafeCall(cudaFree(packed_words_CDC_dev));                   
			cutilSafeCall(cudaFree(word_count_CDC_dev));
			cutilSafeCall(cudaFree(leftover_numbits_CDC_dev));
			cutilSafeCall(cudaFree(leftover_value_CDC_dev));

			cutilSafeCall(cudaFree(packed_words_CAC_dev));                   
			cutilSafeCall(cudaFree(word_count_CAC_dev));
			cutilSafeCall(cudaFree(leftover_numbits_CAC_dev));
			cutilSafeCall(cudaFree(leftover_value_CAC_dev));
			
			cutilSafeCall(cudaFree(out_index_dev));
			cutilSafeCall(cudaFree(shift_bits_dev));
			cutilSafeCall(cudaFree(total_packet_word));
			cutilSafeCall(cudaFree(leftover_numbits_slice));
			cutilSafeCall(cudaFree(leftover_value_slice));
			cutilSafeCall(cudaFree(word_num_slice));

			cutilSafeCall(cudaFree(total_packet_word_mb));
			cutilSafeCall(cudaFree(total_word_count_mb));
			cutilSafeCall(cudaFree(total_leftover_numbits_mb));
			cutilSafeCall(cudaFree(total_leftover_value_mb));

			cutilSafeCall(cudaFree(HeaderCodeBits_dev));
			cutilSafeCall(cudaFree(CBPTable_dev));

			cudaThreadSynchronize();
			end = clock();
			p_enc->new_timers.cavlc_timers += (end-start);
			p_enc->new_timers.rc_total += (end-start);

			cutilSafeCall(cudaFree(pMBContextOut_LumaDC_dev));

			cutilSafeCall(cudaFree(pMBContextOut_LumaAC_dev));
			cutilSafeCall(cudaFree(SkipBlock));
			cutilSafeCall(cudaFree(PrevSkipMB_dev));

			cutilSafeCall(cudaFree(pMBContextOut_ChromaDC_dev));

			cutilSafeCall(cudaFree(pDctCoefs_ZigZag_ChromaAC));
			cutilSafeCall(cudaFree(pMBContextOut_ChromaAC_dev));

			cutilSafeCall(cudaFree(pTextureSymbols_LumaDC_dev));
			cutilSafeCall(cudaFree(pLevelSymbolSuffixLength0_LumaDC_dev));
			cutilSafeCall(cudaFree(pLevelSymbols_LumaDC_dev));
			cutilSafeCall(cudaFree(pRunSymbols_LumaDC_dev));

			cutilSafeCall(cudaFree(pTextureSymbols_LumaAC_dev));
			cutilSafeCall(cudaFree(pLevelSymbolSuffixLength0_LumaAC_dev));
			cutilSafeCall(cudaFree(pLevelSymbols_LumaAC_dev));
			cutilSafeCall(cudaFree(pRunSymbols_LumaAC_dev));

			cutilSafeCall(cudaFree(pTextureSymbols_ChromaDC_dev));
			cutilSafeCall(cudaFree(pLevelSymbolSuffixLength0_ChromaDC_dev));
			cutilSafeCall(cudaFree(pLevelSymbols_ChromaDC_dev));
			cutilSafeCall(cudaFree(pRunSymbols_ChromaDC_dev));

			cutilSafeCall(cudaFree(pTextureSymbols_ChromaAC_dev));
			cutilSafeCall(cudaFree(pLevelSymbolSuffixLength0_ChromaAC_dev));
			cutilSafeCall(cudaFree(pLevelSymbols_ChromaAC_dev));
			cutilSafeCall(cudaFree(pRunSymbols_ChromaAC_dev));

			cutilSafeCall(cudaFree(pCodes_LumaDC_dev));

			cutilSafeCall(cudaFree(pCodes_LumaAC_dev));

			cutilSafeCall(cudaFree(pCodes_ChromaAC_dev));

			cutilSafeCall(cudaFree(CoeffTokenTable_dev));
			cutilSafeCall(cudaFree(TotalZerosTable_dev));
			cutilSafeCall(cudaFree(RunIndexTable_dev));
			cutilSafeCall(cudaFree(RunTable_dev));

			cutilSafeCall(cudaFree(pCodes_ChromaDC_dev));
			cutilSafeCall(cudaFree(CoeffTokenChromaDCTable_dev));
			cutilSafeCall(cudaFree(TotalZerosChromaDCTable_dev));

			cutilSafeCall(cudaFree(dev_input));
			cutilSafeCall(cudaFree(dev_dct_coefs));
			cutilSafeCall(cudaFree(dev_dc_coefs));
					
			cutilSafeCall(cudaFree(dev_Quant_tab));
			cutilSafeCall(cudaFree(dev_Dquant_tab));
			cutilSafeCall(cudaFree(dev_QpData));

			cutilSafeCall(cudaFree(dev_input_uv));
			cutilSafeCall(cudaFree(Quant_tab_uv));
			cutilSafeCall(cudaFree(Dquant_tab_uv));
			cutilSafeCall(cudaFree(dev_QpData_uv));
			cutilSafeCall(cudaFree(dev_dct_coefs_uv));
			cutilSafeCall(cudaFree(dev_dc_coefs_uv));
			cutilSafeCall(cudaFree(dev_ZigZag));

			free(pCodes_packed);
			free(PrevSkipMB);
			free(word_count);
			free(leftover_numbits);
			free(left_value);
			end1 = clock();
			p_enc->new_timers.prep_encode_frame += (end1 - start1);
			E_ERR err = ERR_SUCCESS;
			start = clock();

			S_BLK_MB_INFO *pBlkMBInfo          = p_enc->pBlkMBInfo;
			int disable_deblocking_filter_idc  = p_enc->loopfilter_params.disable_flag;
			int slice_alpha_c0_offset          = p_enc->loopfilter_params.alpha_c0_offset;
			int slice_beta_offset              = p_enc->loopfilter_params.beta_offset;
			yuv_frame_t *frame                 = p_enc->pRecFrame; // Input & Output

			unsigned char *BSRef_d;

			unsigned char *QP_TO_Chroma_dev;
			unsigned char *ALPHA_Table_dev;
			unsigned char *BETA_Table_dev;
			unsigned char *CLIP_Table_dev;
		
			cutilSafeCall(cudaMalloc((void**)&BSRef_d,sizeof(unsigned char)*2*BLOCKS_PER_MB*num_mb_hor*num_mb_ver));

			cutilSafeCall(cudaMalloc((void**)&ALPHA_Table_dev,sizeof(unsigned char)*NUM_QP));
			cutilSafeCall(cudaMalloc((void**)&BETA_Table_dev,sizeof(unsigned char)*NUM_QP));
			cutilSafeCall(cudaMalloc((void**)&CLIP_Table_dev,sizeof(unsigned char)*NUM_QP*5));
			cutilSafeCall(cudaMalloc((void**)&QP_TO_Chroma_dev,sizeof(unsigned char)*NUM_QP));

			cutilSafeCall(cudaMemcpy(ALPHA_Table_dev,ALPHA_TABLE,sizeof(unsigned char)*NUM_QP,cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(BETA_Table_dev,BETA_TABLE,sizeof(unsigned char)*NUM_QP,cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(CLIP_Table_dev,CLIP_TAB,sizeof(unsigned char)*NUM_QP*5,cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(QP_TO_Chroma_dev,QP_TO_CHROMA_MAPPING,sizeof(unsigned char)*NUM_QP,cudaMemcpyHostToDevice));


			dim3 dimblock(BLOCKS_PER_MB,8,1); //一个线程处理一个边的一条边界的横向和纵向边界强度，一个宏块需要的线程数为16，每个线程块处理8个MB
			dim3 dimgrid(num_mb_hor/8,num_mb_ver,1);
			dim3 block_ver(16,2,1);
			dim3 grid_ver((num_mb_ver+1)>>1,2,1);
			dim3 block_hor(16,2,1);
			dim3 grid_hor(num_mb_hor>>1,2,1);

			cudaCalcBoundaryStrength_kernel<<<dimgrid,dimblock>>>
																(dev_blk_mb_info,
																BSRef_d,
																disable_deblocking_filter_idc,
																num_mb_hor,
																num_mb_ver,
															    p_enc->i_slice_num,
																I_Slice);

			cudaDeblockMB_kernel_ver<<<grid_ver,block_ver>>>
										(
											 BSRef_d,
											 QP,
											 num_mb_hor,
											 num_mb_ver,
											 width_ref,
											 height_ref,
											 RECON_FRAME_Y_OFFSET* width_ref + RECON_FRAME_X_OFFSET,
											 RECON_FRAME_Y_OFFSET_C * width_ref_c + RECON_FRAME_X_OFFSET_C,
											 dev_recon,
											 dev_recon_uv,
											 dev_recon_uv+(width_ref*height_ref>>2),
											 QP_TO_Chroma_dev,
											 ALPHA_Table_dev,
											 BETA_Table_dev,
											 CLIP_Table_dev
										 );

			cudaDeblockMB_kernel_hor<<<grid_hor,block_hor>>>
										(
											 BSRef_d,
											 QP,
											 num_mb_hor,
											 num_mb_ver,
											 width_ref,
											 height_ref,
											 RECON_FRAME_Y_OFFSET* width_ref + RECON_FRAME_X_OFFSET,
											 RECON_FRAME_Y_OFFSET_C * width_ref_c + RECON_FRAME_X_OFFSET_C,
											 dev_recon,
											 dev_recon_uv,
											 dev_recon_uv+(width_ref*height_ref>>2),
											 QP_TO_Chroma_dev,
											 ALPHA_Table_dev,
											 BETA_Table_dev,
											 CLIP_Table_dev
										 );
			cutilSafeCall(cudaMemcpy(frame->y,dev_recon,width_ref*height_ref*sizeof(unsigned char),cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(frame->u,dev_recon_uv,width_ref*height_ref*sizeof(unsigned char)>>2,cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(frame->v,dev_recon_uv+(width_ref*height_ref>>2),width_ref*height_ref*sizeof(unsigned char)>>2,cudaMemcpyDeviceToHost));


			cutilSafeCall(cudaFree(BSRef_d));

			cutilSafeCall(cudaFree(QP_TO_Chroma_dev));
			cutilSafeCall(cudaFree(ALPHA_Table_dev));
			cutilSafeCall(cudaFree(CLIP_Table_dev));
			cutilSafeCall(cudaFree(BETA_Table_dev));
			cutilSafeCall(cudaFree(dev_blk_mb_info));

			cutilSafeCall(cudaFree(dev_recon));
			cutilSafeCall(cudaFree(dev_recon_uv));
			
		
			pad_deblock_out_frame(p_enc->pRecFrame, REFERENCE_FRAME_PAD_AMT); // dec
			end = clock();
			p_enc->new_timers.de_block +=(end - start);

}
