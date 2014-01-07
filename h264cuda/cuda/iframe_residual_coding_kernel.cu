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


#include "intra_rc_luma_kernel.cu"

__global__ void Intra16x16Prediction_global(
				unsigned char *dev_input,
				unsigned char *dev_top_neighbor,
				unsigned char *dev_left_neighbor,
				S_BLK_MB_INFO *dev_blk_mb_info,
				int           InStride,
				int           PredStride,
				unsigned char *dev_recon
				)
{
    
				__shared__  uchar4 s_top_neighbor[16];
    			__shared__  uchar4 s_left_neighbor[16];
    			__shared__ unsigned char s_in[256];
				__shared__ int Sad[16];
				uchar4 Rec_Row[4];

	 			int MinSad,intra_16_pred_mode;
	 			int TopAvailable, LeftAvailable;
	 			S_BLK_MB_INFO BlkMBInfoIntra16;
    		
	 			int tid_x,tid_y,tid;
	 			tid_x = threadIdx.x;
	 			tid_y = threadIdx.y;
    		     	
	 			tid = tid_x + tid_y * blockDim.x;

	 		//将全局存储器中的数据加载到共享存储器 
				s_left_neighbor[tid].x = dev_left_neighbor[tid_y*4];  //获得左边宏块最右边一列子宏块的最右列重建像素值threads：3，7，11，15 
				s_left_neighbor[tid].y = dev_left_neighbor[tid_y*4+1];
				s_left_neighbor[tid].z = dev_left_neighbor[tid_y*4+2];
				s_left_neighbor[tid].w = dev_left_neighbor[tid_y*4+3];
			
     			//Left_mb_type = 0;
	 			for(int count =0;count<1;count++)
				{
     			//将一个宏块的原始数据以光栅的形式导入共享存储器
	  				for (int i=0;i<16;i++)
	  				{
	    					s_in[tid+i*16]= dev_input[tid+i*16+count*16];
	  				}
					s_top_neighbor[tid].x = dev_top_neighbor[tid_x*4+count*16];   //获得上边宏块最下边一列子宏块的最后一行重建像素值threads：12，13，14，15 
					s_top_neighbor[tid].y = dev_top_neighbor[tid_x*4+1+count*16];
					s_top_neighbor[tid].z = dev_top_neighbor[tid_x*4+2+count*16];
					s_top_neighbor[tid].w = dev_top_neighbor[tid_x*4+3+count*16];

	   				BlkMBInfoIntra16 = dev_blk_mb_info[tid+count*BLOCKS_PER_MB];
       				TopAvailable  = (BlkMBInfoIntra16.Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
       				LeftAvailable = (BlkMBInfoIntra16.Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;
       
	   		// 16x16帧内预测
	  				Intra16x16Prediction_cu(
												s_top_neighbor,
												s_left_neighbor,
												s_in+tid_x*4+tid_y*64,
												TopAvailable,
												LeftAvailable,
												Rec_Row,MinSad,
												Sad, 
												intra_16_pred_mode,
												tid_x,
												tid_y
											);
	  
      				BlkMBInfoIntra16.MinSAD = MinSad;
	  				BlkMBInfoIntra16.Type  = INTRA_LARGE_BLOCKS_MB_TYPE;
	  				BlkMBInfoIntra16.SubType  = intra_16_pred_mode;

      				dev_blk_mb_info[tid+count*BLOCKS_PER_MB]=BlkMBInfoIntra16 ;

     				for(int i=0;i<4;i++)
	 				{
	    	 			dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16] = Rec_Row[i].x;
		 				dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16 +1] = Rec_Row[i].y;
		 				dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16 +2] = Rec_Row[i].z;
		 				dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16 +3] = Rec_Row[i].w;
	 				}
				}
}


__global__ void Intra4x4Prediction_global(
						unsigned char 	*dev_input,
						unsigned char 	*dev_top_neighbor,
						unsigned char 	*dev_left_neighbor,
						unsigned char 	Top_Left_Pix,
						int				*dev_neighbor_info,
						S_BLK_MB_INFO 	*dev_blk_mb_info,
						short			*Quant_tab,
						short			*Dquant_tab,
						S_QP_DATA     	*pQpData,
						int           	InStride,
						int           	PredStride,
						int				constrained_intra,
						unsigned char 	*dev_recon,
						short		    *dev_dct_coefs
					)
{
				__shared__ 	uchar4 	s_top_neighbor[BLOCKS_PER_MB];
				__shared__  uchar4 	s_left_neighbor[BLOCKS_PER_MB];
				__shared__	unsigned char 	s_in[256];
				__shared__	int	      	left_blk_type[16];
				__shared__	int	      	top_blk_type[16];
				__shared__	int	      	Sad[16];
				__shared__	int         PreferedPredMode[16];
				__shared__ short        Quant_tables[16];
				__shared__ short        DQuant_tables[16];
				
				short2		Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
							Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;
				uchar4 Rec_row[4];

				int  	Quant_Add;
				int  	Quant_Shift;
				int  	Dquant_Shift;
				int  	pred_penalty;
				int  	intra_4_sad,intra_4_pred_mode;
				int  	top_blk_available,left_blk_available,dcOnlyPredictionFlag;
				int  	Left_mb_type,Top_mb_type;
				S_BLK_MB_INFO BlkMBInfoIntra16;
				int tid_x,tid_y;
				tid_x = threadIdx.x;
				tid_y = threadIdx.y;
				int  tid = tid_x + tid_y * blockDim.x;
				
				s_left_neighbor[tid].x = dev_left_neighbor[tid_y*4];  //获得左边宏块最右边一列子宏块的最右列重建像素值threads：3，7，11，15 
				s_left_neighbor[tid].y = dev_left_neighbor[tid_y*4+1];
				s_left_neighbor[tid].z = dev_left_neighbor[tid_y*4+2];
				s_left_neighbor[tid].w = dev_left_neighbor[tid_y*4+3];
				left_blk_type [tid] = dev_neighbor_info[tid_y+1]; //左边子宏块预测方式，初始时只有最左一列有效

        	    Quant_tables[tid] = Quant_tab[tid];
				DQuant_tables[tid] = Dquant_tab[tid];

				Left_mb_type = dev_neighbor_info[0];
				Quant_Add = pQpData->QuantAdd;
				Quant_Shift = pQpData->QuantShift;
				Dquant_Shift = pQpData->DQuantShift;
				pred_penalty = pQpData->PredPenalty;

		for(int count =0;count<1;count++)
		{
       			//将一个宏块的原始数据以光栅的形式导入共享存储器
				for (int i=0;i<16;i++)
				{
					s_in[tid+i*16]= dev_input[tid+i*16+count*16];
				}
				top_blk_type[tid] = dev_neighbor_info[(count+1)*5+tid_x+1];  //上边子宏块预测值，只有最后一行有效
      	
				s_top_neighbor[tid].x = dev_top_neighbor[tid_x*4+count*16];   //获得上边宏块最下边一列子宏块的最后一行重建像素值threads：12，13，14，15 
				s_top_neighbor[tid].y = dev_top_neighbor[tid_x*4+1+count*16];
				s_top_neighbor[tid].z = dev_top_neighbor[tid_x*4+2+count*16];
				s_top_neighbor[tid].w = dev_top_neighbor[tid_x*4+3+count*16];
				
				BlkMBInfoIntra16 = dev_blk_mb_info[tid+count*BLOCKS_PER_MB];
      	
				left_blk_available = (BlkMBInfoIntra16.Loc & LOC_BLK_LEFT_EDGE) ? 0 : 1;
				top_blk_available = (BlkMBInfoIntra16.Loc & LOC_BLK_TOP_EDGE) ? 0 : 1;
      	
				Top_mb_type = dev_neighbor_info[(count+1)*5];	
      	
				dcOnlyPredictionFlag = (((BlkMBInfoIntra16.Loc & LOC_BLK_LEFT_OR_TOP_EDGE) != 0) ||
                                    		((	(Top_mb_type == INTER_SMALL_BLOCKS_MB_TYPE) ||
                                       			(Top_mb_type == INTER_LARGE_BLOCKS_MB_TYPE) ||
                                       			(Left_mb_type == INTER_SMALL_BLOCKS_MB_TYPE) ||
                                       			(Left_mb_type == INTER_LARGE_BLOCKS_MB_TYPE)) && constrained_intra )
                                   		);

				Intra4x4Prediction_cu(
										s_top_neighbor,
										s_left_neighbor,
										Top_Left_Pix,
										s_in+tid_x*4+tid_y*64,
										top_blk_available, 
										left_blk_available,
										dcOnlyPredictionFlag,
										Top_mb_type,
										top_blk_type,
										Left_mb_type,
										left_blk_type,
										pred_penalty,
										Quant_Add,
										Quant_Shift,
										Dquant_Shift,
										Quant_tables,
										DQuant_tables,
										Rec_row,
										Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
										Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23,
										intra_4_sad,
										Sad,
										PreferedPredMode,
										intra_4_pred_mode,
										tid_x,
										tid_y
									);

				BlkMBInfoIntra16.MinSAD = intra_4_sad;
	  			BlkMBInfoIntra16.Type  = INTRA_SMALL_BLOCKS_MB_TYPE;
	  			BlkMBInfoIntra16.SubType  = intra_4_pred_mode;

      			dev_blk_mb_info[tid+count*BLOCKS_PER_MB]=BlkMBInfoIntra16 ;

				{
				dev_recon[tid_x*4 + (tid_y*4) * 16 + count*16 +0] = Rec_row[0].x;
		 		dev_recon[tid_x*4 + (tid_y*4) * 16 + count*16 +1] = Rec_row[0].y;
		 		dev_recon[tid_x*4 + (tid_y*4) * 16 + count*16 +2] = Rec_row[0].z;
		 		dev_recon[tid_x*4 + (tid_y*4) * 16 + count*16 +3] = Rec_row[0].w;
				dev_recon[tid_x*4 + (tid_y*4+1) * 16 + count*16] = Rec_row[1].x;
		 		dev_recon[tid_x*4 + (tid_y*4+1) * 16 + count*16 +1] = Rec_row[1].y;
		 		dev_recon[tid_x*4 + (tid_y*4+1) * 16 + count*16 +2] = Rec_row[1].z;
		 		dev_recon[tid_x*4 + (tid_y*4+1) * 16 + count*16 +3] = Rec_row[1].w;
				dev_recon[tid_x*4 + (tid_y*4+2) * 16 + count*16] = Rec_row[2].x;
		 		dev_recon[tid_x*4 + (tid_y*4+2) * 16 + count*16 +1] = Rec_row[2].y;
		 		dev_recon[tid_x*4 + (tid_y*4+2) * 16 + count*16 +2] = Rec_row[2].z;
		 		dev_recon[tid_x*4 + (tid_y*4+2) * 16 + count*16 +3] = Rec_row[2].w;
				dev_recon[tid_x*4 + (tid_y*4+3) * 16 + count*16] = Rec_row[3].x;
		 		dev_recon[tid_x*4 + (tid_y*4+3) * 16 + count*16 +1] = Rec_row[3].y;
		 		dev_recon[tid_x*4 + (tid_y*4+3) * 16 + count*16 +2] = Rec_row[3].z;
		 		dev_recon[tid_x*4 + (tid_y*4+3) * 16 + count*16 +3] = Rec_row[3].w;
				
				}
				
				dev_dct_coefs[tid*16+count*MB_TOTAL_SIZE]   = Qcoef_0_01.x;
				dev_dct_coefs[tid*16+1+count*MB_TOTAL_SIZE] = Qcoef_0_01.y;
				dev_dct_coefs[tid*16+2+count*MB_TOTAL_SIZE] = Qcoef_0_23.x;
				dev_dct_coefs[tid*16+3+count*MB_TOTAL_SIZE] = Qcoef_0_23.y;
				dev_dct_coefs[tid*16+4+count*MB_TOTAL_SIZE] = Qcoef_1_01.x;
				dev_dct_coefs[tid*16+5+count*MB_TOTAL_SIZE] = Qcoef_1_01.y;
				dev_dct_coefs[tid*16+6+count*MB_TOTAL_SIZE] = Qcoef_1_23.x;
				dev_dct_coefs[tid*16+7+count*MB_TOTAL_SIZE] = Qcoef_1_23.y;
				dev_dct_coefs[tid*16+8+count*MB_TOTAL_SIZE] = Qcoef_2_01.x;
				dev_dct_coefs[tid*16+9+count*MB_TOTAL_SIZE] = Qcoef_2_01.y;
				dev_dct_coefs[tid*16+10+count*MB_TOTAL_SIZE] = Qcoef_2_23.x;
				dev_dct_coefs[tid*16+11+count*MB_TOTAL_SIZE] = Qcoef_2_23.y;
				dev_dct_coefs[tid*16+12+count*MB_TOTAL_SIZE] = Qcoef_3_01.x;
				dev_dct_coefs[tid*16+13+count*MB_TOTAL_SIZE] = Qcoef_3_01.y;
				dev_dct_coefs[tid*16+14+count*MB_TOTAL_SIZE] = Qcoef_3_23.x;
				dev_dct_coefs[tid*16+15+count*MB_TOTAL_SIZE] = Qcoef_3_23.y;

		}

}


__global__ void Intra16x16_transcoding_cu (
											unsigned char *dev_input,
											unsigned char *dev_pred,
											short *Quant_tab,
											short *Dquant_tab,
											S_QP_DATA *dev_QpData,
											unsigned char *dev_recon,
											short *dev_dct_coefs,
											short *dev_dc_coefs
										  )
{
				
				__shared__	unsigned char 	s_in[256];
				__shared__ short        Quant_tables[16];
				__shared__ short        DQuant_tables[16];
				
				short2		Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
							Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;

				uchar4		  PredRow[4],Rec_Row[4];
				__shared__ short	QcoefDc[16],TempDC[16];

				int  	Quant_Add;
				int  	QuantAddDC;
				int  	Quant_Shift;
				int  	Dquant_Shift;
				int tid_x,tid_y;
				tid_x = threadIdx.x;
				tid_y = threadIdx.y;
				int  tid = tid_x + tid_y * blockDim.x;

        	    Quant_tables[tid] = Quant_tab[tid];
				DQuant_tables[tid] = Dquant_tab[tid];

				Quant_Add = dev_QpData->QuantAdd;
				QuantAddDC = dev_QpData->QuantAddDC;
				Quant_Shift = dev_QpData->QuantShift;
				Dquant_Shift = dev_QpData->DQuantShift;

		for(int count =0;count<1;count++)
		{
       			//将一个宏块的原始数据以光栅的形式导入共享存储器
				for (int i=0;i<16;i++)
				{
					s_in[tid+i*16]= dev_input[tid+i*16+count*16];
				}
				for (int i=0;i<BLK_HEIGHT;i++)
				{
					PredRow[i].x = dev_pred[tid_x*4+tid_y*64+i*16];
					PredRow[i].y = dev_pred[tid_x*4+tid_y*64+i*16+1];
					PredRow[i].z = dev_pred[tid_x*4+tid_y*64+i*16+2];
					PredRow[i].w = dev_pred[tid_x*4+tid_y*64+i*16+3];
				}
	
				intra16x16_transforms_cu (
											s_in+tid_x*4+tid_y*64,
											PredRow,
											Rec_Row,
											Quant_Add,
											QuantAddDC,
											Quant_Shift,
											Dquant_Shift,
											Quant_tables,
											DQuant_tables,
											Qcoef_0_01,
											Qcoef_0_23,
											Qcoef_1_01,
											Qcoef_1_23,
											Qcoef_2_01,
											Qcoef_2_23,
											Qcoef_3_01,
											Qcoef_3_23,
											QcoefDc,
											TempDC,
											tid_x,
											tid_y
										 );
				
				for(int i=0;i<4;i++)
	 			{
	    	 		dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16] = Rec_Row[i].x;
		 			dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16 +1] = Rec_Row[i].y;
		 			dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16 +2] = Rec_Row[i].z;
		 			dev_recon[tid_x*4 + (tid_y*4+i) * 16 + count*16 +3] = Rec_Row[i].w;
	 			}
				
				dev_dct_coefs[tid*16+count*MB_TOTAL_SIZE]   = Qcoef_0_01.x;
				dev_dct_coefs[tid*16+1+count*MB_TOTAL_SIZE] = Qcoef_0_01.y;
				dev_dct_coefs[tid*16+2+count*MB_TOTAL_SIZE] = Qcoef_0_23.x;
				dev_dct_coefs[tid*16+3+count*MB_TOTAL_SIZE] = Qcoef_0_23.y;
				dev_dct_coefs[tid*16+4+count*MB_TOTAL_SIZE] = Qcoef_1_01.x;
				dev_dct_coefs[tid*16+5+count*MB_TOTAL_SIZE] = Qcoef_1_01.y;
				dev_dct_coefs[tid*16+6+count*MB_TOTAL_SIZE] = Qcoef_1_23.x;
				dev_dct_coefs[tid*16+7+count*MB_TOTAL_SIZE] = Qcoef_1_23.y;
				dev_dct_coefs[tid*16+8+count*MB_TOTAL_SIZE] = Qcoef_2_01.x;
				dev_dct_coefs[tid*16+9+count*MB_TOTAL_SIZE] = Qcoef_2_01.y;
				dev_dct_coefs[tid*16+10+count*MB_TOTAL_SIZE] = Qcoef_2_23.x;
				dev_dct_coefs[tid*16+11+count*MB_TOTAL_SIZE] = Qcoef_2_23.y;
				dev_dct_coefs[tid*16+12+count*MB_TOTAL_SIZE] = Qcoef_3_01.x;
				dev_dct_coefs[tid*16+13+count*MB_TOTAL_SIZE] = Qcoef_3_01.y;
				dev_dct_coefs[tid*16+14+count*MB_TOTAL_SIZE] = Qcoef_3_23.x;
				dev_dct_coefs[tid*16+15+count*MB_TOTAL_SIZE] = Qcoef_3_23.y;
				dev_dc_coefs[tid] = QcoefDc[tid];

		}



}

//一个I帧色度宏块的帧内预测编码global kernel函数
__global__ void iframe_residual_code_one_mb_cu (
												unsigned char *dev_input,
												int           in_stride,
												int           enc_width,
												unsigned char *dev_top_neighbor,
												unsigned char top_left_pix,
												int			  *dev_neighbor_info,
												unsigned char *dev_recon,
												int           out_stride,
												short         *dev_dct_coefs,
												short         *dev_dc_coefs,
												S_BLK_MB_INFO *dev_blk_mb_info,
												short		  *quant_tab,
												short         *d_quant_tab,
												S_QP_DATA     *pQpData,
												int           constrained_intra,
												int           intra_pred_select,
												int           strip_size

												)
{
	__shared__ 	uchar4 	s_top_neighbor[BLOCKS_PER_MB];
	__shared__  uchar4 	s_left_neighbor[BLOCKS_PER_MB];
	__shared__	unsigned char 	s_in[256];
	__shared__	int	      	left_blk_type[16];
	__shared__	int	      	top_blk_type[16];
	__shared__	int	      	Sad[16];
	__shared__  int         PreferedPredMode[16];
	__shared__ short        Quant_tables[16];
	__shared__ short        DQuant_tables[16];
	__shared__ short	QcoefDc[16],TempDC[16];

	 uchar4 Rec_Row[4],Pred_Row[4];
	 short2 Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
            Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;
	 int Temp_top_left;

     int  Quant_Add;
	 int  QuantAddDC;
     int  Quant_Shift;
     int  Dquant_Shift;
     int  pred_penalty;
	 int ChooseIntra_16,Type,SubType;
	 int MinSad,intra_16_sad,intra_4_sad,intra_16_pred_mode,intra_4_pred_mode;
	 int TopAvailable, LeftAvailable,top_blk_available,left_blk_available,dcOnlyPredictionFlag;
	 int Left_mb_type,Top_mb_type;
	 S_BLK_MB_INFO BlkMBInfoIntra16;
	 int tid_x,tid_y;
	 tid_x = threadIdx.x;
	 tid_y = threadIdx.y;

	 int  tid = tid_x + tid_y * blockDim.x;
	 //将全局存储器中的数据加载到共享存储器
     s_left_neighbor[tid].x = 0;
	 s_left_neighbor[tid].y = 0;
	 s_left_neighbor[tid].z = 0;
	 s_left_neighbor[tid].w = 0;
     Quant_tables[tid] = quant_tab[tid];
     DQuant_tables[tid] = d_quant_tab[tid];  
	 left_blk_type [tid] = 0;

     Left_mb_type = 0;
     Quant_Add = pQpData->QuantAdd;
	 QuantAddDC = pQpData->QuantAddDC;
     Quant_Shift = pQpData->QuantShift;
     Dquant_Shift = pQpData->DQuantShift;
	 pred_penalty = pQpData->PredPenalty;

	for(int count =0;count<strip_size;count++)
	{
       //将一个宏块的原始数据以光栅的形式导入共享存储器
	   for (int i=0;i<16;i++)
	   {
		   
	      s_in[tid+i*16]= dev_input[tid+i*in_stride+count*MB_WIDTH];
	   }
	   top_blk_type[tid] = dev_neighbor_info[count*5+tid_x+1];  //上边子宏块预测值，只有最后一行有效
	   Top_mb_type = dev_neighbor_info[count*5];

	   s_top_neighbor[tid].x = dev_top_neighbor[tid_x*4+count*MB_WIDTH];
	   s_top_neighbor[tid].y = dev_top_neighbor[tid_x*4+1+count*MB_WIDTH];
	   s_top_neighbor[tid].z = dev_top_neighbor[tid_x*4+2+count*MB_WIDTH];
	   s_top_neighbor[tid].w = dev_top_neighbor[tid_x*4+3+count*MB_WIDTH];

	   BlkMBInfoIntra16 = dev_blk_mb_info[tid+count*BLOCKS_PER_MB];
       TopAvailable  = (BlkMBInfoIntra16.Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
       LeftAvailable = (BlkMBInfoIntra16.Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;
       
	   // 16x16帧内预测
	   Intra16x16Prediction_cu(
									s_top_neighbor,
									s_left_neighbor,
									s_in+tid_x*4+tid_y*64,
									TopAvailable,
									LeftAvailable,
									Pred_Row,
									intra_16_sad,
									Sad, 
									intra_16_pred_mode,
									tid_x,
									tid_y
								);

	   left_blk_available = (BlkMBInfoIntra16.Loc & LOC_BLK_LEFT_EDGE) ? 0 : 1;
       top_blk_available = (BlkMBInfoIntra16.Loc & LOC_BLK_TOP_EDGE) ? 0 : 1;

       dcOnlyPredictionFlag = (((BlkMBInfoIntra16.Loc & LOC_BLK_LEFT_OR_TOP_EDGE) != 0) ||
                                    (( (Top_mb_type == INTER_SMALL_BLOCKS_MB_TYPE) ||
                                       (Top_mb_type == INTER_LARGE_BLOCKS_MB_TYPE) ||
                                       (Left_mb_type == INTER_SMALL_BLOCKS_MB_TYPE) ||
                                       (Left_mb_type == INTER_LARGE_BLOCKS_MB_TYPE)) && constrained_intra )
                                   );
       //Sad[tid] = 0;
	   Temp_top_left = s_top_neighbor[15].w;
       Intra4x4Prediction_cu(
								s_top_neighbor,
								s_left_neighbor,
								top_left_pix,
								s_in+tid_x*4+tid_y*64,
								top_blk_available, 
								left_blk_available,
								dcOnlyPredictionFlag,
								Top_mb_type,
								top_blk_type,
								Left_mb_type,
								left_blk_type,
								pred_penalty,
								Quant_Add,
								Quant_Shift,
								Dquant_Shift,
								Quant_tables,
								DQuant_tables,
								Rec_Row,
								Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
								Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23,
								intra_4_sad,
								Sad,
								PreferedPredMode,
								intra_4_pred_mode,
								tid_x,
								tid_y
							);


     //intra_4_sad = intra_4_sad + 6*pred_penalty;
	 ChooseIntra_16 = (intra_16_sad < intra_4_sad) ? 1 : 0;

	 Type    = (ChooseIntra_16 == 0) ? INTRA_SMALL_BLOCKS_MB_TYPE : INTRA_LARGE_BLOCKS_MB_TYPE;
	 MinSad  = (ChooseIntra_16 == 0) ? intra_4_sad : intra_16_sad;
	 SubType = (ChooseIntra_16 == 0) ? intra_4_pred_mode : intra_16_pred_mode;
	 
     QcoefDc[tid] = 0;
     if (ChooseIntra_16 > 0)
	 {
	     intra16x16_transforms_cu (
										s_in+tid_x*4+tid_y*64,
										Pred_Row,
										Rec_Row,
										Quant_Add,
										QuantAddDC,
										Quant_Shift,
										Dquant_Shift,
										Quant_tables,
										DQuant_tables,
										Qcoef_0_01,
										Qcoef_0_23,
										Qcoef_1_01,
										Qcoef_1_23,
										Qcoef_2_01,
										Qcoef_2_23,
										Qcoef_3_01,
										Qcoef_3_23,
										QcoefDc,
										TempDC,
										tid_x,
										tid_y
									);
	 }

	 //写回结果，交流和直流系数等
     dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE] = Qcoef_0_01.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+1] = Qcoef_0_01.y;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+2] = Qcoef_0_23.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+3] = Qcoef_0_23.y;

	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+4] = Qcoef_1_01.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+5] = Qcoef_1_01.y;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+6] = Qcoef_1_23.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+7] = Qcoef_1_23.y;

	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+8] = Qcoef_2_01.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+9] = Qcoef_2_01.y;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+10] = Qcoef_2_23.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+11] = Qcoef_2_23.y;

	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+12] = Qcoef_3_01.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+13] = Qcoef_3_01.y;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+14] = Qcoef_3_23.x;
	 dev_dct_coefs[tid*BLK_SIZE+count*MB_TOTAL_SIZE+15] = Qcoef_3_23.y;
     
	 dev_dc_coefs [tid+count*16] = QcoefDc[tid];
	 // Write back reconstructed block
	 for(int i=0;i< 4;i++)
	 {
	    dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + count*16] = Rec_Row[i].x;
		dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + count*16 +1] = Rec_Row[i].y;
		dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + count*16 +2] = Rec_Row[i].z;
		dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + count*16 +3] = Rec_Row[i].w;
	 }
	 // write top pixels for next mb row
	 if(tid_y == 3)
	 {
	    dev_top_neighbor[tid_x*BLK_WIDTH+count*MB_WIDTH] = Rec_Row[3].x;
		dev_top_neighbor[tid_x*BLK_WIDTH+count*MB_WIDTH+1] = Rec_Row[3].y;
		dev_top_neighbor[tid_x*BLK_WIDTH+count*MB_WIDTH+2] = Rec_Row[3].z;
		dev_top_neighbor[tid_x*BLK_WIDTH+count*MB_WIDTH+3] = Rec_Row[3].w;
		dev_neighbor_info[count*5+tid_x+1] = SubType;
	 }
        // write mbinfo for next mb row
	 if(tid_x == 3)
	 {
		s_left_neighbor [tid-3].x = s_left_neighbor [tid-2].x = s_left_neighbor [tid-1].x = s_left_neighbor [tid].x = Rec_Row[0].w;
		s_left_neighbor [tid-3].y = s_left_neighbor [tid-2].y = s_left_neighbor [tid-1].y = s_left_neighbor [tid].y = Rec_Row[1].w;
		s_left_neighbor [tid-3].z = s_left_neighbor [tid-2].z = s_left_neighbor [tid-1].z = s_left_neighbor [tid].z = Rec_Row[2].w;
		s_left_neighbor [tid-3].w = s_left_neighbor [tid-2].w = s_left_neighbor [tid-1].w = s_left_neighbor [tid].w = Rec_Row[3].w;
	 }
	 if(tid == 0 )
	{
		dev_neighbor_info[count*5] = Type;
	}
		Left_mb_type = Type;
		left_blk_type[tid] = SubType;

		top_left_pix = Temp_top_left;
		dev_blk_mb_info[tid+count*16].Type = Type;
		dev_blk_mb_info[tid+count*16].SubType = SubType;
		dev_blk_mb_info[tid+count*16].MinSAD = MinSad;
		dev_blk_mb_info[tid+count*16].Pred_mode = PreferedPredMode[tid];
	}
}


__global__ void iframe_residual_code_luma_kernel (
													unsigned char *dev_input,
													int           width,
													int           num_mb_hor,
													int           num_mb_ver,
													unsigned char *dev_recon,
													int           out_stride,
													short         *dev_dct_coefs,
													short         *dev_dc_coefs,
													S_BLK_MB_INFO *dev_blk_mb_info,
													short		  *quant_tab,
													short         *d_quant_tab,
													S_QP_DATA     *pQpData,
													int           constrained_intra,
													int           intra_pred_select,
													int			  slice_num
												  )
{
	__shared__ 	uchar4 	s_top_neighbor[BLOCKS_PER_MB];
	__shared__  uchar4 	s_left_neighbor[BLOCKS_PER_MB];
	__shared__	unsigned char 	s_in[256];
	__shared__	int	      	left_blk_type[16];
	__shared__	int	      	top_blk_type[16];
	__shared__	int	      	Sad[16];
	__shared__  int         PreferedPredMode[16];
	__shared__ short        Quant_tables[16];
	__shared__ short        DQuant_tables[16];
	__shared__ short	QcoefDc[16],TempDC[16];

	 uchar4 Rec_Row[4],Pred_Row[4];
	 short2 Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
            Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;
	 unsigned char Temp_top_left,top_left_pix;

     int  Quant_Add;
	 int  QuantAddDC;
     int  Quant_Shift;
     int  Dquant_Shift;
     int  pred_penalty;
	 int ChooseIntra_16,Type,SubType;
	 int MinSad,intra_16_sad,intra_4_sad,intra_16_pred_mode,intra_4_pred_mode;
	 int TopAvailable, LeftAvailable,top_blk_available,left_blk_available,dcOnlyPredictionFlag;
	 int Left_mb_type,Top_mb_type;
	 int i;
	 S_BLK_MB_INFO BlkMBInfoIntra16;
	 int tid_x,tid_y;
	 tid_x = threadIdx.x;
	 tid_y = threadIdx.y;
	 int first_mb;

	 int  tid = tid_x + tid_y * blockDim.x;
	 //将全局存储器中的数据加载到共享存储器
     s_left_neighbor[tid].x = 0;
	 s_left_neighbor[tid].y = 0;
	 s_left_neighbor[tid].z = 0;
	 s_left_neighbor[tid].w = 0;
     Quant_tables[tid] = quant_tab[tid];
     DQuant_tables[tid] = d_quant_tab[tid];  
	 left_blk_type [tid] = 0;
	 top_left_pix = 0;
     Left_mb_type = 0;
     Quant_Add = pQpData->QuantAdd;
	 QuantAddDC = pQpData->QuantAddDC;
     Quant_Shift = pQpData->QuantShift;
     Dquant_Shift = pQpData->DQuantShift;
	 pred_penalty = pQpData->PredPenalty;

	 first_mb = blockIdx.x*num_mb_hor* (num_mb_ver/slice_num);
	for(int j =0;j< num_mb_ver/slice_num;j++)
	{
		for(int k =0;k< num_mb_hor;k++)
		{
			BlkMBInfoIntra16 = dev_blk_mb_info[tid + k*BLOCKS_PER_MB + j*num_mb_hor*BLOCKS_PER_MB  + first_mb*BLOCKS_PER_MB];
			TopAvailable  = (BlkMBInfoIntra16.Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
			LeftAvailable = (BlkMBInfoIntra16.Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;

			left_blk_available = (BlkMBInfoIntra16.Loc & LOC_BLK_LEFT_EDGE) ? 0 : 1;
			top_blk_available = (BlkMBInfoIntra16.Loc & LOC_BLK_TOP_EDGE) ? 0 : 1;

			dcOnlyPredictionFlag = (((BlkMBInfoIntra16.Loc & LOC_BLK_LEFT_OR_TOP_EDGE) != 0) ||
                                    (( (Top_mb_type == INTER_SMALL_BLOCKS_MB_TYPE) ||
                                       (Top_mb_type == INTER_LARGE_BLOCKS_MB_TYPE) ||
                                       (Left_mb_type == INTER_SMALL_BLOCKS_MB_TYPE) ||
                                       (Left_mb_type == INTER_LARGE_BLOCKS_MB_TYPE)) && constrained_intra )
                                   );
			if(TopAvailable)
			{
				top_blk_type[tid] = dev_blk_mb_info[ (k+(j-1)*num_mb_hor)*BLOCKS_PER_MB+12 + tid_x + first_mb*BLOCKS_PER_MB ].SubType;
				Top_mb_type = dev_blk_mb_info[(k+(j-1)*num_mb_hor)*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Type;

				s_top_neighbor[tid].x = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
				s_top_neighbor[tid].y = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + 1 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
				s_top_neighbor[tid].z = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + 2 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
				s_top_neighbor[tid].w = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + 3 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
				top_left_pix = s_top_neighbor[15].w;
				
				//top_blk_type[tid] = dev_neighbor_info[k*5+tid_x+1];  //上边子宏块预测值，只有最后一行有效
				//Top_mb_type = dev_neighbor_info[k*5];
				//s_top_neighbor[tid].x = dev_top_neighbor[tid_x*4 + k*MB_WIDTH];
				//s_top_neighbor[tid].y = dev_top_neighbor[tid_x*4 + k*MB_WIDTH + 1];
				//s_top_neighbor[tid].z = dev_top_neighbor[tid_x*4 + k*MB_WIDTH + 2];
				//s_top_neighbor[tid].w = dev_top_neighbor[tid_x*4 + k*MB_WIDTH + 3];
			}

			//将一个宏块的原始数据以光栅的形式导入共享存储器
			for ( i=0;i<16;i++)
			{
				s_in[tid+i*16]= dev_input[tid + i*width + k*MB_WIDTH + j*num_mb_hor*MB_TOTAL_SIZE + blockIdx.x*MB_HEIGHT*width*(num_mb_ver/slice_num)];
			}
			// 16x16帧内预测
			Intra16x16Prediction_cu(
									s_top_neighbor,
									s_left_neighbor,
									s_in+tid_x*4+tid_y*64,
									TopAvailable,
									LeftAvailable,
									Pred_Row,
									intra_16_sad,
									Sad, 
									intra_16_pred_mode,
									tid_x,
									tid_y
								);
			//Sad[tid] = 0;
			
			Intra4x4Prediction_cu(
								s_top_neighbor,
								s_left_neighbor,
								top_left_pix,
								s_in+tid_x*4+tid_y*64,
								top_blk_available, 
								left_blk_available,
								dcOnlyPredictionFlag,
								Top_mb_type,
								top_blk_type,
								Left_mb_type,
								left_blk_type,
								pred_penalty,
								Quant_Add,
								Quant_Shift,
								Dquant_Shift,
								Quant_tables,
								DQuant_tables,
								Rec_Row,
								Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
								Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23,
								intra_4_sad,
								Sad,
								PreferedPredMode,
								intra_4_pred_mode,
								tid_x,
								tid_y
							);


			 //intra_4_sad = intra_4_sad + 6*pred_penalty;
			ChooseIntra_16 = (intra_16_sad < intra_4_sad) ? 1 : 0;

			Type    = (ChooseIntra_16 == 0) ? INTRA_SMALL_BLOCKS_MB_TYPE : INTRA_LARGE_BLOCKS_MB_TYPE;
			MinSad  = (ChooseIntra_16 == 0) ? intra_4_sad : intra_16_sad;
			SubType = (ChooseIntra_16 == 0) ? intra_4_pred_mode : intra_16_pred_mode;
	 
			QcoefDc[tid] = 0;
			 if (ChooseIntra_16 > 0)
			{
				 intra16x16_transforms_cu (
										s_in+tid_x*4+tid_y*64,
										Pred_Row,
										Rec_Row,
										Quant_Add,
										QuantAddDC,
										Quant_Shift,
										Dquant_Shift,
										Quant_tables,
										DQuant_tables,
										Qcoef_0_01,
										Qcoef_0_23,
										Qcoef_1_01,
										Qcoef_1_23,
										Qcoef_2_01,
										Qcoef_2_23,
										Qcoef_3_01,
										Qcoef_3_23,
										QcoefDc,
										TempDC,
										tid_x,
										tid_y
									);
			 }

			//写回结果，交流和直流系数等
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE] = Qcoef_0_01.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 1] = Qcoef_0_01.y;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 2] = Qcoef_0_23.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 3] = Qcoef_0_23.y;

			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 4] = Qcoef_1_01.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 5] = Qcoef_1_01.y;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 6] = Qcoef_1_23.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 7] = Qcoef_1_23.y;

			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 8] = Qcoef_2_01.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 9] = Qcoef_2_01.y;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 10] = Qcoef_2_23.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 11] = Qcoef_2_23.y;

			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 12] = Qcoef_3_01.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 13] = Qcoef_3_01.y;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 14] = Qcoef_3_23.x;
			dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 15] = Qcoef_3_23.y;
     
			dev_dc_coefs [tid + k*16 + j*num_mb_hor*BLK_SIZE + first_mb*BLK_SIZE] = QcoefDc[tid];
			 // Write back reconstructed block
			for( i=0;i< 4;i++)
			{
				dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)] = Rec_Row[i].x;
				dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 1] = Rec_Row[i].y;
				dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 2] = Rec_Row[i].z;
				dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 3] = Rec_Row[i].w;
			}
			 // write top pixels for next mb row
			/*if(tid_y == 3)
			{
				dev_top_neighbor[tid_x*BLK_WIDTH + k*MB_WIDTH] = Rec_Row[3].x;
				dev_top_neighbor[tid_x*BLK_WIDTH + k*MB_WIDTH + 1] = Rec_Row[3].y;
				dev_top_neighbor[tid_x*BLK_WIDTH + k*MB_WIDTH + 2] = Rec_Row[3].z;
				dev_top_neighbor[tid_x*BLK_WIDTH + k*MB_WIDTH + 3] = Rec_Row[3].w;
				dev_neighbor_info[k*5 + tid_x + 1] = SubType;
			}*/
			// write mbinfo for next mb row
			if(tid_x == 3)
			{
				s_left_neighbor [tid-3].x = s_left_neighbor [tid-2].x = s_left_neighbor [tid-1].x = s_left_neighbor [tid].x = Rec_Row[0].w;
				s_left_neighbor [tid-3].y = s_left_neighbor [tid-2].y = s_left_neighbor [tid-1].y = s_left_neighbor [tid].y = Rec_Row[1].w;
				s_left_neighbor [tid-3].z = s_left_neighbor [tid-2].z = s_left_neighbor [tid-1].z = s_left_neighbor [tid].z = Rec_Row[2].w;
				s_left_neighbor [tid-3].w = s_left_neighbor [tid-2].w = s_left_neighbor [tid-1].w = s_left_neighbor [tid].w = Rec_Row[3].w;
			}
			
			Left_mb_type = Type;
			left_blk_type[tid] = SubType;

			//top_left_pix = Temp_top_left;
			dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Type = Type;
			dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].SubType = SubType;
			dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MinSAD = MinSad;
			dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Pred_mode = PreferedPredMode[tid];
		}
	}
}
__global__ void pframe_intra_resudial_coding_luma_kernel(
															unsigned char *dev_input,
															unsigned char *dev_pred,
															int           width,
															unsigned char *dev_recon,
															int           out_stride,
															S_BLK_MB_INFO *dev_blk_mb_info,
															short         *dev_dct_coefs,
															short		  *dev_dc_coefs,
															short		  *quant_tab,
															short         *d_quant_tab,
															S_QP_DATA     *pQpData,
															int avg_sad,
															int intra_lambda_fact,
															int num_mb_hor,
															int num_mb_ver,
															int slice_num
															)
{
	__shared__ 	uchar4 	s_top_neighbor[BLOCKS_PER_MB];
	__shared__  uchar4 	s_left_neighbor[BLOCKS_PER_MB];
	__shared__	unsigned char 	src_in[256];
	__shared__	int	    Sad[16];
	__shared__  short    Quant_tables[16];
	__shared__  short    DQuant_tables[16];
	__shared__  short	QcoefDc[16],TempDC[16];
	

	 uchar4 Rec_Row[4],Pred_Row[4];

	 short2 Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
            Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;

     int  Quant_Add;
	 int  QuantAddDC;
     int  Quant_Shift;
     int  Dquant_Shift;
	 int  first_mb;

	 int ChooseIntra;
	 int intra_sad,intra_pred_mode;
	 int TopAvailable, LeftAvailable;
	 int i;
	 S_BLK_MB_INFO BlkMBInfo;
	 int tid_x,tid_y;
	 tid_x = threadIdx.x;
	 tid_y = threadIdx.y;

	 int  tid = tid_x + tid_y * blockDim.x;
	 //将全局存储器中的数据加载到共享存储器
     Quant_tables[tid] = quant_tab[tid];
     DQuant_tables[tid] = d_quant_tab[tid];  

     Quant_Add = pQpData->QuantAdd;
	 QuantAddDC = pQpData->QuantAddDC;
     Quant_Shift = pQpData->QuantShift;
     Dquant_Shift = pQpData->DQuantShift;
	
	 first_mb = blockIdx.x*num_mb_hor* num_mb_ver/slice_num;
	 //first_mb_dev[tid + blockIdx.x* blockDim.x*blockDim.y] = first_mb;

	for(int j =0;j< num_mb_ver/slice_num;j++)
	{
		s_left_neighbor[tid].x = 0;
		s_left_neighbor[tid].y = 0;
		s_left_neighbor[tid].z = 0;
		s_left_neighbor[tid].w = 0;

		for(int k =0;k< num_mb_hor;k++)
		{
			BlkMBInfo = dev_blk_mb_info[tid + k*BLOCKS_PER_MB + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB]; 
			
			intra_sad = 65535;
			//如果当前宏块的帧间预测SAD大于平均值加上一个系数，则进行16*16预测
			if(BlkMBInfo.MinSAD>(avg_sad + intra_lambda_fact))
			{
				//将一个宏块的原始数据以光栅的形式导入共享存储器
				for ( i=0;i<16;i++)
				{
					src_in[tid+i*16]= dev_input[tid + i*width + k*MB_WIDTH + j*MB_HEIGHT*width + blockIdx.x*MB_HEIGHT*width*(num_mb_ver/slice_num)];
				}
				if(j==0)
				{
					s_top_neighbor[tid].x = dev_recon[tid_x*4 + k*MB_WIDTH + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
					s_top_neighbor[tid].y = dev_recon[tid_x*4 + k*MB_WIDTH + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 1];
					s_top_neighbor[tid].z = dev_recon[tid_x*4 + k*MB_WIDTH + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 2];
					s_top_neighbor[tid].w = dev_recon[tid_x*4 + k*MB_WIDTH + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 3];
				}
				else
				{
					s_top_neighbor[tid].x = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
					s_top_neighbor[tid].y = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 1];
					s_top_neighbor[tid].z = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 2];
					s_top_neighbor[tid].w = dev_recon[tid_x*4 + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 3];
				
				}
				TopAvailable  = (BlkMBInfo.Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
				LeftAvailable = (BlkMBInfo.Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;
				/*if(tid==0)
					first_mb_dev[j*num_mb_hor + k + first_mb] = TopAvailable+1;*/
				// 16x16帧内预测
				Intra16x16Prediction_cu(
										s_top_neighbor,
										s_left_neighbor,
										src_in+tid_x*4+tid_y*64,
										TopAvailable,
										LeftAvailable,
										Pred_Row,
										intra_sad,
										Sad, 
										intra_pred_mode,
										tid_x,
										tid_y
									);
				/*if(tid==0)
					first_mb_dev[j*num_mb_hor + k + first_mb] = 100;*/
			}
				ChooseIntra = (intra_sad < BlkMBInfo.MinSAD) ? 1 : 0;
				/*if(tid==0)
					first_mb_dev[j*num_mb_hor + k + first_mb] = intra_sad;*/
				if(ChooseIntra)
				{
					intra16x16_transforms_cu (
												src_in+tid_x*4+tid_y*64,
												Pred_Row,
												Rec_Row,
												Quant_Add,
												QuantAddDC,
												Quant_Shift,
												Dquant_Shift,
												Quant_tables,
												DQuant_tables,
												Qcoef_0_01,
												Qcoef_0_23,
												Qcoef_1_01,
												Qcoef_1_23,
												Qcoef_2_01,
												Qcoef_2_23,
												Qcoef_3_01,
												Qcoef_3_23,
												QcoefDc,
												TempDC,
												tid_x,
												tid_y
											);

					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE] = Qcoef_0_01.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 1] = Qcoef_0_01.y;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 2] = Qcoef_0_23.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 3] = Qcoef_0_23.y;

					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 4] = Qcoef_1_01.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 5] = Qcoef_1_01.y;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 6] = Qcoef_1_23.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 7] = Qcoef_1_23.y;

					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 8] = Qcoef_2_01.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 9] = Qcoef_2_01.y;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 10] = Qcoef_2_23.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 11] = Qcoef_2_23.y;

					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 12] = Qcoef_3_01.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 13] = Qcoef_3_01.y;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 14] = Qcoef_3_23.x;
					dev_dct_coefs[tid*BLK_SIZE + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE + 15] = Qcoef_3_23.y;
		     
					dev_dc_coefs [tid + k*16 + j*num_mb_hor*BLK_SIZE + first_mb*BLK_SIZE ] = QcoefDc[tid];

					for( i=0;i< 4;i++)
					{
						dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)] = Rec_Row[i].x;
						dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 1] = Rec_Row[i].y;
						dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 2] = Rec_Row[i].z;
						dev_recon[tid_x*4 + (tid_y*4+i) * out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num) + 3] = Rec_Row[i].w;
					}

					if(tid_x == 3)
					{
						s_left_neighbor [tid-3].x = s_left_neighbor [tid-2].x = s_left_neighbor [tid-1].x = s_left_neighbor [tid].x = Rec_Row[0].w;
						s_left_neighbor [tid-3].y = s_left_neighbor [tid-2].y = s_left_neighbor [tid-1].y = s_left_neighbor [tid].y = Rec_Row[1].w;
						s_left_neighbor [tid-3].z = s_left_neighbor [tid-2].z = s_left_neighbor [tid-1].z = s_left_neighbor [tid].z = Rec_Row[2].w;
						s_left_neighbor [tid-3].w = s_left_neighbor [tid-2].w = s_left_neighbor [tid-1].w = s_left_neighbor [tid].w = Rec_Row[3].w;
					}
					
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MV.x = 0;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MV.y = 0;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].RefFrameIdx = -1;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MinSAD = intra_sad;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Type  = INTRA_LARGE_BLOCKS_MB_TYPE;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].SubType  = intra_pred_mode;

				}
				
				else
				{
					
					{
						s_left_neighbor [tid].x  = dev_recon[k*16 + j*MB_HEIGHT*out_stride + tid_y*4*out_stride + 15 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
						s_left_neighbor [tid].y  = dev_recon[k*16 + j*MB_HEIGHT*out_stride + tid_y*4*out_stride + out_stride + 15 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
						s_left_neighbor [tid].z  = dev_recon[k*16 + j*MB_HEIGHT*out_stride + tid_y*4*out_stride + 2*out_stride + 15 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
						s_left_neighbor [tid].w  = dev_recon[k*16 + j*MB_HEIGHT*out_stride + tid_y*4*out_stride + 3*out_stride + 15 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
					}

				}
		}
	}
}