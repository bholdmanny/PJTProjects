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

#include "../inc/const_defines.h"
#include "intra_rc_chroma_kernel.cu"


__global__ void iframe_residual_coding_chroam_kernel (	unsigned char *	dev_input_uv,
																	//unsigned char *	dev_top_neighbor_uv,
																	S_BLK_MB_INFO *	dev_blk_mb_info,
																	S_QP_DATA	  *	dev_QpData_uv,
																	short		  *	Quant_tab_uv,
																	short		  *	Dquant_tab_uv,
																	unsigned char *	dev_recon_uv,
																	short		  *	dev_dct_coefs_uv,
																	short		  *	dev_dc_coefs_uv,
																	int				width_c,
																	int				height_c,
																	int				width_ref_c,
																	int				height_ref_c,
																	int				num_mb_hor,
																	int				num_mb_ver,
																	int				slice_num
																	)
{
	
		__shared__  uchar4 s_top_neighbor_uv[8];
		__shared__  uchar4 s_left_neighbor_uv[8];
		__shared__ unsigned char s_in_uv[64*2];
		__shared__ unsigned int Sad[8];
		__shared__ short        Quant_tab[16];
		__shared__ short        DQuant_tab[16];
		short2		Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
					Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;

		uchar4		  Pred_Row[4],Rec_Row[4];
		__shared__ short	QcoefDc[8],TempDC[8];

		int TopAvailable, LeftAvailable,pred_mode;
		S_BLK_MB_INFO BlkMBInfo;
		S_BLK_MB_INFO BlkMBInfo1;
		int  	Quant_Add;
		int  	Quant_Shift;
		int  	Dquant_Shift;
		int		num_mbs = num_mb_hor*num_mb_ver;
		int		input_index,top_index,rec_index,coef_index;
		int		first_mb;

		int tid_x,tid_y;
		tid_x = threadIdx.x;
		tid_y = threadIdx.y;
		int  tid = tid_x + tid_y * blockDim.x;

	    Quant_tab[tid] = Quant_tab_uv[tid];
		Quant_tab[tid+8] = Quant_tab_uv[tid+8];
		DQuant_tab[tid] = Dquant_tab_uv[tid];
		DQuant_tab[tid+8] = Dquant_tab_uv[tid+8];

		Quant_Add = dev_QpData_uv->QuantAdd;
		Quant_Shift = dev_QpData_uv->QuantShift;
		Dquant_Shift = dev_QpData_uv->DQuantShift;
		first_mb = blockIdx.x*num_mb_hor* num_mb_ver/slice_num;

		//将全局存储器中的数据加载到共享存储器 
		s_left_neighbor_uv[tid].x = 0;  //left pix for uv component，first 4 threads for u ,the rest for v
		s_left_neighbor_uv[tid].y = 0;
		s_left_neighbor_uv[tid].z = 0;  
		s_left_neighbor_uv[tid].w = 0;
	
		for(int j =0;j< num_mb_ver/slice_num;j++)
		{
			input_index = tid+j*width_c*MB_HEIGHT_C + blockIdx.x*MB_HEIGHT_C*width_c*(num_mb_ver/slice_num);
			rec_index	= tid_x*BLK_WIDTH + (tid_y&1)*width_ref_c*BLK_HEIGHT + j*MB_WIDTH_C*width_ref_c + (tid_y>>1)*height_ref_c*width_ref_c + blockIdx.x*MB_HEIGHT_C*width_ref_c*(num_mb_ver/slice_num);
			coef_index	= (tid&3)*16+j*num_mb_hor*MB_TOTAL_SIZE_C + (tid_y>>1)*MB_TOTAL_SIZE_C*num_mbs + first_mb*MB_TOTAL_SIZE_C;
			top_index = tid_x*BLK_WIDTH + (j*MB_HEIGHT_C-1)*width_ref_c + (tid_y>>1)*width_ref_c*height_ref_c + blockIdx.x*MB_HEIGHT_C*width_ref_c*(num_mb_ver/slice_num) ;
			for(int k =0;k< num_mb_hor;k++)
			{
				
			//将一个宏块的原始数据以光栅的形式导入共享存储器
				for (int i=0;i<16;i++)
				{
					s_in_uv[tid+i*8]= dev_input_uv[input_index+(i>>3)*width_c*height_c+(i&7)*width_c];
				}
				

				BlkMBInfo	= dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB];
				BlkMBInfo1	= dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB + 8];
				TopAvailable  = (BlkMBInfo.Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
				LeftAvailable = (BlkMBInfo.Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;

				if(TopAvailable)
				{
					s_top_neighbor_uv[tid].x = dev_recon_uv[top_index];   
					s_top_neighbor_uv[tid].y = dev_recon_uv[top_index+1];
					s_top_neighbor_uv[tid].z = dev_recon_uv[top_index+2];
					s_top_neighbor_uv[tid].w = dev_recon_uv[top_index+3];
				}

				// 帧内预测
				IntraChromaPrediction_cu(
											s_top_neighbor_uv,
											s_left_neighbor_uv,
											s_in_uv+tid_x*4+tid_y*32,
											TopAvailable,
											LeftAvailable,
											Pred_Row,
											Sad, 
											pred_mode,
											tid_x,
											tid_y
										);
				
				BlkMBInfo.IntraChromaMode  = pred_mode;
				BlkMBInfo1.IntraChromaMode  = pred_mode;

				dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB]=BlkMBInfo;
				dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB +8]=BlkMBInfo1;

				IntraChromaTransforms_cu(
											s_in_uv+tid_x*4+tid_y*32,
											Pred_Row,
											Rec_Row,
											Quant_Add,
											Quant_Shift,
											Dquant_Shift,
											Quant_tab,
											DQuant_tab,
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
	    	 			dev_recon_uv[rec_index+i*width_ref_c+0] = Rec_Row[i].x;
		 				dev_recon_uv[rec_index+i*width_ref_c+1] = Rec_Row[i].y;
		 				dev_recon_uv[rec_index+i*width_ref_c+2] = Rec_Row[i].z;
		 				dev_recon_uv[rec_index+i*width_ref_c+3] = Rec_Row[i].w;
	 				}
					
					dev_dct_coefs_uv[coef_index]   = Qcoef_0_01.x;
					dev_dct_coefs_uv[coef_index+1] = Qcoef_0_01.y;
					dev_dct_coefs_uv[coef_index+2] = Qcoef_0_23.x;
					dev_dct_coefs_uv[coef_index+3] = Qcoef_0_23.y;
					dev_dct_coefs_uv[coef_index+4] = Qcoef_1_01.x;
					dev_dct_coefs_uv[coef_index+5] = Qcoef_1_01.y;
					dev_dct_coefs_uv[coef_index+6] = Qcoef_1_23.x;
					dev_dct_coefs_uv[coef_index+7] = Qcoef_1_23.y;
					dev_dct_coefs_uv[coef_index+8] = Qcoef_2_01.x;
					dev_dct_coefs_uv[coef_index+9] = Qcoef_2_01.y;
					dev_dct_coefs_uv[coef_index+10] = Qcoef_2_23.x;
					dev_dct_coefs_uv[coef_index+11] = Qcoef_2_23.y;
					dev_dct_coefs_uv[coef_index+12] = Qcoef_3_01.x;
					dev_dct_coefs_uv[coef_index+13] = Qcoef_3_01.y;
					dev_dct_coefs_uv[coef_index+14] = Qcoef_3_23.x;
					dev_dct_coefs_uv[coef_index+15] = Qcoef_3_23.y;

					dev_dc_coefs_uv[(tid%4) + k*BLOCKS_PER_MB_C + j*BLOCKS_PER_MB_C*num_mb_hor +(tid_y>>1)*BLOCKS_PER_MB_C*num_mbs + first_mb*BLOCKS_PER_MB_C] = QcoefDc[tid];

					// write mbinfo for next mb row
					if(tid_x == 1)
					{
						s_left_neighbor_uv [tid-1].x = s_left_neighbor_uv [tid].x = Rec_Row[0].w;
						s_left_neighbor_uv [tid-1].y = s_left_neighbor_uv [tid].y = Rec_Row[1].w;
						s_left_neighbor_uv [tid-1].z = s_left_neighbor_uv [tid].z = Rec_Row[2].w;
						s_left_neighbor_uv [tid-1].w = s_left_neighbor_uv [tid].w = Rec_Row[3].w;
					}

					input_index += MB_WIDTH_C;
					top_index += MB_WIDTH_C;
					rec_index += MB_WIDTH_C;
					coef_index += MB_TOTAL_SIZE_C;
				}
		}
}

inline __device__ void Chroma_inter_HadamardTransformAndQuantize_kernel(
																	short *TempDC,
																	int &Quant_Add,                       
																	int &Quant_Shift,                     
																	short &Quant_tab0,
																	short *QCoefDC,
																	int &tid
																)
{
		short TempCoef;
		int   QAdd;
		int   Sign;
		int   QuantDCShift;

		QAdd = Quant_Add * 2;            
		QuantDCShift = Quant_Shift + 1;
		if((tid&3)==0)
		{
			TempCoef = TempDC[tid] + TempDC[1+tid] + TempDC[2+tid] + TempDC[3+tid];
		}
		else if((tid&3)==1)
		{
			TempCoef = TempDC[0+(tid>>2)*4] - TempDC[1+(tid>>2)*4] + TempDC[2+(tid>>2)*4] - TempDC[3+(tid>>2)*4];
		}
		else if((tid&3)==2)
		{
			TempCoef = TempDC[0+(tid>>2)*4] + TempDC[1+(tid>>2)*4] - TempDC[2+(tid>>2)*4] - TempDC[3+(tid>>2)*4];
		}
		else
		{
			TempCoef = TempDC[0+(tid>>2)*4] - TempDC[1+(tid>>2)*4] - TempDC[2+(tid>>2)*4] + TempDC[3+(tid>>2)*4];
		}

		Sign  = (TempCoef >= 0) ? 1 : -1;
		TempCoef  = (TempCoef >= 0) ? TempCoef : -TempCoef;
		TempCoef  = min (((TempCoef * Quant_tab0 + QAdd) >> QuantDCShift), TempCoef);
		QCoefDC[tid] = Sign * TempCoef;
}
//哈达码变换的逆变换
inline __device__ void Chroma_inter_IHadamard2x2AndDQuant_kernel(	short *TempDC,                  
																	int &DQuant_Shift,                     
																	short &DQuant_tab0,
																	short *QCoefDC,
																	int &tid
																	)
{
		short TempCoef;
		
		QCoefDC[tid] = (TempDC[tid] * DQuant_tab0 << DQuant_Shift);

		if((tid&3)==0)
		{
			TempCoef = QCoefDC[tid] + QCoefDC[1+tid] + QCoefDC[2+tid] + QCoefDC[3+tid];
		}
		else if((tid&3)==1)
		{
			TempCoef = QCoefDC[0+(tid>>2)*4] - QCoefDC[1+(tid>>2)*4] + QCoefDC[2+(tid>>2)*4] - QCoefDC[3+(tid>>2)*4];
			
		}
		else if((tid&3)==2)
		{
			TempCoef = QCoefDC[0+(tid>>2)*4] + QCoefDC[1+(tid>>2)*4] - QCoefDC[2+(tid>>2)*4] - QCoefDC[3+(tid>>2)*4];
		}
		else
		{
			TempCoef = QCoefDC[0+(tid>>2)*4] - QCoefDC[1+(tid>>2)*4] - QCoefDC[2+(tid>>2)*4] + QCoefDC[3+(tid>>2)*4];
		}
		QCoefDC [tid] =  (TempCoef>>1);

}

inline __device__ void Chroma_inter_TransformAndQuantize_kernel(short *Diff,short *Dc_coef,short *Quant_tables,int &QAdd,int &Quant_Shift,int &tx,int &ty,int &tz)
{
		short Sum0,Sum1,Diff0,Diff1; 
		short coef0,coef1,coef2,coef3;
		int tid_index;
		int sign;
		
		//碟形算法垂直部分
		tid_index = tx+(ty&1)*8+(ty>>1)*64+tz*128;
		
		Sum0 = (Diff[tid_index] + Diff[tid_index+48]);
		Sum1 = (Diff[tid_index+16] + Diff[tid_index+32]);
		Diff0 = (Diff[tid_index] - Diff[tid_index+48]);
		Diff1 = (Diff[tid_index+16] - Diff[tid_index+32]);

		Diff[tid_index] = Sum0 + Sum1;
        Diff[tid_index+32] = Sum0 - Sum1;
        Diff[tid_index+16] = 2 * Diff0 + Diff1;
        Diff[tid_index+48] = Diff0 - 2 * Diff1;
		
		__syncthreads(); //同步，也许不需要，因为做水平碟形算法需要通信的线程在一个warp中

		//按照相邻的4个线程处理一个4*4子宏块的方式组织线程
		tid_index = ((tx&3)<<4)+((tx>>2)<<2)+((ty>>1)<<3)+((ty&1)<<6)+(tz<<7); //等效于tid_index = (tx&3)*16+(tx>>2)*4+(ty>>1)*8+(ty&1)*64+tz*128;
		Sum0 = (Diff[tid_index] + Diff[tid_index+3]);
		Sum1 = (Diff[tid_index+1] + Diff[tid_index+2]);
		Diff0 = (Diff[tid_index] - Diff[tid_index+3]);
		Diff1 = (Diff[tid_index+1] - Diff[tid_index+2]);
		 
		coef0= Sum0 + Sum1;
        coef2 = Sum0 - Sum1;
        coef1 = 2 * Diff0 + Diff1;
        coef3 = Diff0 - 2 * Diff1;

		if((tx&3) == 0 ) //保存直流分量,只在0和4号线程处需要处理
		{
			Dc_coef[(tx>>2)+(ty<<1)+(tz<<3)] = coef0;
		}

		//量化,按照8x8块的方式以blk-falt的格式输出(开始时是以光删形式排列的)
		tid_index = tx*4+ty*32+tz*128; 
		
		sign  = (coef0 >= 0) ? 1 : -1;
		coef0  = (coef0 >= 0) ? coef0 : -coef0;
		Diff[tid_index] = sign * ((coef0 * Quant_tables[(tx&3)*4] + QAdd) >> Quant_Shift);

		sign  = (coef1 >= 0) ? 1 : -1;
		coef1  = (coef1 >= 0) ? coef1 : -coef1;
		Diff[tid_index+1] = sign * ((coef1 * Quant_tables[(tx&3)*4+1] + QAdd) >> Quant_Shift);

		sign  = (coef2 >= 0) ? 1 : -1;
		coef2  = (coef2 >= 0) ? coef2 : -coef2;
		Diff[tid_index+2] = sign * ((coef2 * Quant_tables[(tx&3)*4+2] + QAdd) >> Quant_Shift);

		sign  = (coef3 >= 0) ? 1 : -1;
		coef3  = (coef3 >= 0) ? coef3 : -coef3;
		Diff[tid_index+3] = sign * ((coef3 * Quant_tables[(tx&3)*4+3] + QAdd) >> Quant_Shift);
	
}

//色度分量的逆变换和反量化
inline __device__ void Chroma_inter_DQuantAndITransform_kernel(short *dct_coef,short * dc_coef,unsigned int *pred,short *DQuant_tables,int &DQuant_Shift,int &tx,int &ty,int &tz)
{
		short	Sum0,Sum1,Diff0,Diff1;
		short   coef0,coef4,coef8,coef12;
		int tid_index;
		int tid_block = tx+ty*8+tz*32;
		//反量化
		dct_coef[tid_block] = (dct_coef[tid_block] * DQuant_tables[tx+(ty&1)*8]) << DQuant_Shift;
		dct_coef[tid_block+64] = (dct_coef[tid_block+64] * DQuant_tables[tx+(ty&1)*8]) << DQuant_Shift;
		dct_coef[tid_block+128] = (dct_coef[tid_block+128] * DQuant_tables[tx+(ty&1)*8]) << DQuant_Shift;
		dct_coef[tid_block+192] = (dct_coef[tid_block+192] * DQuant_tables[tx+(ty&1)*8]) << DQuant_Shift;
		__syncthreads();
		//替换直流系数
		if(tid_block < 16)
		{
			dct_coef[tid_block*16] = dc_coef[tid_block];
		
		}
		__syncthreads();

		tid_index = tx*4+ty*32+tz*128; //tid_index = tx*4+ty*blockDim.x*4+tz*blockDim.x*blockDim.y*4;
		//横向碟形算法
		Sum0  = dct_coef[tid_index] + dct_coef[tid_index+2];
        Diff0 = dct_coef[tid_index] - dct_coef[tid_index+2];
        Diff1 = (dct_coef[tid_index+1] >> 1) - dct_coef[tid_index+3];
        Sum1  = dct_coef[tid_index+1] + (dct_coef[tid_index+3] >> 1);

        dct_coef[tid_index] = Sum0 + Sum1;
        dct_coef[tid_index+1] = Diff0 + Diff1;
        dct_coef[tid_index+2] = Diff0 - Diff1;
        dct_coef[tid_index+3] = Sum0 - Sum1;
		__syncthreads();
		
		//垂直碟形算法
		tid_index = (tx&3)+((tx>>2)<<4)+ty*32+tz*128;

		Sum0 = (dct_coef[tid_index] + dct_coef[tid_index+8]);
		Sum1 = dct_coef[tid_index+4] + (dct_coef[tid_index+12]>>1);
		Diff0 = (dct_coef[tid_index] - dct_coef[tid_index+8]);
		Diff1 = ((dct_coef[tid_index+4]>>1) - dct_coef[tid_index+12]);
		
		tid_index = tx + ((ty&1)<<6) + ((ty>>1)<<3) + (tz<<7);
		
		coef0 = (Sum0 + Sum1 + 32) >> 6;
        coef0 = coef0 + pred[tid_index];

        coef4 = (Diff0 + Diff1 + 32) >> 6;       
        coef4 = coef4 + pred[tid_index+16];

        coef8 = (Diff0 - Diff1 + 32) >> 6;
        coef8 = coef8 + pred[tid_index+32];

        coef12 = (Sum0 - Sum1 + 32) >> 6;
        coef12 = coef12 + pred[tid_index+48];
		
		pred[tid_index] = (unsigned char)(coef0 < 0 ? 0 :((coef0 > 255) ? 255 : coef0));
		pred[tid_index+16] = (unsigned char)(coef4 < 0 ? 0 :((coef4 > 255) ? 255 : coef4));
		pred[tid_index+32] = (unsigned char)(coef8 < 0 ? 0 :((coef8 > 255) ? 255 : coef8));
		pred[tid_index+48] = (unsigned char)(coef12 < 0 ? 0 :((coef12 > 255) ? 255 : coef12));

}


__global__ void MotionCompensateChroma_kernel( unsigned char *dev_ref_uv,
											   unsigned char *dev_pred_uv,
											   S_BLK_MB_INFO *dev_blk_mb_info,
											   int enc_width_c,
											   int enc_height_c,
											   int ref_width_c,
											   int ref_height_c,
											   int RefStride2BeginUV/*,
											   int *dev_index,
											   unsigned char *dev_pred_ref*/
											)
{
		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		int tid_z = threadIdx.z;
		int tid_blk = tid_x + tid_y*blockDim.x + tid_z*blockDim.x*blockDim.y;

		/*__shared__ unsigned char ref_uv[9*9*2];*/
		__shared__ unsigned int ref_uv[9*9*2];
		__shared__ S_BLK_MB_INFO blk_mb_info[BLOCKS_PER_MB];
		
		int index_blk_info; //每个线程计算的像素所属的2*2块的位置
		int mv_x,mv_y,ref_x,ref_y,FracY,FracX;
		int ref_offset,pre_offset;
		unsigned int left0,right0,left1,right1,result;
		int value0,value1,value2,value3;

		index_blk_info = (tid_x>>1) + (tid_y>>1)*4; 
		pre_offset = tid_x + tid_y*enc_width_c + tid_z*enc_width_c*enc_height_c + blockIdx.x*MB_WIDTH_C+blockIdx.y*MB_HEIGHT_C*enc_width_c;
		
		//为每一个block加载对应的MB info,
		if(tid_blk < 16)
		{
			blk_mb_info[tid_blk] = dev_blk_mb_info[tid_blk + blockIdx.x*BLOCKS_PER_MB+blockIdx.y*(enc_width_c>>3)*BLOCKS_PER_MB];
		}
		__syncthreads();

		mv_x = blk_mb_info[index_blk_info].MV.x;
		mv_y = blk_mb_info[index_blk_info].MV.y;

		ref_x = (mv_x>>3) + tid_x + blockIdx.x*8;
		ref_y = (mv_y>>3) + tid_y + blockIdx.y*8;

		FracY = (mv_y & 0x7);
		FracX = (mv_x & 0x7);
		
		ref_x = (ref_x < -MB_WIDTH_C ) ?  -MB_WIDTH_C :((ref_x > (ref_width_c - 1)) ? (ref_width_c - 1) : ref_x);
		ref_y = (ref_y < -MB_HEIGHT_C ) ?  -MB_HEIGHT_C :((ref_y > (ref_height_c - 1)) ? (ref_height_c - 1) : ref_y);
		ref_offset = ref_x + ref_y*ref_width_c + RefStride2BeginUV+tid_z*ref_width_c*ref_height_c;
		
		//加载一个8*8宏块需要的参考数据
		ref_uv[tid_x+tid_y*9+tid_z*9*9] = (unsigned int)dev_ref_uv[ref_offset];
		//加载每一行的第九个像素，由第8列线程加载
		if(tid_y==0)
		{
			ref_x = (ref_x + (blockDim.x-tid_x));
			ref_y = (ref_y + tid_x);
			ref_x = (ref_x < -MB_WIDTH_C ) ?  -MB_WIDTH_C :((ref_x > (ref_width_c - 1)) ? (ref_width_c - 1) : ref_x);
			ref_y = (ref_y < -MB_HEIGHT_C ) ?  -MB_HEIGHT_C :((ref_y > (ref_height_c - 1)) ? (ref_height_c - 1) : ref_y);

			ref_offset = ref_x + ref_y*ref_width_c + RefStride2BeginUV+tid_z*ref_width_c*ref_height_c;
			ref_uv[tid_x*9+tid_z*9*9+8] = (unsigned int)dev_ref_uv[ref_offset];
		}

		if(tid_y == (blockDim.y - 1))
		{
			ref_x = (ref_x < -MB_WIDTH_C ) ?  -MB_WIDTH_C :((ref_x > (ref_width_c - 1)) ? (ref_width_c - 1) : ref_x);
			ref_y = ((ref_y+1) < -MB_HEIGHT_C ) ?  -MB_HEIGHT_C :(((ref_y+1) > (ref_height_c - 1)) ? (ref_height_c - 1) : (ref_y+1));
			ref_offset = ref_x + ref_y*ref_width_c + RefStride2BeginUV+tid_z*ref_width_c*ref_height_c;
			ref_uv[tid_x + blockDim.y*9+tid_z*9*9] = (unsigned int)dev_ref_uv[ref_offset];
		}

		if((tid_x==(blockDim.x-1))&&(tid_y==(blockDim.y-1)))
		{
			ref_x = (((ref_x+1) < -MB_WIDTH_C ) ?  -MB_WIDTH_C :(((ref_x+1) > (ref_width_c - 1)) ? (ref_width_c - 1) : (ref_x+1)));
			ref_offset = ref_x + ref_y*ref_width_c + RefStride2BeginUV+tid_z*ref_width_c*ref_height_c;
			ref_uv[(tid_z+1)*9*9-1] = (unsigned int)dev_ref_uv[ref_offset];
		}
		__syncthreads();

		left0 = ref_uv[tid_x + tid_y*9+tid_z*81];
		right0 = ref_uv[tid_x + tid_y*9+tid_z*81 + 1];
		left1 = ref_uv[tid_x + (tid_y+1)*9+tid_z*81];
		right1 = ref_uv[tid_x + (tid_y+1)*9+tid_z*81+1];

        value0 = (8 - FracX) * (8 - FracY) * (int)(left0);
        value1 = (    FracX) * (8 - FracY) * (int)(right0);
        value2 = (8 - FracX) * (    FracY) * (int)(left1);
        value3 = (    FracX) * (    FracY) * (int)(right1);
        result = (unsigned char)((value0 + value1 + value2 + value3 + 32) >> 6);

        dev_pred_uv[pre_offset] = result;

}

//P帧色度分量帧间编码
__global__ void ChromaPFrameInterResidualCoding_kernel( unsigned char *dev_input_uv,
														unsigned char *dev_pred_uv,
														unsigned char *dev_recon_uv,
														short		  *dev_dct_coefs_uv,
														short		  *dev_dc_coefs_uv,
														short		  *dev_Quant_tables,
														short	      *dev_Dquant_tables,
														S_QP_DATA	  *dev_QpData,
														int			  enc_width_c,
														int			  enc_height_c,
														int			  ref_width_c,
														int			  ref_height_c,
														int			  num_mb_hor,
														int			  num_mb_ver
														)
{
	//dim3 threads(8,4,2);
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_z = threadIdx.z;
	int tid_blk = tid_x+ (tid_y+tid_z*blockDim.y)*blockDim.x;

	int offset_input = blockIdx.x*2*MB_WIDTH_C+blockIdx.y*MB_HEIGHT_C*enc_width_c + tid_z*enc_width_c*enc_height_c; // 每一个Block开始处理的宏块的起始位置，注意一个block处理两个宏块
	int offset_output = blockIdx.x*2*MB_WIDTH_C+blockIdx.y*MB_HEIGHT_C*ref_width_c +tid_z*ref_width_c*ref_height_c; //每一个Block对应的重建帧的起始位置
	
	int tid_index = tid_z*MB_HEIGHT_C*MB_HEIGHT_C*2+tid_y*8+tid_x; //Block内每一个线程对应像素的位置，
	int out_dct_index,out_rec_index;
//	unsigned char src_input;

	__shared__	unsigned int 	pred_in[MB_TOTAL_SIZE_C*4];
	__shared__  unsigned int  src_in[MB_TOTAL_SIZE_C*4];
	__shared__  short			Diff[MB_TOTAL_SIZE_C*4];
	__shared__  short			Dc_coef[16];
	__shared__  short			Dc_coef_temp[16];
	__shared__	short			Quant_tables[16];
	__shared__	short			DQuant_tables[16];
	
	if(tid_blk < 16)
	{
		Quant_tables[tid_blk] = dev_Quant_tables[tid_blk];
		DQuant_tables[tid_blk] = dev_Dquant_tables[tid_blk];  
	}

     int Quant_Add = dev_QpData->QuantAdd;
     int Quant_Shift = dev_QpData->QuantShift;
     int Dquant_Shift = dev_QpData->DQuantShift;

	//按照两个宏块并排的形式（光栅形式排列，前16个像素分别属于不同的两个宏块，得到16*8的块大小）
	for(int i=0;i<4;i++)
	{
		src_in[tid_index+i*32] = ( unsigned int)dev_input_uv[offset_input + tid_x+(tid_y&1)*8 + (tid_y>>1)*enc_width_c + i*2*enc_width_c];
		pred_in[tid_index+i*32] = (unsigned int)dev_pred_uv[offset_input + tid_x+(tid_y&1)*8 + (tid_y>>1)*enc_width_c + i*2*enc_width_c];
		Diff[tid_index+i*32] = (short)(src_in[tid_index+i*32] - pred_in[tid_index+i*32]);
	}
	__syncthreads();
	
	//色度分量的DCT变换和量化
	Chroma_inter_TransformAndQuantize_kernel(Diff,Dc_coef_temp,Quant_tables,Quant_Add,Quant_Shift,tid_x,tid_y,tid_z);
	__syncthreads();
	
	
	//哈达码变换和逆变换，共4个2*2块的变换，由16个线程完成
	if(tid_blk < 16)
	{
		  int out_dc_index = (tid_blk&7)+blockIdx.x*2*BLOCKS_PER_MB_C+blockIdx.y*BLOCKS_PER_MB_C*num_mb_hor+tid_y*num_mb_hor*num_mb_ver*BLOCKS_PER_MB_C;

		  Chroma_inter_HadamardTransformAndQuantize_kernel(
															Dc_coef_temp,
															Quant_Add,                       
															Quant_Shift,                     
															Quant_tables[0],
															Dc_coef,
															tid_blk
															);
		  dev_dc_coefs_uv[out_dc_index] = Dc_coef[tid_blk];
		  
		

		  Chroma_inter_IHadamard2x2AndDQuant_kernel(
													Dc_coef,
													Dquant_Shift,                     
													DQuant_tables[0],
													Dc_coef_temp,
													tid_blk
													);
	}

	__syncthreads();

	out_dct_index = tid_x+tid_y*blockDim.x+blockIdx.x*2*MB_TOTAL_SIZE_C+blockIdx.y*num_mb_hor*MB_TOTAL_SIZE_C+tid_z*num_mb_hor*num_mb_ver*MB_TOTAL_SIZE_C;
	for(int i=0;i<4;i++)
	{
		dev_dct_coefs_uv[out_dct_index+i*32] = Diff[tid_x+tid_y*blockDim.x + tid_z*2*MB_TOTAL_SIZE_C + i*32];
	}
	__syncthreads();
	//逆变换和反量化
	Chroma_inter_DQuantAndITransform_kernel(Diff,
											Dc_coef_temp,
											pred_in,
											DQuant_tables,
											Dquant_Shift,
											tid_x,
											tid_y,
											tid_z);

	

	out_rec_index = offset_output + tid_x+(tid_y&1)*8 + (tid_y>>1)*ref_width_c;
	for(int i = 0;i <4;i++)
	{
		dev_recon_uv[out_rec_index + i*2*ref_width_c] = pred_in[tid_x +  tid_y*blockDim.x + tid_z*2*MB_TOTAL_SIZE_C + i*32];
	}

}


__global__ void Chroma_PFrame_Intra_ResidualCoding_kernel( unsigned char *dev_input_uv,
														unsigned char *dev_recon_uv,
														S_BLK_MB_INFO *dev_blk_mb_info,
														short		  *dev_dct_coefs_uv,
														short		  *dev_dc_coefs_uv,
														short		  *dev_Quant_tables,
														short	      *dev_Dquant_tables,
														S_QP_DATA	  *dev_QpData,
														int			  enc_width_c,
														int			  enc_height_c,
														int			  ref_width_c,
														int			  ref_height_c,
														int			  num_mb_hor,
														int			  num_mb_ver,
														int			  slice_num
														)
{
	//dim3 threads(2,4,1);
		__shared__  uchar4 s_top_neighbor[8];
		__shared__  uchar4 s_left_neighbor[8];
		__shared__ unsigned char src_in[MB_TOTAL_SIZE_C*2];
		__shared__ unsigned int Sad[8];
		__shared__ short        Quant_tab[16];
		__shared__ short        DQuant_tab[16];
		short2		Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
					Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;

		uchar4		  Pred_Row[4],Rec_Row[4];
		__shared__ short	QcoefDc[8],TempDC[8];

		int TopAvailable, LeftAvailable,pred_mode;
		S_BLK_MB_INFO BlkMBInfo;
		S_BLK_MB_INFO BlkMBInfo1;
		int  	Quant_Add;
		int  	Quant_Shift;
		int  	Dquant_Shift;
		int		num_mbs = num_mb_hor*num_mb_ver;
		int		rec_index,coef_index;
		int		first_mb;

		int tid_x,tid_y;
		tid_x = threadIdx.x;
		tid_y = threadIdx.y;
		int  tid = tid_x + tid_y * blockDim.x;

	    Quant_tab[tid] = dev_Quant_tables[tid];
		Quant_tab[tid+8] = dev_Quant_tables[tid+8];
		DQuant_tab[tid] = dev_Dquant_tables[tid];
		DQuant_tab[tid+8] = dev_Dquant_tables[tid+8];

		Quant_Add = dev_QpData->QuantAdd;
		Quant_Shift = dev_QpData->QuantShift;
		Dquant_Shift = dev_QpData->DQuantShift;
		first_mb = blockIdx.x*num_mb_hor* num_mb_ver/slice_num;

		//将全局存储器中的数据加载到共享存储器 
		for(int j =0;j< num_mb_ver/slice_num;j++)
		{
			//input_index = tid_x*BLK_WIDTH +(tid_y&1)*enc_width_c*BLK_HEIGHT + j*enc_width_c*MB_WIDTH_C + (tid_y>>1)*enc_width_c*enc_height_c;
			rec_index	= tid_x*BLK_WIDTH + (tid_y&1)*ref_width_c*BLK_HEIGHT + j*MB_HEIGHT_C*ref_width_c + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num);
			coef_index	= ((tid&3)<<4)+j*num_mb_hor*MB_TOTAL_SIZE_C + (tid_y>>1)*enc_width_c*enc_height_c + first_mb*MB_TOTAL_SIZE_C;

			s_left_neighbor[tid].x = 0;
			s_left_neighbor[tid].y = 0;
			s_left_neighbor[tid].z = 0;
			s_left_neighbor[tid].w = 0;

			for(int k =0;k< num_mb_hor;k++)
			{
				BlkMBInfo	= dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB];
				BlkMBInfo1	= dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB +8 + first_mb*BLOCKS_PER_MB];

				if((BlkMBInfo.Type == INTRA_LARGE_BLOCKS_MB_TYPE) || (BlkMBInfo.Type == INTRA_SMALL_BLOCKS_MB_TYPE))
				{
					//将一个宏块的原始数据以光栅的形式导入共享存储器
					for (int i=0;i<16;i++)
					{
						src_in[tid+i*8]= dev_input_uv[tid+k*MB_WIDTH_C+j*enc_width_c*MB_WIDTH_C+(i>>3)*enc_width_c*enc_height_c+(i&7)*enc_width_c + blockIdx.x*MB_HEIGHT_C*enc_width_c*(num_mb_ver/slice_num)];
					}

					if(j==0)
					{
						s_top_neighbor[tid].x = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num)];
						s_top_neighbor[tid].y = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num) + 1];
						s_top_neighbor[tid].z = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num) + 2];
						s_top_neighbor[tid].w = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num) + 3];
					}
					else
					{
						s_top_neighbor[tid].x = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (j*MB_HEIGHT_C-1)*ref_width_c + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num)];
						s_top_neighbor[tid].y = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (j*MB_HEIGHT_C-1)*ref_width_c + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num) + 1];
						s_top_neighbor[tid].z = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (j*MB_HEIGHT_C-1)*ref_width_c + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num) + 2];
						s_top_neighbor[tid].w = dev_recon_uv[tid_x*4 + k*MB_WIDTH_C + (j*MB_HEIGHT_C-1)*ref_width_c + (tid_y>>1)*ref_width_c*ref_height_c + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num) + 3];
					
					}

					TopAvailable  = (BlkMBInfo.Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
					LeftAvailable = (BlkMBInfo.Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;

					// 帧内预测
					IntraChromaPrediction_cu(
											s_top_neighbor,
											s_left_neighbor,
											src_in+tid_x*4+tid_y*32,
											TopAvailable,
											LeftAvailable,
											Pred_Row,
											Sad, 
											pred_mode,
											tid_x,
											tid_y
										);

					BlkMBInfo.IntraChromaMode  = pred_mode;
					BlkMBInfo1.IntraChromaMode  = pred_mode;
					dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB]=BlkMBInfo;
				    dev_blk_mb_info[tid+k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB +8 + first_mb*BLOCKS_PER_MB]=BlkMBInfo1;

					IntraChromaTransforms_cu(
											src_in+tid_x*4+tid_y*32,
											Pred_Row,
											Rec_Row,
											Quant_Add,
											Quant_Shift,
											Dquant_Shift,
											Quant_tab,
											DQuant_tab,
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
	    	 			dev_recon_uv[rec_index+i*ref_width_c+0] = Rec_Row[i].x;
		 				dev_recon_uv[rec_index+i*ref_width_c+1] = Rec_Row[i].y;
		 				dev_recon_uv[rec_index+i*ref_width_c+2] = Rec_Row[i].z;
		 				dev_recon_uv[rec_index+i*ref_width_c+3] = Rec_Row[i].w;
	 				}

					dev_dct_coefs_uv[coef_index]   = Qcoef_0_01.x;
					dev_dct_coefs_uv[coef_index+1] = Qcoef_0_01.y;
					dev_dct_coefs_uv[coef_index+2] = Qcoef_0_23.x;
					dev_dct_coefs_uv[coef_index+3] = Qcoef_0_23.y;
					dev_dct_coefs_uv[coef_index+4] = Qcoef_1_01.x;
					dev_dct_coefs_uv[coef_index+5] = Qcoef_1_01.y;
					dev_dct_coefs_uv[coef_index+6] = Qcoef_1_23.x;
					dev_dct_coefs_uv[coef_index+7] = Qcoef_1_23.y;
					dev_dct_coefs_uv[coef_index+8] = Qcoef_2_01.x;
					dev_dct_coefs_uv[coef_index+9] = Qcoef_2_01.y;
					dev_dct_coefs_uv[coef_index+10] = Qcoef_2_23.x;
					dev_dct_coefs_uv[coef_index+11] = Qcoef_2_23.y;
					dev_dct_coefs_uv[coef_index+12] = Qcoef_3_01.x;
					dev_dct_coefs_uv[coef_index+13] = Qcoef_3_01.y;
					dev_dct_coefs_uv[coef_index+14] = Qcoef_3_23.x;
					dev_dct_coefs_uv[coef_index+15] = Qcoef_3_23.y;

					dev_dc_coefs_uv[(tid&3) + k*BLOCKS_PER_MB_C + j*BLOCKS_PER_MB_C*num_mb_hor +(tid_y>>1)*BLOCKS_PER_MB_C*num_mbs + first_mb*BLOCKS_PER_MB_C] = QcoefDc[tid];
					if(tid_x == 1)
					{
						s_left_neighbor [tid-1].x = s_left_neighbor [tid].x = Rec_Row[0].w;
						s_left_neighbor [tid-1].y = s_left_neighbor [tid].y = Rec_Row[1].w;
						s_left_neighbor [tid-1].z = s_left_neighbor [tid].z = Rec_Row[2].w;
						s_left_neighbor [tid-1].w = s_left_neighbor [tid].w = Rec_Row[3].w;
					}
				}
				else
				{
						s_left_neighbor [tid].x  = dev_recon_uv[k*MB_WIDTH_C + j*MB_HEIGHT_C*ref_width_c + (tid_y&1)*4*ref_width_c +(tid_y>>1)*ref_width_c*ref_height_c + 7 + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num)];
						s_left_neighbor [tid].y  = dev_recon_uv[k*MB_WIDTH_C + j*MB_HEIGHT_C*ref_width_c + (tid_y&1)*4*ref_width_c +(tid_y>>1)*ref_width_c*ref_height_c + ref_width_c + 7 + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num)];
						s_left_neighbor [tid].z  = dev_recon_uv[k*MB_WIDTH_C + j*MB_HEIGHT_C*ref_width_c + (tid_y&1)*4*ref_width_c +(tid_y>>1)*ref_width_c*ref_height_c + 2*ref_width_c + 7 + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num)];
						s_left_neighbor [tid].w  = dev_recon_uv[k*MB_WIDTH_C + j*MB_HEIGHT_C*ref_width_c + (tid_y&1)*4*ref_width_c +(tid_y>>1)*ref_width_c*ref_height_c + 3*ref_width_c + 7 + blockIdx.x*MB_HEIGHT_C*ref_width_c*(num_mb_ver/slice_num)];
					
				}
					rec_index += MB_WIDTH_C;
					coef_index += MB_TOTAL_SIZE_C;
			}
		}

}