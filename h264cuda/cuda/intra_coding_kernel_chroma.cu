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
#include <cutil_inline.h>
#include <cutil_inline_runtime.h>
#include <cutil.h>
#include "../inc/const_defines.h"
#include "../inc/mb_info.h"
#include "../inc/residual_coding.h"
inline __device__ void ChromaHadamardTransformAndQuantize(
														short *TempDC,
														int &Quant_Add,                       
														int &Quant_Shift,                     
														short &Quant_tab0,
														short *QCoefDC,
														int &tx,
														int &ty
													)
{
		int   QAdd;
		int   Sign;
		int   QuantDCShift;
		short TempCoef;
		short Out0,Out1,Out2,Out3;

		QAdd = Quant_Add * 2;            
		QuantDCShift = Quant_Shift + 1;

		Out0 = TempDC[(ty<<2)] + TempDC[1+(ty<<2)] + TempDC[2+(ty<<2)] + TempDC[3+(ty<<2)];
		Out1 = TempDC[(ty<<2)] - TempDC[1+(ty<<2)] + TempDC[2+(ty<<2)] - TempDC[3+(ty<<2)];
		Out2 = TempDC[(ty<<2)] + TempDC[1+(ty<<2)] - TempDC[2+(ty<<2)] - TempDC[3+(ty<<2)];
		Out3 = TempDC[(ty<<2)] - TempDC[1+(ty<<2)] - TempDC[2+(ty<<2)] + TempDC[3+(ty<<2)];
		
		TempCoef = ((tx==0) ? Out0 :((tx==1) ? Out1 :((tx==2) ? Out2 : Out3)));
		Sign  = (TempCoef >= 0) ? 1 : -1;
		TempCoef  = (Sign >= 0) ? TempCoef : -TempCoef;
		TempCoef  =(((TempCoef * Quant_tab0 + QAdd) >> QuantDCShift)< TempCoef) ? ((TempCoef * Quant_tab0 + QAdd) >> QuantDCShift) : TempCoef;
		QCoefDC[tx+ty*4] = Sign * TempCoef;
}
inline __device__ void ChromaIHadamard2x2AndDQuant(	short *TempDC,                  
													int &DQuant_Shift,                     
													short &DQuant_tab0,
													short *QCoefDC,
													int &tx,
													int &ty
													)
{
		short TempCoef;
		short Out0,Out1,Out2,Out3;

        QCoefDC[tx+ty*4] = (TempDC[tx+ty*4] * DQuant_tab0 << DQuant_Shift);

		Out0 = QCoefDC[(ty<<2)] + QCoefDC[1+(ty<<2)] + QCoefDC[2+(ty<<2)] + QCoefDC[3+(ty<<2)];
		Out1 = QCoefDC[(ty<<2)] - QCoefDC[1+(ty<<2)] + QCoefDC[2+(ty<<2)] - QCoefDC[3+(ty<<2)];
		Out2 = QCoefDC[(ty<<2)] + QCoefDC[1+(ty<<2)] - QCoefDC[2+(ty<<2)] - QCoefDC[3+(ty<<2)];
		Out3 = QCoefDC[(ty<<2)] - QCoefDC[1+(ty<<2)] - QCoefDC[2+(ty<<2)] + QCoefDC[3+(ty<<2)];
		TempCoef = ((tx==0) ? Out0 :((tx==1) ? Out1 :((tx==2) ? Out2 : Out3)));
		QCoefDC [tx+ty*4] =  (TempCoef>>1);

}

inline __device__ void Chr_Dct4x4TransformAndQuantize (
                                short *DiffRow,       //[i]  
                                int &QuantAdd,                       //[i]
                                int &QuantShift,                     //[i]                     
								short *Quant_tables,
								short *TempQCoef,
                                short  *QCoef,
								short  *QCoefDC,
								int	   &tx,
								int	   &ty,
								int    &tz,
								int    &tid
                                )
{
	int Sign;
	if((ty == 0)&&((tz&1)==0))
	{
		short Sum0,Sum1,Diff0,Diff1;
		Sum0 = (DiffRow[tx] + DiffRow[tx+12]);
		Sum1 = (DiffRow[tx+4] + DiffRow[tx+8]);
		Diff0 = (DiffRow[tx] - DiffRow[tx+12]);
		Diff1 = (DiffRow[tx+4] -  DiffRow[tx+8]);
		QCoef[tx] = Sum0 + Sum1;
		QCoef[tx+4] = (2*Diff0) + Diff1;
		QCoef[tx+8] = Sum0 - Sum1;
		QCoef[tx+12] = Diff0 - (2*Diff1);

		Sum0 = QCoef[tx*4] + QCoef[tx*4+3];
		Sum1 = QCoef[tx*4+1] + QCoef[tx*4+2];
		Diff0 = QCoef[tx*4] - QCoef[tx*4+3];
		Diff1 = QCoef[tx*4+1] - QCoef[tx*4+2];

		DiffRow[tx*4] = Sum0 + Sum1;
		DiffRow[tx*4+1] = (2 * Diff0) + Diff1;
		DiffRow[tx*4+2] = Sum0 - Sum1;
		DiffRow[tx*4+3] = Diff0 - (2 * Diff1);
		if(tx == 0 )
			QCoefDC[(tz >> 1)] = (*(DiffRow+tx));
	}
	Sign  = ((*(DiffRow+(tid&15))) >= 0) ? 1 : -1;
    DiffRow[(tid&15)] = (Sign >= 0) ? (*(DiffRow+(tid&15))) : (-(*(DiffRow+(tid&15)))) ;
	QCoef[(tid&15)] = (short)(Sign * (((*(DiffRow+(tid&15))) * Quant_tables[(tid&15)] + QuantAdd) >> QuantShift));

}

inline __device__ void Chr_IDct4x4AndAdd (
										  short *Coef,
										  short *Temp_Coef,
										  unsigned char *PredRow,
										  unsigned char *RecOutRow,
										  int			&tx,
										  int			&ty,
										  int			&tz
										  )
{
   short	Sum0,Sum1,Diff0,Diff1;
   if((ty == 0)&&((tz&1)==0))
   {
		Sum0  = Coef[tx*4] + Coef[tx*4+2];
        Diff0 = Coef[tx*4] - Coef[tx*4+2];
        Diff1 = ( Coef[tx*4+1] >> 1) -  Coef[tx*4+3];
        Sum1  =  Coef[tx*4+1] + ( Coef[tx*4+3] >> 1);
        Temp_Coef[tx*4] = Sum0  + Sum1;
        Temp_Coef[tx*4+1] = Diff0 + Diff1;
        Temp_Coef[tx*4+2] = Diff0 - Diff1;
        Temp_Coef[tx*4+3] = Sum0  - Sum1;

		Sum0 = Temp_Coef[tx] + Temp_Coef[tx+8];
		Sum1 = Temp_Coef[tx+4] + (Temp_Coef[tx+12] >> 1);
        Diff0 = Temp_Coef[tx] - Temp_Coef[tx+8];
        Diff1 = (Temp_Coef[tx+4] >> 1) - Temp_Coef[tx+12];
	    Coef[tx] = (Sum0 + Sum1 + 32) >> 6;
        Coef[tx+4] = (Diff0 + Diff1 + 32) >> 6;
        Coef[tx+8] = (Diff0 - Diff1 + 32) >> 6;
        Coef[tx+12] = (Sum0 - Sum1 + 32 ) >> 6;

	}
	Temp_Coef[tx + ty*4 +((tz&1)<<3)] = Coef[tx + ty*4 +((tz&1)<<3)] + PredRow[tx + ty*8 +((tz&1)<<4)];
    RecOutRow[tx + ty*8 +((tz&1)<<4)]= (unsigned char)((Temp_Coef[tx + ty*4 +((tz&1)<<3)] < 0) ? 0 :((Temp_Coef[tx + ty*4 +((tz&1)<<3)] > 255) ? 255 : Temp_Coef[tx + ty*4 +((tz&1)<<3)]));
}

inline __device__ void IntraChromaPrediction(
												unsigned char *top_pixels,
												unsigned char *left_pixels,
												unsigned char *s_in_uv,
												int &TopAvailable,
												int &LeftAvailable,
												unsigned char *Pred_Row,
												unsigned char *temp_pred,
												int *Sad, 
												int &pred_mode,
												int &tx,
												int &ty,
												int &tz
											)
{
	
    // Intra8x8 Prediction. 0=DC 1=Vertical, 2=Horizontal, 3=Plane
    // Note: the order of prediction mode differs from that of the luma 16x16.

    // DC. 
    // This mode is always tested regardless availability of the neighbors.
    // However, different equations are used for different availibity map. There are
    // four possible cases 1. Both top and left are available. 2. Only top is available
    // 3. Only left is available, 4. Neither top nor left is available.
    // each pixel is given the value of 128. Which is 1<<(BitDepth - 1). BitDepth=8
    
//   int TopSum, LeftSum, AllSum;
 //   int TopAvg, LeftAvg, AllAvg;
    int PredMode; 
	int tid;
	tid = tx + ty*4+tz*8;

    int MinSad = LARGE_NUMBER;
//   Avg = 0;
    PredMode = PRED_UNKNOWN;
    if(tz < 4)
	{
		Sad[tid] = (((ty+tz)&1)==0) ? (top_pixels[tx+ty*4+((tz>>1)*8)]+left_pixels[tx+ty*4+((tz>>1)<<3)]) :((tz&1)==0)?(top_pixels[tx+4+((tz>>1)<<3)]):left_pixels[tx+4+((tz>>1)<<3)];
		if(tx==0)
		{
			Sad[tid] =Sad[tid]+Sad[tid+1]+Sad[tid+2]+Sad[tid+3];
			Sad[tid] +=(((ty+tz)&1)==0) ? 4 : 2;
			Sad[tid] =(((ty+tz)&1)==0) ? (Sad[tid]>>3) : (Sad[tid]>>2);
		}
	}
	else if(tz < 8)
	{
		Sad[tid] = (tz< 6) ? left_pixels[tid-32] : top_pixels[tid-48];
		if(tx==0)
		{
			Sad[tid] = Sad[tid]+Sad[tid+1]+Sad[tid+2]+Sad[tid+3];
			Sad[tid] += 2;
			Sad[tid] =(Sad[tid]>>2);
		}
	}
	__syncthreads();

	Pred_Row[tid] = (TopAvailable && LeftAvailable) ? Sad[ty*4+((tz>>2)<<3)] : ( (LeftAvailable) ? Sad[((tz>> 2)<<2) + 32] : ((TopAvailable) ? Sad[ty*4+((tz>>3)<<3)+48] : 128 ));
	/*if(TopAvailable && LeftAvailable)
	{
		Pred_Row[tid] = Sad[ty*4+((tz>>2)<<3)];

	}
	else if(LeftAvailable && !TopAvailable)
	{
		Pred_Row[tid] = Sad[tz+32];

	}
	else if(TopAvailable && !LeftAvailable)
	{

		Pred_Row[tid] = Sad[tx+ty*4+((tz>>3)<<3)+48];
	}
	else 
	{
		Pred_Row[tid] = 0;
	}
	__syncthreads();*/
	Sad[tid] = (unsigned int)abs(s_in_uv[tid]-Pred_Row[tid]); /*CalcSadChroma_cu(s_in_uv,Pred_Row);*/
	__syncthreads();

	for(int k = 64 ; k >0 ; k>>=1)
	{
		if(tid<k)
		{
			Sad[tid] +=  Sad[tid+k];
		}
		__syncthreads();
	}
	__syncthreads();

	MinSad    = Sad[0];
	PredMode   = C_PRED_DC;
	__syncthreads();
	// Horizontal prediction
	if(LeftAvailable)
	{
		//temp_pred[tid] = left_pixels[tz];

		Sad[tid] = (unsigned int)abs(s_in_uv[tid]-left_pixels[tz]); /*CalcSadChroma_cu(s_in_uv,Pred_Row);*/
		__syncthreads();

		for(int k = 64 ; k >0 ; k>>=1)
		{
			if(tid<k)
			{
				Sad[tid] +=  Sad[tid+k];
			}
		__syncthreads();
		}
		__syncthreads();

		if (Sad[0] < MinSad)
		{
				MinSad    = Sad[0];
       			PredMode   = L_PRED_HORIZONTAL;
				Pred_Row[tid] = left_pixels[tz];
		}
		__syncthreads();
	}

	if(TopAvailable)
	{
		Sad[tid] = (unsigned int)abs(s_in_uv[tid]-top_pixels[tx+ty*4+((tz>>3)<<3)]);
		__syncthreads();

		for(int k = 64 ; k >0 ; k>>=1)
		{
			if(tid<k)
			{
				Sad[tid] +=  Sad[tid+k];
			}
		__syncthreads();
		}
		__syncthreads();
		if (Sad[0] < MinSad)
		{
				MinSad    = Sad[0];
       			PredMode   = C_PRED_VERTICAL;
				Pred_Row[tid] = top_pixels[tx+ty*4+((tz>>3)<<3)];
		}
		__syncthreads();
	}
	if(tid==0)
	{
		pred_mode = PredMode;
		Sad[0]   = MinSad;
	}

}


inline __device__ void	IntraChromaTransforms(
													unsigned char *InputSrcRow,
													unsigned char *PredRow,
													unsigned char *RecOutRow,
													int &QuantAdd,
													int &QuantShift,
													int	&DquantShift,
													short *Quant_tab,
													short *Dquant_tab,
													short *DiffRow,
													short *QCoef,
													short *TempQCoef,
													short  *QCoefDC,
													short  *TempDC,
													int &tx,
													int &ty,
													int &tz,
													int &tid
												)
{

		DiffRow[tid] = ((short)InputSrcRow[tx+ty*8+((tz&1)<<4)+((tz&2)<<1)+((tz&4)<<3)+((tz>>3)<<6)] - (short)PredRow[tx+ty*8+((tz&1)<<4)+((tz&2)<<1)+((tz&4)<<3)+((tz>>3)<<6)]);
		//QCoef[tid] = DiffRow[tid];
		Chr_Dct4x4TransformAndQuantize (
										DiffRow +((tz>>1)<<4),//[i]  
										QuantAdd,      //[i]
										QuantShift,
										Quant_tab,
										TempQCoef+((tz>>1)<<4),
										QCoef+((tz>>1)<<4),
										TempDC,
										tx,
										ty,
										tz,
										tid
										);
		__syncthreads();
		if( tz == 0 )
		{

			//QCoefDC[tid] = 0/*TempDC[tid]*/;
			ChromaHadamardTransformAndQuantize(
													TempDC,
													QuantAdd,                       
													QuantShift,                     
													Quant_tab[0],
													QCoefDC,
													tx,
													ty
												);
		
			ChromaIHadamard2x2AndDQuant(	QCoefDC,                  
											DquantShift,                     
											Dquant_tab[0],
											TempDC,
											tx,
											ty
											);
		}
		__syncthreads();
		DiffRow[tid] = ((tid&15)==0) ? TempDC[(tz>>1)]:((QCoef[tid] * Dquant_tab[tid&15]) << DquantShift);
		__syncthreads();

        //逆变换
		/* Chr_IDct4x4AndAdd (
							DiffRow+((tz>>1)<<4),
							TempQCoef+((tz>>1)<<4),
							PredRow+((tz&1)<<2)+((tz>>1)<<5),
							RecOutRow+((tz&1)<<2)+((tz>>1)<<5),
							tx,
							ty,
							tz
						);*/
		 Chr_IDct4x4AndAdd (
							DiffRow+((tz>>1)<<4),
							TempQCoef+((tz>>1)<<4),
							PredRow + (((tz>>1)&1)<<2)+((tz>>2)<<5),
							RecOutRow+(((tz>>1)&1)<<2)+((tz>>2)<<5),
							tx,
							ty,
							tz
						);
		 //RecOutRow[tx + ty*8 +((tz&1)<<4)+(((tz>>1)&1)<<2)+((tz>>2)<<5)] = (tx + ty*16 +((tz&1)<<5));

}


//帧内色度分量的编码，每个线程块处理一个U和V，kernel配置如下：grid(slice_num,1,1); block(4,4,8);
__global__ void iframe_residual_coding_chroam (	unsigned char *	dev_input_uv,
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
	
		__shared__ 	unsigned char 	top_pixels[8*2];
		__shared__  unsigned char 	left_pixels[8*2];
		/*__shared__  uchar4 s_top_neighbor_uv[8];
		__shared__  uchar4 s_left_neighbor_uv[8];*/
		__shared__	unsigned char 	input_mb[64*2];
		__shared__	int	      	Sad[128];
		__shared__ short	QcoefDc[4*2];
		__shared__ short    TempDC[4*2];
		__shared__ unsigned char rec_pixels[64*2];
		__shared__ unsigned char pred_pixels[64*2];
		__shared__ unsigned char temp_pred[64*2];
		__shared__ short	QcoefAC[64*2];
		__shared__ short Diff[64*2];
		__shared__ short Temp_Coef[64*2];
		__shared__ short        Quant_tab[16];
		__shared__ short        DQuant_tab[16];
		__shared__ int TopAvailable[16];
		__shared__ int LeftAvailable[16];
		/*short2		Qcoef_0_01, Qcoef_0_23, Qcoef_1_01, Qcoef_1_23,
					Qcoef_2_01, Qcoef_2_23, Qcoef_3_01, Qcoef_3_23;

		uchar4		  Pred_Row[4],Rec_Row[4];*/

		__shared__ int  pred_mode;
		//S_BLK_MB_INFO BlkMBInfo;
		//S_BLK_MB_INFO BlkMBInfo1;
		__shared__ int  	Quant_Add;
		__shared__ int  	Quant_Shift;
		__shared__ int  	Dquant_Shift;
		__shared__ int		first_mb;
		/*__shared__ int  Quant_Add[16];
		 __shared__ int  QuantAddDC[16];
		 __shared__ int  Quant_Shift[16];
		 __shared__ int  Dquant_Shift[16];*/
		int		num_mbs = num_mb_hor*num_mb_ver;
		int		input_index,top_index,rec_index,coef_index;
		

		int tid_x,tid_y,tid_z;
		tid_x = threadIdx.x;
		tid_y = threadIdx.y;
		tid_z = threadIdx.z;
		int  tid = tid_x + (tid_y +tid_z*blockDim.y)* blockDim.x;
		if(tid<8)
		{
			QcoefDc[tid] = 0;
		}
		if(tid<16)
		{
			Quant_tab[tid] = Quant_tab_uv[tid];
			DQuant_tab[tid] = Dquant_tab_uv[tid];

			//将全局存储器中的数据加载到共享存储器 
			left_pixels[tid] = 0;//left pix for uv component，first 8  for u ,the rest for v
			if(tid == 0)
			{
				Quant_Add = dev_QpData_uv->QuantAdd;
				Quant_Shift = dev_QpData_uv->QuantShift;
				Dquant_Shift = dev_QpData_uv->DQuantShift;
				first_mb = blockIdx.x*num_mb_hor* (num_mb_ver/slice_num);
			}
		}
		__syncthreads();
		for(int j =0;j< num_mb_ver/slice_num;j++)
		{
			input_index = tid_x + (tid_y*blockDim.x) + ((tid_z&7)*width_c) +(j*num_mb_hor + first_mb)*MB_TOTAL_SIZE_C/*blockIdx.x*MB_HEIGHT_C*width_c*(num_mb_ver/slice_num)*/;
			rec_index	= tid_x + (tid_y*blockDim.x) + ((tid_z&7)*width_ref_c) + j*MB_WIDTH_C*width_ref_c + blockIdx.x*MB_HEIGHT_C*width_ref_c*(num_mb_ver/slice_num) + ((tid_z>>3)*height_ref_c*width_ref_c) ;
			coef_index	= (tid&(64-1)) + ( j*num_mb_hor + first_mb +(tid_z>>3)*num_mbs)*MB_TOTAL_SIZE_C;
			top_index = (j==0) ? 0 :(tid&7)+ ((j*MB_HEIGHT_C-1)*width_ref_c) + ((tid_z & 1) * width_ref_c * height_ref_c) + blockIdx.x*MB_HEIGHT_C*width_ref_c*(num_mb_ver/slice_num) ;

			for(int k =0;k< num_mb_hor;k++)
			{
			//将一个宏块的原始数据以光栅的形式导入共享存储器
				{
					input_mb[tid]= dev_input_uv[input_index+((tid_z>>3)*width_c*height_c)];
				}
				if(tid<16)
				{
					TopAvailable[tid]  = (dev_blk_mb_info[k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
					LeftAvailable[tid] = (dev_blk_mb_info[k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;
					top_pixels[tid] = (TopAvailable[tid]) ? dev_recon_uv[top_index] : 0;
				}
				__syncthreads();
				// 帧内预测
				//if(tid_z == 0)
				{
					IntraChromaPrediction(
												top_pixels,
												left_pixels,
												input_mb,
												TopAvailable[0],
												LeftAvailable[0],
												pred_pixels,
												temp_pred,
												Sad, 
												pred_mode,
												tid_x,
												tid_y,
												tid_z
											);

				}
				__syncthreads();
				if(tid<16)
				{
					dev_blk_mb_info[tid + k*BLOCKS_PER_MB+j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].IntraChromaMode = pred_mode;
				}
				IntraChromaTransforms(
													input_mb,
													pred_pixels,
													rec_pixels,
													Quant_Add,
													Quant_Shift,
													Dquant_Shift,
													Quant_tab,
													DQuant_tab,
													Diff,
													QcoefAC,
													Temp_Coef,
													QcoefDc,
													TempDC,
													tid_x,
													tid_y,
													tid_z,
													tid
												);
					__syncthreads();
					dev_dct_coefs_uv[coef_index] = QcoefAC[tid];
					dev_recon_uv[rec_index] = rec_pixels[tid];
					if(tid < 8)
					{
						//QcoefDc[tid] = tid;
						dev_dc_coefs_uv[tid_x + k*BLOCKS_PER_MB_C + j*BLOCKS_PER_MB_C*num_mb_hor +(tid_y*BLOCKS_PER_MB_C*num_mbs) + first_mb*BLOCKS_PER_MB_C] = QcoefDc[tid];
						left_pixels[tid] = rec_pixels[tid*8+7];
						left_pixels[tid+8] = rec_pixels[tid*8+7+64];
					}
					input_index += MB_WIDTH_C;
					top_index += MB_WIDTH_C;
					rec_index += MB_WIDTH_C;
					coef_index += MB_TOTAL_SIZE_C;
				}
		}
}