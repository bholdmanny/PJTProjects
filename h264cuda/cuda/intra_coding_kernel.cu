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


inline __device__ unsigned int CalcSad4x4(unsigned char *in,unsigned char *Pred_pixels)
{
   unsigned int SAD =0;
  
   SAD += (unsigned int)abs((*in)-Pred_pixels[0]);
   SAD += (unsigned int)abs((*(in+1))-Pred_pixels[1]);
   SAD += (unsigned int)abs((*(in+2))-Pred_pixels[2]);
   SAD += (unsigned int)abs((*(in+3))-Pred_pixels[3]);

   SAD += (unsigned int)abs((*(in+16))-Pred_pixels[4]);
   SAD += (unsigned int)abs((*(in+1+16))-Pred_pixels[5]);
   SAD += (unsigned int)abs((*(in+2+16))-Pred_pixels[6]);
   SAD += (unsigned int)abs((*(in+3+16))-Pred_pixels[7]);

   SAD += (unsigned int)abs((*(in+16*2))-Pred_pixels[8]);
   SAD += (unsigned int)abs((*(in+1+16*2))-Pred_pixels[9]);
   SAD += (unsigned int)abs((*(in+2+16*2))-Pred_pixels[10]);
   SAD += (unsigned int)abs((*(in+3+16*2))-Pred_pixels[11]);

   SAD += (unsigned int)abs((*(in+16*3))-Pred_pixels[12]);
   SAD += (unsigned int)abs((*(in+1+16*3))-Pred_pixels[13]);
   SAD += (unsigned int)abs((*(in+2+16*3))-Pred_pixels[14]);
   SAD += (unsigned int)abs((*(in+3+16*3))-Pred_pixels[15]);
   return SAD;
}

inline __device__ void HadamardTransformAndQuantize(      
													short *CoefDc,
													int  &QuantAddHadamard,
													int  &QuantShift,
													short &QuantTable0,
													short *QCoefDC,
													int &tx,
													int &ty
												)
{   
		short TempCoef;
		short Sum0,Sum1,Diff0,Diff1;
		int Sign,tid;
		int QuantDC,QuantDCShift;

		tid = tx + ty * 4;
    // Vertical butterfly
		if(ty == 0 )
		{
			Sum0  = CoefDc[tx] + CoefDc[tx+12];
			Sum1  = CoefDc[tx+4] + CoefDc[tx+8];
			Diff0 = CoefDc[tx] - CoefDc[tx+12];
			Diff1 = CoefDc[tx+4] - CoefDc[tx+8];

			QCoefDC[tx] = Sum0 + Sum1;
			QCoefDC[tx+4]  = Diff0 + Diff1;
			QCoefDC[tx+8]  = Sum0 - Sum1;
			QCoefDC[tx+12]  = Diff0 - Diff1;		
		
			Sum0 = QCoefDC[tx*4] + QCoefDC[tx*4+3] ;
			Sum1 = QCoefDC[tx*4+1]  + QCoefDC[tx*4+2] ;
			Diff0 = QCoefDC[tx*4] - QCoefDC[tx*4+3] ;
			Diff1 = QCoefDC[tx*4+1]  - QCoefDC[tx*4+2] ;

			CoefDc[tid*4] = (Sum0 + Sum1) >> 1;
			CoefDc[tid*4+1]  = (Diff0 + Diff1) >> 1;
			CoefDc[tid*4+2]  = (Sum0 - Sum1) >> 1;
			CoefDc[tid*4+3]  = (Diff0 - Diff1) >> 1;
		}       
		//__syncthreads();
        QuantDC = QuantAddHadamard * 2;
        QuantDCShift = QuantShift + 1;
		
        Sign  = ((*(CoefDc+tid)) >= 0) ? 1 : -1;
        TempCoef  = (Sign >= 0) ? (*(CoefDc+tid)) : -(*(CoefDc+tid));
        TempCoef  = min (((TempCoef * QuantTable0 + QuantDC) >> QuantDCShift), CAVLC_LEVEL_LIMIT);
        QCoefDC[tid] = Sign * TempCoef;
}

inline __device__ void  InverseHadamardTransform (
												short *QCoefDC,
												short &QuantTable0,
												int &DQuantShift,
												short *TempDC,
												int &tx,
												int &ty
											)
{
		short Sum0,Sum1,Diff0,Diff1;
		/*__shared__ short temp[];*/
    // Horizontal butterfly
		if(ty == 0)
		{
			Sum0  = QCoefDC[tx*4] + QCoefDC[tx*4+2];
			Sum1  = QCoefDC[tx*4+1] + QCoefDC[tx*4+3];
			Diff0 = QCoefDC[tx*4] - QCoefDC[tx*4+2];
			Diff1 = QCoefDC[tx*4+1] - QCoefDC[tx*4+3];
        
			TempDC[tx*4] = Sum0 + Sum1;
			TempDC[tx*4+1] = Diff0 + Diff1;
			TempDC[tx*4+2] = Diff0 - Diff1;
			TempDC[tx*4+3] = Sum0 - Sum1;

			Sum0  = TempDC[tx] + TempDC[tx+8];
			Sum1  = TempDC[tx+4] + TempDC[tx+12];
			Diff0 = TempDC[tx] - TempDC[tx+8];
			Diff1 = TempDC[tx+4] - TempDC[tx+12];

			TempDC[tx] = (Sum0 + Sum1) ;
			TempDC[tx+4]  = (Diff0 + Diff1);
			TempDC[tx+8]  = (Diff0 - Diff1);
			TempDC[tx+12]  = (Sum0 - Sum1) ;

			/*TempDC[tx]  = (TempDC[tx]  * QuantTable0);
			TempDC[tx] = (TempDC[tx] << DQuantShift);
			TempDC[tx] = (TempDC[tx] + 2) >> 2;

			TempDC[tx+4] = (TempDC[tx+4] * QuantTable0);
			TempDC[tx+4] = (TempDC[tx+4] << DQuantShift);
			TempDC[tx+4] = (TempDC[tx+4] + 2) >> 2;

			TempDC[tx+8] = (TempDC[tx+8] * QuantTable0);
			TempDC[tx+8] = (TempDC[tx+8] << DQuantShift);
			TempDC[tx+8] = (TempDC[tx+8] + 2) >> 2;

			TempDC[tx+12] = (TempDC[tx+12] * QuantTable0);
			TempDC[tx+12] = (TempDC[tx+12] << DQuantShift);
			TempDC[tx+12] = (TempDC[tx+12] + 2) >> 2;*/
		}
		//__syncthreads();
		/*Sum0 = 0;
		TempDC[tx + (ty*4)] = ((TempDC[tx+(ty*4)]) * QuantTable0);
        TempDC[tx + (ty*4)] = ((TempDC[tx+(ty*4)]) << DQuantShift);
        TempDC[tx + (ty*4)] = (((TempDC[tx+(ty*4)]) + 2) >> 2);*/
		TempDC[tx + (ty*4)] = ((*(TempDC+tx +(ty*4))) * QuantTable0);
        TempDC[tx + (ty*4)] = ((*(TempDC+tx +(ty*4))) << DQuantShift);
        TempDC[tx + (ty*4)] = ((*(TempDC+tx +(ty*4)))+ 2) >> 2;
}
inline __device__ void Dct4x4TransformAndQuantize (
                                short *DiffRow,       //[i]  
                                int &QuantAdd,                       //[i]
                                int &QuantShift,                     //[i]                     
								short *Quant_tables,
								short *TempQCoef,
                                short  *QCoef,
								short  &Coef00,
								int	   &tx,
								int	   &ty
                                )
{
    
	int Sign;
	// Vertical butterfly
   //first col
	if(ty == 0)
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
		if(tx==0 )
			Coef00 = DiffRow[tx];
	}
	//QCoef[threadIdx.x+threadIdx.y*blockDim.x] = DiffRow[threadIdx.x+threadIdx.y*blockDim.x];
    Sign  = ((*(DiffRow+tx+ty*4)) >= 0) ? 1 : -1;
    DiffRow[tx+ty*4] = (Sign >= 0) ? (*(DiffRow+tx+ty*4)) : (-(*(DiffRow+tx+ty*4))) ;
	QCoef[tx+ty*4] = (short)(Sign * (((*(DiffRow+tx+ty*4)) * Quant_tables[tx+ty*4] + QuantAdd) >> QuantShift));
	/*if(threadIdx.x==0)
		Coef00 = DiffRow[0];*/
}

inline __device__ void IDct4x4AndAdd (
                                  short *Coef,
								  short *Temp_Coef,
								  unsigned char *PredRow,
                                  unsigned char *RecOutRow,
								  int			&tx,
								  int			&ty
                                  )
{
   short	Sum0,Sum1,Diff0,Diff1;
   if(ty == 0)
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
		Temp_Coef[tx + ty*4] = Coef[tx + ty * 4]+PredRow[tx + ty*16];
        RecOutRow[tx + ty*16]= (unsigned char)(Temp_Coef[tx + ty*4] < 0 ? 0 :((Temp_Coef[tx + ty*4] > 255) ? 255 : Temp_Coef[tx + ty*4]));

}
inline __device__ void Intra4x4BlockTransformAndReconstruct(
															unsigned char *InputSrcRow,
															unsigned char *Pred_row,
															int &QuantAdd,
															int &QuantShift,
															int &DquantShift,
															short *Quant_tables,
															short *DQuant_tables,
															short *DiffRow,
															short *QCoef,
															short *TempQCoef,
															unsigned char *Out_Row,
															int	&tx,
															int &ty
														)
{
	
	short Coef00;
   	DiffRow[tx + ty*4] = ((short)InputSrcRow[tx + ty*16] - (short)Pred_row[tx + ty*16]);
	
	Dct4x4TransformAndQuantize (
                                DiffRow,       //[i]  
                                QuantAdd,      //[i]
                                QuantShift,
								Quant_tables,
								TempQCoef,
                                QCoef,
								Coef00,
								tx,
								ty
								);
 //  // Dquant 
   DiffRow[threadIdx.x + threadIdx.y*blockDim.x] = (QCoef[threadIdx.x + threadIdx.y*blockDim.x] * DQuant_tables[threadIdx.x + threadIdx.y*blockDim.x]) << (int)DquantShift;
   //*(DiffRow+tx + ty*4) = ((*(QCoef+tx + ty*4)) * (*(DQuant_tables+tx + ty*4))) << (int)DquantShift;
   //I DCT
   IDct4x4AndAdd (
                    DiffRow,
					TempQCoef,
					Pred_row,
                    Out_Row,
					tx,
					ty
                   );		

   //TempQCoef[threadIdx.x + threadIdx.y*blockDim.x] = DiffRow[threadIdx.x + threadIdx.y*blockDim.x]+Pred_row[threadIdx.x + threadIdx.y*16];
   //Out_Row[threadIdx.x + threadIdx.y*16]= (unsigned char)(TempQCoef[threadIdx.x + threadIdx.y*blockDim.x] < 0 ? 0 :((TempQCoef[threadIdx.x + threadIdx.y*blockDim.x] > 255) ? 255 : TempQCoef[threadIdx.x + threadIdx.y*blockDim.x]));
	
}


inline __device__ void intra16x16_transforms (
											unsigned char *InputSrcRow,
											unsigned char *PredRow,
											unsigned char *RecOutRow,
											int &QuantAdd,
											int &QuantAddDC,
											int &QuantShift,
											int &DquantShift,
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

	DiffRow[tx + ty*4] = ((short)InputSrcRow[tx + ty*MB_WIDTH] - (short)PredRow[tx + ty*MB_WIDTH]);
	
	Dct4x4TransformAndQuantize (
                                DiffRow,       //[i]  
                                QuantAdd,                       //[i]
                                QuantShift,
								Quant_tab,
								TempQCoef,
                                QCoef,
								TempDC[tz],
								tx,
								ty
								);

	 //Hardamard 变换和量化，使用共享存储器存储16个直流系数，每个线程做同样的事情，冗余操作
	 __syncthreads();

	 if(tz == 0)
	 {
		HadamardTransformAndQuantize(
										TempDC,
										QuantAddDC,                       
										QuantShift,                     
									    Quant_tab[0],
										QCoefDC,
										tx,
										ty
										);	 
	 
		InverseHadamardTransform(
									QCoefDC,
									Dquant_tab[0],
									DquantShift,
									TempDC,
									tx,
									ty
								);
	 }
		__syncthreads();
	   //直流系数替换
		DiffRow[tx + ty*4] = ((tx==0) && (ty==0)) ? TempDC[tz]:((QCoef[tx + ty*4] * Dquant_tab[tx + ty*4]) << DquantShift);
		//QCoef[tx + ty*4] =  DiffRow[tx + ty*4];
		__syncthreads();
        //逆变换
		IDct4x4AndAdd (
                        DiffRow,
					    TempQCoef,
					    PredRow,
                        RecOutRow,
						tx,
						ty
                     );	
						
}

inline __device__ void Intra16x16Prediction(
											unsigned char *top_pixels,
											unsigned char *left_pixels,
											unsigned char *input_pixels,
											int &TopAvailable, 
											int &LeftAvailable,
											unsigned char *pred,
											int &MinSad,
											int *SAD,
											int &PredMode,
											int &tx,
											int &ty,
											int &tz,
											int &tid
											)
{
				int Pred;
    			int SubType;
    			int MinSAD;
				SubType = UNDEFINED_SUBDIV_TYPE;
    			MinSAD = LARGE_NUMBER;
				//垂直预测
				if(TopAvailable)
				{
				   //计算sad值
					pred[tid] = top_pixels[tx+ty*4];
  					SAD[tid] = (unsigned int)abs(input_pixels[tid]-top_pixels[tx+ty*4]);
					__syncthreads();

					for(int k = 128 ; k >0 ; k>>=1)
					{
						if(tid<k)
						{
							/*SAD[tid] += (tid<k) ? SAD[tid+k]:0;*/
							SAD[tid] +=  SAD[tid+k];
						}
						__syncthreads();
					}
					__syncthreads();
					
					SubType   = (SAD[0] < MinSAD) ? L_PRED_VERTICAL : SubType;
					MinSAD    = (SAD[0] < MinSAD) ? SAD[0] : MinSAD;
					

					__syncthreads();
				}
				//水平预测
				if(LeftAvailable)
				{
				   SAD[tid] = (unsigned int)abs(input_pixels[tid] - left_pixels[((tz>>2)<<4)+(tz&3)]);
				   __syncthreads();
				   for(int k = 128 ; k >0 ; k >>= 1)
				   {
					   if(tid<k)
					   {
						/*SAD[tid] += (tid<k) ? SAD[tid+k]:0;*/
							SAD[tid] +=  SAD[tid+k];
					   }
					   __syncthreads();
				   }
				   __syncthreads();
  			
				   if (SAD[0] < MinSAD)
  				  {
  						MinSAD    = SAD[0];
  			       		SubType   = L_PRED_HORIZONTAL;
						pred[tid] = left_pixels[((tz>>2)<<4)+(tz&3)];
  				  }
  				  __syncthreads();
				}
  			  
  				if (LeftAvailable && TopAvailable)
  				{
					if(tid<16)
					{
						SAD[tid] = (unsigned int)(top_pixels[tid]+left_pixels[((tid>>2)<<4)+(tid&3)]);
						for(int k = 8 ; k >0 ; k>>=1)
					    {
						   if(tid<k)
						   {
								SAD[tid] +=  SAD[tid+k];
						   }
					    }
						SAD[tid] += (tid==0) ? 16 : 0;
					}
					__syncthreads();
  			    	Pred = SAD[0] >> 5;
  				}

  				else if (LeftAvailable && !TopAvailable)
  				{
					if(tid<16)
					{
						SAD[tid] = (unsigned int)left_pixels[((tid>>2)<<4)+(tid&3)];
						for(int k = 8 ; k >0 ; k>>=1)
					    {
						   if(tid<k)
						   {
								SAD[tid] +=  SAD[tid+k];
						   }
					    }
						SAD[tid] += (tid==0) ? 8 : 0;
					}
					__syncthreads();
  			    	Pred = (SAD[0] >> 4);
  				}

  				else if (TopAvailable && !LeftAvailable)
  				{

					if(tid<16)
					{
						SAD[tid] = (unsigned int)top_pixels[tid];
						for(int k = 8 ; k >0 ; k>>=1)
					    {
						   if(tid<k)
						   {
								SAD[tid] +=  SAD[tid+k];
						   }
					    }
						SAD[tid] += (tid==0) ? 8 : 0;
					}
					__syncthreads();
  			    	Pred = SAD[0] >> 4;
  				}

  				else // Non of the neighbors are available
  				{
  				    Pred = 128;
  				}
				__syncthreads();

  				SAD[tid] = (unsigned int)abs(input_pixels[tid]-Pred);
				__syncthreads();

				for(int k = 128 ; k >0 ; k>>=1)
				{
					if(tid<k)
					{
						/*SAD[tid] += (tid<k) ? SAD[tid+k]:0;*/
						SAD[tid] +=  SAD[tid+k];
					}
					__syncthreads();
				}

				__syncthreads();	
	
				if (SAD[0] < MinSAD)
  				{
  					MinSAD    = SAD[0];
  					SubType   = 2;
					pred[tid] = Pred;
  				}
			
				if(tid == 0)
				{
					MinSad = MinSAD;
					PredMode = SubType;
				}	
}
//一个4x4块的帧内预测，总共有9种预测方式，只实现7种，其中第3和7不实现
inline __device__ void Intra4x4BlockPrediction (
												unsigned char *Top_pix,         //  neighboring pixels from top neighbor block
												unsigned char *Left_pix,        //  neighboring pixels from left neighbor block
												unsigned char &Top_left_pix,        // Only one byte (byte 3) that is correspond to TopLeft pixel of current block is useful.
												int &prefered_pred_mode,
												int &PredModePenalty,
												int &TopAvailable,
												int &LeftAvailable,
												unsigned char *InputSrcRow,
												int &Sad,
												int &Pred_mode,
												unsigned char *pred_pixels,
												unsigned char *Pred_pixels,
												unsigned char *Pred_temp
											)
{
	int Penalty,Sad_temp,MinSad,PredFlag,PredMode;
	int TopSum,TopAvg,LeftSum,LeftAvg,AllSum,AllAvg,Avg;
	int BothAvailable = (TopAvailable && LeftAvailable);
	MinSad = LARGE_NUMBER;
	PredMode = S_PRED_INVALID_MODE;
	Sad_temp = 0;

	// 进行垂直预测
	if(TopAvailable)
	{
		Penalty = (prefered_pred_mode == S_PRED_VERT) ? 0 : PredModePenalty;
		Pred_pixels[0] = Pred_pixels[4] = Pred_pixels[8] = Pred_pixels[12] =Top_pix[0];
		Pred_pixels[1] = Pred_pixels[5] = Pred_pixels[9] = Pred_pixels[13] = Top_pix[1];
		Pred_pixels[2] = Pred_pixels[6] = Pred_pixels[10] = Pred_pixels[14] = Top_pix[2];
		Pred_pixels[3] = Pred_pixels[7] = Pred_pixels[11] = Pred_pixels[15] = Top_pix[3];

		Sad_temp = CalcSad4x4(
								InputSrcRow,
								Pred_pixels
							 );
		Sad_temp += Penalty;
		PredFlag = (TopAvailable) && (Sad_temp < MinSad);
		MinSad   = (PredFlag ? Sad_temp : MinSad);
		PredMode = (PredFlag ? S_PRED_VERT : PredMode); 
	}
	Sad_temp = 0;
   //水平预测
	if(LeftAvailable)
	{
		Penalty = (prefered_pred_mode == S_PRED_HOR) ? 0 : PredModePenalty;

		Pred_temp[0] = Pred_temp[1] = Pred_temp[2] = Pred_temp[3] =Left_pix[0];
		Pred_temp[4] = Pred_temp[5] = Pred_temp[6] = Pred_temp[7] = Left_pix[1];
		Pred_temp[8] = Pred_temp[9] = Pred_temp[10] = Pred_temp[11] = Left_pix[2];
		Pred_temp[12] = Pred_temp[13] = Pred_temp[14] = Pred_temp[15] = Left_pix[3];
		Sad_temp = CalcSad4x4(
									InputSrcRow,
									Pred_temp
								);
		Sad_temp += Penalty;
		PredFlag = (LeftAvailable && (Sad_temp < MinSad));
		MinSad   = (PredFlag ? Sad_temp : MinSad);
		if(PredFlag)
		{
			for(int i = 0 ;i <16 ; i++ )
				Pred_pixels[i] = Pred_temp[i];
			PredMode =S_PRED_HOR ;   
		}
		
	}
	Sad_temp = 0;
   //DC 预测 总会进行的操作
	Avg = 0 ;

	TopSum = Top_pix[0] + Top_pix[1]  + Top_pix[2]  + Top_pix[3]  + 2;     // Add up 4 top neighbor pixels + 2 for rounding
	TopAvg =  (TopSum) >> 2;
    
	LeftSum = Left_pix[0] + Left_pix[1]  + Left_pix[2]  + Left_pix[3] + 2;
	LeftAvg = (LeftSum) >> 2;     // Add up 4 left neighbor pixels + 2 for rounding

	AllSum  = TopSum + LeftSum;
	AllAvg  = (AllSum) >> 3;
            
	Avg = (BothAvailable ? AllAvg : Avg);
	Avg = ((TopAvailable && (!LeftAvailable)) ? TopAvg : Avg);
	Avg = (((!TopAvailable) && LeftAvailable)  ? LeftAvg : Avg);
	Avg = (((!TopAvailable) && (!LeftAvailable)) ? 128 : Avg);

	for(int i = 0 ;i <16 ; i++ )
		Pred_temp[i] = (unsigned char) Avg;


	Sad_temp = CalcSad4x4(
								InputSrcRow,
								Pred_temp
							);

   Penalty = (prefered_pred_mode == S_PRED_DC) ? 0 : PredModePenalty;
   Sad_temp += Penalty;
   PredFlag = (Sad_temp < MinSad) ? 1 : 0;
   MinSad   = (PredFlag ? Sad_temp : MinSad);
   if(PredFlag)
   {
		for(int i = 0 ;i <16 ; i++ )
			Pred_pixels[i] = Pred_temp[i];
		PredMode = S_PRED_DC; // DC mode
   }
   
  
   Sad_temp = 0;
   //右下对角预测
   if(BothAvailable)
   {
		Pred_temp[0] = Pred_temp[5] = Pred_temp[10] = Pred_temp[15]  = (Left_pix[0] + 2*Top_left_pix + Top_pix[0] +2) >> 2;
		Pred_temp[4] = Pred_temp[9] = Pred_temp[14] = (Left_pix[1] +2*Left_pix[0] + Top_left_pix + 2) >> 2;
		Pred_temp[8] = Pred_temp[13] = (Left_pix[2] + 2*Left_pix[1] + Left_pix[0] + 2 ) >> 2;
		Pred_temp[12] = (Left_pix[3] + 2*Left_pix[2] +Left_pix[1] + 2 ) >> 2;

		Pred_temp[1] = Pred_temp[6] = Pred_temp[11] = ((Top_left_pix + 2 * Top_pix[0] + Top_pix[1] + 2) >> 2);
		Pred_temp[2] = Pred_temp[7] = (Top_pix[0] + 2*Top_pix[1] + Top_pix[2] + 2) >> 2;
		Pred_temp[3] = (Top_pix[1] + 2*Top_pix[2] + Top_pix[3] + 2) >> 2;
   
		Penalty = (prefered_pred_mode == S_PRED_DIAG_DOWN_RIGHT) ? 0 : PredModePenalty;

		Sad_temp = CalcSad4x4(
									InputSrcRow,
									Pred_temp
								);
		Sad_temp += Penalty;
		PredFlag = (BothAvailable && Sad_temp < MinSad);
		MinSad   = (PredFlag ? Sad_temp: MinSad);
		if(PredFlag)
		{
			for(int i = 0 ;i <16 ; i++ )
				Pred_pixels[i] = Pred_temp[i];
			PredMode = S_PRED_DIAG_DOWN_RIGHT ;
		 }
		    
   }
   //垂直向右预测

   if(BothAvailable)
   {
		Pred_temp[0] = Pred_temp[9] = (Top_left_pix + Top_pix[0] +1) >> 1;
		Pred_temp[1] = Pred_temp[10] = (Top_pix[0] + Top_pix[1] + 1) >> 1;
		Pred_temp[2] = Pred_temp[11] = (Top_pix[1] + Top_pix[2] + 1) >> 1;
		Pred_temp[3] = (Top_pix[2] + Top_pix[3] + 1) >> 1;
   
		Pred_temp[4] = Pred_temp[13] =(Left_pix[0] + 2*Top_left_pix + Top_pix[0] +2) >> 2;
		Pred_temp[5] = Pred_temp[14] =(Top_left_pix + 2*Top_pix[0] + Top_pix[1] + 2) >> 2;
		Pred_temp[6] = Pred_temp[15] =(Top_pix[0] + 2*Top_pix[1] +Top_pix[2] + 2) >> 2;
		Pred_temp[7] = (Top_pix[1] + 2*Top_pix[2] + Top_pix[3] + 2) >> 2;
   
		Pred_temp[8] = (Top_left_pix + 2*Left_pix[0] + Left_pix[1] + 2) >> 2;
		Pred_temp[12] = (Left_pix[0] + 2*Left_pix[1] + Left_pix[2] + 2) >> 2;
   
		Penalty = (prefered_pred_mode == S_PRED_VERT_RIGHT) ? 0 : PredModePenalty;
		Sad_temp = 0;
		Sad_temp = CalcSad4x4(
									InputSrcRow,
									Pred_temp
								);
		Sad_temp += Penalty;
		PredFlag = (BothAvailable && Sad_temp < MinSad);
		MinSad   = (PredFlag ? Sad_temp : MinSad);
		if(PredFlag)
		{
			for(int i = 0 ;i <16 ; i++ )
				Pred_pixels[i] = Pred_temp[i];
			MinSad   = Sad_temp;
			PredMode = S_PRED_VERT_RIGHT; 
		 }
		
   }

   //水平向下预测
   if(BothAvailable)
   {
		Pred_temp[0] = Pred_temp[6] = (Top_left_pix + Left_pix[0] +1) >> 1;
		Pred_temp[1] = Pred_temp[7] = (Left_pix[0] + 2*Top_left_pix + Top_pix[0] +2) >> 2;
		Pred_temp[2] = (Top_left_pix + 2*Top_pix[0] + Top_pix[1] + 2) >> 2;
		Pred_temp[3] = (Top_pix[0] + 2*Top_pix[1] +Top_pix[2] + 2) >> 2;
   
		Pred_temp[4] = Pred_temp[10] = (Left_pix[0] + Left_pix[1] +1) >> 1;
		Pred_temp[5] = Pred_temp[11] = (Top_left_pix + 2*Left_pix[0] + Left_pix[1] + 2) >> 2;
   
		Pred_temp[8] = Pred_temp[14] = ( Left_pix[1] + Left_pix[2] + 1) >> 1;
		Pred_temp[9] = Pred_temp[15] = ( Left_pix[0] + 2*Left_pix[1] + Left_pix[2]+ 2) >> 2;
		Pred_temp[12] = (Left_pix[2] + Left_pix[3] + 1) >> 1;
		Pred_temp[13] = (Left_pix[1] + 2*Left_pix[2] +Left_pix[3] + 2) >> 2;
  
		Penalty = (prefered_pred_mode == S_PRED_HOR_DOWN) ? 0 : PredModePenalty;
		Sad_temp = 0;
		Sad_temp = CalcSad4x4(
									InputSrcRow,
									Pred_temp
								);
		Sad_temp += Penalty;
		PredFlag = (BothAvailable && Sad_temp < MinSad);
		if(PredFlag)
		{
			for(int i = 0 ;i <16 ; i++ )
				Pred_pixels[i] = Pred_temp[i];
			MinSad   = Sad_temp;
			PredMode = S_PRED_HOR_DOWN; 
		 }
		
   }
   //水平向上预测
   if(LeftAvailable)
   {
		Pred_temp[0] = ( Left_pix[0] + Left_pix[1] + 1) >> 1;
		Pred_temp[1] = (Left_pix[0] + 2*Left_pix[1]+ Left_pix[2] +2) >> 2;
		Pred_temp[2] = Pred_temp[4] = (Left_pix[1]+ Left_pix[2] +1) >> 1;
		Pred_temp[3] = Pred_temp[5] = (Left_pix[1] + 2*Left_pix[2] +Left_pix[3] + 2) >> 2;
   
		Pred_temp[6] = Pred_temp[8] = (Left_pix[2] + Left_pix[3] +1) >> 1;
		Pred_temp[7] = Pred_temp[9] = ( Left_pix[2] + 3*Left_pix[3] + 2) >> 2;
		Pred_temp[10] = Pred_temp[11] = Pred_temp[12] = Pred_temp[13] = Pred_temp[14] = Pred_temp[15] = Left_pix[3];   
   
		Penalty = (prefered_pred_mode == S_PRED_HOR_UP) ? 0 : PredModePenalty;
		Sad_temp = 0;
		Sad_temp = CalcSad4x4(
									InputSrcRow,
									Pred_temp
								);

		Sad_temp += Penalty;
		PredFlag = (LeftAvailable && Sad_temp < MinSad);

		if(PredFlag)
		{
			for(int i = 0 ;i <16 ; i++ )
				Pred_pixels[i] = Pred_temp[i];
			MinSad   = Sad_temp;
			PredMode = S_PRED_HOR_UP; 
		 }
   }
   Sad = MinSad;
   Pred_mode = PredMode;
   for(int i = 0 ;i <4 ; i++ )
   {
		pred_pixels[0+i*16] = Pred_pixels[0+i*4];
		pred_pixels[1+i*16] = Pred_pixels[1+i*4];
		pred_pixels[2+i*16] = Pred_pixels[2+i*4];
		pred_pixels[3+i*16] = Pred_pixels[3+i*4];
   }
    
}

inline __device__ void Intra4x4Prediction_transform(	unsigned char *topblk_pixels,
				      									unsigned char *leftblk_pixels,
														unsigned char &top_left_pix,
														unsigned char *Input_mb,
														int &Top_blk_Available, 
														int &Left_blk_Available,
														int &dc_only_pred_flag,
														int &Top_mb_type,
														int *Top_blk_type,
														int &Left_mb_type,
														int *Left_blk_type,
														int &pred_penalty,
														int &quant_add,
														int &quant_shift,
														int &dquant_shift,
														short *Quant_tables,
														short *DQuant_tables,
														unsigned char *Rec_row,
														unsigned char  *Pred_row,
														unsigned char  *pred_pixels,
														unsigned char  *pred_temp_pix,
														short *Qcoef_AC,
														short *Diff,
														short *Temp_Coef,
														int &MinSad,
														int *Sad,
														int *PreferedPredMode,
														int &PredMode,
														int &tx,
														int &ty,
														int &tz,
														int &tid
														)

{
			int left_blk_pred_mode;
			int top_blk_pred_mode;
			/*uchar4 	Left_pix,Top_pix;*/
			unsigned char Top_left_pix;
			int Pred,temp_sad,temp_pred;
			int is_left_blk,is_top_blk;
    		int prefered_pred_mode;
			int first_blk_index,is_active_blk,active_blk;
    
			int num_stages,current_stage;
			//int is_valid = 0;   //表明哪一步中哪个线程的结果是合法的
			//temp=0;
			num_stages =7;
			current_stage=0;
  		  
			is_left_blk = ((tz&3) == 0) ? 1 : 0;

			is_top_blk = ((tz>>2) == 0) ? 1 : 0;

			left_blk_pred_mode = (is_left_blk) ? Left_blk_type[(tz + 3 ) & 0xf] : S_PRED_INVALID_MODE;
			left_blk_pred_mode = (dc_only_pred_flag||(Left_mb_type!=INTRA_SMALL_BLOCKS_MB_TYPE)) ? S_PRED_DC : left_blk_pred_mode;
  		 
			top_blk_pred_mode = (is_top_blk) ? Top_blk_type[(tz + 12) & 0xf] : S_PRED_INVALID_MODE;
			top_blk_pred_mode = (dc_only_pred_flag||(Top_mb_type!=INTRA_SMALL_BLOCKS_MB_TYPE)) ? S_PRED_DC : top_blk_pred_mode;

			//__syncthreads();  //同步处理
			// the 7 stages in processing these blocks are 
			// Stage 1: block  0
			// Stage 2: blocks 1, 4
			// Stage 3: blocks 2, 5, 8
			// Stage 4: blocks 3, 6, 9, 12
			// Stage 5: blocks 7, 10, 13
			// Stage 6: blocks 11, 14
			// Stage 7: block  15
	

    	while(num_stages > 0)
		{
			first_blk_index = (current_stage < 4 ) ? current_stage : (((current_stage - 3) << 2) + 3);  //每一阶段的第一个块的索引
			is_active_blk   = (tz == first_blk_index) ? 1 : 0;
        	active_blk      = first_blk_index + 3;  //下一行的活动块
        	is_active_blk   = is_active_blk | ((tz == active_blk) && (active_blk <= (current_stage << 2)));
        	active_blk      = first_blk_index + 6;  
			is_active_blk   = is_active_blk | ((tz == active_blk) && (active_blk <= (current_stage << 2)));
        	active_blk      = first_blk_index + 9;     
        	is_active_blk   = is_active_blk | ((tz == active_blk) && (active_blk <= (current_stage << 2)));
 
			if(is_active_blk)
			{
				Top_left_pix = (is_top_blk ? (is_left_blk ? top_left_pix : topblk_pixels[((tz+11)<<2)+3]) : (is_left_blk ? leftblk_pixels[((tz-1)<<2)+3] : leftblk_pixels[((tz-5)<<2)+3]));
				left_blk_pred_mode = dc_only_pred_flag ? S_PRED_DC : (is_left_blk ? left_blk_pred_mode :Left_blk_type[tz-1]);
				top_blk_pred_mode  = dc_only_pred_flag ? S_PRED_DC : (is_top_blk ? top_blk_pred_mode : Top_blk_type[tz-4]);
				//prefered_pred_mode = (top_blk_pred_mode < left_blk_pred_mode) ? top_blk_pred_mode : left_blk_pred_mode;
				if(tx == 0 && ty==0)
				{
					PreferedPredMode[tz]= (top_blk_pred_mode < left_blk_pred_mode) ? top_blk_pred_mode : left_blk_pred_mode;
					Intra4x4BlockPrediction (
        										(is_top_blk ? topblk_pixels+((tz+12)<<2) : topblk_pixels+((tz-4)<<2)),         //  neighboring pixels from top neighbor block
        										(is_left_blk ? leftblk_pixels+((tz + 3)<<2) : leftblk_pixels+((tz-1)<<2) ),        //  neighboring pixels from left neighbor block
        										Top_left_pix,        // Only one byte (byte 3) that is correspond to TopLeft pixel of current block is useful.
        										PreferedPredMode[tz],
        										pred_penalty,
        										Top_blk_Available,
        										Left_blk_Available,
        										Input_mb,
        										Sad[tz],
        										temp_pred,
        										Pred_row,
												pred_pixels,
												pred_temp_pix
            									);
				}

				Intra4x4BlockTransformAndReconstruct(
														Input_mb,
														Pred_row,
														quant_add,
            											quant_shift,
														dquant_shift,
														Quant_tables,
														DQuant_tables,
														Diff,
														Qcoef_AC,
														Temp_Coef,
														Rec_row,
														tx,
														ty
														);
				
				//leftblk_pixels[(tz<<2)]=Rec_row[0];
				if(ty==0)
				{
					//保存重建块最右边一列像素及其预测方式
					leftblk_pixels[(tz<<2)+tx]=Rec_row[tx*16+3];
					//保存重建块最下边一行像素及其预测方式
					topblk_pixels [(tz<<2)+tx]=Rec_row[tx+3*16];
				}
				if(tx == 0 && ty==0)
				{
					Left_blk_type[tz] =temp_pred ;
					Top_blk_type[tz] = temp_pred ;
					PredMode = temp_pred;
				}
			}
        	
        	current_stage = (current_stage + 1);
        	num_stages = (num_stages - 1);   // Needed...as loop_count would have decremented automatically!
			__syncthreads();  //同步处理
	}
    temp_sad = 0;
	if(tid == 0)
	{
		for(int i = 0;i <16 ;i ++)
			temp_sad += Sad[i];
		MinSad = (temp_sad + 6*pred_penalty);
	}
}


//该kernel完成对I帧亮度分量的预测和残差编码过程，每个线程块一遍循环处理一个MB，block(4,4,16),grid(slice_num,1,1)
__global__ void iframe_luma_residual_coding (
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
	__shared__ 	unsigned char 	top_pixels[16*4];
	__shared__  unsigned char 	left_pixels[16*4];
	__shared__	unsigned char 	input_mb[256];
	__shared__	int	      	left_blk_type[16];
	__shared__	int	      	top_blk_type[16];
	__shared__	int	      	Sad[256];
	__shared__  int         PreferedPredMode[16];
	__shared__ short        Quant_tables[16];
	__shared__ short        DQuant_tables[16];
	__shared__ short	QcoefDc[16];
	__shared__ short    TempDC[16];
	__shared__ unsigned char rec_pixels[256];
	__shared__ unsigned char pred_pixels_4x4[256];
	__shared__ unsigned char pred_pixels_16x16[256];
	__shared__ short	QcoefAC[256];

	__shared__ unsigned char  Pred_row[256];
	__shared__ unsigned char  pred_temp_pix[256];
	__shared__ short Diff[256];
	__shared__ short Temp_Coef[256];
	 unsigned char top_left_pix[16];

     __shared__ int  Quant_Add[16];
	 __shared__ int  QuantAddDC[16];
     __shared__ int  Quant_Shift[16];
     __shared__ int  Dquant_Shift[16];
     __shared__ int  pred_penalty[16];
	 __shared__ int ChooseIntra_16;
	 __shared__ int Type;
	 __shared__ int SubType[16];
	 __shared__ int MinSad;
	 __shared__ int intra_16_sad;
	 __shared__ int	intra_4_sad;
	 __shared__ int	intra_16_pred_mode;
	 __shared__ int intra_4_pred_mode[16];
	 __shared__ int TopAvailable[16];
	 __shared__ int	 LeftAvailable[16];
	 __shared__ int	 top_blk_available[16];
	 __shared__ int	 left_blk_available[16];
	 __shared__ int	 dcOnlyPredictionFlag[16];
	 __shared__ int  Left_mb_type[16];
	 __shared__ int	 Top_mb_type[16];
	 int i;
	 int Loc;
	 int tid_x,tid_y,tid_z;
	 tid_x = threadIdx.x;
	 tid_y = threadIdx.y;
	 tid_z = threadIdx.z;
	 int first_mb;

	 int  tid = tid_x + (tid_y +tid_z*blockDim.y)* blockDim.x;
	 //将全局存储器中的数据加载到共享存储器
	 if(tid<16)
	 {
		left_pixels[tid] = left_pixels[tid+16] = left_pixels[tid+32] = left_pixels[tid+48] = 0;
		Quant_tables[tid] = quant_tab[tid];
		DQuant_tables[tid] = d_quant_tab[tid];  
		left_blk_type [tid] = 0;
	 
		 top_left_pix[tid] = 0;
		 Left_mb_type[tid] = 0;
		 Quant_Add[tid] = pQpData->QuantAdd;
		 QuantAddDC[tid] = pQpData->QuantAddDC;
		 Quant_Shift[tid] = pQpData->QuantShift;
		 Dquant_Shift[tid] = pQpData->DQuantShift;
		 pred_penalty[tid] = pQpData->PredPenalty;
	}
	 /*top_left_pix = 0;*/
	 first_mb = blockIdx.x*num_mb_hor* (num_mb_ver/slice_num);
	for(int j =0;j< num_mb_ver/slice_num;j++)
	{
		for(int k =0;k< num_mb_hor;k++)
		{
			if(tid<16)
			{
				Loc = dev_blk_mb_info[tid + k*BLOCKS_PER_MB + j*num_mb_hor*BLOCKS_PER_MB  + first_mb*BLOCKS_PER_MB].Loc;
				TopAvailable[tid]  = (Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
				LeftAvailable[tid] = (Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;

				left_blk_available[tid] = (Loc & LOC_BLK_LEFT_EDGE) ? 0 : 1;
				top_blk_available[tid] = (Loc & LOC_BLK_TOP_EDGE) ? 0 : 1;

				dcOnlyPredictionFlag[tid] = (((Loc & LOC_BLK_LEFT_OR_TOP_EDGE) != 0) );

				top_blk_type[tid] = dev_blk_mb_info[ (k+(j-1)*num_mb_hor)*BLOCKS_PER_MB+12 + tid_x + first_mb*BLOCKS_PER_MB ].SubType;
				Top_mb_type [tid] = dev_blk_mb_info[(k+(j-1)*num_mb_hor)*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Type;
				top_pixels[tid] = (TopAvailable[tid]) ? dev_recon[tid + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)] : 0;
				top_pixels[tid+16] = top_pixels[tid+32] = top_pixels[tid+48] = top_pixels[tid];
				top_left_pix[tid] = (TopAvailable[tid]) ? dev_recon[ k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)-1] : 0;
			}
			//将一个宏块的原始数据以光栅的形式导入共享存储器
			input_mb[tid]= dev_input[tid_x+ tid_y*blockDim.x + tid_z*width + k*MB_WIDTH + j*num_mb_hor*MB_TOTAL_SIZE + blockIdx.x*MB_HEIGHT*width*(num_mb_ver/slice_num)];
			__syncthreads();
			// 16x16帧内预测
			Intra16x16Prediction(
									top_pixels+48,
									left_pixels+12,
									input_mb/*+(tid_z<<4)*/,
									TopAvailable[0],
									LeftAvailable[0],
									pred_pixels_16x16,
									intra_16_sad,
									Sad, 
									intra_16_pred_mode,
									tid_x,
									tid_y,
									tid_z,
									tid
								);
			__syncthreads();

			Intra4x4Prediction_transform(
											top_pixels,
											left_pixels,
											top_left_pix[tid_z],
											input_mb+((tid_z&3)<<2)+((tid_z>>2)<<6),
											top_blk_available[tid_z], 
											left_blk_available[tid_z],
											dcOnlyPredictionFlag[tid_z],
											Top_mb_type[tid_z],
											top_blk_type,
											Left_mb_type[tid_z],
											left_blk_type,
											pred_penalty[tid_z],
											Quant_Add[tid_z],
											Quant_Shift[tid_z],
											Dquant_Shift[tid_z],
											Quant_tables,
											DQuant_tables,
											rec_pixels+((tid_z&3)<<2)+((tid_z>>2)<<6),
											pred_pixels_4x4+((tid_z&3)<<2)+((tid_z>>2)<<6),
											Pred_row+(tid_z<<4),
											pred_temp_pix+(tid_z<<4),
											QcoefAC+(tid_z*16),
											Diff+(tid_z*16),
											Temp_Coef+(tid_z*16),
											intra_4_sad,
											Sad,
											PreferedPredMode,
											intra_4_pred_mode[tid_z],
											tid_x,
											tid_y,
											tid_z,
											tid
										);

			__syncthreads();
			// //intra_4_sad = intra_4_sad + 6*pred_penalty;
			if(tid==0)
			{
				ChooseIntra_16 = (intra_16_sad < intra_4_sad) ? 1 : 0;
				Type    = (ChooseIntra_16 == 0) ? INTRA_SMALL_BLOCKS_MB_TYPE : INTRA_LARGE_BLOCKS_MB_TYPE;
				MinSad  = (ChooseIntra_16 == 0) ? intra_4_sad : intra_16_sad;
			}
			if(tid_z ==0)
			{
				SubType[tid_x + tid_y * 4] = (ChooseIntra_16 == 0) ? intra_4_pred_mode[tid_x + tid_y * 4] : intra_16_pred_mode;
				QcoefDc[tid_x + tid_y * 4]= 0;
				PreferedPredMode[tid] = (ChooseIntra_16 == 0) ? PreferedPredMode[tid] : 0;
			}
			__syncthreads();
			
			if (ChooseIntra_16 > 0)
			{
				 intra16x16_transforms (
											input_mb+((tid_z&3)<<2)+((tid_z>>2)<<6),
											pred_pixels_16x16+((tid_z&3)<<2)+((tid_z>>2)<<6),
											rec_pixels+((tid_z&3)<<2)+((tid_z>>2)<<6),
											Quant_Add[tid_z],
											QuantAddDC[tid_z],
											Quant_Shift[tid_z],
											Dquant_Shift[tid_z],
											Quant_tables,
											DQuant_tables,
											Diff+tid_z*16,
											QcoefAC+tid_z*16,
											Temp_Coef+tid_z*16,
											QcoefDc,
											TempDC,
											tid_x,
											tid_y,
											tid_z,
											tid
										);
										
			 }
			__syncthreads();
			////写回结果，交流和直流系数等
			dev_dct_coefs[tid + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE] = /*(short)pred_pixels_4x4[tid]*/QcoefAC[tid];
			dev_recon[tid_x+tid_y*4+tid_z*out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)] = rec_pixels[tid_x+tid_y*4+tid_z*16];

			//// write mbinfo for next mb row
			if(tid<16)
			{
				dev_dc_coefs [tid + k*16 + j*num_mb_hor*BLK_SIZE + first_mb*BLK_SIZE] =	/*(short)intra_16_pred_mode*/QcoefDc[tid];
				left_pixels[tid_x+tid_y*16+12] = rec_pixels[tid*16+15];

				dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Type = Type;
				dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].SubType = SubType[tid];
				dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MinSAD = MinSad;
				dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Pred_mode = PreferedPredMode[tid];
				Left_mb_type[tid] = Type;
//				top_left_pix = Temp_top_left;
			}
			//top_left_pix = Temp_top_left;
			__syncthreads();
		}
	}
}

__global__ void pframe_intra_resudial_coding_luma(
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
	__shared__ 	unsigned char 	top_pixels[16];
	__shared__  unsigned char 	left_pixels[16*4];
	__shared__	unsigned char 	input_mb[256];
	__shared__	int	      	Sad[256];
	__shared__ short        Quant_tables[16];
	__shared__ short        DQuant_tables[16];
	__shared__ short	QcoefDc[16];
	__shared__ short    TempDC[16];
	__shared__ unsigned char rec_pixels[256];
	__shared__ unsigned char pred_pixels_16x16[256];
	__shared__ short	QcoefAC[256];

	__shared__ short Diff[256];
	__shared__ short Temp_Coef[256];

     __shared__ int  Quant_Add[16];
	 __shared__ int  QuantAddDC[16];
     __shared__ int  Quant_Shift[16];
     __shared__ int  Dquant_Shift[16];
	 __shared__ int ChooseIntra;
	 __shared__ int intra_16_sad;
	 __shared__ int	intra_16_pred_mode;
	 __shared__ int TopAvailable;
	 __shared__ int	 LeftAvailable;
	 __shared__ int  Inter_SAD;
	 int i;
	// S_BLK_MB_INFO BlkMBInfoIntra16;
	 int tid_x,tid_y,tid_z;
	 tid_x = threadIdx.x;
	 tid_y = threadIdx.y;
	 tid_z = threadIdx.z;
	 int first_mb;

	 int  tid = tid_x + (tid_y +tid_z*blockDim.y)* blockDim.x;
	 //将全局存储器中的数据加载到共享存储器
	 if(tid<16)
	 {
		left_pixels[tid] = left_pixels[tid+16] = left_pixels[tid+32] = left_pixels[tid+48] = 0;
		Quant_tables[tid] = quant_tab[tid];
		DQuant_tables[tid] = d_quant_tab[tid];  

		Quant_Add[tid] = pQpData->QuantAdd;
		QuantAddDC[tid] = pQpData->QuantAddDC;
		Quant_Shift[tid] = pQpData->QuantShift;
		Dquant_Shift[tid] = pQpData->DQuantShift;

	}

	 first_mb = blockIdx.x*num_mb_hor* (num_mb_ver/slice_num);

	for(int j =0;j< num_mb_ver/slice_num;j++)
	{
		for(int k =0;k< num_mb_hor;k++)
		{
			if(tid==0)
			{
				Inter_SAD = dev_blk_mb_info[tid + k*BLOCKS_PER_MB + j*num_mb_hor*BLOCKS_PER_MB  + first_mb*BLOCKS_PER_MB].MinSAD;
				TopAvailable  = (dev_blk_mb_info[tid + k*BLOCKS_PER_MB + j*num_mb_hor*BLOCKS_PER_MB  + first_mb*BLOCKS_PER_MB].Loc & LOC_MB_TOP_EDGE) ? 0 : 1;
				LeftAvailable = (dev_blk_mb_info[tid + k*BLOCKS_PER_MB + j*num_mb_hor*BLOCKS_PER_MB  + first_mb*BLOCKS_PER_MB].Loc & LOC_MB_LEFT_EDGE) ? 0 : 1;
				
				intra_16_sad = 65535;
			}
			__syncthreads();
			//如果当前宏块的帧间预测SAD大于平均值加上一个系数，则进行16*16预测
			if(Inter_SAD>(avg_sad + intra_lambda_fact))
			{
				//将一个宏块的原始数据以光栅的形式导入共享存储器
				input_mb[tid]= dev_input[tid_x+ tid_y*blockDim.x + tid_z*width + k*MB_WIDTH + j*num_mb_hor*MB_TOTAL_SIZE + blockIdx.x*MB_HEIGHT*width*(num_mb_ver/slice_num)];
				if(tid < 16)
				{
					top_pixels[tid] = (TopAvailable) ? dev_recon[tid + k*MB_WIDTH + (j*MB_HEIGHT-1)*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)] : 0;
				}
				__syncthreads();
			// 16x16帧内预测
				Intra16x16Prediction(
									top_pixels,
									left_pixels+12,
									input_mb,
									TopAvailable,
									LeftAvailable,
									pred_pixels_16x16,
									intra_16_sad,
									Sad, 
									intra_16_pred_mode,
									tid_x,
									tid_y,
									tid_z,
									tid
								);
		
			}
			__syncthreads();
			if(tid == 0)
			{
				ChooseIntra = (intra_16_sad < Inter_SAD) ? 1 : 0;
			}
			__syncthreads();

			if(ChooseIntra)
			{

				intra16x16_transforms (
											input_mb+((tid_z&3)<<2)+((tid_z>>2)<<6),
											pred_pixels_16x16+((tid_z&3)<<2)+((tid_z>>2)<<6),
											rec_pixels+((tid_z&3)<<2)+((tid_z>>2)<<6),
											Quant_Add[tid_z],
											QuantAddDC[tid_z],
											Quant_Shift[tid_z],
											Dquant_Shift[tid_z],
											Quant_tables,
											DQuant_tables,
											Diff+tid_z*16,
											QcoefAC+tid_z*16,
											Temp_Coef+tid_z*16,
											QcoefDc,
											TempDC,
											tid_x,
											tid_y,
											tid_z,
											tid
										);
				__syncthreads();
				////写回结果，交流和直流系数等
				dev_dct_coefs[tid + k*MB_TOTAL_SIZE + j*num_mb_hor*MB_TOTAL_SIZE + first_mb*MB_TOTAL_SIZE] = /*(short)pred_pixels_4x4[tid]*/QcoefAC[tid];
				dev_recon[tid_x+tid_y*4+tid_z*out_stride + k*16 + j*MB_HEIGHT*out_stride + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)] = rec_pixels[tid_x+tid_y*4+tid_z*16];
	
				if(tid<16)
				{
					dev_dc_coefs [tid + k*16 + j*num_mb_hor*BLK_SIZE + first_mb*BLK_SIZE] =	/*(short)intra_16_pred_mode*/QcoefDc[tid];
					left_pixels[tid_x+tid_y*16+12] = rec_pixels[tid*16+15];
					
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MV.x = 0;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MV.y = 0;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].RefFrameIdx = -1;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].Type = INTRA_LARGE_BLOCKS_MB_TYPE;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].SubType = intra_16_pred_mode;
					dev_blk_mb_info[tid + k*16 + j*num_mb_hor*BLOCKS_PER_MB + first_mb*BLOCKS_PER_MB].MinSAD = intra_16_sad;
				}
			}
			else
			{
				if(tid<16)
				{
					left_pixels[tid_x+tid_y*16+12] = dev_recon[k*16 + j*MB_HEIGHT*out_stride + tid*out_stride + 15 + blockIdx.x*MB_HEIGHT*out_stride*(num_mb_ver/slice_num)];
				}
			}
			__syncthreads();
		}
	}
}