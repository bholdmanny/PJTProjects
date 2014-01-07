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


#include "chroma_transform_quantize_kernel.cu"

inline __device__ unsigned int CalcSadChroma_cu(unsigned char *in,uchar4 &Pred_row0,uchar4 &Pred_row1,uchar4 &Pred_row2,uchar4 &Pred_row3)
{
   unsigned int SAD =0;
  
   SAD += (unsigned int)abs((*in)-Pred_row0.x);
   SAD += (unsigned int)abs((*(in+1))-Pred_row0.y);
   SAD += (unsigned int)abs((*(in+2))-Pred_row0.z);
   SAD += (unsigned int)abs((*(in+3))-Pred_row0.w);

   SAD += (unsigned int)abs((*(in+MB_WIDTH_C))-Pred_row1.x);
   SAD += (unsigned int)abs((*(in+1+MB_WIDTH_C))-Pred_row1.y);
   SAD += (unsigned int)abs((*(in+2+MB_WIDTH_C))-Pred_row1.z);
   SAD += (unsigned int)abs((*(in+3+MB_WIDTH_C))-Pred_row1.w);

   SAD += (unsigned int)abs((*(in+MB_WIDTH_C*2))-Pred_row2.x);
   SAD += (unsigned int)abs((*(in+1+MB_WIDTH_C*2))-Pred_row2.y);
   SAD += (unsigned int)abs((*(in+2+MB_WIDTH_C*2))-Pred_row2.z);
   SAD += (unsigned int)abs((*(in+3+MB_WIDTH_C*2))-Pred_row2.w);

   SAD += (unsigned int)abs((*(in+MB_WIDTH_C*3))-Pred_row3.x);
   SAD += (unsigned int)abs((*(in+1+MB_WIDTH_C*3))-Pred_row3.y);
   SAD += (unsigned int)abs((*(in+2+MB_WIDTH_C*3))-Pred_row3.z);
   SAD += (unsigned int)abs((*(in+3+MB_WIDTH_C*3))-Pred_row3.w);
   return SAD;
}

inline __device__ void IntraChromaPrediction_cu(
												uchar4 *s_top_neighbor_uv,
												uchar4 *s_left_neighbor_uv,
												unsigned char *s_in_uv,
												int &TopAvailable,
												int &LeftAvailable,
												uchar4 *Pred_Row,
												unsigned int *Sad, 
												int &pred_mode,
												int &tx,
												int &ty
											)
{
	
	uchar4  Pred0, Pred1, Pred2, Pred3;
    // Intra8x8 Prediction. 0=DC 1=Vertical, 2=Horizontal, 3=Plane
    // Note: the order of prediction mode differs from that of the luma 16x16.

    // DC. 
    // This mode is always tested regardless availability of the neighbors.
    // However, different equations are used for different availibity map. There are
    // four possible cases 1. Both top and left are available. 2. Only top is available
    // 3. Only left is available, 4. Neither top nor left is available.
    // each pixel is given the value of 128. Which is 1<<(BitDepth - 1). BitDepth=8
    
    int TopSum, LeftSum, AllSum;
    int TopAvg, LeftAvg, AllAvg;
    int Avg,temp,PredMode; 
	int tid,Select;
	tid = tx + ty*2;

    int MinSad = LARGE_NUMBER;
    Avg = 0;
    PredMode = PRED_UNKNOWN;
    
    TopSum = (s_top_neighbor_uv[tid].x + s_top_neighbor_uv[tid].y + s_top_neighbor_uv[tid].z + s_top_neighbor_uv[tid].w + 2);
    LeftSum = (s_left_neighbor_uv[tid].x + s_left_neighbor_uv[tid].y + s_left_neighbor_uv[tid].z + s_left_neighbor_uv[tid].w + 2);
    AllSum = TopSum + LeftSum;
	TopAvg = TopSum>>2;
	LeftAvg = LeftSum>>2;
	AllAvg = AllSum>>3;
	Select = (tid & 0x3);
	Avg = (Select == 0 || Select == 3) ? AllAvg :((Select == 1) ? TopAvg : LeftAvg);
	if(TopAvailable && LeftAvailable)
	{
		Pred0.x = Pred0.y = Pred0.z = Pred0.w = Avg;
		Pred1.x = Pred1.y = Pred1.z = Pred1.w = Avg;
		Pred2.x = Pred2.y = Pred2.z = Pred2.w = Avg;
		Pred3.x = Pred3.y = Pred3.z = Pred3.w = Avg;
	}
	else if(LeftAvailable && !TopAvailable)
	{
		Pred0.x = Pred0.y = Pred0.z = Pred0.w = LeftAvg;
		Pred1.x = Pred1.y = Pred1.z = Pred1.w = LeftAvg;
		Pred2.x = Pred2.y = Pred2.z = Pred2.w = LeftAvg;
		Pred3.x = Pred3.y = Pred3.z = Pred3.w = LeftAvg;
	}
	else if(TopAvailable && !LeftAvailable)
	{
		Pred0.x = Pred0.y = Pred0.z = Pred0.w = TopAvg;
		Pred1.x = Pred1.y = Pred1.z = Pred1.w = TopAvg;
		Pred2.x = Pred2.y = Pred2.z = Pred2.w = TopAvg;
		Pred3.x = Pred3.y = Pred3.z = Pred3.w = TopAvg;
	}
	else 
	{
		Pred0.x = Pred0.y = Pred0.z = Pred0.w = 128;
		Pred1.x = Pred1.y = Pred1.z = Pred1.w = 128;
		Pred2.x = Pred2.y = Pred2.z = Pred2.w = 128;
		Pred3.x = Pred3.y = Pred3.z = Pred3.w = 128;
	}

	Sad[tid] = CalcSadChroma_cu(s_in_uv,Pred0,Pred1,Pred2,Pred3);
	temp = 0;
	for(int i=0;i<8;i++)
	{
		temp += Sad[i];
	}
	if (temp < MinSad)
	{
		MinSad    = temp;
		PredMode   = C_PRED_DC;
		Pred_Row[0].x = Pred_Row[0].y = Pred_Row[0].z =Pred_Row[0].w = Pred0.x;
		Pred_Row[1].x = Pred_Row[1].y = Pred_Row[1].z =Pred_Row[1].w = Pred1.x;
		Pred_Row[2].x = Pred_Row[2].y = Pred_Row[2].z =Pred_Row[2].w = Pred2.x;
		Pred_Row[3].x = Pred_Row[3].y = Pred_Row[3].z =Pred_Row[3].w = Pred3.x;
		
  	}
	
	// Horizontal prediction
	if(LeftAvailable)
	{
		Pred0.x = Pred0.y = Pred0.z = Pred0.w = s_left_neighbor_uv[tid].x;
		Pred1.x = Pred1.y = Pred1.z = Pred1.w = s_left_neighbor_uv[tid].y;
		Pred2.x = Pred2.y = Pred2.z = Pred2.w = s_left_neighbor_uv[tid].z;
		Pred3.x = Pred3.y = Pred3.z = Pred3.w = s_left_neighbor_uv[tid].w;

		Sad[tid] = CalcSadChroma_cu(s_in_uv,Pred0,Pred1,Pred2,Pred3);
		temp = 0;
		for(int i=0;i<8;i++)
		{
			temp += Sad[i];
		}
		if (temp < MinSad)
		{
			MinSad    = temp;
			PredMode   = C_PRED_HORIZONTAL;
			Pred_Row[0].x = Pred_Row[0].y = Pred_Row[0].z =Pred_Row[0].w = Pred0.x;
			Pred_Row[1].x = Pred_Row[1].y = Pred_Row[1].z =Pred_Row[1].w = Pred1.x;
			Pred_Row[2].x = Pred_Row[2].y = Pred_Row[2].z =Pred_Row[2].w = Pred2.x;
			Pred_Row[3].x = Pred_Row[3].y = Pred_Row[3].z =Pred_Row[3].w = Pred3.x;
			
  		}

	}
	if(TopAvailable)
	{
		Pred0.x = s_top_neighbor_uv[tid].x;
		Pred0.y = s_top_neighbor_uv[tid].y;
		Pred0.z = s_top_neighbor_uv[tid].z;
		Pred0.w = s_top_neighbor_uv[tid].w;

		Pred1 = Pred0;
		Pred2 = Pred0;
		Pred3 = Pred0;
		Sad[tid] = CalcSadChroma_cu(s_in_uv,Pred0,Pred1,Pred2,Pred3);
		temp = 0;
		for(int i=0;i<8;i++)
		{
			temp += Sad[i];
		}
		if (temp < MinSad)
		{
			MinSad    = temp;
			PredMode   = C_PRED_VERTICAL;
			Pred_Row[0].x = Pred0.x;
			Pred_Row[0].y = Pred0.y;
			Pred_Row[0].z = Pred0.z;
			Pred_Row[0].w = Pred0.w;

			Pred_Row[1].x = Pred1.x;
			Pred_Row[1].y = Pred1.y;
			Pred_Row[1].z = Pred1.z;
			Pred_Row[1].w = Pred1.w;

			Pred_Row[2].x = Pred2.x;
			Pred_Row[2].y = Pred2.y;
			Pred_Row[2].z = Pred2.z;
			Pred_Row[2].w = Pred2.w;

			Pred_Row[3].x = Pred3.x;
			Pred_Row[3].y = Pred3.y;
			Pred_Row[3].z = Pred3.z;
			Pred_Row[3].w = Pred3.w;
		}
	}
	pred_mode = PredMode;

}

inline __device__ void ChromaHadamardTransformAndQuantize_cu(
														short *TempDC,
														int &Quant_Add,                       
														int &Quant_Shift,                     
														short &Quant_tab0,
														short *QCoefDC,
														int &tx,
														int &ty
													)
{
		short TempCoef;
		short Out0,Out1,Out2,Out3;
		int tid;
		int   QAdd;
		int   Sign;
		int   QuantDCShift;

		tid = tx + ty * 2;
		QAdd = Quant_Add * 2;            
		QuantDCShift = Quant_Shift + 1;

		Out0 = TempDC[0+(ty>>1)*4] + TempDC[1+(ty>>1)*4] + TempDC[2+(ty>>1)*4] + TempDC[3+(ty>>1)*4];
		Out1 = TempDC[0+(ty>>1)*4] - TempDC[1+(ty>>1)*4] + TempDC[2+(ty>>1)*4] - TempDC[3+(ty>>1)*4];
		Out2 = TempDC[0+(ty>>1)*4] + TempDC[1+(ty>>1)*4] - TempDC[2+(ty>>1)*4] - TempDC[3+(ty>>1)*4];
		Out3 = TempDC[0+(ty>>1)*4] - TempDC[1+(ty>>1)*4] - TempDC[2+(ty>>1)*4] + TempDC[3+(ty>>1)*4];
		
		TempCoef = ((tid%4)==0 ? Out0 :((tid%4)==1 ? Out1 :((tid%4)==2) ? Out2 : Out3));
		Sign  = (TempCoef >= 0) ? 1 : -1;
		TempCoef  = (TempCoef >= 0) ? TempCoef : -TempCoef;
		TempCoef  = min (((TempCoef * Quant_tab0 + QAdd) >> QuantDCShift), TempCoef);
		QCoefDC[tid] = Sign * TempCoef;
}

inline __device__ void ChromaIHadamard2x2AndDQuant_cu(	short *TempDC,                  
														int &DQuant_Shift,                     
														short &DQuant_tab0,
														short *QCoefDC,
														int &tx,
														int &ty
														)
{
		short TempCoef;
		short Out0,Out1,Out2,Out3;
		int tid;
		tid = tx +ty*2;
    
        QCoefDC[tid] = (TempDC[tid] * DQuant_tab0 << DQuant_Shift);

		Out0 = QCoefDC[0+(ty>>1)*4] + QCoefDC[1+(ty>>1)*4] + QCoefDC[2+(ty>>1)*4] + QCoefDC[3+(ty>>1)*4];
		Out1 = QCoefDC[0+(ty>>1)*4] - QCoefDC[1+(ty>>1)*4] + QCoefDC[2+(ty>>1)*4] - QCoefDC[3+(ty>>1)*4];
		Out2 = QCoefDC[0+(ty>>1)*4] + QCoefDC[1+(ty>>1)*4] - QCoefDC[2+(ty>>1)*4] - QCoefDC[3+(ty>>1)*4];
		Out3 = QCoefDC[0+(ty>>1)*4] - QCoefDC[1+(ty>>1)*4] - QCoefDC[2+(ty>>1)*4] + QCoefDC[3+(ty>>1)*4];
		TempCoef = ((tid%4)==0 ? Out0 :((tid%4)==1 ? Out1 :((tid%4)==2) ? Out2 : Out3));
		QCoefDC [tid] =  (TempCoef>>1);

}

inline __device__ void	IntraChromaTransforms_cu(
													unsigned char *InputSrcRow,
													uchar4 *PredRow,
													uchar4 *RecOutRow,
													int &QuantAdd,
													int &QuantShift,
													int	&DquantShift,
													short *Quant_tab,
													short *Dquant_tab,
													short2 &QCoef0_01,
													short2 &QCoef0_23,
													short2 &QCoef1_01,
													short2 &QCoef1_23,
													short2 &QCoef2_01,
													short2 &QCoef2_23,
													short2 &QCoef3_01,
													short2 &QCoef3_23,
													short  *QcoefDC,
													short  *TempDC,
													int &tx,
													int &ty
												)
{

		short2 DiffRow0_01,DiffRow0_23,DiffRow1_01,DiffRow1_23,
			   DiffRow2_01,DiffRow2_23,DiffRow3_01,DiffRow3_23;
		int tid;

		tid = tx + ty * 2;

		DiffRow0_01.x = ((short)InputSrcRow[0] - (short)PredRow[0].x);
   		DiffRow0_01.y = ((short)InputSrcRow[1] - (short)PredRow[0].y);
   		DiffRow0_23.x = ((short)InputSrcRow[2] - (short)PredRow[0].z);
   		DiffRow0_23.y = ((short)InputSrcRow[3] - (short)PredRow[0].w);

   		DiffRow1_01.x = ((short)InputSrcRow[0+MB_WIDTH_C] - (short)PredRow[1].x);
   		DiffRow1_01.y = ((short)InputSrcRow[1+MB_WIDTH_C] - (short)PredRow[1].y);
   		DiffRow1_23.x = ((short)InputSrcRow[2+MB_WIDTH_C] - (short)PredRow[1].z);
   		DiffRow1_23.y = ((short)InputSrcRow[3+MB_WIDTH_C] - (short)PredRow[1].w);
	   	
   		DiffRow2_01.x = ((short)InputSrcRow[0+MB_WIDTH_C*2] - (short)PredRow[2].x);
   		DiffRow2_01.y = ((short)InputSrcRow[1+MB_WIDTH_C*2] - (short)PredRow[2].y);
   		DiffRow2_23.x = ((short)InputSrcRow[2+MB_WIDTH_C*2] - (short)PredRow[2].z);
   		DiffRow2_23.y = ((short)InputSrcRow[3+MB_WIDTH_C*2] - (short)PredRow[2].w);
	   	
   		DiffRow3_01.x = ((short)InputSrcRow[0+MB_WIDTH_C*3] - (short)PredRow[3].x);
   		DiffRow3_01.y = ((short)InputSrcRow[1+MB_WIDTH_C*3] - (short)PredRow[3].y);
   		DiffRow3_23.x = ((short)InputSrcRow[2+MB_WIDTH_C*3] - (short)PredRow[3].z);
   		DiffRow3_23.y = ((short)InputSrcRow[3+MB_WIDTH_C*3] - (short)PredRow[3].w);

		Chr_Dct4x4TransformAndQuantize_cu (
										DiffRow0_01,  DiffRow0_23,       //[i]
										DiffRow1_01,  DiffRow1_23,       //[i]
										DiffRow2_01,  DiffRow2_23,       //[i]
										DiffRow3_01,  DiffRow3_23,       //[i]  
										QuantAdd,                       //[i]
										QuantShift,                     //[i]
										Quant_tab,                       //[i]
										QCoef0_01,  QCoef0_23,           //[o]
										QCoef1_01,  QCoef1_23,           //[o]
										QCoef2_01,  QCoef2_23,         //[o]
										QCoef3_01,  QCoef3_23,        //[o]
										TempDC[tid]                          // dummy
									);
		ChromaHadamardTransformAndQuantize_cu(
												TempDC,
												QuantAdd,                       
												QuantShift,                     
												Quant_tab[0],
												QcoefDC,
												tx,
												ty
											);
		
		ChromaIHadamard2x2AndDQuant_cu(	QcoefDC,                  
										DquantShift,                     
										Dquant_tab[0],
										TempDC,
										tx,
										ty
										);
		DiffRow0_01.x = TempDC[tid];
        //反量化
        
//		DiffRow0_01.x = (QCoef0_01.x * Dquant_tab[0]) << DquantShift;
		DiffRow0_01.y = (QCoef0_01.y * Dquant_tab[1]) << DquantShift;
		DiffRow0_23.x = (QCoef0_23.x * Dquant_tab[2]) << DquantShift;
		DiffRow0_23.y = (QCoef0_23.y * Dquant_tab[3]) << DquantShift;

		DiffRow1_01.x = (QCoef1_01.x * Dquant_tab[4]) << DquantShift;
		DiffRow1_01.y = (QCoef1_01.y * Dquant_tab[5]) << DquantShift;
		DiffRow1_23.x = (QCoef1_23.x * Dquant_tab[6]) << DquantShift;
		DiffRow1_23.y = (QCoef1_23.y * Dquant_tab[7]) << DquantShift;

		DiffRow2_01.x = (QCoef2_01.x * Dquant_tab[8]) << DquantShift;
		DiffRow2_01.y = (QCoef2_01.y * Dquant_tab[9]) << DquantShift;
		DiffRow2_23.x = (QCoef2_23.x * Dquant_tab[10]) << DquantShift;
		DiffRow2_23.y = (QCoef2_23.y * Dquant_tab[11]) << DquantShift;

		DiffRow3_01.x = (QCoef3_01.x * Dquant_tab[12]) << DquantShift;
		DiffRow3_01.y = (QCoef3_01.y * Dquant_tab[13]) << DquantShift;
		DiffRow3_23.x = (QCoef3_23.x * Dquant_tab[14]) << DquantShift;
		DiffRow3_23.y = (QCoef3_23.y * Dquant_tab[15]) << DquantShift;

        //逆变换
		 Chr_IDct4x4AndAdd_cu (
							DiffRow0_01,
							DiffRow0_23,
							DiffRow1_01,
							DiffRow1_23,
							DiffRow2_01,
							DiffRow2_23,
							DiffRow3_01,
							DiffRow3_23,
							PredRow,
							RecOutRow[0],
							RecOutRow[1],
							RecOutRow[2],
							RecOutRow[3]
						);

}