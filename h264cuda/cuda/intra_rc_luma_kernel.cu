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
//--------------------------------------------------------------------
#include "../inc/mb_info.h"
#include "../inc/const_defines.h"
#include <math.h>

#include "transform_quantize_kernel.cu"

//compute the SAD 
inline __device__ unsigned int CalcSad4x4_cu(unsigned char *in,uchar4 &Pred_row0,uchar4 &Pred_row1,uchar4 &Pred_row2,uchar4 &Pred_row3)
{
   unsigned int SAD =0;
  
   SAD += (unsigned int)abs((*in)-Pred_row0.x);
   SAD += (unsigned int)abs((*(in+1))-Pred_row0.y);
   SAD += (unsigned int)abs((*(in+2))-Pred_row0.z);
   SAD += (unsigned int)abs((*(in+3))-Pred_row0.w);

   SAD += (unsigned int)abs((*(in+16))-Pred_row1.x);
   SAD += (unsigned int)abs((*(in+1+16))-Pred_row1.y);
   SAD += (unsigned int)abs((*(in+2+16))-Pred_row1.z);
   SAD += (unsigned int)abs((*(in+3+16))-Pred_row1.w);

   SAD += (unsigned int)abs((*(in+16*2))-Pred_row2.x);
   SAD += (unsigned int)abs((*(in+1+16*2))-Pred_row2.y);
   SAD += (unsigned int)abs((*(in+2+16*2))-Pred_row2.z);
   SAD += (unsigned int)abs((*(in+3+16*2))-Pred_row2.w);

   SAD += (unsigned int)abs((*(in+16*3))-Pred_row3.x);
   SAD += (unsigned int)abs((*(in+1+16*3))-Pred_row3.y);
   SAD += (unsigned int)abs((*(in+2+16*3))-Pred_row3.z);
   SAD += (unsigned int)abs((*(in+3+16*3))-Pred_row3.w);
   return SAD;
}


//实现宏块中每个子宏块的16x16预测，类似与流处理方式
inline __device__ void Intra16x16Prediction_cu(
						uchar4 *S_top_neighbor,
						uchar4 *S_left_neighbor,
						unsigned char *S_in_block,
						int &TopAvailable, 
						int &LeftAvailable,
						uchar4 *pred,
						int &MinSad,
						int *SAD,
						int &PredMode,
						int &tx,
						int &ty
						)
{
  		 // int TopAvailable,LeftAvailable;
    			uchar4 Pred_row0,Pred_row1,Pred_row2,Pred_row3,Pred0,Pred1,Pred2,Pred3;
				int Pred;

				int temp,i,Sum;
    			int SubType = UNDEFINED_SUBDIV_TYPE;
    			int MinSAD;
    			//Type = INTRA_LARGE_BLOCKS_MB_TYPE;
    			MinSAD = LARGE_NUMBER;
				temp=0;
				int tid = tx + ty*4;
			//垂直预测
				if(TopAvailable)
				{
					Pred_row0.x = S_top_neighbor[tid].x;
					Pred_row0.y = S_top_neighbor[tid].y;
					Pred_row0.z = S_top_neighbor[tid].z;
					Pred_row0.w = S_top_neighbor[tid].w;
  			
					Pred_row1 = Pred_row0;
					Pred_row2 = Pred_row0;
					Pred_row3 = Pred_row0;
				   //计算sad值
  			     
					SAD[tx+ty*4]= CalcSad4x4_cu(S_in_block,Pred_row0,Pred_row1,Pred_row2,Pred_row3);
	
					for(i=0;i<16;i++)
					{
						temp += SAD[i];
					}
					if (temp < MinSAD)
  					{
  						MinSAD    = temp;
      					SubType   = L_PRED_VERTICAL;
      			  	}

				}
				temp =0;
				//水平预测
				if(LeftAvailable)
				{
				   Pred0.x = Pred0.y = Pred0.z = Pred0.w = S_left_neighbor[tid].x;
				   Pred1.x = Pred1.y = Pred1.z = Pred1.w = S_left_neighbor[tid].y;
				   Pred2.x = Pred2.y = Pred2.z = Pred2.w = S_left_neighbor[tid].z;
				   Pred3.x = Pred3.y = Pred3.z = Pred3.w = S_left_neighbor[tid].w;
  			
				   SAD[tx+ty*4]= CalcSad4x4_cu(S_in_block,Pred0,Pred1,Pred2,Pred3);
		
				   for(i=0;i<16;i++)
				   {
						temp = temp+SAD[i];
				   }
  			
				   if (temp < MinSAD)
  				  {
  						MinSAD    = temp;
  			       		SubType   = L_PRED_HORIZONTAL;
						Pred_row0 = Pred0; 
						Pred_row1 = Pred1;
						Pred_row2 = Pred2;
						Pred_row3 = Pred3;
  				  }
  			
				}
  			  
				Sum = 0;
  				if (LeftAvailable && TopAvailable)
  				{
  			   		for (i=0; i<4; i++)
  					{
  			        	Sum += S_top_neighbor[i].x + S_left_neighbor[i*4].x;
						Sum += S_top_neighbor[i].y + S_left_neighbor[i*4].y;
						Sum += S_top_neighbor[i].z + S_left_neighbor[i*4].z;
						Sum += S_top_neighbor[i].w + S_left_neighbor[i*4].w;
  			    	}
  			   		Sum += 16;
  			    	Pred = Sum >> 5;
  				}
  				else if (LeftAvailable && !TopAvailable)
  				{
  				    for (i=0; i< 4; i++)
  				    {
  				        Sum += S_left_neighbor[i*4].x;
						Sum += S_left_neighbor[i*4].y;
						Sum += S_left_neighbor[i*4].z;
						Sum += S_left_neighbor[i*4].w;
  				    }
  				    Sum += 8;
  				    Pred = Sum >> 4;
  				}
  				else if (TopAvailable && !LeftAvailable)
  				{
  				    for (i=0; i<4; i++)
  				    {   
						Sum += S_top_neighbor[i].x;
						Sum += S_top_neighbor[i].y;
						Sum += S_top_neighbor[i].z;
						Sum += S_top_neighbor[i].w;
  				    }
  				    Sum += 8;
  				    Pred = Sum >> 4;
  				}
  				else // Non of the neighbors are available
  				{
  				    Pred = 128;
  				}
  				Pred0.x = Pred0.y = Pred0.z = Pred0.w =(unsigned char) Pred;
				Pred1.x = Pred1.y = Pred1.z = Pred1.w =(unsigned char) Pred;
				Pred2.x = Pred2.y = Pred2.z = Pred2.w =(unsigned char) Pred;
				Pred3.x = Pred3.y = Pred3.z = Pred3.w =(unsigned char) Pred;
  			  
  				SAD[tx+ty*4] = CalcSad4x4_cu(S_in_block,Pred0,Pred1,Pred2,Pred3);
	
  			    temp = 0;
				for(i=0;i<16;i++)
				{
				    temp = temp+SAD[i];
				}
				if (temp < MinSAD)
  				{
  					MinSAD    = temp;
  					SubType   = L_PRED_DC;
					Pred_row0 = Pred0; 
					Pred_row1 = Pred1;
					Pred_row2 = Pred2;
					Pred_row3 = Pred3;
  				}
	
				MinSad = MinSAD;
				PredMode = SubType;
				pred[0] = Pred_row0;
				pred[1] = Pred_row1;
				pred[2] = Pred_row2;
				pred[3] = Pred_row3;
	}




//一个4x4块的帧内预测，总共有9种预测方式，只实现7种，其中第3和7不实现
inline __device__ void Intra4x4BlockPrediction_cu (
												uchar4 &Top_pix,         //  neighboring pixels from top neighbor block
												uchar4 &Left_pix,        //  neighboring pixels from left neighbor block
												unsigned char &Top_left_pix,        // Only one byte (byte 3) that is correspond to TopLeft pixel of current block is useful.
												int &prefered_pred_mode,
												int &PredModePenalty,
												int &TopAvailable,
												int &LeftAvailable,
												unsigned char *InputSrcRow,
												int &Sad,
												int &Pred_mode,
												uchar4 *Pred_row
											)
{
    //uchar4 InputSrcRow0, InputSrcRow1, InputSrcRow2, InputSrcRow3;
    uchar4 PredRow0, PredRow1, PredRow2, PredRow3;
    uchar4  Pred0, Pred1, Pred2, Pred3;
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
		PredRow0 = Top_pix;
		PredRow1 = Top_pix;
		PredRow2 = Top_pix;
		PredRow3 = Top_pix;

		Sad_temp = CalcSad4x4_cu(
								InputSrcRow,
								PredRow0,PredRow1,PredRow2,PredRow3
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
		Pred0.x = Pred0.y = Pred0.z = Pred0.w = Left_pix.x;
		Pred1.x = Pred1.y = Pred1.z = Pred1.w = Left_pix.y;
		Pred2.x = Pred2.y = Pred2.z = Pred2.w = Left_pix.z;
		Pred3.x = Pred3.y = Pred3.z = Pred3.w = Left_pix.w;
   
		Sad_temp = CalcSad4x4_cu(
									InputSrcRow,
									Pred0,Pred1,Pred2,Pred3
								);
		Sad_temp += Penalty;
		PredFlag = (LeftAvailable && (Sad_temp < MinSad));
		MinSad   = (PredFlag ? Sad_temp : MinSad);
		PredRow0 = (PredFlag ? Pred0 : PredRow0);
		PredRow1 = (PredFlag ? Pred1 : PredRow1);
		PredRow2 = (PredFlag ? Pred2 : PredRow2);
		PredRow3 = (PredFlag ? Pred3 : PredRow3);
		PredMode = (PredFlag ? S_PRED_HOR : PredMode);   
	}
	Sad_temp = 0;
   //DC 预测 总会进行的操作
	Avg = 0 ;

	TopSum = Top_pix.x + Top_pix.y + Top_pix.z + Top_pix.w + 2;     // Add up 4 top neighbor pixels + 2 for rounding
	TopAvg =  (TopSum) >> 2;
    
	LeftSum = Left_pix.x + Left_pix.y + Left_pix.z + Left_pix.w + 2;
	LeftAvg = (LeftSum) >> 2;     // Add up 4 left neighbor pixels + 2 for rounding

	AllSum  = TopSum + LeftSum;
	AllAvg  = (AllSum) >> 3;
            
	Avg = (BothAvailable ? AllAvg : Avg);
	Avg = ((TopAvailable && (!LeftAvailable)) ? TopAvg : Avg);
	Avg = (((!TopAvailable) && LeftAvailable)  ? LeftAvg : Avg);
	Avg = (((!TopAvailable) && (!LeftAvailable)) ? 128 : Avg);

	Pred0.x = Pred0.y = Pred0.z = Pred0.w =(unsigned char) Avg;

	Pred1 = Pred0;
	Pred2 = Pred0;
	Pred3 = Pred0;

	Sad_temp = CalcSad4x4_cu(
								InputSrcRow,
								Pred0,Pred1,Pred2,Pred3
							);

   Penalty = (prefered_pred_mode == S_PRED_DC) ? 0 : PredModePenalty;
   Sad_temp += Penalty;
   PredFlag = (Sad_temp < MinSad) ? 1 : 0;
   MinSad   = (PredFlag ? Sad_temp : MinSad);
   PredRow0 = (PredFlag ? Pred0 : PredRow0);
   PredRow1 = (PredFlag ? Pred1 : PredRow1);
   PredRow2 = (PredFlag ? Pred2 : PredRow2);
   PredRow3 = (PredFlag ? Pred3 : PredRow3);
   PredMode = (PredFlag ? S_PRED_DC : PredMode);        // DC mode
  
	 Sad_temp = 0;
   //右下对角预测
   if(BothAvailable)
   {
		Pred0.x = Pred1.y = Pred2.z = Pred3.w = (Left_pix.x + 2*Top_left_pix + Top_pix.x +2) >> 2;
		Pred1.x = Pred2.y = Pred3.z = (Left_pix.y +2*Left_pix.x + Top_left_pix + 2) >> 2;
		Pred2.x = Pred3.y = (Left_pix.z + 2*Left_pix.y + Left_pix.x + 2 ) >> 2;
		Pred3.x = (Left_pix.w + 2*Left_pix.z +Left_pix.y + 2 ) >> 2;

		Pred0.y = Pred1.z = Pred2.w = ((Top_left_pix + 2 * Top_pix.x + Top_pix.y + 2) >> 2);
		Pred0.z = Pred1.w = (Top_pix.x + 2*Top_pix.y + Top_pix.z + 2) >> 2;
		Pred0.w = (Top_pix.y + 2*Top_pix.z + Top_pix.w + 2) >> 2;
   
		Penalty = (prefered_pred_mode == S_PRED_DIAG_DOWN_RIGHT) ? 0 : PredModePenalty;

		Sad_temp = CalcSad4x4_cu(
									InputSrcRow,
									Pred0,Pred1,Pred2,Pred3
								);
		Sad_temp += Penalty;
		PredFlag = (BothAvailable && Sad_temp < MinSad);
		MinSad   = (PredFlag ? Sad_temp: MinSad);
		PredRow0 = (PredFlag ? Pred0 : PredRow0);
		PredRow1 = (PredFlag ? Pred1 : PredRow1);
		PredRow2 = (PredFlag ? Pred2 : PredRow2);
		PredRow3 = (PredFlag ? Pred3 : PredRow3);
		PredMode = (PredFlag ? S_PRED_DIAG_DOWN_RIGHT : PredMode);    
   }
   //垂直向右预测

   if(BothAvailable)
   {
		Pred0.x = Pred2.y = (Top_left_pix + Top_pix.x +1) >> 1;
		Pred0.y = Pred2.z = (Top_pix.x + Top_pix.y + 1) >> 1;
		Pred0.z = Pred2.w = (Top_pix.y + Top_pix.z + 1) >> 1;
		Pred0.w = (Top_pix.z + Top_pix.w + 1) >> 1;
   
		Pred1.x = Pred3.y =(Left_pix.x + 2*Top_left_pix + Top_pix.x +2) >> 2;
		Pred1.y = Pred3.z =(Top_left_pix + 2*Top_pix.x + Top_pix.y + 2) >> 2;
		Pred1.z = Pred3.w =(Top_pix.x + 2*Top_pix.y +Top_pix.z + 2) >> 2;
		Pred1.w = (Top_pix.y + 2*Top_pix.z + Top_pix.w + 2) >> 2;
   
		Pred2.x = (Top_left_pix + 2*Left_pix.x + Left_pix.y + 2) >> 2;
		Pred3.x = (Left_pix.x + 2*Left_pix.y + Left_pix.z + 2) >> 2;
   
		Penalty = (prefered_pred_mode == S_PRED_VERT_RIGHT) ? 0 : PredModePenalty;
		Sad_temp = 0;
		Sad_temp = CalcSad4x4_cu(
									InputSrcRow,
									Pred0,Pred1,Pred2,Pred3
								);
		Sad_temp += Penalty;
		PredFlag = (BothAvailable && Sad_temp < MinSad);
		MinSad   = (PredFlag ? Sad_temp : MinSad);
		PredRow0 = (PredFlag ? Pred0 : PredRow0);
		PredRow1 = (PredFlag ? Pred1 : PredRow1);
		PredRow2 = (PredFlag ? Pred2 : PredRow2);
		PredRow3 = (PredFlag ? Pred3 : PredRow3);
		PredMode = (PredFlag ? S_PRED_VERT_RIGHT : PredMode); 
   }

   //水平向下预测
   if(BothAvailable)
   {
		Pred0.x = Pred1.z = (Top_left_pix + Left_pix.x +1) >> 1;
		Pred0.y = Pred1.w = (Left_pix.x + 2*Top_left_pix + Top_pix.x +2) >> 2;
		Pred0.z = (Top_left_pix + 2*Top_pix.x + Top_pix.y + 2) >> 2;
		Pred0.w = (Top_pix.x + 2*Top_pix.y +Top_pix.z + 2) >> 2;
   
		Pred1.x = Pred2.z = (Left_pix.x + Left_pix.y +1) >> 1;
		Pred1.y = Pred2.w = (Top_left_pix + 2*Left_pix.x + Left_pix.y + 2) >> 2;
   
		Pred2.x = Pred3.z = ( Left_pix.y + Left_pix.z + 1) >> 1;
		Pred2.y = Pred3.w = ( Left_pix.x + 2*Left_pix.y + Left_pix.z+ 2) >> 2;
		Pred3.x = (Left_pix.z + Left_pix.w + 1) >> 1;
		Pred3.y = (Left_pix.y + 2*Left_pix.z +Left_pix.w + 2) >> 2;
  
		Penalty = (prefered_pred_mode == S_PRED_HOR_DOWN) ? 0 : PredModePenalty;
		Sad_temp = 0;
		Sad_temp = CalcSad4x4_cu(
									InputSrcRow,
									Pred0,Pred1,Pred2,Pred3
								);
		Sad_temp += Penalty;
		PredFlag = (BothAvailable && Sad_temp < MinSad);
		MinSad   = (PredFlag ? Sad_temp: MinSad);
		PredRow0 = (PredFlag ? Pred0 : PredRow0);
		PredRow1 = (PredFlag ? Pred1 : PredRow1);
		PredRow2 = (PredFlag ? Pred2 : PredRow2);
		PredRow3 = (PredFlag ? Pred3 : PredRow3);
		PredMode = (PredFlag ? S_PRED_HOR_DOWN : PredMode); 
   }
   //水平向上预测
   if(LeftAvailable)
   {
		Pred0.x = ( Left_pix.x + Left_pix.y + 1) >> 1;
		Pred0.y = (Left_pix.x + 2*Left_pix.y+ Left_pix.z +2) >> 2;
		Pred0.z = Pred1.x = (Left_pix.y+ Left_pix.z +1) >> 1;
		Pred0.w = Pred1.y = (Left_pix.y + 2*Left_pix.z +Left_pix.w + 2) >> 2;
   
		Pred1.z = Pred2.x = (Left_pix.z + Left_pix.w +1) >> 1;
		Pred1.w = Pred2.y = ( Left_pix.z + 3*Left_pix.w + 2) >> 2;
		Pred2.z = Pred2.w = Pred3.x = Pred3.y = Pred3.z = Pred3.w = Left_pix.w;   
   
		Penalty = (prefered_pred_mode == S_PRED_HOR_UP) ? 0 : PredModePenalty;
		Sad_temp = 0;
		Sad_temp = CalcSad4x4_cu(
									InputSrcRow,
									Pred0,Pred1,Pred2,Pred3
								);

		Sad_temp += Penalty;
		PredFlag = (LeftAvailable && Sad_temp < MinSad);

		MinSad   = (PredFlag ? Sad_temp: MinSad);
		PredRow0 = (PredFlag ? Pred0 : PredRow0);
		PredRow1 = (PredFlag ? Pred1 : PredRow1);
		PredRow2 = (PredFlag ? Pred2 : PredRow2);
		PredRow3 = (PredFlag ? Pred3 : PredRow3);
		PredMode = (PredFlag ? S_PRED_HOR_UP : PredMode); 
   }
   Sad = MinSad;
   Pred_mode = PredMode;
   Pred_row[0] = PredRow0;
   Pred_row[1] = PredRow1;
   Pred_row[2] = PredRow2;
   Pred_row[3] = PredRow3;
    
}

inline __device__ void Intra4x4BlockTransformAndReconstruct_cu(
															unsigned char *InputSrcRow,
															uchar4 *Pred_row,
															int &QuantAdd,
															int &QuantShift,
															int &DquantShift,
															short *Quant_tables,
															short *DQuant_tables,
															short2 &TempQCoef0_01,
															short2 &TempQCoef0_23,
															short2 &TempQCoef1_01,
															short2 &TempQCoef1_23,
															short2 &TempQCoef2_01,
															short2 &TempQCoef2_23,
															short2 &TempQCoef3_01,
															short2 &TempQCoef3_23,
															uchar4 *Out_Row
														)
{
	short2 		DiffRow0_01,DiffRow0_23,DiffRow1_01,DiffRow1_23,
		   			DiffRow2_01,DiffRow2_23,DiffRow3_01,DiffRow3_23;
	short 		Coef00;

   	DiffRow0_01.x = ((short)InputSrcRow[0] - (short)Pred_row[0].x);
   	DiffRow0_01.y = ((short)InputSrcRow[1] - (short)Pred_row[0].y);
   	DiffRow0_23.x = ((short)InputSrcRow[2] - (short)Pred_row[0].z);
   	DiffRow0_23.y = ((short)InputSrcRow[3] - (short)Pred_row[0].w);

   	DiffRow1_01.x = ((short)InputSrcRow[0+MB_WIDTH] - (short)Pred_row[1].x);
   	DiffRow1_01.y = ((short)InputSrcRow[1+MB_WIDTH] - (short)Pred_row[1].y);
   	DiffRow1_23.x = ((short)InputSrcRow[2+MB_WIDTH] - (short)Pred_row[1].z);
   	DiffRow1_23.y = ((short)InputSrcRow[3+MB_WIDTH] - (short)Pred_row[1].w);
   	
   	DiffRow2_01.x = ((short)InputSrcRow[0+MB_WIDTH*2] - (short)Pred_row[2].x);
   	DiffRow2_01.y = ((short)InputSrcRow[1+MB_WIDTH*2] - (short)Pred_row[2].y);
   	DiffRow2_23.x = ((short)InputSrcRow[2+MB_WIDTH*2] - (short)Pred_row[2].z);
   	DiffRow2_23.y = ((short)InputSrcRow[3+MB_WIDTH*2] - (short)Pred_row[2].w);
   	
   	DiffRow3_01.x = ((short)InputSrcRow[0+MB_WIDTH*3] - (short)Pred_row[3].x);
   	DiffRow3_01.y = ((short)InputSrcRow[1+MB_WIDTH*3] - (short)Pred_row[3].y);
   	DiffRow3_23.x = ((short)InputSrcRow[2+MB_WIDTH*3] - (short)Pred_row[3].z);
   	DiffRow3_23.y = ((short)InputSrcRow[3+MB_WIDTH*3] - (short)Pred_row[3].w);
	
    Dct4x4TransformAndQuantize_cu (
									DiffRow0_01,  DiffRow0_23,       //[i]
									DiffRow1_01,  DiffRow1_23,       //[i]
									DiffRow2_01,  DiffRow2_23,       //[i]
									DiffRow3_01,  DiffRow3_23,       //[i]  
									QuantAdd,                       //[i]
									QuantShift,                     //[i]
									Quant_tables,
									TempQCoef0_01,  TempQCoef0_23,           //[o]
									TempQCoef1_01,  TempQCoef1_23,           //[o]
									TempQCoef2_01,  TempQCoef2_23,         //[o]
									TempQCoef3_01,  TempQCoef3_23,        //[o]
									Coef00                          // dummy
								);
	
   // Dquant 
   DiffRow0_01.x = (TempQCoef0_01.x * DQuant_tables[0]) << (int)DquantShift;
   DiffRow0_01.y = (TempQCoef0_01.y * DQuant_tables[1]) << (int)DquantShift;
   DiffRow0_23.x = (TempQCoef0_23.x * DQuant_tables[2]) << (int)DquantShift;
   DiffRow0_23.y = (TempQCoef0_23.y * DQuant_tables[3]) << (int)DquantShift;

   DiffRow1_01.x = (TempQCoef1_01.x * DQuant_tables[4]) << (int)DquantShift;
   DiffRow1_01.y = (TempQCoef1_01.y * DQuant_tables[5]) << (int)DquantShift;
   DiffRow1_23.x = (TempQCoef1_23.x * DQuant_tables[6]) << (int)DquantShift;
   DiffRow1_23.y = (TempQCoef1_23.y * DQuant_tables[7]) << (int)DquantShift;

   DiffRow2_01.x = (TempQCoef2_01.x * DQuant_tables[8]) << (int)DquantShift;
   DiffRow2_01.y = (TempQCoef2_01.y * DQuant_tables[9]) << (int)DquantShift;
   DiffRow2_23.x = (TempQCoef2_23.x * DQuant_tables[10]) << (int)DquantShift;
   DiffRow2_23.y = (TempQCoef2_23.y * DQuant_tables[11]) << (int)DquantShift;

   DiffRow3_01.x = (TempQCoef3_01.x * DQuant_tables[12]) << (int)DquantShift;
   DiffRow3_01.y = (TempQCoef3_01.y * DQuant_tables[13]) << (int)DquantShift;
   DiffRow3_23.x = (TempQCoef3_23.x * DQuant_tables[14]) << (int)DquantShift;
   DiffRow3_23.y = (TempQCoef3_23.y * DQuant_tables[15]) << (int)DquantShift;
  
   //I DCT
   IDct4x4AndAdd_cu (
						DiffRow0_01,
						DiffRow0_23,
						DiffRow1_01,
						DiffRow1_23,
						DiffRow2_01,
						DiffRow2_23,
						DiffRow3_01,
						DiffRow3_23,
						Pred_row,
						Out_Row[0],
						Out_Row[1],
						Out_Row[2],
						Out_Row[3]
              );			  
	
}

// 4x4帧内预测以及残差编码的cuda实现，一个宏块通过7步实现16个4x4子宏块的预测和残差编码
inline __device__ void Intra4x4Prediction_cu(	uchar4 *blk_Botton_Row,
				      							uchar4 *blk_Right_Col,
												unsigned char &top_left_pix,
												unsigned char *InputSrcRow,
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
												//unsigned char &Rec00,unsigned char &Rec01,unsigned char &Rec02,unsigned char &Rec03,
												//unsigned char &Rec10,unsigned char &Rec11,unsigned char &Rec12,unsigned char &Rec13,
												//unsigned char &Rec20,unsigned char &Rec21,unsigned char &Rec22,unsigned char &Rec23,
												//unsigned char &Rec30,unsigned char &Rec31,unsigned char &Rec32,unsigned char &Rec33,
												uchar4 *Rec_row,
												short2 &Qcoef_0_01, short2 &Qcoef_0_23, short2 &Qcoef_1_01, short2 &Qcoef_1_23,
												short2 &Qcoef_2_01, short2 &Qcoef_2_23, short2 &Qcoef_3_01, short2 &Qcoef_3_23, 
												int &MinSad,
												int *Sad,
												int *PreferedPredMode,
												int &PredMode,
												int &tx,
												int &ty)

{
    		uchar4 Pred_row[4];
			//uchar4 Pred_row1,Pred_row2,Pred_row3,Pred0,Pred1,Pred2,Pred3;
			uchar4 Out_Row[4];
			int left_blk_pred_mode;
			int top_blk_pred_mode;
			uchar4 	Left_pix,Top_pix;
			unsigned char Top_left_pix;
			short2 	TempQCoef0_01,
           			TempQCoef0_23,
           			TempQCoef1_01,
           			TempQCoef1_23,
           			TempQCoef2_01,
           			TempQCoef2_23,
           			TempQCoef3_01,
           			TempQCoef3_23;
			int Pred,temp_sad,temp_pred;
			int is_left_blk,is_top_blk;
    		int prefered_pred_mode;
			int first_blk_index,is_active_blk,active_blk;
    
    		int tid = tx + ty*4;
			int num_stages,current_stage;
			//int is_valid = 0;   //表明哪一步中哪个线程的结果是合法的
			//temp=0;
			num_stages =7;
			current_stage=0;
  		  
			is_left_blk = (tx == 0) ? 1 : 0;

			is_top_blk = (ty == 0) ? 1 : 0;

			left_blk_pred_mode = (is_left_blk) ? Left_blk_type[(tid + 3 ) & 0xf] : S_PRED_INVALID_MODE;
			left_blk_pred_mode = (dc_only_pred_flag||(Left_mb_type!=INTRA_SMALL_BLOCKS_MB_TYPE)) ? S_PRED_DC : left_blk_pred_mode;
  		 
			top_blk_pred_mode = (is_top_blk) ? Top_blk_type[(tid + 12) & 0xf] : S_PRED_INVALID_MODE;
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
			is_active_blk   = (tid == first_blk_index) ? 1 : 0;
        	active_blk      = first_blk_index + 3;  //下一行的活动块
        	is_active_blk   = is_active_blk | ((tid == active_blk) && (active_blk <= (current_stage << 2)));
        	active_blk      = first_blk_index + 6;  
			is_active_blk   = is_active_blk | ((tid == active_blk) && (active_blk <= (current_stage << 2)));
        	active_blk      = first_blk_index + 9;     
        	is_active_blk   = is_active_blk | ((tid == active_blk) && (active_blk <= (current_stage << 2)));
 
			Left_pix = (is_left_blk ? blk_Right_Col[tid + 3] : blk_Right_Col[tid-1] );
			Top_pix  = (is_top_blk ? blk_Botton_Row[tid + 12] : blk_Botton_Row[tid-4] ) ;
			Top_left_pix = (is_top_blk ? (is_left_blk ? top_left_pix : blk_Botton_Row[tid+11].w) : (is_left_blk ? blk_Right_Col[tid-1].w : blk_Right_Col[tid-5].w ));
					
			left_blk_pred_mode = dc_only_pred_flag ? S_PRED_DC : (is_left_blk ? left_blk_pred_mode :Left_blk_type[tid-1]);
			top_blk_pred_mode  = dc_only_pred_flag ? S_PRED_DC : (is_top_blk ? top_blk_pred_mode : Top_blk_type[tid-4]);
			prefered_pred_mode = (top_blk_pred_mode < left_blk_pred_mode) ? top_blk_pred_mode : left_blk_pred_mode;

			Intra4x4BlockPrediction_cu (
            								Top_pix,         //  neighboring pixels from top neighbor block
            								Left_pix,        //  neighboring pixels from left neighbor block
            								Top_left_pix,        // Only one byte (byte 3) that is correspond to TopLeft pixel of current block is useful.
            								prefered_pred_mode,
            								pred_penalty,
            								Top_blk_Available,
            								Left_blk_Available,
            								InputSrcRow,
            								temp_sad,
            								Pred,
            								Pred_row
            							);
			Sad[tx+ty*4] = (is_active_blk ==1) ? temp_sad : Sad[tx+ty*4];
			PreferedPredMode[tx+ty*4] = (is_active_blk ==1) ? prefered_pred_mode : PreferedPredMode[tx+ty*4];
			temp_pred = (is_active_blk ==1) ? Pred : temp_pred;
			
			Intra4x4BlockTransformAndReconstruct_cu (
            											InputSrcRow,
            											Pred_row,
            											quant_add,
            											quant_shift,
														dquant_shift,
														Quant_tables,
														DQuant_tables,
            											TempQCoef0_01,
            											TempQCoef0_23,
            											TempQCoef1_01,
            											TempQCoef1_23,
            											TempQCoef2_01,
            											TempQCoef2_23,
            											TempQCoef3_01,
            											TempQCoef3_23,
            											Out_Row
            										);

			Qcoef_0_01.x = (is_active_blk==1 ? TempQCoef0_01.x : Qcoef_0_01.x);
			Qcoef_0_01.y = (is_active_blk==1 ? TempQCoef0_01.y : Qcoef_0_01.y);
			Qcoef_0_23.x = (is_active_blk==1 ? TempQCoef0_23.x : Qcoef_0_23.x);
			Qcoef_0_23.y = (is_active_blk==1 ? TempQCoef0_23.y : Qcoef_0_23.y);
        	Qcoef_1_01.x = (is_active_blk==1 ? TempQCoef1_01.x : Qcoef_1_01.x);
			Qcoef_1_01.y = (is_active_blk==1 ? TempQCoef1_01.y : Qcoef_1_01.y);
			Qcoef_1_23.x = (is_active_blk==1 ? TempQCoef1_23.x : Qcoef_1_23.x);
			Qcoef_1_23.y = (is_active_blk==1 ? TempQCoef1_23.y : Qcoef_1_23.y);
			Qcoef_2_01.x = (is_active_blk==1 ? TempQCoef2_01.x : Qcoef_2_01.x);
			Qcoef_2_01.y = (is_active_blk==1 ? TempQCoef2_01.y : Qcoef_2_01.y);
			Qcoef_2_23.x = (is_active_blk==1 ? TempQCoef2_23.x : Qcoef_2_23.x);
			Qcoef_2_23.y = (is_active_blk==1 ? TempQCoef2_23.y : Qcoef_2_23.y);
        	Qcoef_3_01.x = (is_active_blk==1 ? TempQCoef3_01.x : Qcoef_3_01.x);
			Qcoef_3_01.y = (is_active_blk==1 ? TempQCoef3_01.y : Qcoef_3_01.y);
			Qcoef_3_23.x = (is_active_blk==1 ? TempQCoef3_23.x : Qcoef_3_23.x);
			Qcoef_3_23.y = (is_active_blk==1 ? TempQCoef3_23.y : Qcoef_3_23.y);
    
			Rec_row[0].x = (is_active_blk==1 ? Out_Row[0].x : Rec_row[0].x);
			Rec_row[0].y = (is_active_blk==1 ? Out_Row[0].y : Rec_row[0].y);
			Rec_row[0].z = (is_active_blk==1 ? Out_Row[0].z : Rec_row[0].z);
			Rec_row[0].w = (is_active_blk==1 ? Out_Row[0].w : Rec_row[0].w);
			Rec_row[1].x = (is_active_blk==1 ? Out_Row[1].x : Rec_row[1].x);
			Rec_row[1].y = (is_active_blk==1 ? Out_Row[1].y : Rec_row[1].y);
			Rec_row[1].z = (is_active_blk==1 ? Out_Row[1].z : Rec_row[1].z);
			Rec_row[1].w = (is_active_blk==1 ? Out_Row[1].w : Rec_row[1].w);
			Rec_row[2].x = (is_active_blk==1 ? Out_Row[2].x : Rec_row[2].x);
			Rec_row[2].y = (is_active_blk==1 ? Out_Row[2].y : Rec_row[2].y);
			Rec_row[2].z = (is_active_blk==1 ? Out_Row[2].z : Rec_row[2].z);
			Rec_row[2].w = (is_active_blk==1 ? Out_Row[2].w : Rec_row[2].w);
			Rec_row[3].x = (is_active_blk==1 ? Out_Row[3].x : Rec_row[3].x);
			Rec_row[3].y = (is_active_blk==1 ? Out_Row[3].y : Rec_row[3].y);
			Rec_row[3].z = (is_active_blk==1 ? Out_Row[3].z : Rec_row[3].z);
			Rec_row[3].w = (is_active_blk==1 ? Out_Row[3].w : Rec_row[3].w);
			
      
			//保存重建块最右边一列像素及其预测方式
			blk_Right_Col[tid].x = (is_active_blk ? Out_Row[0].w : blk_Right_Col[tid].x);
			blk_Right_Col[tid].y = (is_active_blk ? Out_Row[1].w : blk_Right_Col[tid].y);
			blk_Right_Col[tid].z = (is_active_blk ? Out_Row[2].w : blk_Right_Col[tid].z);
			blk_Right_Col[tid].w = (is_active_blk ? Out_Row[3].w : blk_Right_Col[tid].w);
			Left_blk_type[tid] = (is_active_blk ? temp_pred : Left_blk_type[tid]);
        
			//保存重建块最下边一行像素及其预测方式
			blk_Botton_Row[tid].x = (is_active_blk ? Out_Row[3].x : blk_Botton_Row[tid].x);
			blk_Botton_Row[tid].y = (is_active_blk ? Out_Row[3].y : blk_Botton_Row[tid].y);
			blk_Botton_Row[tid].z = (is_active_blk ? Out_Row[3].z : blk_Botton_Row[tid].z);
			blk_Botton_Row[tid].w = (is_active_blk ? Out_Row[3].w : blk_Botton_Row[tid].w);
			Top_blk_type[tid] = (is_active_blk ? temp_pred : Top_blk_type[tid]);
        	//__syncthreads();  //同步处理
        	current_stage = (current_stage + 1);
        	num_stages = (num_stages - 1);   // Needed...as loop_count would have decremented automatically!
	}
    temp_sad = 0;
	for(int i = 0; i < 16 ;i++)
	{
	  	temp_sad += Sad[i];
	}

	MinSad = (temp_sad + 6*pred_penalty);
	//MinSad = Sad[tid];
	PredMode = temp_pred;
    
}
