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

inline __device__ unsigned int CalcSad4x4_cu(unsigned char *in,uchar4 &Pred_row0,uchar4 &Pred_row1,uchar4 &Pred_row2,uchar4 &Pred_row3);

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
						);

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
											);

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
														);

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
												int &PredMode,
												int &tx,
												int &ty);
