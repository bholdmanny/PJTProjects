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

inline __device__ void Dct4x4TransformAndQuantize_cu (
                                short2 &DiffRow0_01, short2 &DiffRow0_23,       //[i]
                                short2 &DiffRow1_01, short2 &DiffRow1_23,       //[i]
                                short2 &DiffRow2_01, short2 &DiffRow2_23,       //[i]
                                short2 &DiffRow3_01, short2 &DiffRow3_23,       //[i]  
                                int &QuantAdd,                       //[i]
                                int &QuantShift,                     //[i]                     
								short *Quant_tables,
                                short2 &QCoef0_01, short2 &QCoef0_23,           //[o]
                                short2 &QCoef1_01, short2 &QCoef1_23,           //[o]
                                short2 &QCoef2_01, short2 &QCoef2_23,           //[o]
                                short2 &QCoef3_01, short2 &QCoef3_23,           //[o]
                                short &Coef00                          // dummy
                                );
inline __device__ void IDct4x4AndAdd_cu (
                                  short2 &Coef0_01,
                                  short2 &Coef0_23,
                                  short2 &Coef1_01,
                                  short2 &Coef1_23,
                                  short2 &Coef2_01,
                                  short2 &Coef2_23,
                                  short2 &Coef3_01,
                                  short2 &Coef3_23,
								  uchar4 *PredRow,
                                  uchar4 &RecOutRow0,
                                  uchar4 &RecOutRow1,
                                  uchar4 &RecOutRow2,
                                  uchar4 &RecOutRow3
                                  );

inline __device__ void HadamardTransformAndQuantize_cu(      
													short *CoefDc,
													int  &QuantAddHadamard,
													int  &QuantShift,
													short &QuantTable0,
													short *QCoefDC,
													int &tx,
													int &ty
												);
//¹þ´ïÂëÄæ±ä»»
__device__ void  InverseHadamardTransform_cu (
												short *QCoefDC,
												short &QuantTable0,
												int &DQuantShift,
												short *TempDC,
												int &tx,
												int &ty
											);

// 16x16Ô¤²â
inline __device__ void intra16x16_transforms_cu (
											unsigned char *InputSrcRow,
											uchar4 *PredRow,
											uchar4 *RecOutRow,
											int &QuantAdd,
											int &QuantAddDC,
											int &QuantShift,
											int &DquantShift,
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
											short  *QCoefDC,
											short  *TempDC,
											int &tx,
											int &ty
										);