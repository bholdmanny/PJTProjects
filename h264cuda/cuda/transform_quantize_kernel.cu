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

#include "../inc/mb_info.h"
#include "../inc/const_defines.h"
#include "../inc/residual_coding.h"
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
                                )
{
    short2 TempRow0_01,TempRow0_23,TempRow1_01,TempRow1_23,
		   TempRow2_01,TempRow2_23,TempRow3_01,TempRow3_23;
    short Sum0,Sum1,Diff0,Diff1;
	int Sign;
	// Vertical butterfly
   //first col
   Sum0 = (DiffRow0_01.x + DiffRow3_01.x);
   Sum1 = (DiffRow1_01.x + DiffRow2_01.x);
   Diff0 = (DiffRow0_01.x - DiffRow3_01.x);
   Diff1 = (DiffRow1_01.x - DiffRow2_01.x);
   TempRow0_01.x = Sum0 + Sum1;
   TempRow1_01.x = (2*Diff0) + Diff1;
   TempRow2_01.x = Sum0 - Sum1;
   TempRow3_01.x = Diff0 - (2*Diff1);

   //second col
   Sum0 = DiffRow0_01.y + DiffRow3_01.y;
   Sum1 = DiffRow1_01.y + DiffRow2_01.y;
   Diff0 = DiffRow0_01.y - DiffRow3_01.y;
   Diff1 = DiffRow1_01.y - DiffRow2_01.y;
   TempRow0_01.y = Sum0 + Sum1;
   TempRow1_01.y = (2*Diff0) + Diff1;
   TempRow2_01.y = Sum0 - Sum1;
   TempRow3_01.y = Diff0 - (2*Diff1);
    
   Sum0 = DiffRow0_23.x + DiffRow3_23.x;
   Sum1 = DiffRow1_23.x + DiffRow2_23.x;
   Diff0 = DiffRow0_23.x - DiffRow3_23.x;
   Diff1 = DiffRow1_23.x - DiffRow2_23.x;
   TempRow0_23.x = Sum0 + Sum1;
   TempRow1_23.x = (2*Diff0) + Diff1;
   TempRow2_23.x = Sum0 - Sum1;
   TempRow3_23.x = Diff0 - (2 * Diff1);

   Sum0 = DiffRow0_23.y + DiffRow3_23.y;
   Sum1 = DiffRow1_23.y + DiffRow2_23.y;
   Diff0 = DiffRow0_23.y - DiffRow3_23.y;
   Diff1 = DiffRow1_23.y - DiffRow2_23.y;
   TempRow0_23.y = Sum0 + Sum1;
   TempRow1_23.y = (2 * Diff0) + Diff1;
   TempRow2_23.y = Sum0 - Sum1;
   TempRow3_23.y = Diff0 - (2 * Diff1);

    //  Horizontal butterfly
    Sum0 = TempRow0_01.x + TempRow0_23.y;
    Sum1 = TempRow0_01.y + TempRow0_23.x;
    Diff0 = TempRow0_01.x - TempRow0_23.y;
    Diff1 = TempRow0_01.y - TempRow0_23.x;
	DiffRow0_01.x = Sum0 + Sum1;
    DiffRow0_23.x = Sum0 - Sum1;
    DiffRow0_01.y = (2 * Diff0) + Diff1;
    DiffRow0_23.y = Diff0 - (2 * Diff1);

	Sum0 = TempRow1_01.x + TempRow1_23.y;
    Sum1 = TempRow1_01.y + TempRow1_23.x;
    Diff0 = TempRow1_01.x - TempRow1_23.y;
    Diff1 = TempRow1_01.y - TempRow1_23.x;
	DiffRow1_01.x = Sum0 + Sum1;
    DiffRow1_23.x = Sum0 - Sum1;
    DiffRow1_01.y = (2 * Diff0) + Diff1;
    DiffRow1_23.y = Diff0 - (2 * Diff1);
    
	Sum0 = TempRow2_01.x + TempRow2_23.y;
    Sum1 = TempRow2_01.y + TempRow2_23.x;
    Diff0 = TempRow2_01.x - TempRow2_23.y;
    Diff1 = TempRow2_01.y - TempRow2_23.x;
	DiffRow2_01.x = Sum0 + Sum1;
    DiffRow2_23.x = Sum0 - Sum1;
    DiffRow2_01.y = (2 * Diff0) + Diff1;
    DiffRow2_23.y = Diff0 - (2 * Diff1);

	Sum0 = TempRow3_01.x + TempRow3_23.y;
    Sum1 = TempRow3_01.y + TempRow3_23.x;
    Diff0 = TempRow3_01.x - TempRow3_23.y;
    Diff1 = TempRow3_01.y - TempRow3_23.x;
	DiffRow3_01.x = Sum0 + Sum1;
    DiffRow3_23.x = Sum0 - Sum1;
    DiffRow3_01.y = (2 * Diff0) + Diff1;
    DiffRow3_23.y = Diff0 - (2 * Diff1);

	Coef00 = DiffRow0_01.x;  //输出直流系数

    Sign  = (DiffRow0_01.x >= 0) ? 1 : -1;
    DiffRow0_01.x = (DiffRow0_01.x >= 0) ? DiffRow0_01.x : (-DiffRow0_01.x) ;
	QCoef0_01.x = (short)(Sign * ((DiffRow0_01.x * Quant_tables[0] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow0_01.y >= 0) ? 1 : -1;
    DiffRow0_01.y = (DiffRow0_01.y >= 0) ? DiffRow0_01.y : (-DiffRow0_01.y) ;
    QCoef0_01.y =(short)(Sign * ((DiffRow0_01.y * Quant_tables[1] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow0_23.x >= 0) ? 1 : -1;
    DiffRow0_23.x   = (DiffRow0_23.x >= 0) ? DiffRow0_23.x : (-DiffRow0_23.x) ;
    QCoef0_23.x = (short)(Sign * ((DiffRow0_23.x * Quant_tables[2] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow0_23.y >= 0) ? 1 : -1;
    DiffRow0_23.y = (DiffRow0_23.y >= 0) ? DiffRow0_23.y : (-DiffRow0_23.y) ;
    QCoef0_23.y =(short) (Sign * ((DiffRow0_23.y  * Quant_tables[3] + QuantAdd) >> QuantShift));
    
	//second row
	Sign  = (DiffRow1_01.x >= 0) ? 1 : -1;
    DiffRow1_01.x = (DiffRow1_01.x >= 0) ? DiffRow1_01.x : (-DiffRow1_01.x) ;
    QCoef1_01.x = (short)(Sign * ((DiffRow1_01.x * Quant_tables[4] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow1_01.y >= 0) ? 1 : -1;
    DiffRow1_01.y = (DiffRow1_01.y >= 0) ? DiffRow1_01.y : (-DiffRow1_01.y) ;
    QCoef1_01.y = (short)(Sign * ((DiffRow1_01.y * Quant_tables[5] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow1_23.x >= 0) ? 1 : -1;
    DiffRow1_23.x = (DiffRow1_23.x >= 0) ? DiffRow1_23.x : (-DiffRow1_23.x) ;
    QCoef1_23.x = (short)(Sign * ((DiffRow1_23.x * Quant_tables[6] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow1_23.y >= 0) ? 1 : -1;
    DiffRow1_23.y = (DiffRow1_23.y >= 0) ? DiffRow1_23.y : (-DiffRow1_23.y) ;
    QCoef1_23.y = (short)(Sign * ((DiffRow1_23.y * Quant_tables[7] + QuantAdd) >> QuantShift));

   //third row
    Sign  = (DiffRow2_01.x >= 0) ? 1 : -1;
    DiffRow2_01.x = (DiffRow2_01.x >= 0) ? DiffRow2_01.x : (-DiffRow2_01.x) ;
    QCoef2_01.x = (short)(Sign * ((DiffRow2_01.x * Quant_tables[8] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow2_01.y >= 0) ? 1 : -1;
    DiffRow2_01.y = (DiffRow2_01.y >= 0) ? DiffRow2_01.y : (-DiffRow2_01.y) ;
    QCoef2_01.y = (short)(Sign * ((DiffRow2_01.y * Quant_tables[9] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow2_23.x >= 0) ? 1 : -1;
    DiffRow2_23.x = (DiffRow2_23.x >= 0) ? DiffRow2_23.x : (-DiffRow2_23.x) ;
    QCoef2_23.x = (short)(Sign * ((DiffRow2_23.x * Quant_tables[10] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow2_23.y >= 0) ? 1 : -1;
    DiffRow2_23.y = (DiffRow2_23.y >= 0) ? DiffRow2_23.y : (-DiffRow2_23.y) ;
    QCoef2_23.y = (short)(Sign * ((DiffRow2_23.y * Quant_tables[11] + QuantAdd) >> QuantShift));
    
	//forth row
	Sign  = (DiffRow3_01.x >= 0) ? 1 : -1;
    DiffRow3_01.x = (DiffRow3_01.x >= 0) ? DiffRow3_01.x : (-DiffRow3_01.x) ;
    QCoef3_01.x = (short)(Sign * ((DiffRow3_01.x * Quant_tables[12] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow3_01.y >= 0) ? 1 : -1;
    DiffRow3_01.y = (DiffRow3_01.y >= 0) ? DiffRow3_01.y : (-DiffRow3_01.y) ;
    QCoef3_01.y = (short)(Sign * ((DiffRow3_01.y * Quant_tables[13] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow3_23.x >= 0) ? 1 : -1;
    DiffRow3_23.x = (DiffRow3_23.x >= 0) ? DiffRow3_23.x : (-DiffRow3_23.x) ;
    QCoef3_23.x = (short)(Sign * ((DiffRow3_23.x * Quant_tables[14] + QuantAdd) >> QuantShift));
    
	Sign  = (DiffRow3_23.y >= 0) ? 1 : -1;
    DiffRow3_23.y = (DiffRow3_23.y >= 0) ? DiffRow3_23.y : (-DiffRow3_23.y) ;
    QCoef3_23.y = (short)(Sign * ((DiffRow3_23.y * Quant_tables[15] + QuantAdd) >> QuantShift));

}

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
                                  )
{
   short2	TempRow0_01,TempRow0_23,TempRow1_01,TempRow1_23,
			TempRow2_01,TempRow2_23,TempRow3_01,TempRow3_23;

   short	Sum0,Sum1,Diff0,Diff1;

   int  Rec00, Rec01, Rec02, Rec03,
        Rec10, Rec11, Rec12, Rec13,
        Rec20, Rec21, Rec22, Rec23,
        Rec30, Rec31, Rec32, Rec33;
	//PredRow0,PredRow1,PredRow2,PredRow3;
   //Horizontal butterfly
        Sum0  = Coef0_01.x + Coef0_23.x;
        Diff0 = Coef0_01.x - Coef0_23.x;
        Diff1 = (Coef0_01.y >> 1) - Coef0_23.y;
        Sum1  = Coef0_01.y + (Coef0_23.y >> 1);

        TempRow0_01.x = Sum0  + Sum1;
        TempRow0_01.y = Diff0 + Diff1;
        TempRow0_23.x = Diff0 - Diff1;
        TempRow0_23.y = Sum0  - Sum1;

		Sum0  = Coef1_01.x + Coef1_23.x;
        Diff0 = Coef1_01.x - Coef1_23.x;
        Diff1 = (Coef1_01.y >> 1) - Coef1_23.y;
        Sum1  = Coef1_01.y + (Coef1_23.y >> 1);

        TempRow1_01.x = Sum0  + Sum1;
        TempRow1_01.y = Diff0 + Diff1;
        TempRow1_23.x = Diff0 - Diff1;
        TempRow1_23.y = Sum0  - Sum1;

		Sum0  = Coef2_01.x + Coef2_23.x;
        Diff0 = Coef2_01.x - Coef2_23.x;
        Diff1 = (Coef2_01.y >> 1) - Coef2_23.y;
        Sum1  = Coef2_01.y + (Coef2_23.y >> 1);

        TempRow2_01.x = Sum0  + Sum1;
        TempRow2_01.y = Diff0 + Diff1;
        TempRow2_23.x = Diff0 - Diff1;
        TempRow2_23.y = Sum0  - Sum1;

		Sum0  = Coef3_01.x + Coef3_23.x;
        Diff0 = Coef3_01.x - Coef3_23.x;
        Diff1 = (Coef3_01.y >> 1) - Coef3_23.y;
        Sum1  = Coef3_01.y + (Coef3_23.y >> 1);

        TempRow3_01.x = Sum0  + Sum1;
        TempRow3_01.y = Diff0 + Diff1;
        TempRow3_23.x = Diff0 - Diff1;
        TempRow3_23.y = Sum0  - Sum1;

	//vertical butterfly
        Sum0 = TempRow0_01.x + TempRow2_01.x;
        Sum1 = TempRow1_01.x + (TempRow3_01.x >> 1);
        Diff0 = TempRow0_01.x - TempRow2_01.x;
        Diff1 = (TempRow1_01.x >> 1) - TempRow3_01.x;
	    Rec00 = (Sum0 + Sum1 + 32) >> 6;
        Rec10 = (Diff0 + Diff1 + 32) >> 6;
        Rec20 = (Diff0 - Diff1 + 32) >> 6;
        Rec30 = (Sum0 - Sum1 + 32 ) >> 6;
        
		//second col
		Sum0 = TempRow0_01.y + TempRow2_01.y;
        Sum1 = TempRow1_01.y + (TempRow3_01.y >> 1);
        Diff0 = TempRow0_01.y - TempRow2_01.y;
        Diff1 = (TempRow1_01.y >> 1) - TempRow3_01.y;
	    Rec01 = (Sum0 + Sum1 + 32) >> 6;
        Rec11 = (Diff0 + Diff1 + 32) >> 6;
        Rec21 = (Diff0 - Diff1 + 32) >> 6;
        Rec31 = (Sum0 - Sum1 + 32 ) >> 6;

		Sum0 = TempRow0_23.x + TempRow2_23.x;
        Sum1 = TempRow1_23.x + (TempRow3_23.x >> 1);
        Diff0 = TempRow0_23.x - TempRow2_23.x;
        Diff1 = (TempRow1_23.x >> 1) - TempRow3_23.x;
	    Rec02 = (Sum0 + Sum1 + 32) >> 6;
        Rec12 = (Diff0 + Diff1 + 32) >> 6;
        Rec22 = (Diff0 - Diff1 + 32) >> 6;
        Rec32 = (Sum0 - Sum1 + 32 ) >> 6;

		Sum0 = TempRow0_23.y + TempRow2_23.y;
        Sum1 = TempRow1_23.y + (TempRow3_23.y >> 1);
        Diff0 = TempRow0_23.y - TempRow2_23.y;
        Diff1 = (TempRow1_23.y >> 1) - TempRow3_23.y;
	    Rec03 = (Sum0 + Sum1 + 32) >> 6;
        Rec13 = (Diff0 + Diff1 + 32) >> 6;
        Rec23 = (Diff0 - Diff1 + 32) >> 6;
        Rec33 = (Sum0 - Sum1 + 32 ) >> 6;

        Rec00 = (Rec00 + (int)PredRow[0].x);
		Rec01 = (Rec01 + (int)PredRow[0].y);
		Rec02 = (Rec02 + (int)PredRow[0].z);
		Rec03 = (Rec03 + (int)PredRow[0].w);

		Rec10 = (Rec10 + (int)PredRow[1].x);
		Rec11 = (Rec11 + (int)PredRow[1].y);
		Rec12 = (Rec12 + (int)PredRow[1].z);
		Rec13 = (Rec13 + (int)PredRow[1].w);

		Rec20 = (Rec20 + (int)PredRow[2].x);
		Rec21 = (Rec21 + (int)PredRow[2].y);
		Rec22 = (Rec22 + (int)PredRow[2].z);
		Rec23 = (Rec23 + (int)PredRow[2].w);

		Rec30 = (Rec30 + (int)PredRow[3].x);
		Rec31 = (Rec31 + (int)PredRow[3].y);
		Rec32 = (Rec32 + (int)PredRow[3].z);
		Rec33 = (Rec33 + (int)PredRow[3].w);

		RecOutRow0.x = (unsigned char)(Rec00 < 0 ? 0 :((Rec00 > 255) ? 255 : Rec00));
		RecOutRow0.y = (unsigned char)(Rec01 < 0 ? 0 :((Rec01 > 255) ? 255 : Rec01));
		RecOutRow0.z = (unsigned char)(Rec02 < 0 ? 0 :((Rec02 > 255) ? 255 : Rec02));
		RecOutRow0.w = (unsigned char)(Rec03 < 0 ? 0 :((Rec03 > 255) ? 255 : Rec03));

		RecOutRow1.x = (unsigned char)(Rec10 < 0 ? 0 :((Rec10 > 255) ? 255 : Rec10));
		RecOutRow1.y = (unsigned char)(Rec11 < 0 ? 0 :((Rec11 > 255) ? 255 : Rec11));
		RecOutRow1.z = (unsigned char)(Rec12 < 0 ? 0 :((Rec12 > 255) ? 255 : Rec12));
		RecOutRow1.w = (unsigned char)(Rec13 < 0 ? 0 :((Rec13 > 255) ? 255 : Rec13));

		RecOutRow2.x = (unsigned char)(Rec20 < 0 ? 0 :((Rec20 > 255) ? 255 : Rec20));
		RecOutRow2.y = (unsigned char)(Rec21 < 0 ? 0 :((Rec21 > 255) ? 255 : Rec21));
		RecOutRow2.z = (unsigned char)(Rec22 < 0 ? 0 :((Rec22 > 255) ? 255 : Rec22));
		RecOutRow2.w = (unsigned char)(Rec23 < 0 ? 0 :((Rec23 > 255) ? 255 : Rec23));

		RecOutRow3.x = (unsigned char)(Rec30 < 0 ? 0 :((Rec30 > 255) ? 255 : Rec30));
		RecOutRow3.y = (unsigned char)(Rec31 < 0 ? 0 :((Rec31 > 255) ? 255 : Rec31));
		RecOutRow3.z = (unsigned char)(Rec32 < 0 ? 0 :((Rec32 > 255) ? 255 : Rec32));
		RecOutRow3.w = (unsigned char)(Rec33 < 0 ? 0 :((Rec33 > 255) ? 255 : Rec33));

}

inline __device__ void HadamardTransformAndQuantize_cu(      
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

			QCoefDC[tid*4] = (Sum0 + Sum1) >> 1;
			QCoefDC[tid*4+1]  = (Diff0 + Diff1) >> 1;
			QCoefDC[tid*4+2]  = (Sum0 - Sum1) >> 1;
			QCoefDC[tid*4+3]  = (Diff0 - Diff1) >> 1;
			

		}       
		__syncthreads();
        QuantDC = QuantAddHadamard * 2;
        QuantDCShift = QuantShift + 1;
		
        Sign  = (QCoefDC[tid] >= 0) ? 1 : -1;
        TempCoef  = (QCoefDC[tid] >= 0) ? QCoefDC[tid] : -(QCoefDC[tid]);
        TempCoef  = min (((TempCoef * QuantTable0 + QuantDC) >> QuantDCShift), CAVLC_LEVEL_LIMIT);
        QCoefDC[tid] = Sign * TempCoef;
}

//哈达码逆变换
__device__ void  InverseHadamardTransform_cu (
												short *QCoefDC,
												short &QuantTable0,
												int &DQuantShift,
												short *TempDC,
												int &tx,
												int &ty
											)
{
		short Sum0,Sum1,Diff0,Diff1;
		int tid;;

		tid = tx + ty * 4;
    // Horizontal butterfly
		if(ty == 0)
		{
			Sum0  = QCoefDC[tid*4] + QCoefDC[tid*4+2];
			Sum1  = QCoefDC[tid*4+1] + QCoefDC[tid*4+3];
			Diff0 = QCoefDC[tid*4] - QCoefDC[tid*4+2];
			Diff1 = QCoefDC[tid*4+1] - QCoefDC[tid*4+3];
        
			TempDC[tid*4] = Sum0 + Sum1;
			TempDC[tid*4+1] = Diff0 + Diff1;
			TempDC[tid*4+2] = Diff0 - Diff1;
			TempDC[tid*4+3] = Sum0 - Sum1;

			Sum0  = TempDC[tid] + TempDC[tid+8];
			Sum1  = TempDC[tid+4] + TempDC[tid+12];
			Diff0 = TempDC[tid] - TempDC[tid+8];
			Diff1 = TempDC[tid+4] - TempDC[tid+12];

			TempDC[tid] = (Sum0 + Sum1) ;
			TempDC[tid+4]  = (Diff0 + Diff1);
			TempDC[tid+8]  = (Diff0 - Diff1);
			TempDC[tid+12]  = (Sum0 - Sum1) ;
			
		}
		__syncthreads();
		TempDC[tid] = (TempDC[tid] * QuantTable0);
        TempDC[tid] = (TempDC[tid] << DQuantShift);
        TempDC[tid] = (TempDC[tid] + 2) >> 2;
		
}

// 16x16预测
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
										)
{
    short2 DiffRow0_01,DiffRow0_23,DiffRow1_01,DiffRow1_23,
		   DiffRow2_01,DiffRow2_23,DiffRow3_01,DiffRow3_23;
	int tid;

	tid = tx + ty * 4;

	DiffRow0_01.x = ((short)InputSrcRow[0] - (short)PredRow[0].x);
   	DiffRow0_01.y = ((short)InputSrcRow[1] - (short)PredRow[0].y);
   	DiffRow0_23.x = ((short)InputSrcRow[2] - (short)PredRow[0].z);
   	DiffRow0_23.y = ((short)InputSrcRow[3] - (short)PredRow[0].w);

   	DiffRow1_01.x = ((short)InputSrcRow[0+MB_WIDTH] - (short)PredRow[1].x);
   	DiffRow1_01.y = ((short)InputSrcRow[1+MB_WIDTH] - (short)PredRow[1].y);
   	DiffRow1_23.x = ((short)InputSrcRow[2+MB_WIDTH] - (short)PredRow[1].z);
   	DiffRow1_23.y = ((short)InputSrcRow[3+MB_WIDTH] - (short)PredRow[1].w);
   	
   	DiffRow2_01.x = ((short)InputSrcRow[0+MB_WIDTH*2] - (short)PredRow[2].x);
   	DiffRow2_01.y = ((short)InputSrcRow[1+MB_WIDTH*2] - (short)PredRow[2].y);
   	DiffRow2_23.x = ((short)InputSrcRow[2+MB_WIDTH*2] - (short)PredRow[2].z);
   	DiffRow2_23.y = ((short)InputSrcRow[3+MB_WIDTH*2] - (short)PredRow[2].w);
   	
   	DiffRow3_01.x = ((short)InputSrcRow[0+MB_WIDTH*3] - (short)PredRow[3].x);
   	DiffRow3_01.y = ((short)InputSrcRow[1+MB_WIDTH*3] - (short)PredRow[3].y);
   	DiffRow3_23.x = ((short)InputSrcRow[2+MB_WIDTH*3] - (short)PredRow[3].z);
   	DiffRow3_23.y = ((short)InputSrcRow[3+MB_WIDTH*3] - (short)PredRow[3].w);

	Dct4x4TransformAndQuantize_cu (
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

	 //Hardamard 变换和量化，使用共享存储器存储16个直流系数，每个线程做同样的事情，冗余操作
	 
     HadamardTransformAndQuantize_cu(
										TempDC,
										QuantAddDC,                       
										QuantShift,                     
									    Quant_tab[0],
										QCoefDC,
										tx,
										ty
									);	 
	 
     InverseHadamardTransform_cu (
									QCoefDC,
									Dquant_tab[0],
									DquantShift,
									TempDC,
									tx,
									ty
								);

	   //直流系数替换
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
		 IDct4x4AndAdd_cu (
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
