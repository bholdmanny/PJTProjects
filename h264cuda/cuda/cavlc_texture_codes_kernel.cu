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


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cutil_inline.h>
#include <cutil_inline_runtime.h>
#include <cutil.h>

#include "../inc/mb_info.h"
#include "../inc/cavlc_data.h"
#include "../inc/const_defines.h"
__device__ SINGLE_CODE CalcRunCode_kernel(
                        char Run,
                        char ZerosLeft,
						unsigned int *RunIndexTab,
						unsigned char *RunTab
                        )
{
    SINGLE_CODE Code;
    if (Run != 0xf) 
	{
        // Look up base index into run table based on ZerosLeft
        int ZerosLeftIdx = (ZerosLeft > 7) ? 7 : ZerosLeft;
        int RunIndexBase = RunIndexTab[ZerosLeftIdx-1];
        // Look up actual run now, offset from the new base
        Code.length = RunTab[(RunIndexBase + Run)*2];
        Code.value = RunTab[(RunIndexBase + Run)*2 + 1];
    } 
	else 
	{
        Code.length = 0;
        Code.value = 0;
    }
    return (Code);
}


__global__ void cavlc_texture_codes_luma_DC_kernel(
												     S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_LumaDC,
													 S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_LuamDC,
													 S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_LumaDC,
													 S_RUN_SYMBOLS_BLOCK     *pRunSymbols_LumaDC,
													 unsigned char *CoeffTokenTable_dev,
													 unsigned char *TotalZerosTable_dev,
													 unsigned int *RunIndexTable_dev,
													 unsigned char *RunTable_dev,
													 int   *SkipBlock,
													 SINGLE_CODE  *pCodes_LumaDC,
													 int *TotalCodeBits,
													 int num_mbs
												   )
{

		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y +  blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
		__shared__ unsigned char CoeffTokenTab[3*4*17*2];
		__shared__ unsigned char TotalZerosTab[15*16*2];
		__shared__ unsigned int RunIndexTab[7];
		__shared__ unsigned char RunTab[44*2];
		__shared__ unsigned short Codes[80*CODE_PAIRS_PER_LANE];

		SINGLE_CODE Code,Code1;
		int out_idx,total_Zero;
		int grid_size = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y;
		int MB_factor = grid_size/num_mbs;
		int skip_Mb;
		
        //////////////////////////////////////////////////////////////////
		for(int i = 0;i < 5;i++)
		{
			CoeffTokenTab[tid_blk+80*i] = CoeffTokenTable_dev[tid_blk+80*i];
			TotalZerosTab[tid_blk+80*i] = TotalZerosTable_dev[tid_blk+80*i];
		}
		TotalZerosTab[tid_blk+400] = TotalZerosTable_dev[tid_blk+400] ;
		RunTab[tid_blk] = RunTable_dev[tid_blk] ;
		if(tid_blk < 8)
		{
			CoeffTokenTab[tid_blk+400] = CoeffTokenTable_dev[tid_blk+400];
			RunTab[tid_blk+80] = RunTable_dev[tid_blk+80] ;
		}
		if(tid_blk < 7)
		{
			RunIndexTab[tid_blk] = RunIndexTable_dev[tid_blk];
		}
		__syncthreads();

		skip_Mb = SkipBlock[(tid_grid/MB_factor)];
		if(skip_Mb==0)
		{
        int NumCoeff = pTextureSymbols_LumaDC[tid_grid].NumCoeff;
		int ContextCoeffToken = pTextureSymbols_LumaDC[tid_grid].ContextCoeffToken;
		
        if (NumCoeff != -1) 
		{
            // If this block is to be encoded...
            int NumTrailingOnes = pTextureSymbols_LumaDC[tid_grid].NumTrailingOnes;
			if (ContextCoeffToken == 3) 
			{ 
				Code.length = 6;
				Code.value =(NumCoeff == 0) ?  3 : (unsigned short)(4*(NumCoeff-1) + NumTrailingOnes);
			}
			else 
			{
				Code.length = CoeffTokenTab[ContextCoeffToken*136+NumTrailingOnes*34+NumCoeff*2];
				Code.value  = CoeffTokenTab[ContextCoeffToken*136+NumTrailingOnes*34+NumCoeff*2+1];
			}
			pCodes_LumaDC[COEFFTOKEN_T1SIGN_CODE_IDX+tid_grid*CODE_PAIRS_PER_LANE].length = Code.length;
			pCodes_LumaDC[COEFFTOKEN_T1SIGN_CODE_IDX+tid_grid*CODE_PAIRS_PER_LANE].value  = Code.value;

            if (NumCoeff > 0) 
			{
           
				Code1.length = (short)(NumTrailingOnes>0) ? (short)NumTrailingOnes : 0;
				Code1.value =  (short)(NumTrailingOnes>0) ? pTextureSymbols_LumaDC[tid_grid].TrailingOnesSigns : 0;
			
				Code.length = Code.length + Code1.length;
				Code.value  = Code.value << Code1.length;
				Code.value  = Code.value | Code1.value;
				pCodes_LumaDC[COEFFTOKEN_T1SIGN_CODE_IDX+tid_grid*CODE_PAIRS_PER_LANE].length = Code.length;
				pCodes_LumaDC[COEFFTOKEN_T1SIGN_CODE_IDX+tid_grid*CODE_PAIRS_PER_LANE].value  = Code.value;
				//  *TotalCodeBits += Code.length;
				// Encode levels
				for (int k = 0; k < 16; k++) 
				{
					int valid_coeff = 0;
					int Coeff;
					if ((k == 0) && pLevelSymbolSuffixLength0_LuamDC[tid_grid].valid)
					{
						valid_coeff = 1;
						Coeff = (pLevelSymbolSuffixLength0_LuamDC[tid_grid].coeff);
						int SignValue = ((Coeff >= 0) ? 0 : 1);
						Coeff = abs(Coeff);
						Code.length = (Coeff < 8) ? (short)(2*Coeff + SignValue - 1) :((Coeff < 16) ? 19 : 28);
						Code.value  = (Coeff < 8) ? 1 : ((Coeff < 16) ?(unsigned short)(0x10 | (2*(Coeff - 8)) | SignValue) :(unsigned short)(0x1000 | (2*(Coeff - 16)) | SignValue) );
					}
					else if (pLevelSymbols_LumaDC[tid_grid*16 + k].suffix_length != -1)
					{
						valid_coeff = 1;
						Coeff = pLevelSymbols_LumaDC[tid_grid*16 + k].coeff;
						int SignValue = ((Coeff >= 0) ? 0 : 1);
						Coeff = abs(Coeff);
						if(pLevelSymbols_LumaDC[tid_grid*16 + k].suffix_length== 0)
						{
							//Coeff = pLevelSymbols_LumaDC[tid_grid*16 + k].coeff;
							
							Code.length = (Coeff < 8) ? (short)(2*Coeff + SignValue - 1) :((Coeff < 16) ? 19 : 28);
							Code.value  = (Coeff < 8) ? 1 : ((Coeff < 16) ?(unsigned short)(0x10 | (2*(Coeff - 8)) | SignValue) :(unsigned short)(0x1000 | (2*(Coeff - 16)) | SignValue) );
						}
						else
						{
							int ShiftVal = pLevelSymbols_LumaDC[tid_grid*16 + k].suffix_length - 1;
							int EscapeThreshold = (15 << ShiftVal) + 1;
							int Suffix = (Coeff - 1) & (~(0xffffffff << ShiftVal));
							Code.length =(Coeff < EscapeThreshold) ? (short)(((Coeff - 1) >> ShiftVal) + ShiftVal + 2) :28;
							Code.value = (Coeff < EscapeThreshold) ? (unsigned short)((1 << (ShiftVal+1)) | (Suffix << 1) | SignValue) : (unsigned short)(0x1000 | ((Coeff - EscapeThreshold) << 1) | SignValue);
						}
						
					}
					if (valid_coeff) 
					{
						out_idx  = FIRST_LEVEL_CODE_IDX + k;
						pCodes_LumaDC[out_idx+tid_grid*CODE_PAIRS_PER_LANE].length = Code.length;
						pCodes_LumaDC[out_idx+tid_grid*CODE_PAIRS_PER_LANE].value  = Code.value;
					}
				}             
				// Encode total zeros, if necessary
				total_Zero = pTextureSymbols_LumaDC[tid_grid].TotalZeros;
				Code.length = TotalZerosTab[(NumCoeff-1)*32 + total_Zero*2];
				Code.value = TotalZerosTab[(NumCoeff-1)*32 + total_Zero*2+1];

				pCodes_LumaDC[tid_grid*CODE_PAIRS_PER_LANE+TOTALZEROS_CODE_IDX].length = Code.length;
				pCodes_LumaDC[tid_grid*CODE_PAIRS_PER_LANE+TOTALZEROS_CODE_IDX].value  = Code.value;

				 // Encode runs.  The first run (k == 0) is always
				 // invalid.  That is, it will never contribute
				 Code = CalcRunCode_kernel((pRunSymbols_LumaDC[tid_grid*16+1].run&0xf),
											 (pRunSymbols_LumaDC[tid_grid*16+1].zeros_left&0xf),RunIndexTab,RunTab);

				 pCodes_LumaDC[tid_grid*CODE_PAIRS_PER_LANE+FIRST_RUN_CODE_IDX] =  Code;

					for (int k = 2; k < 16; k +=2)
					{
						 Code = CalcRunCode_kernel((pRunSymbols_LumaDC[tid_grid*16+k].run&0xf),
											 (pRunSymbols_LumaDC[tid_grid*16+k].zeros_left&0xf),RunIndexTab,RunTab);

						 Code1 = CalcRunCode_kernel((pRunSymbols_LumaDC[tid_grid*16+k+1].run&0xf),
											 (pRunSymbols_LumaDC[tid_grid*16+k+1].zeros_left&0xf),RunIndexTab,RunTab);

						 out_idx  = FIRST_RUN_CODE_IDX + (k >> 1);
						 Code.length = Code.length + Code1.length;
						 Code.value  = Code.value << Code1.length;
						 Code.value  = Code.value | Code1.value;
						 pCodes_LumaDC[tid_grid*CODE_PAIRS_PER_LANE+out_idx] =  Code;
					}
			}
		}
		}

}
//由于该函数数处理的是8160个MB,每个线程处理一个MB,我们定义一个block的大小为(16,10,1),grid(8160/80,1,1);
//每个线程的输出数据为8个SINGLE_CODE,1+4+1+2
__global__ void cavlc_texture_codes_chroam_DC_kernel(
												     S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_ChromaDC,
													 S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_ChromaDC,
													 S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_ChromaDC,
													 S_RUN_SYMBOLS_BLOCK     *pRunSymbols_ChromaDC,
													 unsigned char *CoeffTokenChromaDCTable_dev,
													 unsigned char *TotalZerosChromaDCTable_dev,
													 unsigned int *RunIndexTable_dev,
													 unsigned char *RunTable_dev,
													 int   *SkipBlock,

													 SINGLE_CODE           *pCodes_ChromaDC,
													 int *TotalCodeBits
												   )
{
		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y +  blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
		__shared__ unsigned char CoeffTokenChromaDCTab[4*5*2];
		__shared__ unsigned char TotalZerosChromaDCTab[3*4*2];
		__shared__ unsigned int RunIndexTab[7];
		__shared__ unsigned char RunTab[44*2];
		//__shared__ unsigned short Codes[80*CODE_PAIRS_PER_LANE];

		SINGLE_CODE Code,Code1;
		int out_idx,total_Zero;
		int NumCoeff,ContextCoeffToken;
		int skip_Mb;
        //////////////////////////////////////////////////////////////////
		if(tid_blk < 40)
		{
			CoeffTokenChromaDCTab[tid_blk] = CoeffTokenChromaDCTable_dev[tid_blk];
		}
		else if(tid_blk < 64)
		{
			TotalZerosChromaDCTab[tid_blk-40] = TotalZerosChromaDCTable_dev[tid_blk-40];
		}
		else if(tid_blk < 152)
		{
			RunTab[tid_blk-64] = RunTable_dev[tid_blk-64];
		}
		else if(tid_blk < 159)
		{
			RunIndexTab[tid_blk-152] = RunIndexTable_dev[tid_blk-152];
		}
		__syncthreads();
		skip_Mb = SkipBlock[tid_grid>>1];
		if(skip_Mb==0)
		{
			NumCoeff = pTextureSymbols_ChromaDC[tid_grid].NumCoeff;
			//ContextCoeffToken = pTextureSymbols_ChromaDC[tid_grid].ContextCoeffToken;
			if (NumCoeff != -1) 
			{
					// If this block is to be encoded...
					int NumTrailingOnes = pTextureSymbols_ChromaDC[tid_grid].NumTrailingOnes;
									// Encode coeff_token and rest of block, if necessary
					Code.length = CoeffTokenChromaDCTab[NumTrailingOnes*10+NumCoeff*2];
					Code.value = CoeffTokenChromaDCTab[NumTrailingOnes*10+NumCoeff*2+1];
	       
					pCodes_ChromaDC[tid_grid*8+COEFFTOKEN_T1SIGN_CODE_IDX].length = Code.length;
					pCodes_ChromaDC[tid_grid*8+COEFFTOKEN_T1SIGN_CODE_IDX].value  = Code.value;
					if (NumCoeff > 0) 
					{
						Code1.length = (short)(NumTrailingOnes>0) ? (short)NumTrailingOnes : 0;
						Code1.value =  (short)(NumTrailingOnes>0) ? pTextureSymbols_ChromaDC[tid_grid].TrailingOnesSigns : 0;
					
						Code.length = Code.length + Code1.length;
						Code.value  = Code.value << Code1.length;
						Code.value  = Code.value | Code1.value;
						pCodes_ChromaDC[tid_grid*8+COEFFTOKEN_T1SIGN_CODE_IDX].length = Code.length;
						pCodes_ChromaDC[tid_grid*8+COEFFTOKEN_T1SIGN_CODE_IDX].value  = Code.value;
						// Encode levels
						for (int k = 0; k < 4; k++) 
						{
							int valid_coeff = 0;
							int Coeff;
							if ((k == 0) && pLevelSymbolSuffixLength0_ChromaDC[tid_grid].valid)
							{
								valid_coeff = 1;
								Coeff = (pLevelSymbolSuffixLength0_ChromaDC[tid_grid].coeff);
								int SignValue = ((Coeff >= 0) ? 0 : 1);
								Coeff = abs(Coeff);
								Code.length = (Coeff < 8) ? (short)(2*Coeff + SignValue - 1) :((Coeff < 16) ? 19 : 28);
								Code.value  = (Coeff < 8) ? 1 : ((Coeff < 16) ?(unsigned short)(0x10 | (2*(Coeff - 8)) | SignValue) :(unsigned short)(0x1000 | (2*(Coeff - 16)) | SignValue) );
							}
							else if (pLevelSymbols_ChromaDC[tid_grid*4 + k].suffix_length != -1)
							{
								valid_coeff = 1;
								Coeff = pLevelSymbols_ChromaDC[tid_grid*4 + k].coeff;
								int SignValue = ((Coeff >= 0) ? 0 : 1);
								Coeff = abs(Coeff);
								if(pLevelSymbols_ChromaDC[tid_grid*4 + k].suffix_length== 0)
								{
									//Coeff = pLevelSymbols_LumaDC[tid_grid*16 + k].coeff;
									
									Code.length = (Coeff < 8) ? (short)(2*Coeff + SignValue - 1) :((Coeff < 16) ? 19 : 28);
									Code.value  = (Coeff < 8) ? 1 : ((Coeff < 16) ?(unsigned short)(0x10 | (2*(Coeff - 8)) | SignValue) :(unsigned short)(0x1000 | (2*(Coeff - 16)) | SignValue) );
								}
								else
								{
									int ShiftVal = pLevelSymbols_ChromaDC[tid_grid*4 + k].suffix_length - 1;
									int EscapeThreshold = (15 << ShiftVal) + 1;
									int Suffix = (Coeff - 1) & (~(0xffffffff << ShiftVal));
									Code.length =(Coeff < EscapeThreshold) ? (short)(((Coeff - 1) >> ShiftVal) + ShiftVal + 2) :28;
									Code.value = (Coeff < EscapeThreshold) ? (unsigned short)((1 << (ShiftVal+1)) | (Suffix << 1) | SignValue) : (unsigned short)(0x1000 | ((Coeff - EscapeThreshold) << 1) | SignValue);
								}
							}
							if (valid_coeff) 
							{
								out_idx  = FIRST_LEVEL_CODE_IDX + k;
								pCodes_ChromaDC[out_idx+tid_grid*8].length = Code.length;
								pCodes_ChromaDC[out_idx+tid_grid*8].value  = Code.value;
							}
						}
						
						total_Zero = pTextureSymbols_ChromaDC[tid_grid].TotalZeros;
						Code.length = TotalZerosChromaDCTab[(NumCoeff-1)*8 + total_Zero*2];
						Code.value = TotalZerosChromaDCTab[(NumCoeff-1)*8 + total_Zero*2+1];

						pCodes_ChromaDC[tid_grid*8+5].length = Code.length;
						pCodes_ChromaDC[tid_grid*8+5].value  = Code.value;

						 // Encode runs.  The first run (k == 0) is always
						 // invalid.  That is, it will never contribute
						 Code = CalcRunCode_kernel((pRunSymbols_ChromaDC[tid_grid*4+1].run&0xf),
													 (pRunSymbols_ChromaDC[tid_grid*4+1].zeros_left&0xf),RunIndexTab,RunTab);

						 pCodes_ChromaDC[tid_grid*8+6] =  Code;

						 Code = CalcRunCode_kernel((pRunSymbols_ChromaDC[tid_grid*4+2].run&0xf),
													 (pRunSymbols_ChromaDC[tid_grid*4+2].zeros_left&0xf),RunIndexTab,RunTab);

						 Code1 = CalcRunCode_kernel((pRunSymbols_ChromaDC[tid_grid*4+3].run&0xf),
													 (pRunSymbols_ChromaDC[tid_grid*4+3].zeros_left&0xf),RunIndexTab,RunTab);

						 Code.length = Code.length + Code1.length;
						 Code.value  = Code.value << Code1.length;
						 Code.value  = Code.value | Code1.value;
						 pCodes_ChromaDC[tid_grid*8+7] =  Code;
					}
			}
		}
}

