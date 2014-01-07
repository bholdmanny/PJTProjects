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

__device__ short CalcTotalNonZeroCoeffs_block_kernel(
                             short *pCoeffs,
                             int BlockSize
                             )
{
    int i;
    short TotalNonZero = 0;
    for (i = 0; i < BlockSize; i++) 
	{
        if (pCoeffs[i]) 
		{ 
			TotalNonZero++; 
		}
    }
    return (TotalNonZero);
}

__device__ short CalcContextCoeffToken_kernel(
                            int nC
                            )
{
    short Context;

    if (nC == -1)     { Context = -1; }
    else if (nC <= 1) { Context =  0; }
    else if (nC <= 3) { Context =  1; }
    else if (nC <= 7) { Context =  2; }
    else              { Context =  3; }

    return (Context);
}


//由于该函数数处理的是8160个MB,每个线程处理一个MB,我们定义一个block的大小为(16,5,1),grid(8160/80,1,1);
__global__ void cavlc_texture_symbols_luma_DC_kernel(
												   short *pCoefBlksZigZag_LumaDC,
												   S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_LumaDC,
												   int *SkipBlock,
												   S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_LumaDC,
												   S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_LumaDC,
												   S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_LumaDC,
												   S_RUN_SYMBOLS_BLOCK     *pRunSymbols_LumaDC
												   )
{

		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y +  blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
		__shared__ short pCoefBlks_LumaDC[80*16];

		short Coeff,Sign,CoeffToStore;
		int InIdx,blockIndex;
		int BlockSize,TotalNonZeroCoeffs;
		int TrailingOnes;
		int	TrailingOnesSigns;
		int	TotalZeros ;
		int	SuffixLength;
		int	LevelIdx;
		int	RunIdx;
		int	RunBefore;
		int	ZerosLeft;
		int skip_block;

		blockIndex = BLK_SIZE * (blockDim.x * blockDim.y * blockIdx.x + blockIdx.y*gridDim.x*blockDim.x*blockDim.y);
		skip_block = SkipBlock[tid_grid];
		
		for (int k = 0; k < 16; k++) 
		{
			pLevelSymbols_LumaDC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].coeff = 0;
			pLevelSymbols_LumaDC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].suffix_length = -1;
			pRunSymbols_LumaDC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].run =-1;
			pRunSymbols_LumaDC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].zeros_left =-1;
			pCoefBlks_LumaDC[blockDim.x * blockDim.y*k+tid_blk] = pCoefBlksZigZag_LumaDC[blockDim.x * blockDim.y*k+tid_blk + blockIndex];
		}
		pLevelSymbolSuffixLength0_LumaDC[tid_grid].valid = 0;
		pLevelSymbolSuffixLength0_LumaDC[tid_grid].coeff = 0;
		__syncthreads();
		if(!skip_block)
		{
			BlockSize = pMBContextOut_LumaDC[tid_grid].BlockSize;
			TotalNonZeroCoeffs = (BlockSize!=0) ?  CalcTotalNonZeroCoeffs_block_kernel(pCoefBlks_LumaDC+tid_blk*16, BlockSize) : -1;
	          
			pTextureSymbols_LumaDC[tid_grid].NumCoeff = TotalNonZeroCoeffs;

			// Calculate context for coeff_token
			pTextureSymbols_LumaDC[tid_grid].ContextCoeffToken = CalcContextCoeffToken_kernel(pMBContextOut_LumaDC[tid_grid].nC);
			//pCoefBlks_LumaDC_dev[tid_grid] =  CalcContextCoeffToken_kernel(pMBContextOut_LumaDC[tid_grid].nC);

			if(BlockSize != 0)
			{
				TrailingOnes = 0;
				TrailingOnesSigns = 0;
				TotalZeros = -1;
				SuffixLength = -1;
				LevelIdx = 0;
				RunIdx = 0;
				RunBefore = -1;
				ZerosLeft = -1;

				for (int k = (BlockSize-1); k >= 0; k--) 
				{
					Coeff = pCoefBlks_LumaDC[tid_blk*16 + k];
					Sign = ((Coeff >= 0) ? 1 : -1);

					if (Coeff != 0) 
					{
						if (((Coeff == -1) || (Coeff == 1)) && (TrailingOnes < 3) && (SuffixLength == -1)) 
						{
							TrailingOnes++;
							TrailingOnesSigns = (TrailingOnesSigns << 1) + (Sign == -1);
						} 
						else 
						{
							if (SuffixLength == -1) 
							{
								if ((TotalNonZeroCoeffs > 10) && (TrailingOnes < 3)) 
								{
									SuffixLength = 1;
								}
								else 
								{
									SuffixLength = 0;
								}
							}

							if ((LevelIdx == 0) && ((TotalNonZeroCoeffs <= 3) || (TrailingOnes < 3))) 
							{
								CoeffToStore = (short)((abs(Coeff)-1)*Sign);
							} 
							else 
							{
								CoeffToStore = Coeff;
							}
							LevelIdx++;

							if (SuffixLength == 0) 
							{
								pLevelSymbolSuffixLength0_LumaDC[tid_grid].valid = -1;
								pLevelSymbolSuffixLength0_LumaDC[tid_grid].coeff = CoeffToStore;
								SuffixLength++;
							} 
							else 
							{
								pLevelSymbols_LumaDC[tid_grid*16+15-k].coeff = CoeffToStore;
								pLevelSymbols_LumaDC[tid_grid*16+15-k].suffix_length = SuffixLength;
							}

							if ((abs(Coeff) > (3 << (SuffixLength-1))) && (SuffixLength < 6)) 
							{
								SuffixLength++;
							}
						}
						if (TotalZeros == -1) 
						{
							if (TotalNonZeroCoeffs < BlockSize) 
							{
								TotalZeros = (short)(k - TotalNonZeroCoeffs + 1);  //第一个非零系数前0的个数
							}
							ZerosLeft = TotalZeros;
						}
						else
						{
							if (ZerosLeft > 0) 
							{
								int k2 = 15-k;
								pRunSymbols_LumaDC[k2+tid_grid*16].run = RunBefore;
								pRunSymbols_LumaDC[k2+tid_grid*16].zeros_left = ZerosLeft;
								RunIdx++;
								ZerosLeft = ZerosLeft - RunBefore;
							}
						}
						RunBefore = 0;
					} 
					else 
					{
						RunBefore++;
					}
				}
				pTextureSymbols_LumaDC[tid_grid].NumTrailingOnes = TrailingOnes;
				pTextureSymbols_LumaDC[tid_grid].TrailingOnesSigns = TrailingOnesSigns;
				pTextureSymbols_LumaDC[tid_grid].TotalZeros = TotalZeros;
			}
			else
			{
				 pTextureSymbols_LumaDC[tid_grid].NumTrailingOnes = 0;
				 pTextureSymbols_LumaDC[tid_grid].TrailingOnesSigns = 0;
				 pTextureSymbols_LumaDC[tid_grid].TotalZeros = -1;
			}
		}
}

///由于该函数数处理的是8160个MB,每个线程处理一个MB的一个子宏块,我们定义一个block的大小为(16,8,1),grid(120/8,68,1);
__global__ void cavlc_texture_symbols_luma_AC_kernel(
												   short *pCoefBlksZigZag_LumaAC,
												   S_CAVLC_CONTEXT_BLOCK *pMBContextOut_LumaAC,
												   int *SkipBlock,
					                               
												   // Output
												   S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_LumaAC,
												   S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_LumaAC,
												   S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_LumaAC,
												   S_RUN_SYMBOLS_BLOCK     *pRunSymbols_LumaAC
												   )
{

		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y +  blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
		__shared__ short pCoefBlks_LumaAC[128*16];

		short Coeff,Sign,CoeffToStore;
		int InIdx,blockIndex;
		int BlockSize,TotalNonZeroCoeffs;
		int TrailingOnes;
		int	TrailingOnesSigns;
		int	TotalZeros ;
		int	SuffixLength;
		int	LevelIdx;
		int	RunIdx;
		int	RunBefore;
		int	ZerosLeft;
		int skip_block;

		blockIndex = BLK_SIZE * (blockDim.x * blockDim.y * blockIdx.x + blockIdx.y*gridDim.x*blockDim.x*blockDim.y);
		skip_block = SkipBlock[tid_y+blockIdx.x*blockDim.y+blockIdx.y*gridDim.x*blockDim.y];
		

		for (int k = 0; k < 16; k++) 
		{
			pLevelSymbols_LumaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].coeff = 0;
			pLevelSymbols_LumaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].suffix_length = -1;
			pRunSymbols_LumaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].run =-1;
			pRunSymbols_LumaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].zeros_left =-1;
			pCoefBlks_LumaAC[blockDim.x * blockDim.y*k+tid_blk] = pCoefBlksZigZag_LumaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex];
		}
		pLevelSymbolSuffixLength0_LumaAC[tid_grid].valid = 0;
		pLevelSymbolSuffixLength0_LumaAC[tid_grid].coeff = 0;
		__syncthreads();
		if(skip_block==0)
		{

			BlockSize = pMBContextOut_LumaAC[tid_grid].BlockSize;
			TotalNonZeroCoeffs = (BlockSize!=0) ?  CalcTotalNonZeroCoeffs_block_kernel(pCoefBlks_LumaAC+tid_blk*16, BlockSize) : -1;
	          
			pTextureSymbols_LumaAC[tid_grid].NumCoeff = TotalNonZeroCoeffs;

			// Calculate context for coeff_token
			pTextureSymbols_LumaAC[tid_grid].ContextCoeffToken = CalcContextCoeffToken_kernel(pMBContextOut_LumaAC[tid_grid].nC);
			//pCoefBlks_LumaAC_dev[tid_grid] =  CalcContextCoeffToken_kernel(pMBContextOut_LumaAC[tid_grid].nC);

			if(BlockSize != 0)
			{
				TrailingOnes = 0;
				TrailingOnesSigns = 0;
				// This is -1 until the first non-zero coefficient is reached
				TotalZeros = -1;
				// This is -1 until the first non-trailing-one coefficient is reached
				SuffixLength = -1;
				LevelIdx = 0;
				RunIdx = 0;
				RunBefore = -1;
				ZerosLeft = -1;

				// Traverse coefficients in revese order
				for (int k = 15; k >= 0; k--) 
				{
					// Get this coefficient
					Coeff = pCoefBlks_LumaAC[tid_blk*16 + k];
					Sign = ((Coeff >= 0) ? 1 : -1);

					if (Coeff != 0) 
					{
						// Do this main clause if we have a non-zero coefficient

						if (((Coeff == -1) || (Coeff == 1)) && (TrailingOnes < 3) && (SuffixLength == -1)) 
						{
							// We have a trailing-one if: abs(Coeff)==1, if we haven't see three trailing ones
							// already, and if we haven't seen a coefficient > 1 already.
							TrailingOnes++;
							TrailingOnesSigns = (TrailingOnesSigns << 1) + (Sign == -1);
						} 
						else 
						{
							// Otherwise, we have a "level" coefficient
							if (SuffixLength == -1) 
							{
								// If this is the first level we have seen, then initialize SuffixLength
								if ((TotalNonZeroCoeffs > 10) && (TrailingOnes < 3)) 
								{
									SuffixLength = 1;
								}
								else 
								{
									SuffixLength = 0;
								}
							}

							// The first non-zero non-trailing-ones coefficient gets special handling
							if ((LevelIdx == 0) && ((TotalNonZeroCoeffs <= 3) || (TrailingOnes < 3))) 
							{
								CoeffToStore = (short)((abs(Coeff)-1)*Sign);
							} 
							else 
							{
								CoeffToStore = Coeff;
							}
							LevelIdx++;

							// If SuffixLength is zero, store for special handling in the next kernel
							if (SuffixLength == 0) 
							{
								pLevelSymbolSuffixLength0_LumaAC[tid_grid].valid = -1;
								pLevelSymbolSuffixLength0_LumaAC[tid_grid].coeff = CoeffToStore;
								SuffixLength++;
							} 
							else 
							{
								pLevelSymbols_LumaAC[tid_grid*16+15-k].coeff = CoeffToStore;
								pLevelSymbols_LumaAC[tid_grid*16+15-k].suffix_length = SuffixLength;
							}

							if ((abs(Coeff) > (3 << (SuffixLength-1))) && (SuffixLength < 6)) 
							{
								// If we exceeded the threshold, update context
								SuffixLength++;
							}
						}
						if (TotalZeros == -1) 
						{
							// If this is the first non-zero coefficient we've seen (either trailing-one or level), then
							// calculate TotalZeros and initialize context associated with RunBefore (i.e., ZerosLeft)
							if (TotalNonZeroCoeffs < BlockSize) 
							{
								// TotalZeros is only encoded if the
								// number of non-zero coefficients is less
								// than the maximum possible blocksize.
								TotalZeros = (short)(k - TotalNonZeroCoeffs + 1);  //第一个非零系数前0的个数
							}
							ZerosLeft = TotalZeros;
						}
						else
						{
							// Otherwise, if this is not the first level we've seen, collect the number of zeros
							// between this non-zero coefficient and the previous non-zero coefficient and store
							// that as RunBefore.

							if (ZerosLeft > 0) 
							{
								int k2 = 15-k;
								pRunSymbols_LumaAC[k2+tid_grid*16].run = RunBefore;
								pRunSymbols_LumaAC[k2+tid_grid*16].zeros_left = ZerosLeft;
								//set_run_symbol(pRunSymbols[i][k2/4].run, j, k2, RunBefore);
							   // set_run_symbol(pRunSymbols[i][k2/4].zeros_left, j, k2, ZerosLeft);
								RunIdx++;
								ZerosLeft = ZerosLeft - RunBefore;
							}
						}
						RunBefore = 0;
					} 
					else 
					{
						RunBefore++;
					}
				}
				pTextureSymbols_LumaAC[tid_grid].NumTrailingOnes = TrailingOnes;
				pTextureSymbols_LumaAC[tid_grid].TrailingOnesSigns = TrailingOnesSigns;
				pTextureSymbols_LumaAC[tid_grid].TotalZeros = TotalZeros;
			}
			else
			{

				 pTextureSymbols_LumaAC[tid_grid].NumTrailingOnes = 0;
				 pTextureSymbols_LumaAC[tid_grid].TrailingOnesSigns = 0;
				 pTextureSymbols_LumaAC[tid_grid].TotalZeros = -1;
			}
		}
}

//由于该函数数处理的是8160个MB,每个线程处理一个MB的一种分量,我们定义一个block的大小为(16,5,2),grid(8160/80,1,1);
__global__ void cavlc_texture_symbols_chroma_DC_kernel(
												   short *pCoefBlksZigZag_ChromaDC,
												   S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_ChromaDC,
												   int *SkipBlock,
					                               
												   // Output
												   S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_ChromaDC,
												   S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_ChromaDC,
												   S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_ChromaDC,
												   S_RUN_SYMBOLS_BLOCK     *pRunSymbols_ChromaDC,
												   int num_mbs
												   )
{

		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		int tid_z = threadIdx.z;
		int tid_component = tid_x + tid_y*blockDim.x;
		int tid_blk = tid_x + tid_y*blockDim.x+tid_z*blockDim.y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y*blockDim.z +  blockIdx.y*gridDim.x*blockDim.x*blockDim.y*blockDim.z;
		__shared__ short pCoefBlks_ChromaDC[4*80*2];

		short Coeff,Sign,CoeffToStore;
		int InIdx,blockIndex;
		int BlockSize,TotalNonZeroCoeffs;
		int TrailingOnes;
		int	TrailingOnesSigns;
		int	TotalZeros ;
		int	SuffixLength;
		int	LevelIdx;
		int	RunIdx;
		int	RunBefore;
		int	ZerosLeft;
		int My_luma_MB = blockIdx.x*blockDim.x*blockDim.y +  blockIdx.y*gridDim.x*blockDim.x*blockDim.y*blockDim.z + (tid_blk>>1);
		int skip_block;
		skip_block = SkipBlock[My_luma_MB];

		blockIndex = BLOCKS_PER_MB_C * (blockDim.x * blockDim.y *blockDim.z* blockIdx.x + blockIdx.y*gridDim.x*blockDim.x*blockDim.y*blockDim.z)/*+tid_z*gridDim.y*gridDim.x*blockDim.x*blockDim.y*BLOCKS_PER_MB_C*/;

		for (int k = 0; k < 4; k++) 
		{
			pLevelSymbols_ChromaDC[blockDim.x * blockDim.y*blockDim.z*k+tid_blk + blockIndex].coeff = 0;
			pLevelSymbols_ChromaDC[blockDim.x * blockDim.y*blockDim.z*k+tid_blk + blockIndex].suffix_length = -1;
			pRunSymbols_ChromaDC[blockDim.x * blockDim.y*blockDim.z*k+tid_blk + blockIndex].run =-1;
			pRunSymbols_ChromaDC[blockDim.x * blockDim.y*blockDim.z*k+tid_blk + blockIndex].zeros_left =-1;
			pCoefBlks_ChromaDC[blockDim.x * blockDim.y*k +tid_component + tid_z*80*4] = pCoefBlksZigZag_ChromaDC[blockDim.x * blockDim.y*k+tid_component +(tid_z*num_mbs*4)+(blockIndex>>1)];
			//if(blockIdx.x==0)
				//pCoefBlks_ChromaDC_dev[80*k+tid_blk] = pCoefBlksZigZag_ChromaDC[blockDim.x * blockDim.y*k+tid_blk + blockIndex];

		}
		pLevelSymbolSuffixLength0_ChromaDC[tid_grid].valid = 0;
		pLevelSymbolSuffixLength0_ChromaDC[tid_grid].coeff = 0;

		__syncthreads();
		if(skip_block==0)
		{
			BlockSize = pMBContextOut_ChromaDC[My_luma_MB].BlockSize;
			TotalNonZeroCoeffs = (BlockSize!=0) ?  CalcTotalNonZeroCoeffs_block_kernel(pCoefBlks_ChromaDC+(tid_blk&1)*320+(tid_blk>>1)*4, BlockSize) : -1;
	          
			pTextureSymbols_ChromaDC[tid_grid].NumCoeff = TotalNonZeroCoeffs;

			// Calculate context for coeff_token
			pTextureSymbols_ChromaDC[tid_grid].ContextCoeffToken = CalcContextCoeffToken_kernel(pMBContextOut_ChromaDC[My_luma_MB].nC);
			//pCoefBlks_ChromaDC_dev[tid_grid] =  CalcContextCoeffToken_kernel(pMBContextOut_ChromaDC[tid_grid].nC);

			if(BlockSize != 0)
			{
				TrailingOnes = 0;
				TrailingOnesSigns = 0;
				// This is -1 until the first non-zero coefficient is reached
				TotalZeros = -1;
				// This is -1 until the first non-trailing-one coefficient is reached
				SuffixLength = -1;
				LevelIdx = 0;
				RunIdx = 0;
				RunBefore = -1;
				ZerosLeft = -1;

				// Traverse coefficients in revese order
				for (int k = (BlockSize-1); k >= 0; k--) 
				{
					// Get this coefficient
					Coeff = pCoefBlks_ChromaDC[(tid_blk&1)*320+(tid_blk>>1)*4 + k];
					Sign = ((Coeff >= 0) ? 1 : -1);

					if (Coeff != 0) 
					{
						// Do this main clause if we have a non-zero coefficient

						if (((Coeff == -1) || (Coeff == 1)) && (TrailingOnes < 3) && (SuffixLength == -1)) 
						{
							// We have a trailing-one if: abs(Coeff)==1, if we haven't see three trailing ones
							// already, and if we haven't seen a coefficient > 1 already.
							TrailingOnes++;
							TrailingOnesSigns = (TrailingOnesSigns << 1) + (Sign == -1);
						} 
						else 
						{
							// Otherwise, we have a "level" coefficient
							if (SuffixLength == -1) 
							{
								// If this is the first level we have seen, then initialize SuffixLength
								if ((TotalNonZeroCoeffs > 10) && (TrailingOnes < 3)) 
								{
									SuffixLength = 1;
								}
								else 
								{
									SuffixLength = 0;
								}
							}

							// The first non-zero non-trailing-ones coefficient gets special handling
							if ((LevelIdx == 0) && ((TotalNonZeroCoeffs <= 3) || (TrailingOnes < 3))) 
							{
								CoeffToStore = (short)((abs(Coeff)-1)*Sign);
							} 
							else 
							{
								CoeffToStore = Coeff;
							}
							LevelIdx++;

							// If SuffixLength is zero, store for special handling in the next kernel
							if (SuffixLength == 0) 
							{
								pLevelSymbolSuffixLength0_ChromaDC[tid_grid].valid = -1;
								pLevelSymbolSuffixLength0_ChromaDC[tid_grid].coeff = CoeffToStore;
								SuffixLength++;
							} 
							else 
							{
								pLevelSymbols_ChromaDC[tid_grid*4+3-k].coeff = CoeffToStore;
								pLevelSymbols_ChromaDC[tid_grid*4+3-k].suffix_length = SuffixLength;
							}

							if ((abs(Coeff) > (3 << (SuffixLength-1))) && (SuffixLength < 6)) 
							{
								// If we exceeded the threshold, update context
								SuffixLength++;
							}
						}
						if (TotalZeros == -1) 
						{
							// If this is the first non-zero coefficient we've seen (either trailing-one or level), then
							// calculate TotalZeros and initialize context associated with RunBefore (i.e., ZerosLeft)
							if (TotalNonZeroCoeffs < BlockSize) 
							{
								// TotalZeros is only encoded if the
								// number of non-zero coefficients is less
								// than the maximum possible blocksize.
								TotalZeros = (short)(k - TotalNonZeroCoeffs + 1);  //第一个非零系数前0的个数
							}
							ZerosLeft = TotalZeros;
						}
						else
						{
							// Otherwise, if this is not the first level we've seen, collect the number of zeros
							// between this non-zero coefficient and the previous non-zero coefficient and store
							// that as RunBefore.

							if (ZerosLeft > 0) 
							{
								int k2 = 3-k;
								pRunSymbols_ChromaDC[k2+tid_grid*4].run = RunBefore;
								pRunSymbols_ChromaDC[k2+tid_grid*4].zeros_left = ZerosLeft;
								//set_run_symbol(pRunSymbols[i][k2/4].run, j, k2, RunBefore);
							   // set_run_symbol(pRunSymbols[i][k2/4].zeros_left, j, k2, ZerosLeft);
								RunIdx++;
								ZerosLeft = ZerosLeft - RunBefore;
							}
						}
						RunBefore = 0;
					} 
					else 
					{
						RunBefore++;
					}
				}
				pTextureSymbols_ChromaDC[tid_grid].NumTrailingOnes = TrailingOnes;
				pTextureSymbols_ChromaDC[tid_grid].TrailingOnesSigns = TrailingOnesSigns;
				pTextureSymbols_ChromaDC[tid_grid].TotalZeros = TotalZeros;
			}
			else
			{

				 pTextureSymbols_ChromaDC[tid_grid].NumTrailingOnes = 0;
				 pTextureSymbols_ChromaDC[tid_grid].TrailingOnesSigns = 0;
				 pTextureSymbols_ChromaDC[tid_grid].TotalZeros = -1;
			}
		}
}
//由于该函数数处理的是8160个MB,每个线程处理一个MB,我们定义一个block的大小为(16,5,1),grid(8160/80,1,1);
__global__ void cavlc_texture_symbols_chroma_AC_kernel(
												   short *pCoefBlksZigZag_ChromaAC,
												   S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_ChromaAC,
												   int *SkipBlock,
					                               
												   // Output
												   S_TEXTURE_SYMBOLS_BLOCK *pTextureSymbols_ChromaAC,
												   S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK *pLevelSymbolSuffixLength0_ChromaAC,
												   S_LEVEL_SYMBOLS_BLOCK   *pLevelSymbols_ChromaAC,
												   S_RUN_SYMBOLS_BLOCK     *pRunSymbols_ChromaAC
												   )
{

		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y +  blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
		__shared__ short pCoefBlks_ChromaAC[80*16];

		short Coeff,Sign,CoeffToStore;
		int InIdx,blockIndex;
		int BlockSize,TotalNonZeroCoeffs;
		int TrailingOnes;
		int	TrailingOnesSigns;
		int	TotalZeros ;
		int	SuffixLength;
		int	LevelIdx;
		int	RunIdx;
		int	RunBefore;
		int	ZerosLeft;
		int skip_block;

		blockIndex = BLK_SIZE * (blockDim.x * blockDim.y * blockIdx.x + blockIdx.y*gridDim.x*blockDim.x*blockDim.y);
		skip_block = SkipBlock[(tid_grid>>3)];
		

		for (int k = 0; k < 16; k++) 
		{
			pLevelSymbols_ChromaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].coeff = 0;
			pLevelSymbols_ChromaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].suffix_length = -1;
			pRunSymbols_ChromaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].run =-1;
			pRunSymbols_ChromaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex].zeros_left =-1;
			pCoefBlks_ChromaAC[blockDim.x * blockDim.y*k+tid_blk] = pCoefBlksZigZag_ChromaAC[blockDim.x * blockDim.y*k+tid_blk + blockIndex];
		}
		if(skip_block == 0)
		{
			pLevelSymbolSuffixLength0_ChromaAC[tid_grid].valid = 0;
			pLevelSymbolSuffixLength0_ChromaAC[tid_grid].coeff = 0;
			__syncthreads();

			BlockSize = pMBContextOut_ChromaAC[tid_grid].BlockSize;
			TotalNonZeroCoeffs = (BlockSize!=0) ?  CalcTotalNonZeroCoeffs_block_kernel(pCoefBlks_ChromaAC+tid_blk*16, BlockSize) : -1;
	          
			pTextureSymbols_ChromaAC[tid_grid].NumCoeff = TotalNonZeroCoeffs;

			// Calculate context for coeff_token
			pTextureSymbols_ChromaAC[tid_grid].ContextCoeffToken = CalcContextCoeffToken_kernel(pMBContextOut_ChromaAC[tid_grid].nC);
			//pCoefBlks_ChromaAC_dev[tid_grid] =  CalcContextCoeffToken_kernel(pMBContextOut_ChromaAC[tid_grid].nC);

			if(BlockSize != 0)
			{
				TrailingOnes = 0;
				TrailingOnesSigns = 0;
				// This is -1 until the first non-zero coefficient is reached
				TotalZeros = -1;
				// This is -1 until the first non-trailing-one coefficient is reached
				SuffixLength = -1;
				LevelIdx = 0;
				RunIdx = 0;
				RunBefore = -1;
				ZerosLeft = -1;

				// Traverse coefficients in revese order
				for (int k = (BlockSize-1); k >= 0; k--) 
				{
					// Get this coefficient
					Coeff = pCoefBlks_ChromaAC[tid_blk*16 + k];
					Sign = ((Coeff >= 0) ? 1 : -1);

					if (Coeff != 0) 
					{
						// Do this main clause if we have a non-zero coefficient

						if (((Coeff == -1) || (Coeff == 1)) && (TrailingOnes < 3) && (SuffixLength == -1)) 
						{
							// We have a trailing-one if: abs(Coeff)==1, if we haven't see three trailing ones
							// already, and if we haven't seen a coefficient > 1 already.
							TrailingOnes++;
							TrailingOnesSigns = (TrailingOnesSigns << 1) + (Sign == -1);
						} 
						else 
						{
							// Otherwise, we have a "level" coefficient
							if (SuffixLength == -1) 
							{
								// If this is the first level we have seen, then initialize SuffixLength
								if ((TotalNonZeroCoeffs > 10) && (TrailingOnes < 3)) 
								{
									SuffixLength = 1;
								}
								else 
								{
									SuffixLength = 0;
								}
							}

							// The first non-zero non-trailing-ones coefficient gets special handling
							if ((LevelIdx == 0) && ((TotalNonZeroCoeffs <= 3) || (TrailingOnes < 3))) 
							{
								CoeffToStore = (short)((abs(Coeff)-1)*Sign);
							} 
							else 
							{
								CoeffToStore = Coeff;
							}
							LevelIdx++;

							// If SuffixLength is zero, store for special handling in the next kernel
							if (SuffixLength == 0) 
							{
								pLevelSymbolSuffixLength0_ChromaAC[tid_grid].valid = -1;
								pLevelSymbolSuffixLength0_ChromaAC[tid_grid].coeff = CoeffToStore;
								SuffixLength++;
							} 
							else 
							{
								pLevelSymbols_ChromaAC[tid_grid*16+15-k].coeff = CoeffToStore;
								pLevelSymbols_ChromaAC[tid_grid*16+15-k].suffix_length = SuffixLength;
							}

							if ((abs(Coeff) > (3 << (SuffixLength-1))) && (SuffixLength < 6)) 
							{
								// If we exceeded the threshold, update context
								SuffixLength++;
							}
						}
						if (TotalZeros == -1) 
						{
							// If this is the first non-zero coefficient we've seen (either trailing-one or level), then
							// calculate TotalZeros and initialize context associated with RunBefore (i.e., ZerosLeft)
							if (TotalNonZeroCoeffs < BlockSize) 
							{
								// TotalZeros is only encoded if the
								// number of non-zero coefficients is less
								// than the maximum possible blocksize.
								TotalZeros = (short)(k - TotalNonZeroCoeffs + 1);  //第一个非零系数前0的个数
							}
							ZerosLeft = TotalZeros;
						}
						else
						{
							// Otherwise, if this is not the first level we've seen, collect the number of zeros
							// between this non-zero coefficient and the previous non-zero coefficient and store
							// that as RunBefore.

							if (ZerosLeft > 0) 
							{
								int k2 = 15-k;
								pRunSymbols_ChromaAC[k2+tid_grid*16].run = RunBefore;
								pRunSymbols_ChromaAC[k2+tid_grid*16].zeros_left = ZerosLeft;
								//set_run_symbol(pRunSymbols[i][k2/4].run, j, k2, RunBefore);
							   // set_run_symbol(pRunSymbols[i][k2/4].zeros_left, j, k2, ZerosLeft);
								RunIdx++;
								ZerosLeft = ZerosLeft - RunBefore;
							}
						}
						RunBefore = 0;
					} 
					else 
					{
						RunBefore++;
					}
				}
				pTextureSymbols_ChromaAC[tid_grid].NumTrailingOnes = TrailingOnes;
				pTextureSymbols_ChromaAC[tid_grid].TrailingOnesSigns = TrailingOnesSigns;
				pTextureSymbols_ChromaAC[tid_grid].TotalZeros = TotalZeros;
			}
			else
			{

				 pTextureSymbols_ChromaAC[tid_grid].NumTrailingOnes = 0;
				 pTextureSymbols_ChromaAC[tid_grid].TrailingOnesSigns = 0;
				 pTextureSymbols_ChromaAC[tid_grid].TotalZeros = -1;
			}
		}
}