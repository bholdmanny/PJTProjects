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

__device__ SINGLE_CODE exp_golomb_unsigned_kernel
											(
											 unsigned int code
											 )
{
	SINGLE_CODE s;
	short i = 32;

	while (i > 0) 
	{
		if ((code+1) & (1 << (i-1))) 
		{
			break;
		}
		i--;
	}

	s.length = (short)((i*2)-1);
	s.value = (unsigned short)(code+1);
	return (s);
}


//处理I帧所有宏块的头信息，每个宏块包括mbtype,16个subtype(四个记录),色度模式，CBP，delta_quant。kernel设计时每个线程编码一个宏块
//线程块大小为                                                                                   
__global__ void cavlc_header_codes_Iframe
							(
                            // Input context
                            S_CAVLC_CONTEXT_BLOCK *pMBContext,
							int *DecodeOrderMapping_dev,//解码顺序
							unsigned char*CBPTable_dev,
							int head_size,
							// Updated codes (input and output)
                            SINGLE_CODE          *pCodes_Header_IMB, //8 element for a I MB 1+4+1+1+1(mbtype,subtype(4*4),CHROMAMODE,CBP,delta_quant) 
                            // Scalar parameters
                            int *TotalCodeBits
                            )
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	
	int tid_blk = tid_x + tid_y*blockDim.x;
	int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y;

	int mb_type,misc;
	int CBP;
	unsigned int CBPCodeNum,temp;
	int out_idx;
	SINGLE_CODE Code,Code1;
	short i = 32;

	mb_type = pMBContext[tid_grid*BLOCKS_PER_MB].MBModes[HDR_CONTEXT_TYPE];
	misc = pMBContext[tid_grid*BLOCKS_PER_MB].Misc;
	
	if (mb_type == INTRA_LARGE_BLOCKS_MB_TYPE) 
	{
        temp = ((((misc & 15) > 0) ? 0xd : 1)
                + ((misc >> 2) & 0xc)
                + pMBContext[tid_grid*BLOCKS_PER_MB].MBModes[HDR_CONTEXT_SUBTYPE]);
    } 
	else if (mb_type == INTRA_SMALL_BLOCKS_MB_TYPE) 
	{
        temp = 0 ;
    }
	/*else 
	{
        printf("Invalid macroblock type!\n");
        exit(-1);
    }*/
    // Mb type coding
	Code = exp_golomb_unsigned_kernel(temp);
    pCodes_Header_IMB[tid_grid*head_size].length = Code.length;
    pCodes_Header_IMB[tid_grid*head_size].value  = Code.value;
    TotalCodeBits[tid_grid] += Code.length;
	Code.length = 0;
	Code.value = 0;

    ////////////////////////////////////////////////////////
    // Intra4x4 Luma prediction mode
    if (mb_type == INTRA_SMALL_BLOCKS_MB_TYPE) 
	{
        for (int ii = 0; ii < 16; ii++) 
		{
            // i is the lane number which should be the ii'th one to be encoded
            i = DecodeOrderMapping_dev[ii];

            if (pMBContext[tid_grid*BLOCKS_PER_MB+i].MBModes[HDR_CONTEXT_SUBTYPE] == -1)
			{
                // Use predicted mode
                Code1.length = 1;
                Code1.value  = 1;
            }
			else 
			{
                // Use non-predicted mode
                Code1.length = 4;
                Code1.value  = (unsigned short)(pMBContext[tid_grid*BLOCKS_PER_MB+i].MBModes[HDR_CONTEXT_SUBTYPE]);
            }
            
			Code.length = Code.length + Code1.length;
			Code.value  = Code.value << Code1.length;
			Code.value  = Code.value | Code1.value;

			if( ((ii+1)&3) == 0) //每4个子宏块的sub_type合成到一个元素中
			{
				out_idx  =  (ii>>2)+1;
				pCodes_Header_IMB[tid_grid*head_size+out_idx].length = Code.length;
				pCodes_Header_IMB[tid_grid*head_size+out_idx].value = Code.value;
				TotalCodeBits[tid_grid] += Code.length;
				Code.length = 0;
				Code.value = 0;
			}
        }
    }
        ////////////////////////////////////////////////////////
        // Intra prediction mode for chroma
		temp = pMBContext[tid_grid*BLOCKS_PER_MB].MBModes[HDR_CONTEXT_INTRACHROMAMODE];

		Code = exp_golomb_unsigned_kernel(temp);

		pCodes_Header_IMB[tid_grid*head_size+5].length = Code.length;
		pCodes_Header_IMB[tid_grid*head_size+5].value  = Code.value;
		TotalCodeBits[tid_grid] += Code.length;

        ////////////////////////////////////////////////////////
        // Coded block pattern (CBP)

        if (mb_type != INTRA_LARGE_BLOCKS_MB_TYPE) 
		{
            CBP = (misc & 0xff);
            CBPCodeNum = (unsigned int)CBPTable_dev[CBP*2];

			Code = exp_golomb_unsigned_kernel(CBPCodeNum);

            pCodes_Header_IMB[tid_grid*head_size+6].length = Code.length;
            pCodes_Header_IMB[tid_grid*head_size+6].value  = Code.value;
            TotalCodeBits[tid_grid] += Code.length;
        }

        ////////////////////////////////////////////////////////
        // Delta-quant
        if (((mb_type == INTRA_LARGE_BLOCKS_MB_TYPE) || ((misc & 0xff) != 0))) 
		{
			int Delta_quant =((((int)((misc >> 8) & 0xff)) << 24) >> 24);
			unsigned int Delta_num = (abs(Delta_quant)*2) - ((Delta_quant > 0) ? 1 : 0);

			Code = exp_golomb_unsigned_kernel(Delta_num);

            pCodes_Header_IMB[tid_grid*head_size+7].length = Code.length;
            pCodes_Header_IMB[tid_grid*head_size+7].value  = Code.value;
            TotalCodeBits[tid_grid] += Code.length;
        }
}

__global__ void cavlc_header_codes_Pframe
							(
                            // Input context
                            S_CAVLC_CONTEXT_BLOCK *pMBContext,
							int *SkipBlock,
							int *DecodeOrderMapping_dev,//解码顺序
							unsigned char*CBPTable_dev,
							int head_size,
							// Updated codes (input and output)
                            SINGLE_CODE          *pCodes_Header_PMB, //13s element for a P MB 
                            // Scalar parameters
                            int *TotalCodeBits
                            )
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	
	int tid_blk = tid_x + tid_y*blockDim.x;
	int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y;

	int mb_type,misc,sub_type,IsIntra;
	int skip_block;
	int CBP;
	unsigned int CBPCodeNum,temp;
	int out_idx;
	SINGLE_CODE Code,Code1;
	short i = 32;
	skip_block = SkipBlock[tid_grid];
	TotalCodeBits[tid_grid] = 0;
	if(!skip_block)
	{

		mb_type = pMBContext[tid_grid*BLOCKS_PER_MB].MBModes[HDR_CONTEXT_TYPE];
		sub_type = pMBContext[tid_grid*BLOCKS_PER_MB].MBModes[HDR_CONTEXT_SUBTYPE];
		misc = pMBContext[tid_grid*BLOCKS_PER_MB].Misc;
		IsIntra = ((mb_type == INTRA_LARGE_BLOCKS_MB_TYPE) || (mb_type == INTRA_SMALL_BLOCKS_MB_TYPE));
		
		//num_skip
		Code = exp_golomb_unsigned_kernel((misc >> 16) & 0xffff);
		pCodes_Header_PMB[tid_grid*head_size].length = Code.length;
		pCodes_Header_PMB[tid_grid*head_size].value = Code.value;
		TotalCodeBits[tid_grid] += Code.length;
		
		// Mb type
		if (mb_type == INTRA_LARGE_BLOCKS_MB_TYPE) 
		{
			temp = ((((misc & 15) > 0) ? 0xd : 1)
					+ ((misc >> 2) & 0xc)
					+ sub_type + 5);
		} 
		else if (mb_type == INTRA_SMALL_BLOCKS_MB_TYPE) 
		{
			temp = 5 ;
		}
		else if (mb_type == INTER_LARGE_BLOCKS_MB_TYPE) 
		{
			temp = sub_type ;
		} 
		else if (mb_type == INTER_SMALL_BLOCKS_MB_TYPE) 
		{
			temp =  4;
		}
		Code = exp_golomb_unsigned_kernel(temp);
		pCodes_Header_PMB[tid_grid*head_size+1].length = Code.length;
		pCodes_Header_PMB[tid_grid*head_size+1].value  = Code.value;
		TotalCodeBits[tid_grid] += Code.length;
		Code.length = 0;
		Code.value = 0;

		////////////////////////////////////////////////////////
		// Intra4x4 Luma prediction mode
		if (mb_type == INTRA_SMALL_BLOCKS_MB_TYPE) 
		{
			for (int ii = 0; ii < 16; ii++) 
			{
				// i is the lane number which should be the ii'th one to be encoded
				i = DecodeOrderMapping_dev[ii];

				if (pMBContext[tid_grid*BLOCKS_PER_MB+i].MBModes[HDR_CONTEXT_SUBTYPE] == -1)
				{
					// Use predicted mode
					Code1.length = 1;
					Code1.value  = 1;
				}
				else 
				{
					// Use non-predicted mode
					Code1.length = 4;
					Code1.value  = (unsigned short)(pMBContext[tid_grid*BLOCKS_PER_MB+i].MBModes[HDR_CONTEXT_SUBTYPE]);
				}
	            
				Code.length = Code.length + Code1.length;
				Code.value  = Code.value << Code1.length;
				Code.value  = Code.value | Code1.value;

				if( ((ii+1)&3) == 0) //每4个子宏块的sub_type合成到一个元素中
				{
					out_idx  =  (ii>>2) + ((ii>>2)>0) ? 4 : 2;
					
					pCodes_Header_PMB[tid_grid*head_size+out_idx].length = Code.length;
					pCodes_Header_PMB[tid_grid*head_size+out_idx].value = Code.value;
					TotalCodeBits[tid_grid] += Code.length;
					Code.length = 0;
					Code.value = 0;
				}
			}
		}
			////////////////////////////////////////////////////////
			// Intra prediction mode for chroma
		 if(IsIntra)
		 {
			temp = pMBContext[tid_grid*BLOCKS_PER_MB].MBModes[HDR_CONTEXT_INTRACHROMAMODE];

			Code = exp_golomb_unsigned_kernel(temp);

			pCodes_Header_PMB[tid_grid*head_size+8].length = Code.length;
			pCodes_Header_PMB[tid_grid*head_size+8].value = Code.value;
			TotalCodeBits[tid_grid] += Code.length;
		 }

		 ////////////////////////////////////////////////////////
		// Macroblock Subdivision Type.  Only output if INTER_SMALL
		// macroblock type, and then only for 4x4 blocks 0, 2, 8, 10
		// (i.e., the top-left 4x4 block in each 8x8 mb
		// sub-partition).
		 //本程序均采用16x16划分，这种情况never appear
	 //   if ((mb_type == INTER_SMALL_BLOCKS_MB_TYPE)) 
		//{
	 //       for (ii = 0; ii < 4; ii++) 
		//	{
	 //           // i is the lane number which contains the top-left
	 //           // 4x4 block of the ii'th 8x8 block to be encoded
	 //           i = DecodeOrderMapping8x8[ii];

	 //           code = exp_golomb_unsigned(pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE]);

	 //           out_idx  = 3 + (ii / 2);
	 //           concatenate_codes_two_code(&pCodes_Header_PMB[out_idx], code);
	 //           *TotalCodeBits += code.length;
	 //       }
	 //   }
			//本文采用16x16划分，所以只需要传第一个子宏块的MV
			////////////////////////////////////////////////////////
			// Differential motion vector (dmv).  Only output if this 4x4
			// block is the top-left block of the macroblock partition
			// that it is a part of.
			if (!IsIntra) 
			{
				// Handle the x-component
				int mv = pMBContext[tid_grid*BLOCKS_PER_MB].DeltaMV[0];
				temp = (abs(mv)*2) - ((mv > 0) ? 1 : 0);
				Code = exp_golomb_unsigned_kernel(temp);

				pCodes_Header_PMB[tid_grid*head_size+3].length = Code.length;
				pCodes_Header_PMB[tid_grid*head_size+3].value  = Code.value;
				TotalCodeBits[tid_grid] += Code.length;

				// Handle y-component
				mv = pMBContext[tid_grid*BLOCKS_PER_MB].DeltaMV[1];
				temp = (abs(mv)*2) - ((mv > 0) ? 1 : 0);
				Code = exp_golomb_unsigned_kernel(temp);

				pCodes_Header_PMB[tid_grid*head_size+4].length = Code.length;
				pCodes_Header_PMB[tid_grid*head_size+4].value  = Code.value;
				TotalCodeBits[tid_grid] += Code.length;
			}

			////////////////////////////////////////////////////////
			// Coded block pattern (CBP)

			if (mb_type != INTRA_LARGE_BLOCKS_MB_TYPE) 
			{
				CBP = (misc & 0xff);
				CBPCodeNum = CBPTable_dev[CBP*2+(IsIntra ? 0:1)];

				Code = exp_golomb_unsigned_kernel(CBPCodeNum);

				pCodes_Header_PMB[tid_grid*head_size+9].length = Code.length;
				pCodes_Header_PMB[tid_grid*head_size+9].value  = Code.value;
				TotalCodeBits[tid_grid] += Code.length;
			}

			////////////////////////////////////////////////////////
			// Delta-quant
			if (((mb_type == INTRA_LARGE_BLOCKS_MB_TYPE) || ((misc & 0xff) != 0))) 
			{
				int Delta_quant =((((int)((misc >> 8) & 0xff)) << 24) >> 24);
				unsigned int Delta_num = (abs(Delta_quant)*2) - ((Delta_quant > 0) ? 1 : 0);

				Code = exp_golomb_unsigned_kernel(Delta_num);

				pCodes_Header_PMB[tid_grid*head_size+10].length = Code.length;
				pCodes_Header_PMB[tid_grid*head_size+10].value  = Code.value;
				TotalCodeBits[tid_grid] += Code.length;
			}
		}
}
