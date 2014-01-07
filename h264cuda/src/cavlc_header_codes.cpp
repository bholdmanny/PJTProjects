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



#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../inc/cavlc_data.h"
#include "../inc/cavlc.h"

#include "../inc/encoder_tables.h"

// Mapping from decoding order of 4x4 blocks to raster order.  The
// header context for each 4x4 block in a macroblock are stored in
// raster order across the clusters.  However, the H.264 bitstream
// requires 4x4 blocks to be encoded in raster order within each 8x8
// block, and each 8x8 block to then be encoded in raster order.
// Intended usage: when encoding the header information for the Nth
// 4x4 block in the eventual bitstream, then read the data from block
// DecodeOrderMapping[N] for this macroblock.  A similar usage model
// applies for decoding, except that this table indicates which block
// to write to (instead of which block to read from).
const int DecodeOrderMapping[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};


// Mapping from decoding order of 8x8 blocks to raster order.  This is
// similar to the table above except it gives the lane number of the
// top-left 4x4 block in the nth 8x8 block to be encoded.
const int DecodeOrderMapping8x8[4] = {0, 2, 8, 10};


//--------------------------------------------------------------------
// Local Protocols
//--------------------------------------------------------------------

//------------------------------------------------------------------
SINGLE_CODE exp_golomb_unsigned
//------------------------------------------------------------------
// This is implemented according to Section 9.1 of the 03/2005 version
// of the H.264 encoder specification.
//
// The return bit string consists of the string size in the top 8 bits
// of the 32-bit int, and the string value in the lower 24 bits.
// //------------------------------------------------------------------
(
 unsigned int code
 )
    //------------------------------------------------------------------
{
    SINGLE_CODE s;
    short i = 32;

    // Find the first non-zero bit of code+1.  At the end of this
    // loop, i will indicate which bit was the first non-zero bit,
    // where 32 is the MSB, 1 is the LSB, and 0 indicates no non-zero
    // bits.
    while (i > 0) {
        if ((code+1) & (1 << (i-1))) {
            break;
        }
        i--;
    }
    assert(i > 0);

    s.length = (short)((i*2)-1);
    s.value = (unsigned short)(code+1);
    return (s);
}


//------------------------------------------------------------------
SINGLE_CODE exp_golomb_signed
//------------------------------------------------------------------
// This is implemented according to Section 9.1.1 of the 03/2005
// version of the H.264 encoder specification.
//
// The return bit string consists of the string size in the top 8 bits
// of the 32-bit int, and the string value in the lower 24 bits.
//------------------------------------------------------------------
(
 int code
 )
    //------------------------------------------------------------------
{
    unsigned int code_num = (abs(code)*2) - ((code > 0) ? 1 : 0);
    return (exp_golomb_unsigned(code_num));
}


//------------------------------------------------------------------
unsigned int gen_mb_type_value
//------------------------------------------------------------------
(
 S_CAVLC_CONTEXT *pHeaderInfo,
 int NumRefs,
 int I_Slice
 )
    //------------------------------------------------------------------
{
    int code = 0;
    
    if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_LARGE_BLOCKS_MB_TYPE) 
	{
        code = ((((pHeaderInfo->Misc & 15) > 0) ? 0xd : 1)
                + ((pHeaderInfo->Misc >> 2) & 0xc)
                + pHeaderInfo->MBModes[HDR_CONTEXT_SUBTYPE]
                + (I_Slice ? 0 : 5));
    } 
	else if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_SMALL_BLOCKS_MB_TYPE) 
	{
        code = (I_Slice ? 0 : 5);
    }
	else if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTER_LARGE_BLOCKS_MB_TYPE) 
	{
        code = pHeaderInfo->MBModes[HDR_CONTEXT_SUBTYPE] + 1 + (I_Slice ? 0 : -1);
    } 
	else if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTER_SMALL_BLOCKS_MB_TYPE) 
	{
        code = ((NumRefs > 1) ? 4 : 5) + (I_Slice ? 0 : -1);
    }
	else 
	{
        printf("Invalid macroblock type!\n");
        exit(-1);
    }

    return (code);
}

unsigned int gen_mb_type_value_b
//------------------------------------------------------------------
(
 S_CAVLC_CONTEXT_BLOCK *pHeaderInfo,
 int NumRefs,
 int I_Slice
 )
    //------------------------------------------------------------------
{
    int code = 0;
    
    if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_LARGE_BLOCKS_MB_TYPE) 
	{
        code = ((((pHeaderInfo->Misc & 15) > 0) ? 0xd : 1)
                + ((pHeaderInfo->Misc >> 2) & 0xc)
                + pHeaderInfo->MBModes[HDR_CONTEXT_SUBTYPE]
                + (I_Slice ? 0 : 5));
    } 
	else if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_SMALL_BLOCKS_MB_TYPE) 
	{
        code = (I_Slice ? 0 : 5);
    }
	else if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTER_LARGE_BLOCKS_MB_TYPE) 
	{
        code = pHeaderInfo->MBModes[HDR_CONTEXT_SUBTYPE] + 1 + (I_Slice ? 0 : -1);
    } 
	else if (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTER_SMALL_BLOCKS_MB_TYPE) 
	{
        code = ((NumRefs > 1) ? 4 : 5) + (I_Slice ? 0 : -1);
    }
	else 
	{
        printf("Invalid macroblock type!\n");
        exit(-1);
    }

    return (code);
}

//--------------------------------------------------------------------
int is_topleft_block_of_partition
//--------------------------------------------------------------------
// Figures out whether the given 4x4 block index (block_num) is the
// top-left block of the macroblock partition that it is within.  Only
// one block (we chose the top-left one) in each partition will
// actually write its motion vector to the bitstream, since all the
// blocks in the same partition will have identical motion vectors.
//
// If the return value is true, the motion vector of this 4x4 block
// should be written to the bitstream.
//--------------------------------------------------------------------
(
 S_CAVLC_CONTEXT *pHeaderInfo,
 int block_num
 )
    //--------------------------------------------------------------------
{
    int LargeSingleBlk = (block_num == 0);
    int LargeHorBlk    = LargeSingleBlk || (block_num == 8);
    int LargeVertBlk   = LargeSingleBlk || (block_num == 2);

    int SmallSingleBlk = LargeHorBlk || LargeVertBlk || (block_num == 10);
    int SmallHorBlk    = ((block_num & 0x1) == 0);
    int SmallVertBlk   = ((block_num & 0x4) == 0);

    int SingleBlockType = pHeaderInfo->MBModes[HDR_CONTEXT_SUBTYPE] == SINGLE_BLOCK_SUBDIV_TYPE;
    int HorBlockType = pHeaderInfo->MBModes[HDR_CONTEXT_SUBTYPE] == HOR_SUBDIV_TYPE;

    int LargeBlockMVFlag = (SingleBlockType ? LargeSingleBlk :
                             HorBlockType ? LargeHorBlk :
                             // else VERT_BLOCK_SUBDIV_TYPE
                             LargeVertBlk);

    int SmallBlockMVFlag = (SingleBlockType ? SmallSingleBlk :
                             HorBlockType ? SmallHorBlk :
                             // else VERT_BLOCK_SUBDIV_TYPE
                             SmallVertBlk);

    int MVFlag = ((pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTER_LARGE_BLOCKS_MB_TYPE) ? LargeBlockMVFlag :
                  (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTER_SMALL_BLOCKS_MB_TYPE) ? SmallBlockMVFlag :
                   // If not an INTER block
                   0);

    return(MVFlag);
}



//--------------------------------------------------------------------
// Exported Functions
//--------------------------------------------------------------------
void concatenate_codes_two_code(
                           SINGLE_CODE *code1,      // will be updated    
                           SINGLE_CODE code2    // will be cat'd to code1
                           )
{
    assert(code1->value < (1 << (16 - code2.length)));

    code1->length = code1->length + code2.length;
    code1->value  = code1->value << code2.length;
    code1->value  = code1->value  | code2.value;
}

void cavlc_header_codes_ISlice(
                            // Input context
                            S_CAVLC_CONTEXT_BLOCK  *pHeaderInfo,
                            
                            // Updated codes (input and output)
                            SINGLE_CODE          *pCodes_Header_IMB, //8 element for a I MB 1+4+1+1+1
                            
                            // Scalar parameters
                            int *TotalCodeBits, int mbs_coded
                            )
//------------------------------------------------------------------
{
    int mb_num;
    for(mb_num = 0; mb_num < mbs_coded; mb_num++)
	{
        int i = -1, ii, out_lane, out_idx, out_blk;
        SINGLE_CODE code;
		for (i = 0; i < 8; i++) 
		{
				pCodes_Header_IMB[i].length = 0;
				pCodes_Header_IMB[i].value  = 0;
		}

        ////////////////////////////////////////////////////////
        // Mb type
        code = exp_golomb_unsigned(gen_mb_type_value_b(pHeaderInfo, 1, 1));
        pCodes_Header_IMB[0].length = code.length;
        pCodes_Header_IMB[0].value  = code.value;
        *TotalCodeBits += code.length;


        ////////////////////////////////////////////////////////
        // Intra4x4 Luma prediction mode
        if (pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] == INTRA_SMALL_BLOCKS_MB_TYPE) 
		{
            for (ii = 0; ii < 16; ii++) 
			{
                // i is the lane number which should be the ii'th one to be encoded
                i = DecodeOrderMapping[ii];

                if (pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE] == -1)
				{
                    // Use predicted mode
                    code.length = 1;
                    code.value  = 1;
                }
				else 
				{
                    // Use non-predicted mode
                    code.length = 4;
                    code.value  = (unsigned short)(pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE]);
                }
                out_idx  =  ii/4+1;
				concatenate_codes_two_code(
											pCodes_Header_IMB + out_idx,  // will be updated    
											code    // will be cat'd to code1
											 );
                *TotalCodeBits += code.length;
            }
        }


        ////////////////////////////////////////////////////////
        // Intra prediction mode for chroma

        code = exp_golomb_unsigned(pHeaderInfo[0].MBModes[HDR_CONTEXT_INTRACHROMAMODE]);
        pCodes_Header_IMB[5].length = code.length;
        pCodes_Header_IMB[5].value  = code.value;
        *TotalCodeBits += code.length;

        ////////////////////////////////////////////////////////
        // Coded block pattern (CBP)
        if (pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] != INTRA_LARGE_BLOCKS_MB_TYPE) 
		{
            int CBP = pHeaderInfo[15].Misc & 0xff;
            int CBPCodeNum = CBPTable[CBP][0];
            code = exp_golomb_unsigned(CBPCodeNum);
            pCodes_Header_IMB[6].length = code.length;
            pCodes_Header_IMB[6].value  = code.value;
            *TotalCodeBits += code.length;
        }


        ////////////////////////////////////////////////////////
        // Delta-quant
        if (((pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] == INTRA_LARGE_BLOCKS_MB_TYPE) || ((pHeaderInfo[15].Misc & 0xff) != 0))) 
		{
	    code = exp_golomb_signed((((int)((pHeaderInfo[15].Misc >> 8) & 0xff)) << 24) >> 24);
            pCodes_Header_IMB[7].length = code.length;
            pCodes_Header_IMB[7].value  = code.value;
            *TotalCodeBits += code.length;
        }
       
    }
}


void cavlc_header_codes_PSlice(
                            // Input context
                            S_CAVLC_CONTEXT_BLOCK  *pHeaderInfo,
                            
                            // Updated codes (input and output)
                            SINGLE_CODE          *pCodes_Header_PMB,
                            
                            // Scalar parameters
                            int  NumRefs,
                            int *TotalCodeBits, int mbs_coded
                            )
//------------------------------------------------------------------
{
    int mb_num;
    for(mb_num = 0; mb_num < mbs_coded; mb_num++)
	{
        int i = -1, ii, out_lane, out_idx, out_blk;
        SINGLE_CODE code;
		for (i = 0; i < 11; i++) 
		{
				pCodes_Header_PMB[i].length = 0;
				pCodes_Header_PMB[i].value  = 0;
		}
        int IsIntra = ((pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_LARGE_BLOCKS_MB_TYPE)
                        || (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_SMALL_BLOCKS_MB_TYPE));

		
		code = exp_golomb_unsigned((pHeaderInfo[0].Misc >> 16) & 0xffff);
            pCodes_Header_PMB[0].length = code.length;
            pCodes_Header_PMB[0].value = code.value;
            *TotalCodeBits += code.length;
       
        ////////////////////////////////////////////////////////
        // Mb type
        code = exp_golomb_unsigned(gen_mb_type_value_b(pHeaderInfo+14, NumRefs, 0));
        pCodes_Header_PMB[1].length = code.length;
        pCodes_Header_PMB[1].value  = code.value;
        *TotalCodeBits += code.length;


        ////////////////////////////////////////////////////////
        // Intra4x4 Luma prediction mode
        if (pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] == INTRA_SMALL_BLOCKS_MB_TYPE) 
		{
            for (ii = 0; ii < 16; ii++) 
			{
                // i is the lane number which should be the ii'th one to be encoded
                i = DecodeOrderMapping[ii];

                if (pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE] == -1) 
				{
                    // Use predicted mode
                    code.length = 1;
                    code.value  = 1;
                } 
				else 
				{
                    // Use non-predicted mode
                    code.length = 4;
                    code.value  = (unsigned short)(pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE]);
                }
                out_idx  =  (ii/4)+2;
				if((ii/4) >= 1)
				{
					 out_idx += 2;
				}
				concatenate_codes_two_code(
											pCodes_Header_PMB +  out_idx,  // will be updated    
											code    // will be cat'd to code1
											 );
                *TotalCodeBits += code.length;
            }
        }


        ////////////////////////////////////////////////////////
        // Intra prediction mode for chroma
        if (IsIntra) 
		{
            code = exp_golomb_unsigned(pHeaderInfo[0].MBModes[HDR_CONTEXT_INTRACHROMAMODE]);
            pCodes_Header_PMB[8].length = code.length;
            pCodes_Header_PMB[8].value  = code.value;
            *TotalCodeBits += code.length;
        }


        ////////////////////////////////////////////////////////
        // Macroblock Subdivision Type.  Only output if INTER_SMALL
        // macroblock type, and then only for 4x4 blocks 0, 2, 8, 10
        // (i.e., the top-left 4x4 block in each 8x8 mb
        // sub-partition).
  //      if ((pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] == INTER_SMALL_BLOCKS_MB_TYPE)) 
		//{
  //          for (ii = 0; ii < 4; ii++) 
		//	{
  //              // i is the lane number which contains the top-left
  //              // 4x4 block of the ii'th 8x8 block to be encoded
  //              i = DecodeOrderMapping8x8[ii];

  //              code = exp_golomb_unsigned(pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE]);

  //              out_idx  = 3 + (ii / 2);
  //              concatenate_codes_two_code(&pCodes_Header_PMB[out_idx], code);
  //              *TotalCodeBits += code.length;
  //          }
  //      }


        ////////////////////////////////////////////////////////
        // Differential motion vector (dmv).  Only output if this 4x4
        // block is the top-left block of the macroblock partition
        // that it is a part of.
        if (!IsIntra) 
		{
            /*for (ii = 0; ii < 16; ii++) 
			{*/
                // i is the lane number which should be the ii'th one to be encoded
               // i = DecodeOrderMapping[ii];
				i = 0;

                //if (is_topleft_block_of_partition(pHeaderInfo+i, i)) {

                    // Handle the x-component
                    code = exp_golomb_signed(pHeaderInfo[i].DeltaMV[0]);
                    //out_lane = 14 + ii/8;
                    out_idx  = 3;
                    //out_blk  = (ii/4)%2;
                    pCodes_Header_PMB[out_idx].length = code.length;
                    pCodes_Header_PMB[out_idx].value  = code.value;
                    *TotalCodeBits += code.length;

                    // Handle y-component
                    code = exp_golomb_signed(pHeaderInfo[i].DeltaMV[1]);
                    out_idx++;
                    pCodes_Header_PMB[out_idx].length = code.length;
                    pCodes_Header_PMB[out_idx].value  = code.value;
                    *TotalCodeBits += code.length;
               // }
            //}
        }


        ////////////////////////////////////////////////////////
        // Coded block pattern (CBP)
        if (pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] != INTRA_LARGE_BLOCKS_MB_TYPE) 
		{
            int CBP = pHeaderInfo[0].Misc & 0xff;
            int CBPCodeNum = CBPTable[CBP][IsIntra ? 0 : 1];
            code = exp_golomb_unsigned(CBPCodeNum);
            pCodes_Header_PMB[9].length = code.length;
            pCodes_Header_PMB[9].value  = code.value;
            *TotalCodeBits += code.length;
        }

        ////////////////////////////////////////////////////////
        // Delta-quant
        if (((pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] == INTRA_LARGE_BLOCKS_MB_TYPE) || ((pHeaderInfo[0].Misc & 0xff) != 0))) 
		{
	    code = exp_golomb_signed((((int)((pHeaderInfo[0].Misc >> 8) & 0xff)) << 24) >> 24);
            pCodes_Header_PMB[10].length = code.length;
            pCodes_Header_PMB[10].value  = code.value;
            *TotalCodeBits += code.length;
        }
        pHeaderInfo += BLOCKS_PER_MB;
        pCodes_Header_PMB += 11;
    }
}

void concatenate_codes_ref(
                           struct S_CODES_T *code1,      // will be updated
                           int j,                 // which block
                           SINGLE_CODE code2    // will be cat'd to code1
                           )
{
    assert(code1->value[j] < (1 << (16 - code2.length)));

    code1->length[j] = code1->length[j] + code2.length;
    code1->value[j]  = code1->value[j] << code2.length;
    code1->value[j]  = code1->value[j]  | code2.value;
}


//------------------------------------------------------------------
void cavlc_header_codes_ref(
                            // Input context
                            S_CAVLC_CONTEXT  *pHeaderInfo,
                            
                            // Updated codes (input and output)
                            S_CODES          *pCodes,
                            
                            // Scalar parameters
                            int  NumRefs,
                            int I_Slice,
                            int *TotalCodeBits, int mbs_coded
                            )
//------------------------------------------------------------------
{
    int mb_num;
    for(mb_num = 0; mb_num < mbs_coded; mb_num++){
        int i = -1, ii, out_lane, out_idx, out_blk;
        SINGLE_CODE code;

        int IsIntra = ((pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_LARGE_BLOCKS_MB_TYPE)
                        || (pHeaderInfo->MBModes[HDR_CONTEXT_TYPE] == INTRA_SMALL_BLOCKS_MB_TYPE));


        ////////////////////////////////////////////////////////
        // Number of preceding macroblocks that are skipped
        if (!I_Slice) {
	  code = exp_golomb_unsigned((pHeaderInfo[14].Misc >> 16) & 0xffff);
            pCodes[14][MBSKIPRUN_CODE_IDX].length[0] = code.length;
            pCodes[14][MBSKIPRUN_CODE_IDX].value[0]  = code.value;
            *TotalCodeBits += code.length;
        }


        ////////////////////////////////////////////////////////
        // Mb type
        code = exp_golomb_unsigned(gen_mb_type_value(pHeaderInfo+14, NumRefs, I_Slice));
        pCodes[14][MBTYPE_CODE_IDX].length[0] = code.length;
        pCodes[14][MBTYPE_CODE_IDX].value[0]  = code.value;
        *TotalCodeBits += code.length;


        ////////////////////////////////////////////////////////
        // Intra4x4 Luma prediction mode
        if (pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] == INTRA_SMALL_BLOCKS_MB_TYPE) {
            for (ii = 0; ii < 16; ii++) {
                // i is the lane number which should be the ii'th one to be encoded
                i = DecodeOrderMapping[ii];

                if (pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE] == -1) {
                    // Use predicted mode
                    code.length = 1;
                    code.value  = 1;
                } else {
                    // Use non-predicted mode
                    code.length = 4;
                    code.value  = (unsigned short)(pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE]);
                }
                out_lane = 14 + ii/8;
                out_idx  =  + ii%4;
                out_blk  = (ii/4)%2;
                concatenate_codes_ref(&pCodes[out_lane][ILUMA4X4_MODE_CODE_IDX], out_blk, code);
                *TotalCodeBits += code.length;
            }
        }


        ////////////////////////////////////////////////////////
        // Intra prediction mode for chroma
        if (IsIntra) {
            code = exp_golomb_unsigned(pHeaderInfo[0].MBModes[HDR_CONTEXT_INTRACHROMAMODE]);
            pCodes[15][ICHROMA_MODE_CODE_IDX].length[1] = code.length;
            pCodes[15][ICHROMA_MODE_CODE_IDX].value[1]  = code.value;
            *TotalCodeBits += code.length;
        }


        ////////////////////////////////////////////////////////
        // Macroblock Subdivision Type.  Only output if INTER_SMALL
        // macroblock type, and then only for 4x4 blocks 0, 2, 8, 10
        // (i.e., the top-left 4x4 block in each 8x8 mb
        // sub-partition).
        if ((pHeaderInfo[0].MBModes[HDR_CONTEXT_TYPE] == INTER_SMALL_BLOCKS_MB_TYPE)) {
            for (ii = 0; ii < 4; ii++) {
                // i is the lane number which contains the top-left
                // 4x4 block of the ii'th 8x8 block to be encoded
                i = DecodeOrderMapping8x8[ii];

                code = exp_golomb_unsigned(pHeaderInfo[i].MBModes[HDR_CONTEXT_SUBTYPE]);

                out_idx  = FIRST_SUBDIV_CODE_IDX + (ii / 2);
                concatenate_codes_ref(&pCodes[14][out_idx], 0, code);
                *TotalCodeBits += code.length;
            }
        }


        ////////////////////////////////////////////////////////
        // Reference frame index: output for cluster 0 for
        // INTER_LARGE, and for 4x4 blocks 0, 2, 8, and 10 for
        // INTER_SMALL.
        if (!IsIntra && (NumRefs > 1)) {
            int shape8x8 = (pHeaderInfo[i].MBModes[HDR_CONTEXT_TYPE] == INTER_SMALL_BLOCKS_MB_TYPE);
            for (ii = 0; ii < (shape8x8 ? 4 : 1); ii++) {
                // i is the lane number which contains the top-left
                // 4x4 block of the ii'th 8x8 block to be encoded
                i = DecodeOrderMapping8x8[ii];

                if (NumRefs > 2) {
                    // pHeaderInfo[i].MBModes[HDR_CONTEXT_REF_IDX] could be greater than 1
                    code = exp_golomb_unsigned(pHeaderInfo[i].MBModes[HDR_CONTEXT_REF_IDX]);
                } else {
                    // pHeaderInfo[i].MBModes[HDR_CONTEXT_REF_IDX] is either 0 or 1.
                    code.length = 1;
                    code.value  = (unsigned short)(1 - pHeaderInfo[i].MBModes[HDR_CONTEXT_REF_IDX]);
                }

                out_idx  = FIRST_REFFRAME_CODE_IDX + (ii / 2);
                concatenate_codes_ref(&pCodes[14][out_idx], 0, code);
                *TotalCodeBits += code.length;
            }
        }


        ////////////////////////////////////////////////////////
        // Differential motion vector (dmv).  Only output if this 4x4
        // block is the top-left block of the macroblock partition
        // that it is a part of.
        if (!IsIntra) {
            for (ii = 0; ii < 16; ii++) {
                // i is the lane number which should be the ii'th one to be encoded
                i = DecodeOrderMapping[ii];

                if (is_topleft_block_of_partition(pHeaderInfo+i, i)) {

                    // Handle the x-component
                    code = exp_golomb_signed(pHeaderInfo[i].DeltaMV[0]);
                    out_lane = 14 + ii/8;
                    out_idx  = FIRST_DMV_CODE_IDX + ((ii%4) * 2);
                    out_blk  = (ii/4)%2;
                    pCodes[out_lane][out_idx].length[out_blk] = code.length;
                    pCodes[out_lane][out_idx].value[out_blk]  = code.value;
                    *TotalCodeBits += code.length;

                    // Handle y-component
                    code = exp_golomb_signed(pHeaderInfo[i].DeltaMV[1]);
                    out_idx++;
                    pCodes[out_lane][out_idx].length[out_blk] = code.length;
                    pCodes[out_lane][out_idx].value[out_blk]  = code.value;
                    *TotalCodeBits += code.length;
                }
            }
        }


        ////////////////////////////////////////////////////////
        // Coded block pattern (CBP)
        if (pHeaderInfo[15].MBModes[HDR_CONTEXT_TYPE] != INTRA_LARGE_BLOCKS_MB_TYPE) {
            int CBP = pHeaderInfo[15].Misc & 0xff;
            int CBPCodeNum = CBPTable[CBP][IsIntra ? 0 : 1];
            code = exp_golomb_unsigned(CBPCodeNum);
            pCodes[15][CBP_CODE_IDX].length[1] = code.length;
            pCodes[15][CBP_CODE_IDX].value[1]  = code.value;
            *TotalCodeBits += code.length;
        }


        ////////////////////////////////////////////////////////
        // Delta-quant
        if (((pHeaderInfo[15].MBModes[HDR_CONTEXT_TYPE] == INTRA_LARGE_BLOCKS_MB_TYPE) || ((pHeaderInfo[15].Misc & 0xff) != 0))) {
	    code = exp_golomb_signed((((int)((pHeaderInfo[15].Misc >> 8) & 0xff)) << 24) >> 24);
            pCodes[15][DQUANT_CODE_IDX].length[1] = code.length;
            pCodes[15][DQUANT_CODE_IDX].value[1]  = code.value;
            *TotalCodeBits += code.length;
        }
        pHeaderInfo += BLOCK_PAIRS_PER_MB;
        pCodes += BLOCK_PAIRS_PER_MB;
    }
}