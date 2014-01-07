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

__device__ int Median3MV
(
 int &X, int &Y, int &Z
 )
{
    int MinXY, MaxXY, MedianTmp, MedianXYZ;
    MinXY = (X < Y) ? X : Y;
    MaxXY = (X > Y) ? X : Y;
    MedianTmp = (MaxXY < Z) ? MaxXY : Z;
    MedianXYZ = (MedianTmp > MinXY) ? MedianTmp : MinXY;
    return (MedianXYZ);
}

//由于该函数数处理的是8160个MB,每个线程处理一个MB,我们定义一个block的大小为(16,5,1),grid(8160/80,1,1);
__global__ void cavlc_block_context_DC_kernel (
													short *pDcCoefs_LumaDC,
													S_BLK_MB_INFO *dev_blk_mb_info,
													int *ZigZagScan,
													short *pCoefBlksZigZag_LumaDC,
													S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_LumaDC,
													S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_Chroma_DC,
													int num_mb_hor
													)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_blk = tid_x + tid_y*blockDim.x;
	//S_BLK_MB_INFO BlkMBInfo,BlkA,BlkB;
	int Type,Loc,CBP;
	short coeff;
	int InIdx,blockIndex;
	int Avalid,Bvalid,nA,nB,nC;

	InIdx  = ZigZagScan[tid_x] + tid_y*blockDim.x;
	blockIndex = BLK_SIZE * blockDim.x * blockDim.y * blockIdx.x;
	//BlkMBInfo = dev_blk_mb_info[tid_blk*BLK_SIZE + blockIndex];
	Type = dev_blk_mb_info[tid_blk*BLK_SIZE + blockIndex].Type;
	Loc = dev_blk_mb_info[tid_blk*BLK_SIZE + blockIndex].Loc;
	CBP = dev_blk_mb_info[tid_blk*BLK_SIZE + blockIndex].CBP;

	//以ZigZag的方式对编码系数重排序
	for(int i = 0;i <BLK_SIZE; i++ )
	{
		coeff = pDcCoefs_LumaDC[InIdx + i*blockDim.x*blockDim.y + blockIndex];
		pCoefBlksZigZag_LumaDC[tid_blk + i*blockDim.x*blockDim.y + blockIndex] = coeff;
	}
	if(Type == INTRA_LARGE_BLOCKS_MB_TYPE)
	{
		pMBContextOut_LumaDC[tid_blk + blockIdx.x*blockDim.x*blockDim.y].BlockSize =16;
		Avalid = ((Loc & LOC_BLK_LEFT_EDGE) == 0);
        Bvalid = ((Loc & LOC_BLK_TOP_EDGE) == 0);

		nA = (Avalid) ? dev_blk_mb_info[tid_blk*BLK_SIZE + blockIndex -13].TotalCoeffLuma :0;
        nB = (Bvalid) ? dev_blk_mb_info[tid_blk*BLK_SIZE + blockIndex -num_mb_hor*16+12].TotalCoeffLuma :0;

        //nA = BlkA.TotalCoeffLuma;
        //nB = BlkB.TotalCoeffLuma;
        //nA = Avalid ? nA : 0;
        //nB = Bvalid ? nB : 0;

        // Calculate nC based on neighbors
        nC = nA + nB;
        nC = (Avalid && Bvalid) ? ((nC + 1) >> 1) : nC;
        pMBContextOut_LumaDC[tid_blk + blockIdx.x*blockDim.x*blockDim.y].nC = (short)(nC);

	}
	else
	{
		pMBContextOut_LumaDC[tid_blk + blockIdx.x*blockDim.x*blockDim.y].BlockSize =0;
		pMBContextOut_LumaDC[tid_blk + blockIdx.x*blockDim.x*blockDim.y].nC = 0;
	}
	//Chroma DC
	if (CBP & 0x30) 
	{
        pMBContextOut_Chroma_DC[tid_blk + blockIdx.x*blockDim.x*blockDim.y].BlockSize= 4;
    }
	else
	{
		 pMBContextOut_Chroma_DC[tid_blk + blockIdx.x*blockDim.x*blockDim.y].BlockSize= 0;
	}
	 //////////////////////////////////////////////////////////////
    // nC
	pMBContextOut_Chroma_DC[tid_blk + blockIdx.x*blockDim.x*blockDim.y].nC= -1;
}

//由于该函数数处理的是8160个MB,每个线程处理一个一个色度分量子宏块,我们定义一个block的大小为(16,4,2),grid(8160/16,1,1);
__global__ void cavlc_block_context_ChromaAC_kernel (
													short *pDctCoefs_ChromaAC,
													S_BLK_MB_INFO *dev_blk_mb_info,
													int *ZigZagScan,
													short *pCoefBlksZigZag_ChromaAC,
													S_CAVLC_CONTEXT_DC_CHROMA *pMBContextOut_ChromaAC,
													int num_mb_hor,
													int num_mb_ver
													)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_z = threadIdx.z;
	int tid_blk = tid_x + tid_y*blockDim.x;
	int tid_grid = tid_x + tid_y*blockDim.x +(tid_z+blockIdx.x*blockDim.z)*blockDim.x*blockDim.y;
	S_BLK_MB_INFO BlkMBInfo,BlkA,BlkB;
	short coeff;
	int InIdx,blockIndex;
	int Avalid,Bvalid,nA,nB,nC,BlockSize;
	int myChromaBlock,myLumaBlock;
	int myLumaMB;
	int LeftBlock , TopBlock ;

	InIdx  = ZigZagScan[((tid_x+1)&0x0f)] + tid_y*blockDim.x;
	blockIndex = MB_TOTAL_SIZE_C * blockDim.x * blockIdx.x;

	myChromaBlock = ((tid_blk+tid_z*blockDim.x*blockDim.y)&0x7);
	myLumaBlock = (((myChromaBlock & 0x3) << 1) & 0x4) + ((myChromaBlock & 0x3) << 1); //0,2,8,10
	myLumaMB = ((tid_blk+tid_z*blockDim.x*blockDim.y)>>3); 
	LeftBlock =((myChromaBlock - 1) & 1) + (myChromaBlock & 0xe);
	TopBlock = ((myChromaBlock - 2) & 0x3) + (myChromaBlock & 0xc);

	//以ZigZag的方式对编码系数重排序
	//u component
	for(int i = 0;i <8; i++ )
	{
		coeff = pDctCoefs_ChromaAC[InIdx + tid_z*MB_TOTAL_SIZE_C + i*blockDim.x*blockDim.y*blockDim.z + blockIndex];

		pCoefBlksZigZag_ChromaAC[tid_blk + (i*2+tid_z)*blockDim.x*blockDim.y*blockDim.z + blockIdx.x*64*32] = coeff;
	}
	//v component
	for(int i = 0;i <8; i++ )
	{
		coeff = pDctCoefs_ChromaAC[InIdx + tid_z*MB_TOTAL_SIZE_C + i*blockDim.x*blockDim.y*blockDim.z + blockIndex + MB_TOTAL_SIZE_C*num_mb_ver*num_mb_hor];
		pCoefBlksZigZag_ChromaAC[tid_blk + (i*2+tid_z)*blockDim.x*blockDim.y*blockDim.z + blockIdx.x*64*32 + MB_TOTAL_SIZE_C] = coeff;
	}
	//compute block Size

	pMBContextOut_ChromaAC[tid_grid ].BlockSize = (dev_blk_mb_info[myLumaMB*BLK_SIZE + (blockIndex>>2)].CBP & 0x20) ? 15 : 0;

	Avalid = ((dev_blk_mb_info[myLumaBlock + myLumaMB*BLK_SIZE + (blockIndex>>2)].Loc & LOC_BLK_LEFT_EDGE) == 0);
	Bvalid = ((dev_blk_mb_info[myLumaBlock + myLumaMB*BLK_SIZE + (blockIndex>>2)].Loc & LOC_BLK_TOP_EDGE) == 0);

	nA = ((Avalid)?(((LeftBlock & 1) == 1)
			? dev_blk_mb_info[myLumaMB*BLK_SIZE + (blockIndex>>2) - 16 + LeftBlock].TotalCoeffChroma
			: dev_blk_mb_info[myLumaMB*BLK_SIZE + (blockIndex>>2) + LeftBlock].TotalCoeffChroma) : 0 );

	// Do something similar for nB now.
	nB = ((Bvalid) ? (((TopBlock & 2) == 2)
			? dev_blk_mb_info[myLumaMB*BLK_SIZE + (blockIndex>>2)+ TopBlock - num_mb_hor*16].TotalCoeffChroma
			:dev_blk_mb_info[myLumaMB*BLK_SIZE + (blockIndex>>2)+ TopBlock].TotalCoeffChroma) :0 );
	// Calculate nC based on neighbors
	nC = nA + nB;
	nC = (Avalid && Bvalid) ? ((nC + 1) >> 1) : nC;
	pMBContextOut_ChromaAC[tid_grid ].nC = (short)(nC);
}

//由于I帧的skipMB等于0,而且deltaQp为0,deltaMV为0,所以不需要像P帧那样复杂考虑,MB之间也就没有相关性
//由于MB之间没有相关性,故可以任意定义kernel的配置参数,暂时定义为threads(16,8,1), grid(num_hor/8,num_ver,1);
//每16个线程处理自己的宏块
__global__ void cavlc_block_context_iframe_LumaAC_kernel (
															short *pDctCoefs_LumaAC,
															S_BLK_MB_INFO *dev_blk_mb_info,
															int *ZigZagScan_tab,
															short *pDctCoefs_ZigZag_LumaAC,
															S_CAVLC_CONTEXT_BLOCK *pMBContextOut_LumaAC,
															int num_mb_hor
															)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	
	int tid_blk = tid_x + tid_y*blockDim.x;
	int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
	//S_BLK_MB_INFO BlkMBInfo;
	int Type,SubType,IntraChromaMode,RefFrameIdx,CBP,Pred_mode,Loc;
	short coeff;
	int InIdx,blk_coeff_index;
	int Avalid,Bvalid,nA,nB,nC,BlkSize,BlkStart;
	int my8x8Block;
	int myLumaMB;
	int LeftBlock , TopBlock ;

	blk_coeff_index = MB_TOTAL_SIZE *(threadIdx.y + blockDim.y*blockIdx.x + blockIdx.y *gridDim.x * blockDim.y);
	//BlkMBInfo = dev_blk_mb_info[tid_grid];
	myLumaMB = tid_y + blockDim.y*blockIdx.x + blockIdx.y *gridDim.x * blockDim.y;

	Type = dev_blk_mb_info[tid_grid].Type;
	SubType = dev_blk_mb_info[tid_grid].SubType;
	Pred_mode = dev_blk_mb_info[tid_grid].Pred_mode;
	CBP = dev_blk_mb_info[tid_grid].CBP;
	Loc = dev_blk_mb_info[tid_grid].Loc;
	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_TYPE] = Type;
	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_SUBTYPE] = SubType;
	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_INTRACHROMAMODE] = dev_blk_mb_info[tid_grid].IntraChromaMode;
	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_REF_IDX] = dev_blk_mb_info[tid_grid].RefFrameIdx;
	//delta Qp and skipMV==0
	pMBContextOut_LumaAC[tid_grid].Misc = (CBP & 0xff);
	//delta MV ==0
	pMBContextOut_LumaAC[tid_grid].DeltaMV[0] = 0;
	pMBContextOut_LumaAC[tid_grid].DeltaMV[1] = 0;

	if (Type == INTRA_SMALL_BLOCKS_MB_TYPE)
	{
        int UsePredicted4x4Mode, Signal4x4Mode;
       
		UsePredicted4x4Mode = (SubType == Pred_mode);
        Signal4x4Mode = ((SubType < Pred_mode)
                             ? SubType : (SubType - 1));
        // Overwrite SubType
		pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_SUBTYPE] = UsePredicted4x4Mode ? -1 : Signal4x4Mode; 
    }
   
	//
	BlkSize = (Type == INTRA_LARGE_BLOCKS_MB_TYPE) ? 15 : 16;
	BlkStart = (Type == INTRA_LARGE_BLOCKS_MB_TYPE) ? 1 : 0;

	InIdx = ZigZagScan_tab[(tid_x + BlkStart)&15];       
     // Copy data in zig-zag order
     for (int j = 0; j < BLOCKS_PER_MB; j++) 
	{
        coeff = pDctCoefs_LumaAC[InIdx + j*16 + blk_coeff_index];
		if((tid_x+BlkStart) ==16)
			coeff = 0;
		pDctCoefs_ZigZag_LumaAC[tid_x + j*16 + blk_coeff_index] = coeff;
     }

	my8x8Block = ((tid_x & 2)>>1) + ((tid_x >> 3) <<1);
	pMBContextOut_LumaAC[tid_grid].BlockSize = (CBP & (1 << my8x8Block)) ? BlkSize : 0;

	////nC, Calculate neighbor indices
    LeftBlock = ((tid_x - 1) & 3) + (tid_x & 0xc);
    TopBlock = ((tid_x - 4) & 0xf);
	Avalid = ((Loc & LOC_BLK_LEFT_EDGE) == 0);
    Bvalid = ((Loc & LOC_BLK_TOP_EDGE) == 0);

	nA = ((Avalid) ?(((LeftBlock & 3) == 3)
			? dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB - 16 + LeftBlock].TotalCoeffLuma
			: dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB + LeftBlock].TotalCoeffLuma):0);

    // Do something similar for nB now.
	nB = ((Bvalid) ? (((TopBlock >> 2) == 3)
			? dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB + TopBlock - num_mb_hor*16].TotalCoeffLuma
			:dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB + TopBlock].TotalCoeffLuma): 0);

            // Calculate nC based on neighbors
    nC = nA + nB;
    nC = (Avalid && Bvalid) ? ((nC + 1) >> 1) : nC;
	pMBContextOut_LumaAC[tid_grid].nC = (short)(nC);
}


//__global__ void cavlc_block_context_iframe_LumaAC_kernel (
//															short *pDctCoefs_LumaAC,
//															S_BLK_MB_INFO *dev_blk_mb_info,
//															int *ZigZagScan_tab,
//															short *pDctCoefs_ZigZag_LumaAC,
//															S_CAVLC_CONTEXT_BLOCK *pMBContextOut_LumaAC
//															)
//{
//	int tid_x = threadIdx.x;
//	int tid_y = threadIdx.y;
//	
//	int tid_blk = tid_x + tid_y*blockDim.x;
//	int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
//	S_BLK_MB_INFO BlkMBInfo;
//	short coeff;
//	int InIdx,blk_coeff_index;
//	int Avalid,Bvalid,nA,nB,nC,BlkSize,BlkStart;
//	int my8x8Block;
//	int myLumaMB;
//	int LeftBlock , TopBlock ;
//
//	blk_coeff_index = MB_TOTAL_SIZE *(threadIdx.y + blockDim.y*blockIdx.x + blockIdx.y *gridDim.x * blockDim.y);
//	BlkMBInfo = dev_blk_mb_info[tid_grid];
//	myLumaMB = tid_y + blockDim.y*blockIdx.x + blockIdx.y *gridDim.x * blockDim.y;
//
//	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_TYPE] = BlkMBInfo.Type;
//	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_SUBTYPE] = BlkMBInfo.SubType;
//	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_INTRACHROMAMODE] = BlkMBInfo.IntraChromaMode;
//	pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_REF_IDX] = BlkMBInfo.RefFrameIdx;
//	//delta Qp and skipMV==0
//	pMBContextOut_LumaAC[tid_grid].Misc = (BlkMBInfo.CBP & 0xff);
//	//delta MV ==0
//	pMBContextOut_LumaAC[tid_grid].DeltaMV[0] = 0;
//	pMBContextOut_LumaAC[tid_grid].DeltaMV[1] = 0;
//
//	if (BlkMBInfo.Type == INTRA_SMALL_BLOCKS_MB_TYPE)
//	{
//        int UsePredicted4x4Mode, Signal4x4Mode;
//       
//		UsePredicted4x4Mode = (BlkMBInfo.SubType == BlkMBInfo.Pred_mode);
//        Signal4x4Mode = ((BlkMBInfo.SubType < BlkMBInfo.Pred_mode)
//                             ? BlkMBInfo.SubType : (BlkMBInfo.SubType - 1));
//        // Overwrite SubType
//		pMBContextOut_LumaAC[tid_grid].MBModes[HDR_CONTEXT_SUBTYPE] = UsePredicted4x4Mode ? -1 : Signal4x4Mode; 
//    }
//   
//	//
//	BlkSize = (BlkMBInfo.Type == INTRA_LARGE_BLOCKS_MB_TYPE) ? 15 : 16;
//	BlkStart = (BlkMBInfo.Type == INTRA_LARGE_BLOCKS_MB_TYPE) ? 1 : 0;
//
//	InIdx = ZigZagScan_tab[(tid_x + BlkStart)&15];       
//     // Copy data in zig-zag order
//     for (int j = 0; j < BLOCKS_PER_MB; j++) 
//	{
//        coeff = pDctCoefs_LumaAC[InIdx + j*16 + blk_coeff_index];
//		if((tid_x+BlkStart) ==16)
//			coeff = 0;
//		pDctCoefs_ZigZag_LumaAC[tid_x + j*16 + blk_coeff_index] = coeff;
//     }
//
//	my8x8Block = ((tid_x & 2)>>1) + ((tid_x >> 3) <<1);
//	pMBContextOut_LumaAC[tid_grid].BlockSize = (BlkMBInfo.CBP & (1 << my8x8Block)) ? BlkSize : 0;
//
//	////nC, Calculate neighbor indices
//    LeftBlock = ((tid_x - 1) & 3) + (tid_x & 0xc);
//    TopBlock = ((tid_x - 4) & 0xf);
//	Avalid = ((BlkMBInfo.Loc & LOC_BLK_LEFT_EDGE) == 0);
//    Bvalid = ((BlkMBInfo.Loc & LOC_BLK_TOP_EDGE) == 0);
//
//	nA = ((Avalid) ?(((LeftBlock & 3) == 3)
//			? dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB - 16 + LeftBlock].TotalCoeffLuma
//			: dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB + LeftBlock].TotalCoeffLuma):0);
//
//    // Do something similar for nB now.
//	nB = ((Bvalid) ? (((TopBlock >> 2) == 3)
//			? dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB + TopBlock - 120*16].TotalCoeffLuma
//			:dev_blk_mb_info[myLumaMB*BLOCKS_PER_MB + TopBlock].TotalCoeffLuma): 0);
//
//            // Calculate nC based on neighbors
//    nC = nA + nB;
//    nC = (Avalid && Bvalid) ? ((nC + 1) >> 1) : nC;
//	pMBContextOut_LumaAC[tid_grid].nC = (short)(nC);
//}

//由于本程序处理的帧间宏块均为16*16模式,并没有进行更细的划分,所以,不需要对每一个子宏块进行处理,每一MB只需要处理一个即可,该MB中的其余15个子宏块的数据与其相同
//所以每个MB需要一个线程,threads(80,1,1); grid(8160/80,1,1);
__global__ void CalcPredictedMVRef_16x16_kernel( S_BLK_MB_INFO *dev_blk_mb_info,
												/*short* DeltaMV,*/
												S_CAVLC_CONTEXT_BLOCK *pMBContextOut_LumaAC,
												int *SkipBlock,
												int num_mb_hor
												)
{
	int tid_x = threadIdx.x;
	int tid_grid = threadIdx.x + blockIdx.x*blockDim.x;
    S_MV ZeroMV;
	int Avalid, Bvalid, Cvalid, Dvalid;
    int RefA_diff, RefB_diff, RefC_diff, OnlyRefA_same, OnlyRefB_same, OnlyRefC_same, OnlyOneRefSame;
    int SkipSpecialCondition;
    S_BLK_MB_INFO Blk, *BlkA, *BlkB, *BlkC, *BlkD;
    S_MV SkipPredMV,PredMV,Delta_MV;

    // Declare the neighboring motion vectors and reference frame
    // indices, as well as predicted motion vector
    S_MV MVA, MVB, MVC;
    int RefA, RefB, RefC;

    ZeroMV.x = 0;
    ZeroMV.y = 0;
	
    Blk = dev_blk_mb_info[tid_grid*16]; //每个线程获得自己MB
	if(Blk.Type == INTER_LARGE_BLOCKS_MB_TYPE)
	{
		Avalid = ((dev_blk_mb_info[tid_grid*16].Loc & LOC_BLK_LEFT_EDGE) == 0);
		Bvalid = ((dev_blk_mb_info[tid_grid*16].Loc & LOC_BLK_TOP_EDGE) == 0);

		Cvalid = !(((dev_blk_mb_info[tid_grid*16+3].Loc & LOC_BLK_TOP_PICTURE_EDGE) != 0)
						|| ((dev_blk_mb_info[tid_grid*16+3].Loc & LOC_BLK_RIGHT_PICTURE_EDGE) != 0));
		Dvalid = !(((dev_blk_mb_info[tid_grid*16].Loc & LOC_BLK_LEFT_OR_TOP_PICTURE_EDGE) != 0));
		
		MVA = Avalid ? dev_blk_mb_info[tid_grid*16 -13].MV : ZeroMV;
		RefA = Avalid ? dev_blk_mb_info[tid_grid*16 -13].RefFrameIdx : -1;
		MVB = Bvalid ? dev_blk_mb_info[tid_grid*16-num_mb_hor*16+12].MV : ZeroMV;
		RefB = Bvalid ? dev_blk_mb_info[tid_grid*16-num_mb_hor*16+12].RefFrameIdx :  -1;
		MVC = Cvalid ? dev_blk_mb_info[tid_grid*16-num_mb_hor*16+12+16].MV :(Dvalid ? dev_blk_mb_info[tid_grid*16-num_mb_hor*16-1].MV : ZeroMV);
		RefC =Cvalid ? dev_blk_mb_info[tid_grid*16-num_mb_hor*16+12+16].RefFrameIdx :(Dvalid ? dev_blk_mb_info[tid_grid*16-num_mb_hor*16-1].RefFrameIdx : -1);
		//DeltaMV[tid_grid].x = Cvalid;
		//DeltaMV[tid_grid].y = Dvalid;
		Cvalid = (Cvalid || Dvalid);

		// Precalculate which neighbors have the same reference frame
		// index as the current partition.
		RefA_diff = (RefA != Blk.RefFrameIdx);
		RefB_diff = (RefB != Blk.RefFrameIdx);
		RefC_diff = (RefC != Blk.RefFrameIdx);
		OnlyRefA_same = !RefA_diff && RefB_diff && RefC_diff;
		OnlyRefB_same = RefA_diff && !RefB_diff && RefC_diff;
		OnlyRefC_same = RefA_diff && RefB_diff && !RefC_diff;
		OnlyOneRefSame = (OnlyRefA_same || OnlyRefB_same || OnlyRefC_same);
	             
		if (Avalid && !Bvalid && !Cvalid) 
		{
			PredMV = MVA;
		} 
		else if (OnlyOneRefSame) 
		{
			PredMV = OnlyRefA_same ? MVA : (OnlyRefB_same ? MVB : MVC);
		} 
		else 
		{
			PredMV.x = Median3MV(MVA.x, MVB.x, MVC.x);
			PredMV.y = Median3MV(MVA.y, MVB.y, MVC.y);
		}
		SkipSpecialCondition = (!Avalid || !Bvalid
								|| ((RefA == 0) && (MVA.x == 0) && (MVA.y == 0))
								|| ((RefB == 0) && (MVB.x == 0) && (MVB.y == 0)));
		SkipPredMV = SkipSpecialCondition ? ZeroMV : (PredMV);
	    
		SkipBlock[tid_grid] = 0;
		if ((Blk.CBP == 0)&&(Blk.MV.x == SkipPredMV.x)&&(Blk.MV.y == SkipPredMV.y))
		{
			  SkipBlock[tid_grid] = 1;
		}
		Delta_MV.x = SkipBlock[tid_grid] ? 0 :(short)Blk.MV.x - PredMV.x;
		Delta_MV.y = SkipBlock[tid_grid] ? 0: (short)Blk.MV.y - PredMV.y;
		for(int i = 0; i< 16 ;i++)
		{
			pMBContextOut_LumaAC[tid_grid*16+i].DeltaMV[0] = (short)Delta_MV.x;
			pMBContextOut_LumaAC[tid_grid*16+i].DeltaMV[1] = (short)Delta_MV.y;
		}
	}
	else
	{
		//DeltaMV[tid_grid*2] = ZeroMV;
		SkipBlock[tid_grid] = 0;
		for(int i = 0; i< 16;i++)
		{
			pMBContextOut_LumaAC[tid_grid*16+i].DeltaMV[0] = 0;
			pMBContextOut_LumaAC[tid_grid*16+i].DeltaMV[1] = 0;
		}
	}
}

//多少个Slice就申明多少个block ,每个block一个线程,或者申明一个block,包含slice个线程
__global__ void cavlc_block_context_PrevSkipMB_kernel (
														int *SkipBlock,
														int *PrevSkipMB,
														S_CAVLC_CONTEXT_BLOCK *pMBContextOut_LumaAC,
														int num_mbs
													)
{
	int tid_x = threadIdx.x;
	int blk_idx = blockIdx.x;
	int Skip_index;
	Skip_index = (num_mbs/gridDim.x)*blk_idx;
	int PrevSkip = 0;

	for(int i=0;i<(num_mbs/gridDim.x);i++)
	{	
		
		pMBContextOut_LumaAC[(Skip_index+i)*16+tid_x].Misc |= (PrevSkip & 0xffff) << 16;
		if (SkipBlock[Skip_index+i])
		{
			PrevSkip += 1;
		}
		else
		{
			PrevSkip = 0;
		}
	}
	if(tid_x ==0)
	{
		PrevSkipMB[blk_idx] = PrevSkip;
	}
}