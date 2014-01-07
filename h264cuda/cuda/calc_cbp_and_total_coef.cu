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
//#include "cacl_cbp_and_total_coef_kernel.h"

__device__ void Calc_CBP_and_TotalCoeff_Luma_kernel(short *QCoef,
											int	  &Type,
											int   &TotalCoeff,
											int   &CBP,
											int	  &val
											)
{
	int TempFirst = (Type ==INTRA_LARGE_BLOCKS_MB_TYPE) ? 0 : QCoef[0];
	TotalCoeff +=((TempFirst == 0) ? 0 : 1 );

	for(int i=1; i<BLK_SIZE; i++)
	{
		TotalCoeff +=((QCoef[i] == 0) ? 0 : 1 );
	}
	
	CBP |= (TotalCoeff !=0) ? val : 0;
	//CBP = ((Type ==INTRA_LARGE_BLOCKS_MB_TYPE)&&(CBP!=0)) ? 15 : CBP;

}

__device__ void Calc_CB_Pand_TotalCoeff_Chroma_kernel(short *QCoef,
											int   &TotalCoeff,
											int   &ACHasNonZero
											)
{
	for(int i=1; i<BLK_SIZE; i++)
	{
		TotalCoeff +=((QCoef[i] == 0) ? 0 : 1 );
	}
	ACHasNonZero = (TotalCoeff !=0) ? 1 : 0;

}

//__global__ void CalcCBP_and_TotalCoeff_Luma_cuda(short  *pQCoefs_dev,
//												S_BLK_MB_INFO *pMBInfo_dev/*,
//											  int *CBP_dev*/
//											  )
//{
//		__shared__  int CBP[16*6];
//		__shared__  short pQcoefs[MB_TOTAL_SIZE*6];
//		S_BLK_MB_INFO BlkMBInfo;
//		int val,TotalCoeff;
//
//		int tid_x,tid_y,tid_z,tid;
//		tid_x = threadIdx.x;
//		tid_y = threadIdx.y;
//		tid_z = threadIdx.z;
//		tid = tid_x + tid_y * blockDim.x + tid_z*blockDim.x*blockDim.y;
//		for(int i =0; i<16;i++)
//		{
//			pQcoefs[tid+i*96] = pQCoefs_dev[tid +i*96 +blockIdx.x*blockDim.z*MB_TOTAL_SIZE + blockIdx.y*gridDim.x*blockDim.z*MB_TOTAL_SIZE];
//		}
//
//		//int input_index = tid*BLK_SIZE + blockIdx.x*blockDim.z*MB_TOTAL_SIZE + blockIdx.y*gridDim.x*blockDim.z*MB_TOTAL_SIZE;
//		BlkMBInfo = pMBInfo_dev[tid+blockIdx.x*blockDim.z*BLOCKS_PER_MB+blockIdx.y*gridDim.x*blockDim.z*BLOCKS_PER_MB];
//		
//		TotalCoeff = 0;
//		val = 1<<(((tid_y>>1)<<1) + (tid_x>>1));
//		CBP[tid] = 0;
//		__syncthreads();
//
//		Calc_CBP_and_TotalCoeff_Luma_kernel(pQcoefs+tid*BLK_SIZE,
//											BlkMBInfo.Type,
//											TotalCoeff,
//											CBP[tid],
//											val
//											);
//		__syncthreads();
//		if((tid &15)==0)
//		{
//			//int first_CBP_MB = tid_z* blockDim.x*blockDim.y;/*tid* blockDim.x*blockDim.y;*/
//			for(int i=1; i < BLK_SIZE; i++)
//			{
//				CBP[tid] |= CBP[tid+i];
//			}
//			CBP[tid] = ((BlkMBInfo.Type ==INTRA_LARGE_BLOCKS_MB_TYPE)&&(CBP[tid]!=0)) ? 15 : CBP[tid];
//		
//		}
//		__syncthreads();
//		BlkMBInfo.TotalCoeffLuma = TotalCoeff;
//		BlkMBInfo.CBP = CBP[tid_z* blockDim.x*blockDim.y];
//		pMBInfo_dev[tid+blockIdx.x*blockDim.z*BLOCKS_PER_MB+blockIdx.y*gridDim.x*blockDim.z*BLOCKS_PER_MB] = BlkMBInfo;
//		//CBP_dev [tid+blockIdx.x*blockDim.z*BLOCKS_PER_MB + blockIdx.y*gridDim.x*blockDim.z*BLOCKS_PER_MB] = CBP[tid_z* blockDim.x*blockDim.y];
//}

__global__ void CalcCBP_and_TotalCoeff_Luma_cuda(short  *pQCoefs_dev,
												S_BLK_MB_INFO *pMBInfo_dev
											  
											  )
{
		__shared__  int CBP[16*8];
		__shared__  short pQcoefs[MB_TOTAL_SIZE*8];
		//S_BLK_MB_INFO BlkMBInfo;
		int val,TotalCoeff,type;

		int tid_x,tid_y,tid_z,tid;
		tid_x = threadIdx.x;
		tid_y = threadIdx.y;
		tid_z = threadIdx.z;
		tid = tid_x + tid_y * blockDim.x + tid_z*blockDim.x*blockDim.y;
		for(int i =0; i<16;i++)
		{
			pQcoefs[tid+i*128] = pQCoefs_dev[tid +i*128 +blockIdx.x*blockDim.z*MB_TOTAL_SIZE + blockIdx.y*gridDim.x*blockDim.z*MB_TOTAL_SIZE];
		}

		//int input_index = tid*BLK_SIZE + blockIdx.x*blockDim.z*MB_TOTAL_SIZE + blockIdx.y*gridDim.x*blockDim.z*MB_TOTAL_SIZE;
		type = pMBInfo_dev[(tid)+blockIdx.x*blockDim.z*BLOCKS_PER_MB+blockIdx.y*gridDim.x*blockDim.z*BLOCKS_PER_MB].Type;
		
		TotalCoeff = 0;
		val = 1<<(((tid_y>>1)<<1) + (tid_x>>1));
		CBP[tid] = 0;
		__syncthreads();

		Calc_CBP_and_TotalCoeff_Luma_kernel(pQcoefs+tid*BLK_SIZE,
											type,
											TotalCoeff,
											CBP[tid],
											val
											);
		__syncthreads();
		if((tid &15)==0)
		{
			//int first_CBP_MB = tid_z* blockDim.x*blockDim.y;/*tid* blockDim.x*blockDim.y;*/
			for(int i=1; i < BLK_SIZE; i++)
			{
				CBP[tid] |= CBP[tid+i];
			}
			CBP[tid] = ((type ==INTRA_LARGE_BLOCKS_MB_TYPE)&&(CBP[tid]!=0)) ? 15 : CBP[tid];
		
		}
		__syncthreads();
		/*BlkMBInfo.TotalCoeffLuma = TotalCoeff;
		BlkMBInfo.CBP = CBP[tid_z* blockDim.x*blockDim.y];*/
		pMBInfo_dev[tid+blockIdx.x*blockDim.z*BLOCKS_PER_MB+blockIdx.y*gridDim.x*blockDim.z*BLOCKS_PER_MB].TotalCoeffLuma = TotalCoeff;
		pMBInfo_dev[tid+blockIdx.x*blockDim.z*BLOCKS_PER_MB+blockIdx.y*gridDim.x*blockDim.z*BLOCKS_PER_MB].CBP =  CBP[tid_z* blockDim.x*blockDim.y];
		//CBP_dev [tid+blockIdx.x*blockDim.z*BLOCKS_PER_MB + blockIdx.y*gridDim.x*blockDim.z*BLOCKS_PER_MB] = CBP[tid_z* blockDim.x*blockDim.y];
}

//__global__ void CalcCBP_and_TotalCoeff_Chroma_cuda(short  *dev_dct_coefs_uv,
//												   short  *dev_dc_coefs_uv,
//												   S_BLK_MB_INFO *pMBInfo_dev/*,
//												   int *CBP_dev*/
//											  )
//{
//		__shared__  int CBP[8];
//		__shared__  int DCHasNonZero[8*2];
//		__shared__  int ACHasNonZero[BLOCKS_PER_MB_C*8*2];
//		__shared__  short pQcoefs_uv[MB_TOTAL_SIZE_C*8*2];
//		__shared__  short pQcoefs_dc_uv[BLOCKS_PER_MB_C*8*2];
//		S_BLK_MB_INFO BlkMBInfo,BlkMBInfo1;
//		int TotalCoeff;
//
//		int tid_x,tid_y,tid_z,tid;
//		tid_x = threadIdx.x;
//		tid_y = threadIdx.y;
//		tid_z = threadIdx.z;
//		tid = tid_x + tid_y * blockDim.x;
//		for(int i =0; i< 16;i++)
//		{
//			pQcoefs_uv[tid+i*blockDim.x*blockDim.y+tid_z*MB_TOTAL_SIZE_C*8] = dev_dct_coefs_uv[tid +i*blockDim.x*blockDim.y +blockIdx.x*8*MB_TOTAL_SIZE_C + blockIdx.y*120*MB_TOTAL_SIZE_C+tid_z*MB_TOTAL_SIZE_C*120*68];
//		}
//		pQcoefs_dc_uv[tid+tid_z*32] = dev_dc_coefs_uv[tid + blockIdx.x*8*BLOCKS_PER_MB_C + blockIdx.y*120*BLOCKS_PER_MB_C + tid_z*BLOCKS_PER_MB_C*120*68];
//
//		BlkMBInfo = pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*120*BLOCKS_PER_MB];
//		BlkMBInfo1 = pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*120*BLOCKS_PER_MB + 8];
//
//		TotalCoeff = 0;
//		__syncthreads();
//		
//		Calc_CB_Pand_TotalCoeff_Chroma_kernel( pQcoefs_uv+tid*BLK_SIZE+tid_z*MB_TOTAL_SIZE_C*8,
//												TotalCoeff,
//												ACHasNonZero[tid+tid_z*32]
//												);
//		__syncthreads();
//
//		if((tid < 16) && (tid_z==0))
//		{
//			DCHasNonZero[tid] = (pQcoefs_dc_uv[tid*4])||(pQcoefs_dc_uv[tid*4+1]) ||(pQcoefs_dc_uv[tid*4+2]) ||(pQcoefs_dc_uv[tid*4+3]);
//			ACHasNonZero[tid*4] = (ACHasNonZero[tid*4])||(ACHasNonZero[tid*4+1]) ||(ACHasNonZero[tid*4+2]) ||(ACHasNonZero[tid*4+3]);
//		}
//
//		//__syncthreads();
//		if((tid < 8) && (tid_z==0))
//		{
//
//			DCHasNonZero[tid] = (DCHasNonZero[tid])||(DCHasNonZero[tid+8]);
//			ACHasNonZero[tid*4] = (ACHasNonZero[tid*4])||(ACHasNonZero[tid*4+32]);
//			CBP[tid] = ((DCHasNonZero[tid]==0)&&(ACHasNonZero[tid*4]==0)) ? 0 :(((DCHasNonZero[tid]==1)&&(ACHasNonZero[tid*4]==0)) ? 1 : 2);
//		}
//		__syncthreads();
//
//		BlkMBInfo.TotalCoeffChroma = TotalCoeff;
//		BlkMBInfo1.TotalCoeffChroma = 0;
//		BlkMBInfo.CBP |= (CBP[tid_y] << 4);
//		BlkMBInfo1.CBP |= (CBP[tid_y] << 4);
//
//		pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*120*BLOCKS_PER_MB] = BlkMBInfo;
//		pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*120*BLOCKS_PER_MB + 8] = BlkMBInfo1;
//
//}

__global__ void CalcCBP_and_TotalCoeff_Chroma_cuda(short  *dev_dct_coefs_uv,
												   short  *dev_dc_coefs_uv,
												   S_BLK_MB_INFO *pMBInfo_dev,
												   int	   width,
												   int	   height
											  )
{
		__shared__  int CBP[8];
		__shared__  int DCHasNonZero[8*2];
		__shared__  int ACHasNonZero[BLOCKS_PER_MB_C*8*2];
		__shared__  short pQcoefs_uv[MB_TOTAL_SIZE_C*8*2];
		__shared__  short pQcoefs_dc_uv[BLOCKS_PER_MB_C*8*2];
		//S_BLK_MB_INFO BlkMBInfo,BlkMBInfo1;
		int TotalCoeff;

		int tid_x,tid_y,tid_z,tid;
		tid_x = threadIdx.x;
		tid_y = threadIdx.y;
		tid_z = threadIdx.z;
		tid = tid_x + tid_y * blockDim.x;
		int num_mb_hor,num_mb_ver;
		num_mb_hor = (width>>3);
		num_mb_ver = (height >>3);
		for(int i =0; i< 16;i++)
		{
			pQcoefs_uv[tid+i*blockDim.x*blockDim.y+tid_z*MB_TOTAL_SIZE_C*8] = dev_dct_coefs_uv[tid +i*blockDim.x*blockDim.y +blockIdx.x*8*MB_TOTAL_SIZE_C + blockIdx.y*num_mb_hor*MB_TOTAL_SIZE_C+tid_z*MB_TOTAL_SIZE_C*num_mb_hor*num_mb_ver];
		}
		pQcoefs_dc_uv[tid+tid_z*32] = dev_dc_coefs_uv[tid + blockIdx.x*8*BLOCKS_PER_MB_C + blockIdx.y*num_mb_hor*BLOCKS_PER_MB_C + tid_z*BLOCKS_PER_MB_C*num_mb_hor*num_mb_ver];

		//BlkMBInfo = pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*num_mb_hor*BLOCKS_PER_MB];
		//BlkMBInfo1 = pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*num_mb_hor*BLOCKS_PER_MB + 8];

		TotalCoeff = 0;
		__syncthreads();
		
		Calc_CB_Pand_TotalCoeff_Chroma_kernel( pQcoefs_uv+tid*BLK_SIZE+tid_z*MB_TOTAL_SIZE_C*8,
												TotalCoeff,
												ACHasNonZero[tid+tid_z*32]
												);
		__syncthreads();

		if((tid < 16) && (tid_z==0))
		{
			DCHasNonZero[tid] = (pQcoefs_dc_uv[tid*4])||(pQcoefs_dc_uv[tid*4+1]) ||(pQcoefs_dc_uv[tid*4+2]) ||(pQcoefs_dc_uv[tid*4+3]);
			ACHasNonZero[tid*4] = (ACHasNonZero[tid*4])||(ACHasNonZero[tid*4+1]) ||(ACHasNonZero[tid*4+2]) ||(ACHasNonZero[tid*4+3]);
		}

		//__syncthreads();
		if((tid < 8) && (tid_z==0))
		{

			DCHasNonZero[tid] = (DCHasNonZero[tid])||(DCHasNonZero[tid+8]);
			ACHasNonZero[tid*4] = (ACHasNonZero[tid*4])||(ACHasNonZero[tid*4+32]);
			CBP[tid] = ((DCHasNonZero[tid]==0)&&(ACHasNonZero[tid*4]==0)) ? 0 :(((DCHasNonZero[tid]==1)&&(ACHasNonZero[tid*4]==0)) ? 1 : 2);
		}
		__syncthreads();

		/*BlkMBInfo.TotalCoeffChroma = TotalCoeff;
		BlkMBInfo1.TotalCoeffChroma = 0;
		BlkMBInfo.CBP |= (CBP[tid_y] << 4);
		BlkMBInfo1.CBP |= (CBP[tid_y] << 4);*/

		pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*num_mb_hor*BLOCKS_PER_MB].TotalCoeffChroma = TotalCoeff;
		pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*num_mb_hor*BLOCKS_PER_MB].CBP |= (CBP[tid_y] << 4);
		pMBInfo_dev[tid_x + tid_y * BLOCKS_PER_MB + tid_z*4 + blockIdx.x*8*BLOCKS_PER_MB+blockIdx.y*num_mb_hor*BLOCKS_PER_MB + 8] .CBP |= (CBP[tid_y] << 4);

}

