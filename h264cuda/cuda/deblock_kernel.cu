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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "../inc/encoder_tables.h"

#include "../inc/mb_info.h"
#include "../inc/encoder_context.h"

__device__ int Select_Middle(int min,int max,int val)
{
	return (((val)<(min)) ? (min) : (((val)>(max)) ? (max) : (val)));
}
__global__ void cudaCalcBoundaryStrength_kernel(S_BLK_MB_INFO *pBlkMBInfo,
												 unsigned char *BSRef,
												 int disable_deblocking_filter_idc,
												 int MBNumX,
												 int MBNumY,
												 int Slice_num,
												 int Slice_flag
										 )
{
		int MBcurr, MBcurrLeft, MBcurrTop;
		int tid_x = threadIdx.x;
		int	tid_y = threadIdx.y;
		int blk_x = blockIdx.x;
		int	blk_y = blockIdx.y;
		int first_edge_mb,left_edge_pic,top_edge_pic;

		unsigned char BoundySth_ver,BoundySth_hor;
		int out_position,first_row_slice;
		first_row_slice = (blk_y&((MBNumY/Slice_num)-1))==0;

		MBcurr = blk_y*MBNumX + blk_x*blockDim.y+tid_y; //计算当前线程处理的宏块位置
        MBcurrLeft = (tid_y == 0 && blk_x==0) ? (MBcurr) : (MBcurr - 1); //old
        MBcurrTop = (blk_y == 0) ? (MBcurr) : (MBcurr - MBNumX);
		first_edge_mb = ((tid_x>>2)==0) ? 1 : 0; //前4个线程对应宏块的最左边的4条边界
		left_edge_pic = first_edge_mb&&(tid_y == 0 && blk_x==0);
		top_edge_pic = first_edge_mb&&first_row_slice;

		if(Slice_flag) //I帧，边界两边都是帧内宏块，只需要判断该边界是否是最左边或最上边的边
		{
			BoundySth_ver = (left_edge_pic) ? 0 :(first_edge_mb ? 4 : 3);
			BoundySth_hor = (top_edge_pic) ? 0 :(first_edge_mb ? 4 : 3);

			out_position = tid_x+(blk_x*blockDim.y+tid_y)*BLOCKS_PER_MB*MBNumY+blk_y*BLOCKS_PER_MB;
			BSRef[out_position] = BoundySth_ver;//按照列的方式对宏块的数值边界强度输出
			out_position = tid_x+(blk_x*blockDim.y+tid_y)*BLOCKS_PER_MB+blk_y*BLOCKS_PER_MB*MBNumX+MBNumX*MBNumY*BLOCKS_PER_MB;
			BSRef[out_position] = BoundySth_hor;
		}

		else //P帧处理过程，由于P帧存在帧内和帧间两种情况，因此会复杂一些，需要根据每条边两边宏块的内型进行判断
		{
			//vertical boundry strengths,为了提高数据的重用效果，按照光栅的形式对一个MB进行处理，水平边界与C程序一直，
			//垂直边界输出边界强度结果时进行调整，block：（0-0，1-4，2-8，3-12，4-1，5-2...14-10,15-15）前面为线程数，后面为边界强度输出的顺序
			int block_mb = ((tid_x&3)<<2) + (tid_x>>2);
			S_BLK_MB_INFO p_Blk,q_Blk;
			int q_Type,p_Type,q_TotalCoeffLuma,p_TotalCoeffLuma;
			S_MV pMV,qMV;
			int BlkIsIntra,NeighborIsIntra;
			//MBInfo = pBlkMBInfo[MBcurr*BLOCKS_PER_MB];
			//NeighborBlk = pBlkMBInfo[MBcurrLeft*BLOCKS_PER_MB];

			q_Type = pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x].Type;	//每个线程对应的子宏块
			p_Type = ((tid_x&3)==0) ? pBlkMBInfo[MBcurrLeft*BLOCKS_PER_MB+tid_x+3].Type:pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x-1].Type; //边界左边的子宏块

			q_TotalCoeffLuma = pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x].TotalCoeffLuma;
			p_TotalCoeffLuma = ((tid_x&3)==0) ? pBlkMBInfo[MBcurrLeft*BLOCKS_PER_MB+tid_x+3].TotalCoeffLuma:pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x-1].TotalCoeffLuma;

			BlkIsIntra = ((q_Type == INTRA_LARGE_BLOCKS_MB_TYPE)
                          || (q_Type == INTRA_SMALL_BLOCKS_MB_TYPE));
			NeighborIsIntra = ((p_Type == INTRA_LARGE_BLOCKS_MB_TYPE)
                               || (p_Type == INTRA_SMALL_BLOCKS_MB_TYPE));

            // Get MVs for each 4x4 block to the left and above
            pMV = ((tid_x&3)==0) ? pBlkMBInfo[MBcurrLeft*BLOCKS_PER_MB+tid_x+3].MV:pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x-1].MV;
            qMV = pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x].MV;
            // Get Reference frames for each 4x4 block to the left and above
            // Check for coded coefficients in each 4x4 block to the left and above
            int pCodedCoeffs = (p_TotalCoeffLuma != 0);
            int qCodedCoeffs = (q_TotalCoeffLuma != 0);
			BoundySth_ver = ((tid_x&3) == 0 && (NeighborIsIntra || BlkIsIntra)) ? 4:((BlkIsIntra) ? 3 :((pCodedCoeffs || qCodedCoeffs) ? 2 :(((abs(pMV.x - qMV.x) >= 4)|| (abs(pMV.y - qMV.y) >= 4)) ? 1:0)));
			BoundySth_ver = ((tid_x&3) == 0&&(tid_y == 0 && blk_x==0)) ? 0 : BoundySth_ver;

			//horizontal boundary strenghts
			p_Type  = ((tid_x>>2)==0)?pBlkMBInfo[MBcurrTop*BLOCKS_PER_MB+tid_x+12].Type:pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x-4].Type; //边界上边子宏块索引
			NeighborIsIntra = ((p_Type == INTRA_LARGE_BLOCKS_MB_TYPE)
                               || (p_Type == INTRA_SMALL_BLOCKS_MB_TYPE));
			pMV = ((tid_x>>2)==0)?pBlkMBInfo[MBcurrTop*BLOCKS_PER_MB+tid_x+12].MV:pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x-4].MV;
			p_TotalCoeffLuma = ((tid_x>>2)==0)?pBlkMBInfo[MBcurrTop*BLOCKS_PER_MB+tid_x+12].TotalCoeffLuma:pBlkMBInfo[MBcurr*BLOCKS_PER_MB+tid_x-4].TotalCoeffLuma;
			pCodedCoeffs = (p_TotalCoeffLuma != 0);

			BoundySth_hor = ((tid_x>>2) == 0 && (NeighborIsIntra || BlkIsIntra)) ? 4:( BlkIsIntra ? 3 :((pCodedCoeffs || qCodedCoeffs) ? 2 :(((abs(pMV.x - qMV.x) >= 4)|| (abs(pMV.y - qMV.y) >= 4)) ? 1:0)));
			BoundySth_hor = top_edge_pic ? 0 : BoundySth_hor;

			out_position = block_mb+(blk_x*blockDim.y+tid_y)*BLOCKS_PER_MB*MBNumY+blk_y*BLOCKS_PER_MB;
			BSRef[out_position] = BoundySth_ver;//按照列的方式对宏块的数值边界强度输出
			out_position = tid_x+(blk_x*blockDim.y+tid_y)*BLOCKS_PER_MB+blk_y*BLOCKS_PER_MB*MBNumX+MBNumX*MBNumY*BLOCKS_PER_MB;
			BSRef[out_position] = BoundySth_hor;
		}

}

//线程块大小为32(16x2)，即一个线程块处理两个MB，整个网格的前一半线程处理亮度，后一半线程处理两个色度(计算量减半)，
__global__ void cudaDeblockMB_kernel_ver
(
	 unsigned char *BSRef,
	 int QP,
	 int NUM_X,
	 int NUM_Y,
	 int Width_ref,
	 int Height_ref,
	 /*int alpha_c0_offset,
	 int beta_offset,*/
	 int y_basicoffset,
	 int u_basicoffset,
	 unsigned char *Reference_Y,
	 unsigned char *Reference_U,
	 unsigned char *Reference_V,
	///////////////////////////data from frame
	 unsigned char *QP_TO_Chroma_dev,
	 unsigned char *ALPHA_Table_dev,
	 unsigned char *BETA_Table_dev,
	 unsigned char *CLIP_Table_dev
 )
{
	int tid_x = threadIdx.x;
	int	tid_y = threadIdx.y; //由tid_y来区分一个线程块中不同线程处理的MB
	int tid_blk = tid_y*blockDim.x+tid_x;
	int blk_x = blockIdx.x;
	int	blk_y = blockIdx.y;	//由blk_y来区分线程处理的事亮度还是色度分量
	__shared__ unsigned char B_strenght_y[32];
	__shared__ unsigned char B_strenght_uv[32];
	__shared__ unsigned char Ref_pixels_y[20*16*2];
	__shared__ unsigned char Ref_pixels_uv[20*16];
	__shared__ unsigned char ClipTab[5];  

	
    /*int      indexA, indexB;*/
	//亮度分量处理
	if(blockIdx.y==0)
	{
		int     ap, aq,  Strng ;
		int     C0, c0, Delta, dif, AbsDelta ;
		int     L2, L1, L0, R0, R1, R2, RL0 ;
	     
		int     Alpha, Beta;
		int     small_gap;
		int start_ref = y_basicoffset-4+tid_x+(tid_y+blk_x*blockDim.y)*Width_ref*MB_HEIGHT;
		int start_shared_mem = 4+tid_blk*20;
		Alpha = ALPHA_Table_dev[QP];
		Beta=BETA_Table_dev[QP];  
		if(tid_blk<5)
		{
			ClipTab[tid_blk]=CLIP_Table_dev[QP*5+tid_blk];
		}
		if(blockIdx.x != (gridDim.x-1))
		{
			for(int i = 0;i< NUM_X;i++)
			{
				B_strenght_y[tid_blk] = BSRef[blk_x*32+tid_blk+i*NUM_Y*16]; //加载每个宏块对应的边界强度到共享存储区，分四次使用
				for(int j =0 ;j<16;j++)
				{
					Ref_pixels_y[tid_x + j*20 + tid_y*20*16] = Reference_Y[start_ref + j*Width_ref+i*16];
					if(tid_x<4)
						Ref_pixels_y[tid_x + j*20 + tid_y*20*16 + 16] = Reference_Y[start_ref + j*Width_ref +16+i*16];
				}
				__syncthreads();
				for(int k=0;k<4;k++)
				{
					Strng = B_strenght_y[(tid_x>>2)+k*4+tid_y*16];

					if(Strng) //需要滤波
					{
						R0 =Ref_pixels_y[k*4+start_shared_mem];
						R1 =Ref_pixels_y[k*4+start_shared_mem+1];
						R2 =Ref_pixels_y[k*4+start_shared_mem+2];

						L0 =Ref_pixels_y[k*4+start_shared_mem-1];
						L1 =Ref_pixels_y[k*4+start_shared_mem-2];
						L2 =Ref_pixels_y[k*4+start_shared_mem-3];

						AbsDelta  = abs( Delta = (R0 - L0) );
						C0  = (int)ClipTab[ Strng ];
						if( (AbsDelta < Alpha) &&(((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0))
						{
							aq  = (abs( R0 - R2) - Beta ) < 0  ;
							ap  = (abs( L0 - L2) - Beta ) < 0  ;
							RL0 = L0 + R0 ;

							if(Strng == 4 )
							{
								small_gap = (AbsDelta < ((Alpha >> 2) + 2));
								aq &= small_gap;
								ap &= small_gap;

								Ref_pixels_y[k*4+start_shared_mem]   = aq ?(unsigned char)(( L1 + ((R1 + RL0) << 1) +  R2 + 4) >> 3) : (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2);
								Ref_pixels_y[k*4+start_shared_mem+1] = aq ?(unsigned char)(( R2 + R0 + R1 + L0 + 2) >> 2) : (unsigned char)R1;
								Ref_pixels_y[k*4+start_shared_mem+2] = aq ?(unsigned char)((((Ref_pixels_y[k*4+start_shared_mem+3] + R2) <<1) + R2 + R1 + RL0 + 4) >> 3) : (unsigned char)R2;

								Ref_pixels_y[k*4+start_shared_mem-1] = ap ?(unsigned char)(( R1 + ((L1 + RL0) << 1) +  L2 + 4) >> 3) : (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2);
								Ref_pixels_y[k*4+start_shared_mem-2] = ap ?(unsigned char)(( L2  + L1 + L0 + R0 + 2) >> 2) : (unsigned char)L1;
								Ref_pixels_y[k*4+start_shared_mem-3] = ap ?(unsigned char)((((Ref_pixels_y[k*4+start_shared_mem-4] + L2 ) <<1) + L2  + L1 + RL0 + 4) >> 3) : (unsigned char)L2;
							}
							else
							{
								c0  = C0 + ap + aq ;
								dif = ((Delta << 2) + (L1 - R1) + 4) >> 3;
								dif = Select_Middle( -c0,c0,dif);

								Ref_pixels_y[k*4+start_shared_mem-1]  = (unsigned char)Select_Middle(0,255,(L0+dif));
								Ref_pixels_y[k*4+start_shared_mem]	 = (unsigned char)Select_Middle(0,255,(R0 - dif));
								Ref_pixels_y[k*4+start_shared_mem-2] += (ap) ? (unsigned char)Select_Middle(-C0,  C0, ( L2 + ((RL0+1) >> 1) - (L1<<1)) >> 1):0;
								Ref_pixels_y[k*4+start_shared_mem+1] += (aq) ? (unsigned char)Select_Middle(-C0,  C0, ( R2 + ((RL0+1) >> 1) - (R1<<1)) >> 1):0;

							}
						}
					}		
				}
				for(int j =0;j<16;j++)
				{
					 Reference_Y[start_ref + j*Width_ref+ i*16] = Ref_pixels_y[tid_x + j*20 + tid_y*20*16];
					 if(tid_x<4)
						 Reference_Y[start_ref + j*Width_ref + i*16 +16]= Ref_pixels_y[tid_x + j*20 + tid_y*20*16 + 16];
				}	
			}
		}

		else if((tid_blk + blk_x*blockDim.y)<(NUM_Y*16))
		{
			for(int i = 0;i< NUM_X;i++)
			{
				B_strenght_y[tid_blk] = BSRef[blk_x*32+tid_blk+i*NUM_Y*16]; //加载每个宏块对应的边界强度到共享存储区，分四次使用
				for(int j =0 ;j<16;j++)
				{
					Ref_pixels_y[tid_x + j*20 + tid_y*20*16] = Reference_Y[start_ref + j*Width_ref+i*16];
					if(tid_x<4)
						Ref_pixels_y[tid_x + j*20 + tid_y*20*16 + 16] = Reference_Y[start_ref + j*Width_ref +16+i*16];
				}
				__syncthreads();

				for(int k=0;k<4;k++)
				{
					Strng = B_strenght_y[(tid_x>>2)+k*4+tid_y*16];

					if(Strng) //需要滤波
					{
						R0 =Ref_pixels_y[k*4+start_shared_mem];
						R1 =Ref_pixels_y[k*4+start_shared_mem+1];
						R2 =Ref_pixels_y[k*4+start_shared_mem+2];

						L0 =Ref_pixels_y[k*4+start_shared_mem-1];
						L1 =Ref_pixels_y[k*4+start_shared_mem-2];
						L2 =Ref_pixels_y[k*4+start_shared_mem-3];

						AbsDelta  = abs( Delta = (R0 - L0) );
						C0  = (int)ClipTab[ Strng ];
						if( (AbsDelta < Alpha) &&(((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0))
						{
							aq  = (abs( R0 - R2) - Beta ) < 0  ;
							ap  = (abs( L0 - L2) - Beta ) < 0  ;
							RL0 = L0 + R0 ;

							if(Strng == 4 )
							{
								small_gap = (AbsDelta < ((Alpha >> 2) + 2));
								aq &= small_gap;
								ap &= small_gap;

								Ref_pixels_y[k*4+start_shared_mem]   = aq ?(unsigned char)(( L1 + ((R1 + RL0) << 1) +  R2 + 4) >> 3) : (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2);
								Ref_pixels_y[k*4+start_shared_mem+1] = aq ?(unsigned char)(( R2 + R0 + R1 + L0 + 2) >> 2) : (unsigned char)R1;
								Ref_pixels_y[k*4+start_shared_mem+2] = aq ?(unsigned char)((((Ref_pixels_y[k*4+start_shared_mem+3] + R2) <<1) + R2 + R1 + RL0 + 4) >> 3) : (unsigned char)R2;

								Ref_pixels_y[k*4+start_shared_mem-1] = ap ?(unsigned char)(( R1 + ((L1 + RL0) << 1) +  L2 + 4) >> 3) : (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2);
								Ref_pixels_y[k*4+start_shared_mem-2] = ap ?(unsigned char)(( L2  + L1 + L0 + R0 + 2) >> 2) : (unsigned char)L1;
								Ref_pixels_y[k*4+start_shared_mem-3] = ap ?(unsigned char)((((Ref_pixels_y[k*4+start_shared_mem-4] + L2 ) <<1) + L2  + L1 + RL0 + 4) >> 3) : (unsigned char)L2;
							}
							else
							{
								c0  = C0 + ap + aq ;
								dif = ((Delta << 2) + (L1 - R1) + 4) >> 3;
								dif = Select_Middle( -c0,c0,dif);

								Ref_pixels_y[k*4+start_shared_mem-1]  = (unsigned char)Select_Middle(0,255,(L0+dif));
								Ref_pixels_y[k*4+start_shared_mem]	 = (unsigned char)Select_Middle(0,255,(R0 - dif));
								Ref_pixels_y[k*4+start_shared_mem-2] += (ap) ? (unsigned char)Select_Middle(-C0,  C0, ( L2 + ((RL0+1) >> 1) - (L1<<1)) >> 1):0;
								Ref_pixels_y[k*4+start_shared_mem+1] += (aq) ? (unsigned char)Select_Middle(-C0,  C0, ( R2 + ((RL0+1) >> 1) - (R1<<1)) >> 1):0;

							}
						}
					}		
				}
				for(int j =0;j<16;j++)
				{
					 Reference_Y[start_ref + j*Width_ref+ i*16] = Ref_pixels_y[tid_x + j*20 + tid_y*20*16];
					 if(tid_x<4)
						 Reference_Y[start_ref + j*Width_ref + i*16 +16]= Ref_pixels_y[tid_x + j*20 + tid_y*20*16 + 16];
				}	
			}
		}
	}
	//第二部分线程处理色度信息的滤波,每个线程块的前16个线程处理两个宏块的U分量，后16个线程处理V分量,一个边界管两个像素
	else
	{
		int     Strng ;
		int     C0, c0, Delta, dif, AbsDelta ;
		int     L1, L0, R0, R1,  RL0 ;
	     
		int     Alpha, Beta;
		int start_ref = u_basicoffset-2+tid_x+(blk_x*(blockDim.x>>3))*(Width_ref*MB_HEIGHT>>2);
		int start_shared_mem = 2+tid_blk*10;

		int QPcAv = (int)QP_TO_Chroma_dev[QP];
		Alpha = ALPHA_Table_dev[QPcAv];
		Beta=BETA_Table_dev[QPcAv];  
		if(tid_blk<5)
		{
			ClipTab[tid_blk]=CLIP_Table_dev[QPcAv*5+tid_blk];
		}
		if(blockIdx.x != (gridDim.x-1))
		{
			for(int i = 0;i< NUM_X;i++)
			{
				B_strenght_uv[tid_blk] = BSRef[blk_x*32+tid_blk+i*NUM_Y*16]; //加载每个宏块对应的边界强度到共享存储区，分四次使用
				for(int j =0 ;j<16;j++)
				{
					if(tid_x<10)
						Ref_pixels_uv[tid_x + j*10+10*16*tid_y] =(tid_y==0) ? Reference_U[start_ref + (j*Width_ref>>1) +i*8] : Reference_V[start_ref + (j*Width_ref>>1) +i*8];
					/*if(tid_x<10&&tid_y==1)
						Ref_pixels[tid_x + j*20+20*16] = Reference_V[start_row + (j*Width_ref>>1) +i*8];*/
				}
			
				for(int k=0;k<2;k++)
				{
					Strng = B_strenght_uv[((tid_x&7)>>1)+(k<<3)+((tid_x>>3)<<4)]; 
					
					if(Strng) //需要滤波
					{
						R0 =Ref_pixels_uv[k*4+start_shared_mem];
						R1 =Ref_pixels_uv[k*4+start_shared_mem+1];
						L0 =Ref_pixels_uv[k*4+start_shared_mem-1];
						L1 =Ref_pixels_uv[k*4+start_shared_mem-2];
						AbsDelta  = abs( Delta = (R0 - L0) );
						C0  = (int)ClipTab[ Strng ];

						if( (AbsDelta < Alpha) &&(((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0))
						{
							/*aq  = (abs( R0 - R2) - Beta ) < 0  ;
							ap  = (abs( L0 - L2) - Beta ) < 0  ;*/
							RL0 = L0 + R0 ;
							c0  = C0+1;
							dif =( (Delta << 2) + (L1 - R1) + 4) >> 3;
							dif = Select_Middle( -c0,c0,dif);

							Ref_pixels_uv[k*4+start_shared_mem]	= (Strng==4) ? (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2) :(unsigned char)Select_Middle(0,255,(R0 - dif));
							Ref_pixels_uv[k*4+start_shared_mem-1] = (Strng==4) ? (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2) :(unsigned char)Select_Middle(0,255,(L0 + dif));
						}
					
					}
				}
				for(int j =0;j<16;j++)
				{

					if(tid_x<10&&tid_y==0)
						Reference_U[start_ref + (j*Width_ref>>1) +i*8] = Ref_pixels_uv[tid_x + j*10];
					if(tid_x<10&&tid_y==1)
						Reference_V[start_ref + (j*Width_ref>>1) +i*8] = Ref_pixels_uv[tid_x + j*10+10*16];
				}	
			}
		}
		else if((tid_blk + blk_x*blockDim.y) < (NUM_Y*16))
		{
			for(int i = 0;i< NUM_X;i++)
			{
				B_strenght_uv[tid_blk] = BSRef[blk_x*32+tid_blk+i*NUM_Y*16]; //加载每个宏块对应的边界强度到共享存储区，分四次使用
				for(int j =0 ;j<16;j++)
				{
					if(tid_x<10)
						Ref_pixels_uv[tid_x + j*10+10*16*tid_y] =(tid_y==0) ? Reference_U[start_ref + (j*Width_ref>>1) +i*8] : Reference_V[start_ref + (j*Width_ref>>1) +i*8];
					/*if(tid_x<10&&tid_y==1)
						Ref_pixels[tid_x + j*20+20*16] = Reference_V[start_row + (j*Width_ref>>1) +i*8];*/
				}
			
				for(int k=0;k<2;k++)
				{
					Strng = B_strenght_uv[((tid_x&7)>>1)+(k<<3)+((tid_x>>3)<<4)]; 
					
					if(Strng) //需要滤波
					{
						R0 =Ref_pixels_uv[k*4+start_shared_mem];
						R1 =Ref_pixels_uv[k*4+start_shared_mem+1];
						L0 =Ref_pixels_uv[k*4+start_shared_mem-1];
						L1 =Ref_pixels_uv[k*4+start_shared_mem-2];
						AbsDelta  = abs( Delta = (R0 - L0) );
						C0  = (int)ClipTab[ Strng ];

						if( (AbsDelta < Alpha) &&(((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0))
						{
							/*aq  = (abs( R0 - R2) - Beta ) < 0  ;
							ap  = (abs( L0 - L2) - Beta ) < 0  ;*/
							RL0 = L0 + R0 ;
							c0  = C0+1;
							dif =( (Delta << 2) + (L1 - R1) + 4) >> 3;
							dif = Select_Middle( -c0,c0,dif);

							Ref_pixels_uv[k*4+start_shared_mem]	= (Strng==4) ? (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2) :(unsigned char)Select_Middle(0,255,(R0 - dif));
							Ref_pixels_uv[k*4+start_shared_mem-1] = (Strng==4) ? (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2) :(unsigned char)Select_Middle(0,255,(L0 + dif));
						}
					
					}
				}
				for(int j =0;j<16;j++)
				{

					if(tid_x<10&&tid_y==0)
						Reference_U[start_ref + (j*Width_ref>>1) +i*8] = Ref_pixels_uv[tid_x + j*10];
					if(tid_x<10&&tid_y==1)
						Reference_V[start_ref + (j*Width_ref>>1) +i*8] = Ref_pixels_uv[tid_x + j*10+10*16];
				}	
			}
		
		}
	
	}
}

__global__ void cudaDeblockMB_kernel_hor
(
	 unsigned char *BSRef,
	 int QP,
	 int NUM_X,
	 int NUM_Y,
	 int Width_ref,
	 int Height_ref,
	 /*int alpha_c0_offset,
	 int beta_offset,*/
	 int y_basicoffset,
	 int u_basicoffset,
	 unsigned char *Reference_Y,
	 unsigned char *Reference_U,
	 unsigned char *Reference_V,
	///////////////////////////data from frame
	 unsigned char *QP_TO_Chroma_dev,
	 unsigned char *ALPHA_Table_dev,
	 unsigned char *BETA_Table_dev,
	 unsigned char *CLIP_Table_dev
 )
{
	int tid_x = threadIdx.x;
	int	tid_y = threadIdx.y; //由tid_y来区分一个线程块中不同线程处理的MB
	int tid_blk = tid_y*blockDim.x+tid_x;
	int blk_x = blockIdx.x;
	int	blk_y = blockIdx.y;	//由blk_y来区分线程处理的事亮度还是色度分量
	__shared__ unsigned char B_strenght_y[32];
	__shared__ unsigned char B_strenght_uv[32];
	__shared__ unsigned char Ref_pixels_y[16*20*2];
	__shared__ unsigned char Ref_pixels_uv[16*20];
	__shared__ unsigned char ClipTab[5];  

	
    /*int      indexA, indexB;*/

    
	if(blk_y==0)
	{
		int     ap, aq,  Strng ;
		int     C0, c0, Delta, dif, AbsDelta ;
		int     L2, L1, L0, R0, R1, R2, RL0 ;
	     
		int     Alpha, Beta;
		int     small_gap;
		int start_ref = y_basicoffset-4*Width_ref+tid_x+(tid_y+blk_x*blockDim.y)*MB_WIDTH;
		int start_shared_mem = 4*16+tid_x+tid_y*16*20;

		Alpha = ALPHA_Table_dev[QP];
		Beta=BETA_Table_dev[QP];  
		if(tid_blk<5)
		{
			ClipTab[tid_blk]=CLIP_Table_dev[QP*5+tid_blk];
		}

		for(int i = 0;i< NUM_Y;i++)
		{

			B_strenght_y[tid_blk] = BSRef[blk_x*32+tid_blk+i*NUM_X*16+NUM_X*NUM_Y*16]; //加载每个宏块对应的边界强度到共享存储区，分四次使用
			for(int j =0 ;j<20;j++)
			{
				Ref_pixels_y[tid_x + j*16 + tid_y*20*16] = Reference_Y[start_ref + j*Width_ref+i*Width_ref*MB_HEIGHT];	
			}
			//start_row = 4*16+tid_x+tid_y*16*20;
			for(int k=0;k<4;k++)
			{
				Strng = B_strenght_y[(tid_x>>2)+k*4+tid_y*16];

				if(Strng) //需要滤波
				{
					R0 =Ref_pixels_y[k*4*16+start_shared_mem];
					R1 =Ref_pixels_y[k*4*16+start_shared_mem+16];
					R2 =Ref_pixels_y[k*4*16+start_shared_mem+2*16];

					L0 =Ref_pixels_y[k*4*16+start_shared_mem-16];
					L1 =Ref_pixels_y[k*4*16+start_shared_mem-2*16];
					L2 =Ref_pixels_y[k*4*16+start_shared_mem-3*16];

					AbsDelta  = abs( Delta = (R0 - L0) );
					C0  = (int)ClipTab[ Strng ];

					if( (AbsDelta < Alpha) &&(((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0))
					{
						aq  = (abs( R0 - R2) - Beta ) < 0  ;
						ap  = (abs( L0 - L2) - Beta ) < 0  ;
						RL0 = L0 + R0 ;
			
						if(Strng == 4 )
						{
							small_gap = (AbsDelta < ((Alpha >> 2) + 2));
							aq &= small_gap;
							ap &= small_gap;

							Ref_pixels_y[k*4*16+start_shared_mem]	= aq ?(unsigned char)(( L1 + ((R1 + RL0) << 1) +  R2 + 4) >> 3) : (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2);
							Ref_pixels_y[k*4*16+start_shared_mem+1*16] = aq ?(unsigned char)(( R2 + R0 + R1 + L0 + 2) >> 2) : (unsigned char)R1;
							Ref_pixels_y[k*4*16+start_shared_mem+2*16] = aq ?(unsigned char)((((Ref_pixels_y[k*4*16+start_shared_mem+3*16] + R2) <<1) + R2 + R1 + RL0 + 4) >> 3) : (unsigned char)R2;

							Ref_pixels_y[k*4*16+start_shared_mem-1*16]	= ap ?(unsigned char)(( R1 + ((L1 + RL0) << 1) +  L2 + 4) >> 3) : (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2);
							Ref_pixels_y[k*4*16+start_shared_mem-2*16] = ap ?(unsigned char)(( L2  + L1 + L0 + R0 + 2) >> 2) : (unsigned char)L1;
							Ref_pixels_y[k*4*16+start_shared_mem-3*16] = ap ?(unsigned char)((((Ref_pixels_y[k*4*16+start_shared_mem-4*16] + L2 ) <<1) + L2  + L1 + RL0 + 4) >> 3) : (unsigned char)L2;
						}
						else
						{
							c0  = C0 + ap + aq ;
							dif = ((Delta << 2) + (L1 - R1) + 4) >> 3;
							dif = Select_Middle( -c0,c0,dif);

							Ref_pixels_y[k*4*16+start_shared_mem-1*16]  = (unsigned char)Select_Middle(0,255,(L0+dif));
							Ref_pixels_y[k*4*16+start_shared_mem]	 = (unsigned char)Select_Middle(0,255,(R0 - dif));
							Ref_pixels_y[k*4*16+start_shared_mem-2*16] += (ap) ? (unsigned char)Select_Middle(-C0,  C0, ( L2 + ((RL0+1) >> 1) - (L1<<1)) >> 1):0;
							Ref_pixels_y[k*4*16+start_shared_mem+1*16] += (aq) ? (unsigned char)Select_Middle(-C0,  C0, ( R2 + ((RL0+1) >> 1) - (R1<<1)) >> 1):0;
						}
					}
				}		
			}
			for(int j =0 ;j<20;j++)
			{
				Reference_Y[start_ref + j*Width_ref + i*Width_ref*MB_HEIGHT] = Ref_pixels_y[tid_x + j*16 + tid_y*20*16];	
			}	
		}
	}

	//第二部分线程处理色度信息的滤波,每个线程块的前16个线程处理两个宏块的U分量，后16个线程处理V分量,一个边界管两个像素
	else
	{
		int     Strng ;
		int     C0, c0, Delta, dif, AbsDelta ;
		int     L1, L0, R0, R1,  RL0 ;
	     
		int     Alpha, Beta;
		int start_ref = u_basicoffset - Width_ref + tid_x +(blk_x*blockDim.y)*(MB_WIDTH>>1);
		int start_shared_mem = 2*16+tid_x+tid_y*16*10;

		int QPcAv = (int)QP_TO_Chroma_dev[QP];
		Alpha = ALPHA_Table_dev[QPcAv];
		Beta=BETA_Table_dev[QPcAv];  
		if(tid_blk<5)
		{
			ClipTab[tid_blk]=CLIP_Table_dev[QPcAv*5+tid_blk];
		}
		for(int i = 0;i< NUM_Y;i++)
		{
			B_strenght_uv[tid_blk] = BSRef[blk_x*32+tid_blk+i*NUM_X*16+NUM_X*NUM_Y*16];//加载每个宏块对应的边界强度到共享存储区，分2次使用
			for(int j =0 ;j<10;j++)
			{
				Ref_pixels_uv[tid_x + j*16+10*16*tid_y] = (tid_y==0) ? Reference_U[start_ref + (j*Width_ref>>1) + (i*Width_ref*MB_HEIGHT>>2)] : Reference_V[start_ref + (j*Width_ref>>1) +(i*Width_ref*MB_HEIGHT>>2)];
			}
			for(int k=0;k<2;k++)
			{
				//Strng = B_strenght_uv[(tid_x>>1)+(k<<3)+(tid_x>>3)<<4]; 
				Strng = B_strenght_uv[((tid_x&7)>>1)+(k<<3)+((tid_x>>3)<<4)]; 
				if(Strng) //需要滤波
				{
					R0 =Ref_pixels_uv[k*4*16+start_shared_mem];
					R1 =Ref_pixels_uv[k*4*16+start_shared_mem+16];
					L0 =Ref_pixels_uv[k*4*16+start_shared_mem-16];
					L1 =Ref_pixels_uv[k*4*16+start_shared_mem-2*16];
					AbsDelta  = abs( Delta = (R0 - L0) );
					C0  = (int)ClipTab[ Strng ];

					if( AbsDelta < Alpha &&(((abs( R0 - R1) - Beta )  & (abs(L0 - L1) - Beta )) < 0))
					{
						/*aq  = (abs( R0 - R2) - Beta ) < 0  ;
						ap  = (abs( L0 - L2) - Beta ) < 0  ;*/
						RL0 = L0 + R0 ;
						c0  = C0+1;
						dif =( (Delta << 2) + (L1 - R1) + 4) >> 3;
                        dif = Select_Middle( -c0,c0,dif);

						Ref_pixels_uv[k*4*16+start_shared_mem]	= (Strng==4) ? (unsigned char)(((R1 << 1) + R0 + L1 + 2) >> 2) :(unsigned char)Select_Middle(0,255,(R0 - dif));
                        Ref_pixels_uv[k*4*16+start_shared_mem-16] = (Strng==4) ? (unsigned char)(((L1 << 1) + L0 + R1 + 2) >> 2) :(unsigned char)Select_Middle(0,255,(L0 + dif));
					}
				
				}
			}
			for(int j =0;j<10;j++)
			{

				//Ref_pixels[tid_x + j*16+10*16*tid_y] = (tid_y) ? Reference_U[start_row + (j*Width_ref>>1) + (i*Width_ref*MB_HEIGHT>>2)] : Reference_V[start_row + (j*Width_ref>>1) +(i*Width_ref*MB_HEIGHT>>2)];
				if(tid_y==0)
					Reference_U[start_ref + (j*Width_ref>>1) + (i*Width_ref*MB_HEIGHT>>2)] = Ref_pixels_uv[tid_x + j*16];
				if(tid_y==1)
					Reference_V[start_ref + (j*Width_ref>>1) + (i*Width_ref*MB_HEIGHT>>2)] = Ref_pixels_uv[tid_x + j*16+10*16];
			}	
		}
	
	}
}
