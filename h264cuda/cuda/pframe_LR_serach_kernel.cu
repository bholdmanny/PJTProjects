#include "../inc/me_context.h"
#include "../inc/const_defines.h"
#include "math.h"
#include <cutil_inline.h>
#include <cutil_inline_runtime.h>
#include <cutil.h>

inline __device__ short ComputeMedia(short &a,short &b,short &c)
{	
	short min,max,media;
	int sum;
	min = (b<c) ? b : c;
	min = (min<a) ? min : a;
	max = (b>c) ? b : c;
	max = (max>a) ? max : a;
	sum = a + b + c;
	media = (short)(sum - min - max);
	return media;
}


inline __device__ int Cal_QRMBSAD(unsigned int *src_QR,unsigned int *ref_QR,int &stride1,int &stride2)
								    
{
  	
   int SAD=0;
  
   SAD += abs((int)((unsigned char)src_QR[0]-ref_QR[0]));
   SAD += abs((int)((unsigned char)src_QR[1]-ref_QR[1]));
   SAD += abs((int)((unsigned char)src_QR[2]-ref_QR[2]));
   SAD += abs((int)((unsigned char)src_QR[3]-ref_QR[3]));

   SAD += abs((int)((unsigned char)src_QR[0+stride1]-ref_QR[0+stride2]));
   SAD += abs((int)((unsigned char)src_QR[1+stride1]-ref_QR[1+stride2]));
   SAD += abs((int)((unsigned char)src_QR[2+stride1]-ref_QR[2+stride2]));
   SAD += abs((int)((unsigned char)src_QR[3+stride1]-ref_QR[3+stride2]));

   SAD += abs((int)((unsigned char)src_QR[0+stride1*2]-ref_QR[0+stride2*2]));
   SAD += abs((int)((unsigned char)src_QR[1+stride1*2]-ref_QR[1+stride2*2]));
   SAD += abs((int)((unsigned char)src_QR[2+stride1*2]-ref_QR[2+stride2*2]));
   SAD += abs((int)((unsigned char)src_QR[3+stride1*2]-ref_QR[3+stride2*2]));

   SAD += abs((int)((unsigned char)src_QR[0+stride1*3]-ref_QR[0+stride2*3]));
   SAD += abs((int)((unsigned char)src_QR[1+stride1*3]-ref_QR[1+stride2*3]));
   SAD += abs((int)((unsigned char)src_QR[2+stride1*3]-ref_QR[2+stride2*3]));
   SAD += abs((int)((unsigned char)src_QR[3+stride1*3]-ref_QR[3+stride2*3]));
   return SAD;


}

//1/4像素低分辨率全搜索，每个section包含最多6x3的4x4MB，有边界和下边界做特殊处理
//__global__ void me_QR_LowresSearch(unsigned char *src_in, 
//								  unsigned char *ref, 
//								  CUVME_MV_RESULTS *ptr_mvsLocal,
//								  int width,
//								  int height,
//								  int width_c,
//								  int height_c,
//								  int weight,
//								  int BSize,
//								  int ZeroBias, 
//								  int lambda_factor,
//								  int skipbias_factor
//								)
//{
//
//	__shared__ unsigned int ref_QR[40*28];
//	__shared__ unsigned int src_QR[16];
//	__shared__ short2  PredMV[6*3];
//
//	__shared__ int QR_SAD[256];
//	__shared__ int Best_tid_block;
//	short2 CurrMV,Predmv,MV_diff;
//	int temp_tid;
//	int ThisSAD,costs;
//	int BiasForSkip;
//	BiasForSkip = skipbias_factor*5;
//
//	int width_ref_shared=40;
//	int width_src_shared=4;
//	int x,y;
//	int tid_x = threadIdx.x;
//	int tid_y = threadIdx.y;
//	int tid_block = tid_x + tid_y*blockDim.x;
//	int src_index; //每一个block对应的源数据的起始位置。
//	int ref_index_x1,ref_index_x2,ref_index_y1,ref_index_y2; //每个block对应的参考帧的起始位置（第一行和第一列block还要进行修正）
//	int NumColsMB_Section,NumRowsMB_Section;
//
//	int i,j,k;
//	
//	//线程对gmem的访问是按照4跳跃的，但是对共享存储器的写则不存在bank conflict
//	ref_index_x1 =(blockIdx.x==0) ? ((tid_x<=4) ? 0 : (tid_x - 4)) : ((tid_x+(blockIdx.x*6*4)-4));
//	ref_index_y1 =(blockIdx.y==0) ? ((tid_y<4) ? 0 :(tid_y - 4)) : ((tid_y + (blockIdx.y*3*4)-4));
//
//	ref_index_x2 =(blockIdx.x==0) ? (tid_x+12) : ((tid_x+(blockIdx.x*6*4)-4)+16);
//	ref_index_x2 = ((ref_index_x2<(width_c-1)) ? ref_index_x2 : (width_c-1));
//
//	ref_index_y2 =(blockIdx.y==0) ? (tid_y+12) : ((tid_y+(blockIdx.y*3*4)-4)+16);
//	ref_index_y2 = ((ref_index_y2<(height_c-1)) ? ref_index_y2 : (height_c-1));
//		
//	ref_QR[tid_x+(tid_y*40)] = (unsigned int)ref[ref_index_x1 + ref_index_y1*width_c];
//	if(tid_y<12)
//		ref_QR[tid_x+(tid_y+16)*40] = (unsigned int)ref[ref_index_x1 +  ref_index_y2*width_c];
//
//
//	ref_QR[tid_x+(tid_y*40)+16] = (unsigned int)ref[ref_index_x2 + ref_index_y1*width_c];
//	if(tid_y<12)
//		ref_QR[tid_x+(tid_y+16)*40+16] = (unsigned int)ref[ref_index_x2 + ref_index_y2*width_c];
//
//	ref_index_x2 = (ref_index_x2+16);
//	ref_index_x2 = ((ref_index_x2<(width_c-1)) ? ref_index_x2 : (width_c-1));
//
//	if(tid_x < 8)
//	{
//		ref_QR[tid_x+(tid_y*40)+32] = (unsigned int)ref[ref_index_x2 + ref_index_y1*width_c];
//		if(tid_y < 12)
//			ref_QR[tid_x+(tid_y+16)*40+32] = (unsigned int)ref[ref_index_x2 + ref_index_y2*width_c];
//	}
//	__syncthreads();
//	
//	NumColsMB_Section = (blockIdx.x!=(gridDim.x-1)) ? 6 : ((width>>2)- ((gridDim.x-1)*6));
//	NumRowsMB_Section = (blockIdx.y!=(gridDim.y-1)) ? 3 : ((height>>2)- ((gridDim.y-1)*3));
//
//	//每一遍循环处理一个宏块的预测，得到的pred值可能在下一编循环中使用，每个subsection的大小为6*4
//	for(i=0;i<NumRowsMB_Section;i++)
//	{
//		for(j=0;j<NumColsMB_Section;j++)
//		{
//			Predmv.x = 0;
//			Predmv.y = 0;
//
//			x = tid_x - 8;
//			x = ((blockIdx.x == 0)&&(j==0)&&(x < -4)) ? -4 : x;
//			x = ((blockIdx.x == (gridDim.x-1))&&(j==(NumColsMB_Section-1))&&(x >=4)) ? 3 : x;
//
//			y = tid_y - 8;
//			y = ((blockIdx.y == 0)&&(i==0)&&(y < -4)) ? -4 : y;
//			y = ((blockIdx.y == (gridDim.y-1))&&(i==(NumRowsMB_Section-1))&&(y >=4)) ? 3 : y;
//			
//			CurrMV.x = (x<<4);
//			CurrMV.y = (y<<4);
//
//			if(i) //top avail
//			{
//				Predmv.x = PredMV[(i-1)*6+j].x;
//                Predmv.y = PredMV[(i-1)*6+j].y;
//			}
//			if(j) //left avail
//			{
//				 Predmv.x = PredMV[i*6+j-1].x;
//				 Predmv.y = PredMV[i*6+j-1].y;
//			}
//			if((i)&&(j)&&((j+1)<NumColsMB_Section)) //top right avail
//			{
//                Predmv.x = ComputeMedia(PredMV[i*6+j-1].x, 
//                            PredMV[(i-1)*6+j].x,
//                            PredMV[(i-1)*6+j+1].x);
//
//                Predmv.y = ComputeMedia(PredMV[i*6+j-1].y, 
//                            PredMV[(i-1)*6+j].y,
//                            PredMV[(i-1)*6+j+1].y);
//            }
//
//			src_index = blockIdx.x * BSize * 6 + blockIdx.y * width * BSize * 3 + j * BSize + i * BSize * width;
//			if(tid_y==0)
//			{
//				src_QR[tid_x]=src_in[src_index+(tid_x>>2)*width+(tid_x&3)];
//			}
//			__syncthreads();
//
//			ref_index_x1 = tid_x+j*4;
//			ref_index_y1 = tid_y+i*4;
//
//			MV_diff.x = abs(Predmv.x - CurrMV.x);
//			MV_diff.y = abs(Predmv.y - CurrMV.y);
//			costs = (MV_diff.x + MV_diff.y)>>1;
//
//			//ThisSAD= Calc_QRMBSAD(src_row0,src_row1,src_row2,src_row3,ref_row0,ref_row1,ref_row2,ref_row3); //第一个subsection的MB的预测SAD处在基数为偶数的位置
//																											//第二个subsection的MB的预测SAD处在基数为奇数的位置，这样是为了避免bank conflict
//			ThisSAD= Cal_QRMBSAD(src_QR,ref_QR+ref_index_x1+ref_index_y1*width_ref_shared,width_src_shared,width_ref_shared);																				 
//			ThisSAD = ThisSAD + BiasForSkip; 
//			// Subtract new bias if Curr == Pred
//			if((MV_diff.x == 0) && (MV_diff.y == 0))
//			{
//				ThisSAD -= BiasForSkip;
//			}
//			ThisSAD += costs;
//			QR_SAD[tid_block] = (int)ThisSAD;
//			__syncthreads();
//
//			if(tid_block==0)	
//			{
//				Best_tid_block =256;
//			}
//			// select the best SAD
//			for(k = 128 ; k >0 ; k>>=1)
//			{
//				if(tid_block<k)
//				{
//					if((QR_SAD[tid_block] > QR_SAD[tid_block+k]))
//					{
//						QR_SAD[tid_block] = QR_SAD[tid_block+k];
//					}
//				}
//				__syncthreads();
//			}
//			/*__syncthreads();*/
//
//			if(ThisSAD == (short)QR_SAD[0])
//			{
//				temp_tid = atomicMin(&Best_tid_block,tid_block);
//			}
//			__syncthreads();
//			if(tid_block == Best_tid_block)
//			{
//				PredMV[i*6+j].x = CurrMV.x;
//				PredMV[i*6+j].y = CurrMV.y;
//				ptr_mvsLocal[blockIdx.y*(width>>2)*3+blockIdx.x*6+i*(width>>2)+j].MV_X =  CurrMV.x;
//				ptr_mvsLocal[blockIdx.y*(width>>2)*3+blockIdx.x*6+i*(width>>2)+j].MV_Y =  CurrMV.y;
//			}
//			__syncthreads(); //有可能会死锁
//			
//		}
//		__syncthreads(); //有可能会死锁
//	}
//
//}

__global__ void me_QR_LowresSearch(unsigned char *src_in, 
								  unsigned char *ref, 
								  CUVME_MV_RESULTS *ptr_mvsLocal,
								  int width,
								  int height,
								  int width_c,
								  int height_c,
								  int num_mb_hor,
								  int num_mb_ver,
								  int weight,
								  int BSize,
								  int ZeroBias, 
								  int lambda_factor,
								  int skipbias_factor
								)
{

	__shared__ unsigned int ref_QR[40*28];
	__shared__ unsigned int src_QR[16];
	__shared__ short2  PredMV[6*3];

	__shared__ int QR_SAD[256];
	__shared__ int Best_tid_block;
	short2 CurrMV,Predmv,MV_diff;
	int temp_tid;
	int ThisSAD,costs;
	int BiasForSkip;
	BiasForSkip = skipbias_factor*5;

	int width_ref_shared=40;
	int width_src_shared=4;
	int x,y;
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_block = tid_x + tid_y*blockDim.x;
	int src_index; //每一个block对应的源数据的起始位置。
	int ref_index_x1,ref_index_x2,ref_index_y1,ref_index_y2; //每个block对应的参考帧的起始位置（第一行和第一列block还要进行修正）
	int NumColsMB_Section,NumRowsMB_Section;

	int i,j,k;
	
	//线程对gmem的访问是按照4跳跃的，但是对共享存储器的写则不存在bank conflict
	ref_index_x1 =(blockIdx.x==0) ? ((tid_x<=4) ? 0 : (tid_x - 4)) : ((tid_x+(blockIdx.x*6*4)-4));
	ref_index_y1 =(blockIdx.y==0) ? ((tid_y<4) ? 0 :(tid_y - 4)) : ((tid_y + (blockIdx.y*3*4)-4));

	ref_index_x2 =(blockIdx.x==0) ? (tid_x+12) : ((tid_x+(blockIdx.x*6*4)-4)+16);
	ref_index_x2 = ((ref_index_x2< (width_c-1)) ? ref_index_x2 : (width_c-1));

	ref_index_y2 =(blockIdx.y==0) ? (tid_y+12) : ((tid_y+(blockIdx.y*3*4)-4)+16);
	ref_index_y2 = ((ref_index_y2<(height_c-1)) ? ref_index_y2 : (height_c-1));
		
	ref_QR[tid_x+(tid_y*40)] = (unsigned int)ref[ref_index_x1 + ref_index_y1*width_c];
	if(tid_y<12)
		ref_QR[tid_x+(tid_y+16)*40] = (unsigned int)ref[ref_index_x1 +  ref_index_y2*width_c];

	ref_QR[tid_x+(tid_y*40)+16] = (unsigned int)ref[ref_index_x2 + ref_index_y1*width_c];
	if(tid_y<12)
		ref_QR[tid_x+(tid_y+16)*40+16] = (unsigned int)ref[ref_index_x2 + ref_index_y2*width_c];

	ref_index_x2 = (ref_index_x2+16);
	ref_index_x2 = ((ref_index_x2<(width_c-1)) ? ref_index_x2 : (width_c-1));

	if(tid_x < 8)
	{
		ref_QR[tid_x+(tid_y*40)+32] = (unsigned int)ref[ref_index_x2 + ref_index_y1*width_c];
		if(tid_y < 12)
			ref_QR[tid_x+(tid_y+16)*40+32] = (unsigned int)ref[ref_index_x2 + ref_index_y2*width_c];
	}
	__syncthreads();
	
	NumColsMB_Section = (blockIdx.x!=(gridDim.x-1)) ? 6 : (num_mb_hor - ((gridDim.x-1)*6));
	NumRowsMB_Section = (blockIdx.y!=(gridDim.y-1)) ? 3 : (num_mb_ver - ((gridDim.y-1)*3));

	//每一遍循环处理一个宏块的预测，得到的pred值可能在下一编循环中使用，每个subsection的大小为6*4
	for(i=0;i<NumRowsMB_Section;i++)
	{
		for(j=0;j<NumColsMB_Section;j++)
		{
			Predmv.x = 0;
			Predmv.y = 0;

			x = tid_x - 8;
			x = ((blockIdx.x == 0)&&(j==0)&&(x < -4)) ? -4 : x;
			x = ((blockIdx.x == (gridDim.x-1))&&(j==(NumColsMB_Section-1))&&(x >=4)) ? 3 : x;

			y = tid_y - 8;
			y = ((blockIdx.y == 0)&&(i==0)&&(y < -4)) ? -4 : y;
			y = ((blockIdx.y == (gridDim.y-1))&&(i==(NumRowsMB_Section-1))&&(y >=4)) ? 3 : y;
			
			CurrMV.x = (x<<4);
			CurrMV.y = (y<<4);

			if(i) //top avail
			{
				Predmv.x = PredMV[(i-1)*6+j].x;
                Predmv.y = PredMV[(i-1)*6+j].y;
			}
			if(j) //left avail
			{
				 Predmv.x = PredMV[i*6+j-1].x;
				 Predmv.y = PredMV[i*6+j-1].y;
			}
			if((i)&&(j)&&((j+1)<NumColsMB_Section)) //top right avail
			{
                Predmv.x = ComputeMedia(PredMV[i*6+j-1].x, 
                            PredMV[(i-1)*6+j].x,
                            PredMV[(i-1)*6+j+1].x);

                Predmv.y = ComputeMedia(PredMV[i*6+j-1].y, 
                            PredMV[(i-1)*6+j].y,
                            PredMV[(i-1)*6+j+1].y);
            }

			src_index = blockIdx.x * BSize * 6 + blockIdx.y * width * BSize * 3 + j * BSize + i * BSize * width;
			if(tid_y==0)
			{
				src_QR[tid_x]=src_in[src_index+(tid_x>>2)*width+(tid_x&3)];
			}
			__syncthreads();

			ref_index_x1 = tid_x+j*4;
			ref_index_y1 = tid_y+i*4;

			MV_diff.x = abs(Predmv.x - CurrMV.x);
			MV_diff.y = abs(Predmv.y - CurrMV.y);
			costs = (MV_diff.x + MV_diff.y)>>1;

			//ThisSAD= Calc_QRMBSAD(src_row0,src_row1,src_row2,src_row3,ref_row0,ref_row1,ref_row2,ref_row3); //第一个subsection的MB的预测SAD处在基数为偶数的位置
																											//第二个subsection的MB的预测SAD处在基数为奇数的位置，这样是为了避免bank conflict
			ThisSAD= Cal_QRMBSAD(src_QR,ref_QR+ref_index_x1+ref_index_y1*width_ref_shared,width_src_shared,width_ref_shared);																				 
			ThisSAD = ThisSAD + BiasForSkip; 
			// Subtract new bias if Curr == Pred
			if((MV_diff.x == 0) && (MV_diff.y == 0))
			{
				ThisSAD -= BiasForSkip;
			}
			ThisSAD += costs;
			QR_SAD[tid_block] = (int)ThisSAD;
			__syncthreads();

			if(tid_block==0)	
			{
				Best_tid_block =256;
			}
			// select the best SAD
			for(k = 128 ; k >0 ; k>>=1)
			{
				if(tid_block<k)
				{
					if((QR_SAD[tid_block] > QR_SAD[tid_block+k]))
					{
						QR_SAD[tid_block] = QR_SAD[tid_block+k];
					}
				}
				__syncthreads();
			}
			/*__syncthreads();*/

			if(ThisSAD == (short)QR_SAD[0])
			{
				temp_tid = atomicMin(&Best_tid_block,tid_block);
			}
			__syncthreads();
			if(tid_block == Best_tid_block)
			{
				PredMV[i*6+j].x = CurrMV.x;
				PredMV[i*6+j].y = CurrMV.y;
				ptr_mvsLocal[blockIdx.y*3*num_mb_hor+blockIdx.x*6+i*num_mb_hor+j].MV_X =  CurrMV.x;
				ptr_mvsLocal[blockIdx.y*3*num_mb_hor+blockIdx.x*6+i*num_mb_hor+j].MV_Y =  CurrMV.y;
			}
			__syncthreads(); //有可能会死锁
			
		}
		__syncthreads(); //有可能会死锁
	}

}

//1/2像素低分辨率全搜索，每个section包含最多6x3的4x4MB，右边界和下边界做特殊处理
__global__ void me_HR_Candidate_Vote(  unsigned int *HR_SAD_dev,
									  CUVME_MV_RESULTS *ptr_mvsLocal,																													
									  CUVME_MB_INFO *ptr_mb_info,
									  int width,                     																													
									  int height, 
									  int width_ref,                     																													
									  int height_ref,
									  int num_mb_hor,
									  int num_mb_ver,
									  int weight,                    																													
									  int BSize,                     																													
									  int ZeroBias,                  																													
									  int lambda_factor,             																													
									  int skipbias_factor)
{
	__shared__ unsigned int HR_SAD[32];
	__shared__ short2  PredMV[18];
	__shared__ int Best_tid_block;
	short2 CurrMV,MV_diff;
	int temp_tid;
	int ThisSAD;
	__shared__ int BiasForSkip;

	__shared__ int offset_x,offset_y,SearchStart_x,SearchStart_y,SearchEnd_x,SearchEnd_y;
	__shared__ short2 Predmv;
	int	MVinBounds,valid_x,valid_y;
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_z = threadIdx.z;
	int tid_block = tid_x + tid_y*blockDim.x+tid_z*blockDim.x*blockDim.y;

	__shared__ int section_offset_x,section_offset_y,NumColsMB_Section,NumRowsMB_Section; //每一个block对应的源数据的起始位置。
	__shared__ int ref_index_x,ref_index_y;//,load_index; //每个block对应的参考帧的起始位置（第一行和第一列block还要进行修正）

	int i,j,k;
	
	//每个block处理的原始帧的起始位置，注意一个block有两个起始位置，通过tid_z来区分
	if(tid_block == 0)
	{
		section_offset_x = blockIdx.x*BSize*6;
		section_offset_y = blockIdx.y*BSize*3;
		BiasForSkip = skipbias_factor*5*(BSize/4);
		NumColsMB_Section = (blockIdx.x!=(gridDim.x-1)) ? 6 : (num_mb_hor - ((gridDim.x-1)*6));
		NumRowsMB_Section = (blockIdx.y!=(gridDim.y-1)) ? 3 : (num_mb_ver - ((gridDim.y-1)*3));
	}
	//载入1/4分辨率搜索得到的MV值，将每个block要处理的MB的MV值放到共享存储器中
	if(tid_block < 18)
	{
		PredMV[tid_block].x = ((blockIdx.y*3*num_mb_hor+blockIdx.x*6+(tid_block/6)*num_mb_hor+(tid_block % 6))<(num_mb_hor * num_mb_ver)) ? ptr_mvsLocal[blockIdx.y*3*num_mb_hor+blockIdx.x*6+(tid_block/6)*num_mb_hor+(tid_block % 6)].MV_X : 0 ;
		PredMV[tid_block].y = ((blockIdx.y*3*num_mb_hor+blockIdx.x*6+(tid_block/6)*num_mb_hor+(tid_block % 6))<(num_mb_hor * num_mb_ver)) ? ptr_mvsLocal[blockIdx.y*3*num_mb_hor+blockIdx.x*6+(tid_block/6)*num_mb_hor+(tid_block % 6)].MV_Y : 0 ;
	}
	__syncthreads();
   
	//每一遍循环处理一个宏块的预测，得到的pred值可能在下一编循环中使用，每个subsection的大小为6*4
	for(i=0;i<NumRowsMB_Section;i++) 
	{	
		if(tid_block == 0)
		{
			offset_y = section_offset_y+i*BSize;
		}
		for(j=0;j<NumColsMB_Section;j++)
		{
			ThisSAD = HR_SAD_dev[tid_block+(blockIdx.x*6+j+(blockIdx.y*3+i)*num_mb_hor)*32];
			if(tid_block == 0)
			{
				offset_x = section_offset_x + j*BSize;

				ref_index_x = offset_x + ((PredMV[i*6+j].x)>>3); //得到当前ＭＢ在ref中搜索的中心点的X坐标
				ref_index_y = offset_y + ((PredMV[i*6+j].y)>>3); //得到当前ＭＢ在ref中搜索的中心点的Y坐标	

				ref_index_x	= ref_index_x - 4;	//得到X搜索方向的起始位置
				ref_index_y	= ref_index_y - 2;  //得到Y搜索方向的起始为置

				SearchEnd_x = ref_index_x + 8; //水平搜索的结束点，主要用于判断12个线程中哪些有效
				SearchEnd_y = ref_index_y + 4;

				// 限制搜索范围在图像的内部
				ref_index_x	= (ref_index_x< -8) ? -8 : ref_index_x;//左边界
				ref_index_y	= (ref_index_y< -8) ? -8 : ref_index_y;//上边界

				SearchStart_x = ref_index_x; //有效搜索的水平起始位置
				SearchStart_y = ref_index_y;

				ref_index_x = (width < (ref_index_x + 8)) ? (width - 8) : ref_index_x; //保证每一行至少可以加载5个字进行水平搜索，右边界
				ref_index_y = (height < (ref_index_y + 4)) ? (height - 4) : ref_index_y;

				Predmv.x = 0;
				Predmv.y = 0;
				if(i) //top avail
				{
					Predmv.x = PredMV[(i-1)*6+j].x;
					Predmv.y = PredMV[(i-1)*6+j].y;
				}
				if(j) //left avail
				{
					 Predmv.x = PredMV[i*6+j-1].x;
					 Predmv.y = PredMV[i*6+j-1].y;
				}
				if((i)&&(j)&&((j+1)<NumColsMB_Section)) //top right avail
				{
            				Predmv.x = ComputeMedia(PredMV[i*6+j-1].x, 
                        							PredMV[(i-1)*6+j].x,
                        							PredMV[(i-1)*6+j+1].x);

            				Predmv.y = ComputeMedia(PredMV[i*6+j-1].y, 
                        							PredMV[(i-1)*6+j].y,
                        							PredMV[(i-1)*6+j+1].y);
				}
				Best_tid_block = 32;
			}
			__syncthreads();
			
			CurrMV.x = (ref_index_x - offset_x) + tid_x;
			CurrMV.y = (ref_index_y - offset_y) + tid_y;
			MVinBounds = (CurrMV.x< 16)&&
					(CurrMV.y< 16)&&
					(CurrMV.x>= -16)&&
					(CurrMV.y>= -16);

			CurrMV.x = CurrMV.x << 3;
			CurrMV.y = CurrMV.y << 3;
			
			MV_diff.x = abs(Predmv.x - CurrMV.x);
			MV_diff.y = abs(Predmv.y - CurrMV.y);
			//costs = (MV_diff.x + MV_diff.y);
		
			valid_x = ((ref_index_x + tid_x) >= SearchStart_x ) && ((ref_index_x + tid_x) <  SearchEnd_x);
			valid_y = ((ref_index_y + tid_y) >= SearchStart_y ) && ((ref_index_y + tid_y) <  SearchEnd_y);
			if((valid_x==0)||(valid_y==0)||(MVinBounds==0))	
			{
				ThisSAD = 0x7fff;
			}

			ThisSAD = ThisSAD + BiasForSkip; 
			// Subtract new bias if Curr == Pred
			if((MV_diff.x == 0) && (MV_diff.y == 0))
			{
				ThisSAD -= BiasForSkip;
			}
			ThisSAD += (MV_diff.x + MV_diff.y);
			HR_SAD[tid_block] = ThisSAD;
			__syncthreads();

			for(k = 16 ; k > 0 ; k>>=1)
			{
				if(tid_block < k)
				{
					if((HR_SAD[tid_block]) > (HR_SAD[tid_block+k]))
					{
						HR_SAD[tid_block ] = HR_SAD[tid_block +k];
					}
				}
				__syncthreads();
			}
			if(ThisSAD == HR_SAD[0])
			{
				temp_tid = atomicMin(&Best_tid_block,tid_block);
			}
			__syncthreads();
			
			if(tid_block == Best_tid_block)
			{
				PredMV[i*6+j].x = CurrMV.x;
				PredMV[i*6+j].y = CurrMV.y;
				ptr_mvsLocal[blockIdx.y*3*num_mb_hor+blockIdx.x*6+i*num_mb_hor+j].MV_X =  CurrMV.x;
				ptr_mvsLocal[blockIdx.y*3*num_mb_hor+blockIdx.x*6+i*num_mb_hor+j].MV_Y =  CurrMV.y;
				ptr_mb_info[blockIdx.y*3*num_mb_hor+blockIdx.x*6+i*num_mb_hor+j].SAD = ThisSAD;
			}
			__syncthreads();
			
		}
		__syncthreads(); 
	}
	}