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

inline __device__ short me_ComputeMedia_Pred(short &a,short &b,short &c)
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

inline __device__ int Calc_SAD(unsigned char *src_in,unsigned char *ref_in,int &strid_src,int &strid_ref) 
///inline __device__ int Calc_SAD(unsigned int *src_in,unsigned int *ref_in,int &strid_src,int &strid_ref)
{
	int SAD = 0;
	SAD +=(int)abs(src_in[0]-ref_in[0]);
	SAD +=(int)abs(src_in[1]-ref_in[1]);
	SAD +=(int)abs(src_in[2]-ref_in[2]);
	SAD +=(int)abs(src_in[3]-ref_in[3]);

	SAD +=(int)abs(src_in[strid_src]-ref_in[strid_ref]);
	SAD +=(int)abs(src_in[strid_src+1]-ref_in[strid_ref+1]);
	SAD +=(int)abs(src_in[strid_src+2]-ref_in[strid_ref+2]);
	SAD +=(int)abs(src_in[strid_src+3]-ref_in[strid_ref+3]);

	SAD +=(int)abs(src_in[strid_src*2]-ref_in[strid_ref*2]);
	SAD +=(int)abs(src_in[strid_src*2+1]-ref_in[strid_ref*2+1]);
	SAD +=(int)abs(src_in[strid_src*2+2]-ref_in[strid_ref*2+2]);
	SAD +=(int)abs(src_in[strid_src*2+3]-ref_in[strid_ref*2+3]);

	SAD +=(int)abs(src_in[strid_src*3]-ref_in[strid_ref*3]);
	SAD +=(int)abs(src_in[strid_src*3+1]-ref_in[strid_ref*3+1]);
	SAD +=(int)abs(src_in[strid_src*3+2]-ref_in[strid_ref*3+2]);
	SAD +=(int)abs(src_in[strid_src*3+3]-ref_in[strid_ref*3+3]);

	   /*SAD += abs((int)((unsigned char)src_in[0]-ref_in[0]));
	   SAD += abs((int)((unsigned char)src_in[1]-ref_in[1]));
	   SAD += abs((int)((unsigned char)src_in[2]-ref_in[2]));
	   SAD += abs((int)((unsigned char)src_in[3]-ref_in[3]));

	   SAD += abs((int)((unsigned char)src_in[0+strid_src]-ref_in[0+strid_ref]));
	   SAD += abs((int)((unsigned char)src_in[1+strid_src]-ref_in[1+strid_ref]));
	   SAD += abs((int)((unsigned char)src_in[2+strid_src]-ref_in[2+strid_ref]));
	   SAD += abs((int)((unsigned char)src_in[3+strid_src]-ref_in[3+strid_ref]));

	   SAD += abs((int)((unsigned char)src_in[0+strid_src*2]-ref_in[0+strid_ref*2]));
	   SAD += abs((int)((unsigned char)src_in[1+strid_src*2]-ref_in[1+strid_ref*2]));
	   SAD += abs((int)((unsigned char)src_in[2+strid_src*2]-ref_in[2+strid_ref*2]));
	   SAD += abs((int)((unsigned char)src_in[3+strid_src*2]-ref_in[3+strid_ref*2]));

	   SAD += abs((int)((unsigned char)src_in[0+strid_src*3]-ref_in[0+strid_ref*3]));
	   SAD += abs((int)((unsigned char)src_in[1+strid_src*3]-ref_in[1+strid_ref*3]));
	   SAD += abs((int)((unsigned char)src_in[2+strid_src*3]-ref_in[2+strid_ref*3]));
	   SAD += abs((int)((unsigned char)src_in[3+strid_src*3]-ref_in[3+strid_ref*3]));*/
	return SAD;

}
//计算一个宏块的代价，第一个参数是预测MV与当前MV的差，第二个参数是cost系数
inline __device__ int  me_Cacl_Cost(short2 &costMV,int &costFactor,int &SAD_MB)
{
    int CostVector_X,CostVector_Y,lxff1,lyff1,costlx, costly,cost;

	CostVector_X = costMV.x;
	CostVector_Y = costMV.y;
    CostVector_X = (abs(CostVector_X)>255) ? 255 : abs(CostVector_X);
    CostVector_Y = (abs(CostVector_Y)>255) ? 255 : abs(CostVector_Y);
    lxff1 = 32;
    lyff1 = 32;
    while (CostVector_X) 
	{
		CostVector_X>>=1; 
		lxff1 -=1;
	}
    while (CostVector_Y) 
	{
		CostVector_Y>>=1; 
		lyff1 -=1;
	}
    lxff1 =((32-lxff1)<<1)+1;  //32
    lyff1 =((32-lyff1)<<1)+1;
    costlx = costFactor; //default if cliplx = 0
    costly = costFactor;
    if (costMV.x!=0) 
		costlx = lxff1 * costFactor;
    if (costMV.y!=0) 
		costly = lyff1 * costFactor;
    cost = costlx + costly;
	cost += SAD_MB;
    return cost;
}
__global__ void me_ClipVec_ForFrame(CUVME_MV_RESULTS *dev_LRMV,unsigned int *IntegerPelCenterVecs,int search_range_x,
									int search_range_y, int integer_clip_range,int NumRowMBs,int NumColMBs)
{
	int tid_grid_x = threadIdx.x;
	int tid_grid_y = blockIdx.y;
	int tid_grid = tid_grid_x + (blockDim.x*blockIdx.y);

	int MinX;
    int MaxX;
    int MinY;
    int MaxY;
	int width, height;
	unsigned int IntCands0;
	
	short2 MB_mv;
	MB_mv.x = dev_LRMV[tid_grid].MV_X;
	MB_mv.y = dev_LRMV[tid_grid].MV_Y;
	
    width =  NumRowMBs << 4;
    height = NumColMBs<< 4;
	/*if(tid_grid_y <2 )
	{
		dev_ref_index[tid_grid] = tid_grid;
	}*/

    MinX = (-search_range_x + integer_clip_range) << 2;
    MaxX = (search_range_x - integer_clip_range) << 2;
	MinY = (-search_range_y + integer_clip_range) << 2;
    MaxY = (search_range_y - integer_clip_range) << 2;
 
    if ( MB_mv.x < MinX)
        MB_mv.x = MinX;
    if ( MB_mv.x > MaxX)
        MB_mv.x = MaxX;
    if ( MB_mv.y < MinY)
        MB_mv.y = MinY;
    if ( MB_mv.y > MaxY)
        MB_mv.y = MaxY;

	
    // The +/-1 below is to cover the 3x3 search range
    // vec's are usually in subpel precision, so limits are << 2.
	MinX = (-16 - (tid_grid_x << 4) + integer_clip_range) << 2;
	MaxX = (width - (tid_grid_x << 4) - integer_clip_range) << 2;
	MinY = (-16 - (tid_grid_y << 4) + integer_clip_range) << 2;
    MaxY = (height - (tid_grid_y << 4) - integer_clip_range) << 2;
 
    if ( MB_mv.x < MinX)
        MB_mv.x = MinX;
    if ( MB_mv.x > MaxX)
        MB_mv.x = MaxX;
    if ( MB_mv.y < MinY)
        MB_mv.y = MinY;
    if ( MB_mv.y > MaxY)
        MB_mv.y = MaxY;
	
    //to deal with ME_STRIP_SIZE < LenX we need to just go ahead an rid subpel bits here.
	MB_mv.x = (MB_mv.x >> 2) << 2;
	MB_mv.y = (MB_mv.y >> 2) << 2;

	IntCands0 = (MB_mv.x & 0xffff)|((MB_mv.y & 0xffff) << 16 );
	IntegerPelCenterVecs[tid_grid] = IntCands0;

}


__global__ void me_IntegerSimulsadVote_kernel(unsigned char *dev_input,unsigned char *dev_ref,unsigned int *dev_MB_MV,CUVME_MV_RESULTS *integer_mvmap,CUVME_MB_INFO *mb_info,
											  unsigned char *dev_out_pred,int num_mb_hor,int num_mb_ver,int refStride2Begin,int costFactor,int width, int width_ref,S_BLK_MB_INFO *pBlkMBInfo_dev)
{
	
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_z = threadIdx.z;
	int tid_block = tid_x + tid_y*blockDim.x + tid_z*blockDim.x*blockDim.y;
	int src_index ;
	int ref_index ;
	int search_index,tid_index_in_refMB,tid_index_in_srcMB,tid_index_out_refMB;
	int blcok_size = blockDim.x*blockDim.y*blockDim.z;
	tid_index_in_refMB = (tid_x&8)*18+(tid_x&4)*2+(tid_x&2)*2*18 + (tid_x&1)*4 + tid_y*18 +tid_z;//计算每一个threads对应的ref的4x4块的起始位置，相对于shared memory,处理顺序与C相同，先列后行
	tid_index_in_srcMB = tid_x * 16; //计算每一个thread 对应的src的4x4块的起始位置，每一个线程处理一个block,而y,z变量则对应一次16*16的SAD计算
	
	int stride_src = 4;
	int stride_ref =18;

	__shared__ unsigned char src_MB[256];
	__shared__ unsigned char ref_MB[324];//(16+2)*(16*2)
	__shared__ int SAD_4x4[144];
	__shared__ int cost_MB[9];
	int cost_add_SAD;
	__shared__ short2 PredMV,ref_out_offset;
	__shared__ int Best_tid;
	int temp_tid;
	short2 Candidate_MV;
	short2 costMV;
	short west_x,west_y,neast_x,neast_y,north_x,north_y;

	unsigned int center;
	int j;
	if(tid_block==0 )
	{
		{
			PredMV.x =0;
			PredMV.y =0;
		}
	}
	for(int i = 0; i< num_mb_ver; i++ )
	{

		src_index = (blockIdx.x+i*num_mb_hor)*256;
	    ref_index = (blockIdx.x+i*width_ref)*16+refStride2Begin;

		//将每次要处理的原始宏块加载到共享存储器，注意：每一个宏块是block_flat格式的
		src_MB[tid_block] = dev_input[src_index+tid_block];
		if(tid_block <112)
		{
			src_MB[tid_block+blcok_size] = dev_input[src_index+tid_block+blcok_size];
		}	
		center = dev_MB_MV[blockIdx.x+i*num_mb_hor];

		Candidate_MV.x = center& 0xffff;
		Candidate_MV.y = (short)((center >> 16)& 0xffff);

		Candidate_MV.x = (Candidate_MV.x - 4); //搜索的左上角的坐标位置+-1个像素。因为Candidate_MV是从LRSF得到的，故X4
		Candidate_MV.y = (Candidate_MV.y - 4);
		//加载每一个MB对应的参数数据，18*18byte
		search_index = ref_index + ((Candidate_MV.y>>2)*width_ref) + (Candidate_MV.x>>2); //当前处理宏块对应的ref的起始位置，加载18*18大小的块

		ref_MB [tid_x+(tid_y+tid_z*blockDim.y)*18] = dev_ref[tid_x+(tid_y+tid_z*blockDim.y)*width_ref+search_index];
		ref_MB [tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*18] = dev_ref[tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*width_ref+search_index];

		if(tid_x<2)
		{
			ref_MB [tid_x+(tid_y+tid_z*blockDim.y)*(18)+16] = dev_ref[tid_x+(tid_y+tid_z*blockDim.y)*width_ref+16+search_index];
			ref_MB [tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*(18)+16] = dev_ref[tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*width_ref+16+search_index];
		}
		__syncthreads();

		SAD_4x4[tid_block]=Calc_SAD(src_MB+tid_index_in_srcMB,
									ref_MB+tid_index_in_refMB,
									stride_src,
									stride_ref);
		__syncthreads();


		for( j = (blockDim.x>>1);j>0;j>>=1)
		{
			if(tid_x<j)
			{
				SAD_4x4[tid_x+(tid_y+tid_z*blockDim.y)*blockDim.x] += SAD_4x4[tid_x+j+(tid_y+tid_z*blockDim.y)*blockDim.x];
			}
			__syncthreads();
		}

		if(tid_block < 9)
		{
			costMV.x = PredMV.x - (Candidate_MV.x+(tid_block/3)*4);
			costMV.y = PredMV.y - (Candidate_MV.y+(tid_block%3)*4);
			cost_add_SAD = me_Cacl_Cost(costMV,costFactor,SAD_4x4[tid_block*blockDim.x]);
			cost_MB[tid_block] = cost_add_SAD;
		}
		//__syncthreads();

		if(tid_block < 4)
		{
			if(cost_MB[tid_block] > cost_MB[tid_block+4])
			{
				cost_MB [tid_block] = cost_MB[tid_block+4];
			}
		}
		//__syncthreads();

		if(tid_block < 2)
		{
			if(cost_MB[tid_block] > cost_MB[tid_block+2])
			{
				cost_MB [tid_block] = cost_MB[tid_block+2];
			}
		}
		//__syncthreads();

		if(tid_block < 1)
		{
			if(cost_MB[tid_block] > cost_MB[tid_block+1])
			{
				cost_MB [tid_block] = cost_MB[tid_block+1];
			}
		}
		//__syncthreads();

		if(tid_block==0)
		{
			if(cost_MB[tid_block] > cost_MB[tid_block+8])
			{
				cost_MB [tid_block] = cost_MB[tid_block+8];
			}
			Best_tid = 256;
		}
		__syncthreads();

		if((cost_add_SAD == cost_MB[0]) && (tid_block < 9))
		{
			temp_tid = atomicMin(&Best_tid,tid_block);
		}
		__syncthreads();
		if(tid_block == Best_tid&&(tid_block<9))
		{
			mb_info[i*(num_mb_hor)+blockIdx.x].SAD = cost_MB[0];
			integer_mvmap[i*(num_mb_hor)+blockIdx.x].MV_X = (Candidate_MV.x+(tid_block/3)*4);
			integer_mvmap[i*(num_mb_hor)+blockIdx.x].MV_Y = (Candidate_MV.y+(tid_block%3)*4);
			ref_out_offset.x = (tid_block/3);
			ref_out_offset.y = (tid_block%3);
			PredMV.x = (Candidate_MV.x+(tid_block/3)*4);
			PredMV.y = (Candidate_MV.y+(tid_block%3)*4);

		}
		__syncthreads();
		
		//以raster的方式将预测数据存到对应的空间
		tid_index_out_refMB = ref_out_offset.x + ref_out_offset.y*18 +tid_x + tid_y*18 + tid_z*blockDim.y*18;
							  //(tid_x%4)+(tid_x/4)18+计算每一个threads对应的ref的4x4块的起始位置，相对于shared memory,处理顺序与C相同，先列后行
		
		if(tid_block < 128)
		{
			dev_out_pred[(blockIdx.x+i*width)*16+tid_x + tid_y*width + tid_z*blockDim.y*width] = ref_MB[tid_index_out_refMB];
			dev_out_pred[(blockIdx.x+i*width)*16+tid_x + tid_y*width + tid_z*blockDim.y*width+8*width] = ref_MB[tid_index_out_refMB+144];
		}

		if(tid_block < 16)
		{
				pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].Type = INTER_LARGE_BLOCKS_MB_TYPE;
				pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].SubType = SINGLE_BLOCK_SUBDIV_TYPE;
				pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].MinSAD = cost_MB[0];
				pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].MV.x = PredMV.x;
				pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].MV.y = PredMV.y;

				pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].SliceNum         = 0;
	            pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].CBP              = 0;
	            pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].IntraChromaMode  = -1;
	            pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].TotalCoeffLuma	= -1;
	            pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].TotalCoeffChroma = -1;
	            pBlkMBInfo_dev[(i*(num_mb_hor)+blockIdx.x)*BLOCKS_PER_MB+tid_block].RefFrameIdx      = 0;
		}

		}
}


//__global__ void me_IntegerSimulsadVote_kernel(unsigned char *dev_input,unsigned char *dev_ref,unsigned int *dev_MB_MV,SPVME_MV_RESULTS *integer_mvmap,SPVME_MB_INFO *mb_info,
//											  unsigned char *dev_out_pred,int num_mb_hor,int num_mb_ver,int refStride2Begin,int costFactor,int i)
//{
//	
//	int tid_x = threadIdx.x;
//	int tid_y = threadIdx.y;
//	int tid_z = threadIdx.z;
//	int tid_block = tid_x + tid_y*blockDim.x + tid_z*blockDim.x*blockDim.y;
//	int src_index = (blockIdx.x+i*num_mb_hor)*256;
//	int ref_index = (blockIdx.x+i*1952)*16+refStride2Begin;
//	int search_index,tid_index_in_refMB,tid_index_in_srcMB,tid_index_out_refMB;
//	int blcok_size = blockDim.x*blockDim.y*blockDim.z;
//	tid_index_in_refMB = (tid_x&8)*18+(tid_x&4)*2+(tid_x&2)*2*18 + (tid_x&1)*4 + tid_y*18 +tid_z;//计算每一个threads对应的ref的4x4块的起始位置，相对于shared memory,处理顺序与C相同，先列后行
//	tid_index_in_srcMB = tid_x * 16; //计算每一个thread 对应的src的4x4块的起始位置，每一个线程处理一个block,而y,z变量则对应一次16*16的SAD计算
//	
//	int stride_src = 4;
//	int stride_ref =18;
//
//	__shared__ unsigned char src_MB[256];
//	__shared__ unsigned char ref_MB[324];//(16+2)*(16*2)
//	__shared__ int SAD_4x4[144];
//	__shared__ int cost_MB[9];
//	int cost_add_SAD;
//	__shared__ short2 PredMV,ref_out_offset;
//	__shared__ int Best_tid;
//	int temp_tid;
//	short2 Candidate_MV;
//	short2 costMV;
//	short west_x,west_y,neast_x,neast_y,north_x,north_y;
//
//	unsigned int center;
//	int j;
//	//初始化第一行宏块上边宏块预测MV，初始值为0
//	if(tid_block==0 )
//	{
//		if(i==0)
//		{
//			PredMV.x =0;
//			PredMV.y =0;
//		}
//		else
//		{
//		
//			north_x = integer_mvmap[(i-1)*(num_mb_hor)+blockIdx.x].MV_X;
//			north_y = integer_mvmap[(i-1)*(num_mb_hor)+blockIdx.x].MV_Y;
//			west_x	= (blockIdx.x==0) ? 0 : integer_mvmap[(i-1)*(num_mb_hor)+blockIdx.x-1].MV_X;
//			west_y  = (blockIdx.x==0) ? 0 : integer_mvmap[(i-1)*(num_mb_hor)+blockIdx.x-1].MV_Y;
//			neast_x = (blockIdx.x==(gridDim.x-1)) ? 0 : integer_mvmap[(i-1)*(num_mb_hor)+blockIdx.x+1].MV_X;
//			neast_y	= (blockIdx.x==(gridDim.x-1)) ? 0 : integer_mvmap[(i-1)*(num_mb_hor)+blockIdx.x+1].MV_Y;
//			PredMV.x = me_ComputeMedia_Pred(north_x,west_x,neast_x);
//			PredMV.y = me_ComputeMedia_Pred(north_y,west_y,neast_y);
//
//			if (((north_x == 0)
//			 && (north_y == 0))
//			||((west_x == 0)
//			&& (west_y == 0)))
//			{
//				PredMV.x =0;
//				PredMV.y =0;
//			}
//		}
//	}
//
//		//将每次要处理的原始宏块加载到共享存储器，注意：每一个宏块是block_flat格式的
//		src_MB[tid_block] = dev_input[src_index+tid_block];
//		if(tid_block <112)
//		{
//			src_MB[tid_block+blcok_size] = dev_input[src_index+tid_block+blcok_size];
//		}	
//		center = dev_MB_MV[blockIdx.x+i*num_mb_hor];
//
//		Candidate_MV.x = center& 0xffff;
//		Candidate_MV.y = (short)((center >> 16)& 0xffff);
//
//		Candidate_MV.x = (Candidate_MV.x - 4); //搜索的左上角的坐标位置+-1个像素。因为Candidate_MV是从LRSF得到的，故X4
//		Candidate_MV.y = (Candidate_MV.y - 4);
//		//加载每一个MB对应的参数数据，18*18byte
//		search_index = ref_index + ((Candidate_MV.y>>2)*1952) + (Candidate_MV.x>>2); //当前处理宏块对应的ref的起始位置，加载18*18大小的块
//
//		ref_MB [tid_x+(tid_y+tid_z*blockDim.y)*18] = dev_ref[tid_x+(tid_y+tid_z*blockDim.y)*1952+search_index];
//		ref_MB [tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*18] = dev_ref[tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*1952+search_index];
//
//		if(tid_x<2)
//		{
//			ref_MB [tid_x+(tid_y+tid_z*blockDim.y)*(18)+16] = dev_ref[tid_x+(tid_y+tid_z*blockDim.y)*1952+16+search_index];
//			ref_MB [tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*(18)+16] = dev_ref[tid_x+(tid_y+(tid_z+blockDim.z)*blockDim.y)*1952+16+search_index];
//		}
//		__syncthreads();
//
//		SAD_4x4[tid_block]=Calc_SAD(src_MB+tid_index_in_srcMB,
//									ref_MB+tid_index_in_refMB,
//									stride_src,
//									stride_ref);
//		__syncthreads();
//
//
//		for( j = (blockDim.x>>1);j>0;j>>=1)
//		{
//			if(tid_x<j)
//			{
//				SAD_4x4[tid_x+(tid_y+tid_z*blockDim.y)*blockDim.x] += SAD_4x4[tid_x+j+(tid_y+tid_z*blockDim.y)*blockDim.x];
//			}
//			__syncthreads();
//		}
//
//		if(tid_block < 9)
//		{
//			costMV.x = PredMV.x - (Candidate_MV.x+(tid_block/3)*4);
//			costMV.y = PredMV.y - (Candidate_MV.y+(tid_block%3)*4);
//			cost_add_SAD = me_Cacl_Cost(costMV,costFactor,SAD_4x4[tid_block*blockDim.x]);
//			cost_MB[tid_block] = cost_add_SAD;
//		}
//		__syncthreads();
//
//		if(tid_block < 4)
//		{
//			if(cost_MB[tid_block] > cost_MB[tid_block+4])
//			{
//				cost_MB [tid_block] = cost_MB[tid_block+4];
//			}
//		}
//		__syncthreads();
//
//		if(tid_block < 2)
//		{
//			if(cost_MB[tid_block] > cost_MB[tid_block+2])
//			{
//				cost_MB [tid_block] = cost_MB[tid_block+2];
//			}
//		}
//		__syncthreads();
//
//		if(tid_block < 1)
//		{
//			if(cost_MB[tid_block] > cost_MB[tid_block+1])
//			{
//				cost_MB [tid_block] = cost_MB[tid_block+1];
//			}
//		}
//		__syncthreads();
//
//		if(tid_block==0)
//		{
//			if(cost_MB[tid_block] > cost_MB[tid_block+8])
//			{
//				cost_MB [tid_block] = cost_MB[tid_block+8];
//			}
//			Best_tid = 256;
//		}
//		__syncthreads();
//
//		if((cost_add_SAD == cost_MB[0]) && (tid_block < 9))
//		{
//			temp_tid = atomicMin(&Best_tid,tid_block);
//		}
//
//		if(tid_block == Best_tid&&(tid_block<9))
//		{
//			mb_info[i*(num_mb_hor)+blockIdx.x].SAD = cost_MB[0];
//			integer_mvmap[i*(num_mb_hor)+blockIdx.x].MV_X = (Candidate_MV.x+(tid_block/3)*4);
//			integer_mvmap[i*(num_mb_hor)+blockIdx.x].MV_Y = (Candidate_MV.y+(tid_block%3)*4);
//			ref_out_offset.x = (tid_block/3);
//			ref_out_offset.y = (tid_block%3);
//		}
//		__syncthreads();
//
//		//以raster的方式将预测数据存到对应的空间
//		tid_index_out_refMB = ref_out_offset.x + ref_out_offset.y*18 +tid_x + tid_y*18 + tid_z*blockDim.y*18;
//							  //(tid_x%4)+(tid_x/4)18+计算每一个threads对应的ref的4x4块的起始位置，相对于shared memory,处理顺序与C相同，先列后行
//		
//		if(tid_block < 128)
//		{
//			dev_out_pred[(blockIdx.x+i*1920)*16+tid_x + tid_y*1920 + tid_z*blockDim.y*1920] = ref_MB[tid_index_out_refMB];
//			dev_out_pred[(blockIdx.x+i*1920)*16+tid_x + tid_y*1920 + tid_z*blockDim.y*1920+8*1920] = ref_MB[tid_index_out_refMB+144];
//		}
//}