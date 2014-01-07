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


inline __device__ void Dct4x4TransformAndQuantize_kernel(short *Diff,short *Quant_tables,int &QAdd,int &Quant_Shift,int &tx,int &ty,int &tz)
{
		short Sum0,Sum1,Diff0,Diff1; 
		short coef0,coef1,coef2,coef3;
		int tid_index;
		int sign;
		
		//碟形算法垂直部分
		tid_index = tx+ty*4+tz*64;
		
		Sum0 = (Diff[tid_index] + Diff[tid_index+48]);
		Sum1 = (Diff[tid_index+16] + Diff[tid_index+32]);
		Diff0 = (Diff[tid_index] - Diff[tid_index+48]);
		Diff1 = (Diff[tid_index+16] - Diff[tid_index+32]);

		Diff[tid_index] = Sum0 + Sum1;
        Diff[tid_index+32] = Sum0 - Sum1;
        Diff[tid_index+16] = 2 * Diff0 + Diff1;
        Diff[tid_index+48] = Diff0 - 2 * Diff1;
		
		__syncthreads(); //同步，也许不需要，因为做水平碟形算法需要通信的线程在一个warp中
		tid_index = tx*4+ty*16+tz*64;
		
		Sum0 = (Diff[tid_index] + Diff[tid_index+3]);
		Sum1 = (Diff[tid_index+1] + Diff[tid_index+2]);
		Diff0 = (Diff[tid_index] - Diff[tid_index+3]);
		Diff1 = (Diff[tid_index+1] - Diff[tid_index+2]);
		 
		coef0= Sum0 + Sum1;
        coef2 = Sum0 - Sum1;
        coef1 = 2 * Diff0 + Diff1;
        coef3 = Diff0 - 2 * Diff1;

		//量化,按照8x8块的方式以blk-falt的格式输出(开始时是以光删形式排列的)
		tid_index = (ty*4) + ((tx&1)<<4) + ((tx>>1)<<6) + ((tz&1)<<5) + ((tz>>1)<<7); //ty*4+ (tx&1)*16+(tx>>1)*32+(tz&1)*32+(tz>>1)*128
		
		sign  = (coef0 >= 0) ? 1 : -1;
		coef0  = (coef0 >= 0) ? coef0 : -coef0;
		Diff[tid_index] = sign * ((coef0 * Quant_tables[ty*4] + QAdd) >> Quant_Shift);

		sign  = (coef1 >= 0) ? 1 : -1;
		coef1  = (coef1 >= 0) ? coef1 : -coef1;
		Diff[tid_index+1] = sign * ((coef1 * Quant_tables[ty*4+1] + QAdd) >> Quant_Shift);

		sign  = (coef2 >= 0) ? 1 : -1;
		coef2  = (coef2 >= 0) ? coef2 : -coef2;
		Diff[tid_index+2] = sign * ((coef2 * Quant_tables[ty*4+2] + QAdd) >> Quant_Shift);

		sign  = (coef3 >= 0) ? 1 : -1;
		coef3  = (coef3 >= 0) ? coef3 : -coef3;
		Diff[tid_index+3] = sign * ((coef3 * Quant_tables[ty*4+3] + QAdd) >> Quant_Shift);
	
}

inline __device__ void DQuantAndITransform_kernel(short *dct_coef,unsigned int *pred,short *DQuant_tables,int &DQuant_Shift,int &tx,int &ty,int &tz)
{
		short	Sum0,Sum1,Diff0,Diff1;
		short   coef0,coef4,coef8,coef12;
		int tid_index;
		int tid_block = tx+ty*4+tz*16;
		//反量化
		dct_coef[tid_block] = (dct_coef[tid_block] * DQuant_tables[tx+ty*4]) << DQuant_Shift;
		dct_coef[tid_block+64] = (dct_coef[tid_block+64] * DQuant_tables[tx+ty*4]) << DQuant_Shift;
		dct_coef[tid_block+128] = (dct_coef[tid_block+128] * DQuant_tables[tx+ty*4]) << DQuant_Shift;
		dct_coef[tid_block+192] = (dct_coef[tid_block+192] * DQuant_tables[tx+ty*4]) << DQuant_Shift;
		__syncthreads();

		tid_index = tx*4+ty*16+tz*64;

		Sum0  = dct_coef[tid_index] + dct_coef[tid_index+2];
        Diff0 = dct_coef[tid_index] - dct_coef[tid_index+2];
        Diff1 = (dct_coef[tid_index+1] >> 1) - dct_coef[tid_index+3];
        Sum1  = dct_coef[tid_index+1] + (dct_coef[tid_index+3] >> 1);

        dct_coef[tid_index] = Sum0 + Sum1;
        dct_coef[tid_index+1] = Diff0 + Diff1;
        dct_coef[tid_index+2] = Diff0 - Diff1;
        dct_coef[tid_index+3] = Sum0 - Sum1;
		__syncthreads();

		tid_index = tx+ty*16+tz*64;
		Sum0 = (dct_coef[tid_index] + dct_coef[tid_index+8]);
		Sum1 = dct_coef[tid_index+4] + (dct_coef[tid_index+12]>>1);
		Diff0 = (dct_coef[tid_index] - dct_coef[tid_index+8]);
		Diff1 = ((dct_coef[tid_index+4]>>1) - dct_coef[tid_index+12]);
		
		tid_index = tx + ((ty&1)<<2) + ((ty>>1)<<6) + ((tz&1)<<3) + ((tz>>1)<<7);
		
		coef0 = (Sum0 + Sum1 + 32) >> 6;
        coef0 = coef0 + pred[tid_index];

        coef4 = (Diff0 + Diff1 + 32) >> 6;       
        coef4 = coef4 + pred[tid_index+16];

        coef8 = (Diff0 - Diff1 + 32) >> 6;
        coef8 = coef8 + pred[tid_index+32];

        coef12 = (Sum0 - Sum1 + 32) >> 6;
        coef12 = coef12 + pred[tid_index+48];
		
		pred[tid_index] = (unsigned char)(coef0 < 0 ? 0 :((coef0 > 255) ? 255 : coef0));
		pred[tid_index+16] = (unsigned char)(coef4 < 0 ? 0 :((coef4 > 255) ? 255 : coef4));
		pred[tid_index+32] = (unsigned char)(coef8 < 0 ? 0 :((coef8 > 255) ? 255 : coef8));
		pred[tid_index+48] = (unsigned char)(coef12 < 0 ? 0 :((coef12 > 255) ? 255 : coef12));

}

inline __device__ int CalcCoefCostsLumablk_kernel(short *dct_coef,int *ZigZag,int *CalcCoefCosts)
{
		int Costs = 0;
        int Run = 0;
		int Coef,Level;
        // loop over 16 coefficients in the 4x4 block
        for (int i = 0; i < 16; i++)
        {
            Coef = dct_coef[ZigZag[i]];
            if (Coef!=0)
            {
                Level = Coef;
                Costs += ( abs(Level) > 1 ) ? 16 : CalcCoefCosts[Run];
                Run = -1;
            }
            Run++;
        }
		return Costs;
}

__global__ void pframe_inter_resudial_coding_luma_kernel(
															unsigned char *dev_input,
															unsigned char *dev_pred,
															int           width,
															unsigned char *dev_recon,
															int           out_stride,
															short         *dev_dct_coefs,
															short		  *quant_tab,
															short         *d_quant_tab,
															S_QP_DATA     *pQpData,
															int *dev_ZigZag,
															int *dev_CoeffCosts
															)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_z = threadIdx.z;
	int tid_block = tid_x+ (tid_y+tid_z*blockDim.y)*blockDim.x;

	int offset_input = blockIdx.x*4*MB_WIDTH+blockIdx.y*MB_HEIGHT*width; // 每一个Block开始处理的宏块的起始位置
	int offset_output = blockIdx.x*4*MB_WIDTH+blockIdx.y*MB_HEIGHT*out_stride; //每一个Block对应的重建帧的起始位置
	int tid_index = tid_z*width+tid_y*4+tid_x; //Block内每一个线程对应像素的位置，
	int input_index,output_index;
	unsigned char src_input;

	/*__shared__	unsigned char 	pred_in[MB_WIDTH*MB_HEIGHT];*/
	__shared__	unsigned int 	pred_in[MB_WIDTH*MB_HEIGHT];
	__shared__  short Diff[MB_WIDTH*MB_HEIGHT];
	__shared__ short        Quant_tables[16];
	__shared__ short        DQuant_tables[16];
	__shared__ int ZigZag[16];
	__shared__ int CoeffCosts[16];
	__shared__ int CostBlock[16+4+1];

	if(tid_block < 16)
	{
		Quant_tables[tid_block] = quant_tab[tid_block];
		DQuant_tables[tid_block] = d_quant_tab[tid_block];  
		ZigZag[tid_block] = dev_ZigZag[tid_block];
		CoeffCosts[tid_block] = dev_CoeffCosts[tid_block];
	}

     int Quant_Add = pQpData->QuantAdd;
     int Quant_Shift = pQpData->QuantShift;
     int Dquant_Shift = pQpData->DQuantShift;
	 
	 for(int i = 0;i < 4;i++)
	 {
		input_index = offset_input + tid_index+i*MB_WIDTH;
		output_index = offset_output + i* MB_WIDTH;
		 //对于每一个宏块，以光栅的形式存储
		for(int j =0 ;j < 4 ; j++)
		{
			
			pred_in[tid_x + tid_y*4+tid_z*16+j*64] = (unsigned int )dev_pred[input_index];  //由于重建的时候还要使用这部分数据，所以使用共享储存器,
			src_input = dev_input[input_index];
			Diff [tid_x + tid_y*4+tid_z*16+j*64]  = src_input - pred_in[tid_x + tid_y*4+tid_z*16+j*64];
			input_index += (4*width);
		}
		
		__syncthreads();
		//DCT变换量化
		Dct4x4TransformAndQuantize_kernel(Diff,Quant_tables,Quant_Add,Quant_Shift,tid_x,tid_y,tid_z);
		__syncthreads();
		
		//计算各个4x4块的系数的非零值的代价，只用16个线程计算，
		if(tid_z == 0)
		{
			CostBlock[tid_x + tid_y*4] = CalcCoefCostsLumablk_kernel(Diff+tid_x*16+tid_y*64,ZigZag,CoeffCosts);
	
		}
		__syncthreads();

		//计算8x8块的非零值代价
		if(tid_block<4)
		{
			CostBlock [tid_block+16] = (CostBlock[tid_block*4] +CostBlock[tid_block*4+1]+CostBlock[tid_block*4+2] +CostBlock[tid_block*4+3]);
		}
		__syncthreads();
		if(tid_block==0)
		{
			CostBlock [20]=0;
			if(CostBlock[16]>4)  //if(CostBlock[16+tid_z]>MAX_LUMA_COEFFS_COST_8x8)
				CostBlock [20] += CostBlock[16];

			if(CostBlock[17]>4)
				CostBlock [20] += CostBlock[17];

			if(CostBlock[18]>4)
				CostBlock [20] += CostBlock[18];

			if(CostBlock[19]>4)
				CostBlock [20] += CostBlock[19];
		}
		__syncthreads();

	
		if (CostBlock [20] <= 5) //if (CostBlock [20] <= MAX_LUMA_COEFFS_COST_16x16)
        {
            Diff[tid_block] = 0;
			Diff[tid_block+64] = 0;
			Diff[tid_block+128] = 0;
			Diff[tid_block+192] = 0;
        }
		//判断四个8x8块是否应该重置为0
		else if(CostBlock[16+tid_z] <=4 )//else if(CostBlock[16+tid_z] <=MAX_LUMA_COEFFS_COST_8x8 )
		{
			Diff[tid_z*64+tid_x+tid_y*4] = 0;
			Diff[tid_z*64+tid_x+tid_y*4+16] = 0;
			Diff[tid_z*64+tid_x+tid_y*4+32] = 0;
			Diff[tid_z*64+tid_x+tid_y*4+48] = 0;
		}
		__syncthreads();
		
		for(int k=0;k<4;k++)
		{
			dev_dct_coefs[tid_block+k*64+i*MB_TOTAL_SIZE+blockIdx.x*4*MB_TOTAL_SIZE +blockIdx.y*(width>>4)*MB_TOTAL_SIZE] = Diff[tid_x+tid_y*4+(tid_z&1)*16+(tid_z>>1)*64+((k&1)*32)+((k>>1)*128)];
		}
		__syncthreads();

		//反量化和反变换，重建编码的值存储在dct_coef中
		DQuantAndITransform_kernel(Diff,pred_in,DQuant_tables,Dquant_Shift,tid_x,tid_y,tid_z);
		__syncthreads();

		for(int k=0;k<4;k++)
		{
			dev_recon[tid_x+tid_y*4+tid_z*out_stride+k*4*out_stride+output_index] = pred_in[tid_x+tid_y*4+tid_z*16+k*64];
		}
		__syncthreads();

	 }

}
