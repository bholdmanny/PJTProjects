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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cutil_inline.h>
#include <cutil_inline_runtime.h>
#include <cutil.h>

#include "../inc/mb_info.h"
#include "../inc/cavlc_data.h"
#include "../inc/const_defines.h"

__device__ void cavlc_bitpack_kernel
//--------------------------------------------------------------------
(
 // Input codes
 SINGLE_CODE			*pCodes_in,
 int					codes_size,
 // Output packed bits (full-32 bit words)
 unsigned int        *packed_word_buf,
                   
 // Output total number of valid output words
 unsigned int        word_count,
 // Input:  "leftover" bits from previous call to kernel
 // Output: "leftover" bits that didn't fill an entire 32-bit word
 int                 *leftover_numbits,
 unsigned int        *leftover_value
)
{
	SINGLE_CODE curr_code;
	int total_word;
    int numbits = 0;
    unsigned int value = 0;
	for(int i = 0 ;i < codes_size ; i++)
	{
		if(pCodes_in[i].length > 0)
		{
			curr_code.length = pCodes_in[i].length;
			curr_code.value = pCodes_in[i].value;

			numbits += curr_code.length;
			if (numbits >= 32) 
			{
				numbits -= 32;
				packed_word_buf[word_count++] = value | (curr_code.value >> numbits);
				value = 0;
			}
            
			if ((numbits > 0) && (curr_code.length > 0)) 
			{
				value |= curr_code.value << (32 - numbits);
			}
		}
    } 
    *leftover_numbits = numbits;
    *leftover_value = value;
}

//每个宏块对应的头、亮度直流分量、色度直流分量的码流在一个kernel中紧凑化，通过block的y分量控制线程处理的数据
__global__ void cavlc_bitpack_HEAD_DC_cu
										(
										  // Input codes
										 SINGLE_CODE			*pCodes_HEAD,
										 SINGLE_CODE			*pCodes_Luma_DC,
										 SINGLE_CODE			*pCodes_Chroma_DC,
										 int					codes_size_HEAD,
										 int					codes_size_LDC,
										 int					codes_size_CDC,
										 // Output packed bits (full-32 bit words)
										 unsigned int        *packed_word_HEAD,
										 unsigned int        *packed_word_LDC,
										 unsigned int        *packed_word_CDC,
										                   
										 // Output total number of valid output words
										 unsigned int        *word_count_HEAD,
										 unsigned int        *word_count_LDC,
										 unsigned int        *word_count_CDC,

										 // Input:  "leftover" bits from previous call to kernel
										 // Output: "leftover" bits that didn't fill an entire 32-bit word
										 int                 *leftover_numbits_HEAD,
										 int                 *leftover_numbits_LDC,
										 int                 *leftover_numbits_CDC,
										 unsigned int        *leftover_value_HEAD,
										 unsigned int        *leftover_value_LDC,
										 unsigned int        *leftover_value_CDC
										 )
{
		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y;
		int blk_y = blockIdx.y;
		
		if(blk_y == 0)
		{
			cavlc_bitpack_kernel
								(
								 pCodes_HEAD + (tid_grid*codes_size_HEAD),
								 codes_size_HEAD,
								 packed_word_HEAD+ ((threadIdx.x*codes_size_HEAD)),
								                   
								 word_count_HEAD[tid_grid],
								 leftover_numbits_HEAD+tid_grid,
								 leftover_value_HEAD+tid_grid
								);
		}
		if(blk_y == 1)
		{
				cavlc_bitpack_kernel
								(
								 pCodes_Luma_DC + (tid_grid*codes_size_LDC),
								 codes_size_LDC,
								 packed_word_LDC + ((tid_grid*codes_size_LDC)>>1),
								                   
								 word_count_LDC[tid_grid],
								 leftover_numbits_LDC+tid_grid,
								 leftover_value_LDC+tid_grid
								);
		}

		if(blk_y ==2)
		{
				cavlc_bitpack_kernel
								(
								 pCodes_Chroma_DC + (tid_grid*codes_size_CDC),
								 codes_size_CDC,
								 packed_word_CDC + ((tid_grid*codes_size_CDC)>>1),
								                   
								 word_count_CDC[tid_grid],
								 leftover_numbits_CDC+tid_grid,
								 leftover_value_CDC + tid_grid
								);
		
		}

}

//以动态的形式申明共享存储器空间，这样可以使用参数化的形式对所有的分量和宏块头做统一处理
__global__ void cavlc_bitpack_block_cu
										(
										  // Input codes
										 SINGLE_CODE			*pCodes_dev,
										 int					codes_size_blk, //以short为单位
										 // Output packed bits (full-32 bit words)
										 unsigned int        *packed_words_dev,
										                   
										 // Output total number of valid output words
										 unsigned int        *word_count_dev,

										 // Input:  "leftover" bits from previous call to kernel
										 // Output: "leftover" bits that didn't fill an entire 32-bit word
										 int                 *leftover_numbits,
										 unsigned int        *leftover_value
										 )
{
		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y;
		//__shared__ unsigned int packet_word[1664];//申明动态存储器，保存紧凑后各个块对应的码流
		extern __shared__ unsigned int packet_word[];

		SINGLE_CODE curr_code;
		int index_grid = tid_grid*codes_size_blk;  //每个线程访问的编码后数据起始位置
		int index_blk = (tid_blk * ((codes_size_blk + 1)>>1)); //每个线程对应访问的共享存储器数据，先将结果写在共享存储器中，最后在写回全局存储器
		int out_index = tid_grid * ((codes_size_blk + 1)>>1);

		int total_word = 0;
		int numbits = 0;
		unsigned int value = 0;
		for(int i=0;i < ((codes_size_blk + 1)>>1) ;i++)
		{
			packet_word[index_blk+i] = 0;
		}
		for(int i = 0 ;i < codes_size_blk ; i++)
		{
			if(pCodes_dev[index_grid+i].length > 0)
			{
				curr_code.length = pCodes_dev[index_grid+i].length;
				curr_code.value = pCodes_dev[index_grid+i].value;

				numbits += curr_code.length;
				if (numbits >= 32) 
				{
					numbits -= 32;
					packet_word[index_blk + total_word] = value | (curr_code.value >> numbits);
					//packed_words_dev[out_index+total_word]= value | (curr_code.value >> numbits);

					total_word++;
					value = 0;
				}
	            
				if ((numbits > 0) && (curr_code.length > 0)) 
				{
					value |= curr_code.value << (32 - numbits);
				}
			}
		} 
		leftover_numbits[tid_grid] = numbits;
		leftover_value [tid_grid]= value;
		word_count_dev [tid_grid] = total_word;
		for(int i=0;i < ((codes_size_blk + 1)>>1) ;i++)
		{
			packed_words_dev[out_index+i] = packet_word[index_blk+i];
		}
}

__global__ void cavlc_bitpack_MB_cu(
										 //intput packet codes of head,lumadc,lumaac,chromadc...
										 unsigned int        *packed_words_HEAD,
										 unsigned int        *packed_words_LDC,
										 unsigned int        *packed_words_LAC,
										 unsigned int        *packed_words_CDC,
										 unsigned int        *packed_words_CAC,

										 unsigned int        *word_count_HEAD,
										 unsigned int        *word_count_LDC,
										 unsigned int        *word_count_LAC,
										 unsigned int        *word_count_CDC,
										 unsigned int        *word_count_CAC,


										 int                 *leftover_numbits_HEAD,
										 int                 *leftover_numbits_LDC,
										 int                 *leftover_numbits_LAC,
										 int                 *leftover_numbits_CDC,
										 int                 *leftover_numbits_CAC,
										 unsigned int        *leftover_value_HEAD,
										 unsigned int        *leftover_value_LDC,
										 unsigned int        *leftover_value_LAC,
										 unsigned int        *leftover_value_CDC,
										 unsigned int        *leftover_value_CAC,

										 int				 *BlockScan_dev,
										 int				 Max_size_MB_bit,
										 int				 Head_size,
										 int				 *Skip_block,

										//ouput packet words for mb
										 unsigned int		 *total_packet_word_mb,
										 unsigned int		 *total_word_count_mb,
										 int                 *total_leftover_numbits_mb,
										 unsigned int        *total_leftover_value_mb
									 )
{
		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;
		
		int tid_blk = tid_x + tid_y*blockDim.x;
		int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y;

		int in_index ; //每个线程对应访问的共享存储器数据，先将结果写在共享存储器中，最后在写回全局存储器
		int out_index = tid_grid*Max_size_MB_bit;
		
		unsigned int word_count,total_word,numbits,value,value1,i;
		total_word = 0;
		numbits = 0;
		value   = 0;

		if(!Skip_block[tid_grid])
		{
			word_count = word_count_HEAD[tid_grid];
			total_word = word_count;
			in_index = tid_grid*Head_size;
			//header codes
			for( i = 0 ;i < word_count; i++)
			{	
				//if(i< word_count)
				//{
					total_packet_word_mb[out_index+i] = packed_words_HEAD[in_index+i];
				//}
			} 
			//ldc codes
			out_index += word_count;
			numbits = leftover_numbits_HEAD[tid_grid];
			value = leftover_value_HEAD[tid_grid];

			word_count = word_count_LDC[tid_grid];
			total_word += word_count;
			in_index = tid_grid*13;
			for( i = 0 ;i < word_count; i++)
			{	
					value1 = packed_words_LDC[in_index+i]  ;
					total_packet_word_mb[out_index+i] = value|(value1>>numbits);
					value = (value1 <<(32-numbits));
			}
			if((numbits + leftover_numbits_LDC[tid_grid])>=32)	
			{
				value1 = leftover_value_LDC[tid_grid];
				total_packet_word_mb[out_index+word_count] = value|(value1>>numbits);
				total_word +=1;
				word_count +=1;
				value = (value1<<(32-numbits));
				numbits = (numbits + leftover_numbits_LDC[tid_grid])-32;
			}
			else
			{
				value |= (leftover_value_LDC[tid_grid]>>numbits);
				numbits = (numbits + leftover_numbits_LDC[tid_grid]);
				
			}
			out_index += word_count;
			//lac codes
			for(int j = 0 ; j< 16; j++)
			{
				int ii = BlockScan_dev[j];
				word_count = word_count_LAC[tid_grid*16 + ii];
				total_word += word_count;
				in_index = tid_grid*13*16 + ii*13;
				for( i = 0 ;i < word_count; i++)
				{	/*if(i< word_count)
					{*/
						value1 = packed_words_LAC[in_index+i];
						total_packet_word_mb[out_index+i] = value|(value1>>numbits);
						value = (value1 <<(32-numbits));
					/*}*/
				}
				if((numbits + leftover_numbits_LAC[tid_grid*16 + ii])>=32)	
				{
					value1 = leftover_value_LAC[tid_grid*16 + ii];
					total_word +=1;
					total_packet_word_mb[out_index+word_count] = value|(value1>>numbits);
					word_count +=1;
					value = (value1<<(32-numbits));
					numbits = (numbits + leftover_numbits_LAC[tid_grid*16 + ii])-32;
				}
				else
				{
					value |= (leftover_value_LAC[tid_grid*16 + ii]>>numbits);
					numbits = (numbits + leftover_numbits_LAC[tid_grid*16 + ii]);
					/*value |= (leftover_value_LAC[tid_grid*16 + ii]<<(32-numbits));*/
				}
				out_index += word_count;

			}

			//cdc codes
			word_count = word_count_CDC[tid_grid];
			total_word += word_count;
			in_index = tid_grid*8;
			for( i = 0 ;i < word_count; i++)
			{	//if(i< word_count)
				//{
					value1 = packed_words_CDC[in_index+i];
					total_packet_word_mb[out_index+i] = value|(value1>>numbits);
					value = (value1 <<(32-numbits));
				//}
			}
			if((numbits + leftover_numbits_CDC[tid_grid])>=32)	
			{
				value1 = leftover_value_CDC[tid_grid];
				total_word +=1;
				total_packet_word_mb[out_index+word_count] = value|(value1>>numbits);
				word_count +=1;
				value = (value1<<(32-numbits));
				numbits = (numbits + leftover_numbits_CDC[tid_grid])-32;
			}
			else
			{
				value |= (leftover_value_CDC[tid_grid]>>numbits);
				numbits = (numbits + leftover_numbits_CDC[tid_grid]);
				/*value |= (leftover_value_CDC[tid_grid]<<(32-numbits));*/
			}
			out_index += word_count;
			//cac codes
			for(int j = 0 ; j< 8; j++)
			{
				word_count = word_count_CAC[tid_grid*8+j];
				total_word += word_count;
				in_index = tid_grid*13*8 + j*13;
				for( i = 0 ;i < word_count; i++)
				{	
					//if(i< word_count)
					//{
						value1 = packed_words_CAC[in_index+i];
						total_packet_word_mb[out_index+i] = value|(value1>>numbits);
						value = (value1 <<(32-numbits));
					//}
				}
				if((numbits + leftover_numbits_CAC[tid_grid*8 + j])>=32)	
				{
					value1 = leftover_value_CAC[tid_grid*8 + j];
					total_word +=1;
					total_packet_word_mb[out_index+word_count] = value|(value1>>numbits);
					word_count +=1;
					value = (value1<<(32-numbits));
					numbits = (numbits + leftover_numbits_CAC[tid_grid*8 + j])-32;
				}
				else
				{
					value |= (leftover_value_CAC[tid_grid*8 + j]>>numbits);
					numbits = (numbits + leftover_numbits_CAC[tid_grid*8 + j]);
					/*value |= (leftover_value_CAC[tid_grid*8 + j]<<(32-numbits));*/
				}
				out_index += word_count;
			}
			
		}
			total_word_count_mb[tid_grid] = total_word;
			total_leftover_numbits_mb[tid_grid]=numbits;
			total_leftover_value_mb[tid_grid]=value;
		

}

//分别计算各个slice中各个宏块对应的输出位置，以及由前面宏块多余位数导致的移位方式及位数，每个slice对应一个block
//block大小为slice包含的宏块数量，采用规约的方法，每次计算为i的平方个有效地址，第i次计算使用第i-1次计算的结果
__global__ void compute_out_position
									(
										//input: word of mb and leftover_numbits
										 unsigned int		*total_word_count_mb,
										 int                *total_leftover_numbits_mb,
										 //output: out position for mb and shift bits
										 unsigned int		*out_index,
										 int				*shift_bits
									 )
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
		
	int tid_blk = tid_x + tid_y*blockDim.x;
	int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y;
	__shared__ unsigned int out_position[240];
	__shared__ int shift_numbits[240];
	__shared__ int num_iteration,block_size;
	int word_count;
	int left_numbits;
	out_position[tid_blk] = total_word_count_mb[tid_grid];
	shift_numbits[tid_blk] = total_leftover_numbits_mb[tid_grid];
	word_count = 0;
	left_numbits = 0;
	if(tid_blk ==0)
	{
		block_size = blockDim.x*blockDim.y;
		num_iteration = log2((float)block_size);
		num_iteration += ((block_size>>num_iteration) == 0) ? 0 : 1;
		
	}
	__syncthreads();
	for(int i =0 ;i <num_iteration; i++ )
	{
		if(((tid_blk &(1<<i))!=0)&&(tid_blk < block_size))
		{
			word_count  = (out_position[tid_blk] + out_position[(tid_blk|((1<<i)-1)) -(1<<i)]);
			left_numbits = (shift_numbits[tid_blk] + shift_numbits[(tid_blk|((1<<i)-1)) -(1<<i)]);

			shift_numbits[tid_blk] = left_numbits;
			out_position[tid_blk] = word_count;
		}
		__syncthreads();
	}
	__syncthreads();
	 word_count = out_position[tid_blk] - total_word_count_mb[tid_grid];//获得前面有效宏块的整字数之和
	 left_numbits = shift_numbits[tid_blk] - total_leftover_numbits_mb[tid_grid]; 

	while(left_numbits >=32)
	{
		word_count++;
		left_numbits -=32;
	}
	if(left_numbits > 0)
	{
		word_count++;
	}
	 shift_numbits[tid_blk]=left_numbits;
	 out_index[tid_grid] = word_count;
	 shift_bits[tid_grid] = shift_numbits[tid_blk];
}


__global__ void parallel_write
								(

									unsigned int		*total_packet_word_mb,
									unsigned int		*total_word_count_mb,
									int					*Skip_block,

									int                 *total_leftover_numbits_mb,
									unsigned int        *total_leftover_value_mb,
									unsigned int		*out_index,
									int					*shift_bits,
									int					slice_size,
									//out_put packet word for slice
									unsigned int		*total_packet_word,
									unsigned int		 *slice_word_count,
									int					*leftover_numbit_slice,
									unsigned int		*leftover_value_slice
								)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
		
	int tid_blk = tid_x + tid_y*blockDim.x;
	int tid_grid = tid_blk + blockIdx.x*blockDim.x*blockDim.y;
	__shared__ unsigned int out_start_slice;
	__shared__ unsigned int word_num_slice;
	int out_position,left_shift_bits,right_shift_bits,word_count,add_another_word,write_count,leftover_numbits;
	unsigned int value,value1;
	int	block_start,tid_num,valid_last;
	tid_num = tid_blk;
	block_start = blockIdx.x*blockDim.x*blockDim.y;
	valid_last = 0;
	tid_num = 0;
	write_count =0;
	if(tid_blk == 0)
	{
		out_start_slice = 0;
		for(int i = 1;i<= blockIdx.x;i++)
		{
			out_start_slice += out_index[(i*slice_size)-1];
			out_start_slice += total_word_count_mb[(slice_size * i)-1];
			if((shift_bits[(slice_size * i)-1]!=0)&&(shift_bits[(slice_size * i)-1] + total_leftover_numbits_mb[(slice_size * i) -1])< 32)
				out_start_slice--;

		}
		word_num_slice  = out_start_slice + out_index[(blockIdx.x+1)*slice_size-1];
		word_num_slice  += total_word_count_mb[(blockIdx.x+1)*slice_size-1];
		if((shift_bits[(blockIdx.x+1)*slice_size-1]!=0)&&(shift_bits[(blockIdx.x+1)*slice_size-1] + total_leftover_numbits_mb[(blockIdx.x+1)*slice_size-1])< 32)
				word_num_slice--;
		slice_word_count[blockIdx.x] = word_num_slice;
	
	}
	__syncthreads();
	out_position = out_start_slice + out_index[tid_grid];

	if(!Skip_block[tid_grid])
	{
		right_shift_bits = (tid_blk == 0) ? 0 : shift_bits[tid_grid];
		left_shift_bits = (tid_blk == 0||right_shift_bits == 0) ? 0 : (32-right_shift_bits);
		word_count = total_word_count_mb[tid_grid];
		leftover_numbits = total_leftover_numbits_mb[tid_grid];

		add_another_word = ((leftover_numbits>=((32-right_shift_bits)&31))/*&&(tid_blk!=(slice_size-1))*/) ? 1 : 0;
		write_count = word_count + add_another_word;
		value =(word_count==0) ? (total_leftover_value_mb[tid_grid]<< left_shift_bits):(total_packet_word_mb[tid_grid*64]<<left_shift_bits);
		//valid_last = (tid_blk==(slice_size-1)&&(leftover_numbits>=((32-right_shift_bits)&31)||word_count>0))||(add_another_word);

		for(int i = 0; i< (write_count-1); i++ )
		{
			value1 = (i == (word_count-1)) ? total_leftover_value_mb[tid_grid] : total_packet_word_mb[tid_grid*64 + i + 1] ;
			total_packet_word[out_position + i] =(right_shift_bits == 0) ? value : (value|(value1>>right_shift_bits));
			value = (value1<<left_shift_bits);
		}

		value =( add_another_word == 0) ? (value|(total_leftover_value_mb[tid_grid]>>right_shift_bits)): value; //再一次更正value 的值
		leftover_numbits = ( add_another_word == 0 ||right_shift_bits==0 )?(leftover_numbits +right_shift_bits):(leftover_numbits + right_shift_bits -32);
		tid_num = tid_blk + 1;

		while(leftover_numbits>=0&&(write_count>0)&&(tid_num<slice_size))
		{
			if(leftover_numbits==0)
			{
				if(!Skip_block[tid_num+block_start])
				{
					write_count = 0;
				}
				else
				{
					tid_num++;
				}
			}
			else
			{
				if(!Skip_block[tid_num+block_start])
				{
					if(total_word_count_mb[tid_num+block_start]>0)
					{
						total_packet_word[out_position + write_count-1] = (leftover_numbits == 32)? value :(value|(total_packet_word_mb[(tid_num+block_start)*64]>>leftover_numbits));
						leftover_numbits = 0;
					}
					else if(total_leftover_numbits_mb[tid_num+block_start]>0)
					{
						if((leftover_numbits + total_leftover_numbits_mb[tid_num+block_start])>=32)
						{
							total_packet_word[out_position + write_count-1] =(leftover_numbits==32)? value :( value | (total_leftover_value_mb[tid_num+block_start]>>(leftover_numbits&31)));
							leftover_numbits =0;
						}
						else
						{
							value = (value | total_leftover_value_mb[tid_num+block_start]>>(leftover_numbits&31));
							leftover_numbits += total_leftover_numbits_mb[tid_num+block_start];
							tid_num++;
							
						}
					}
				}
				else
				{
					tid_num++;
				}
			}
		}
		
		if((tid_num==slice_size)&&(write_count>0))
		{
			leftover_numbit_slice[blockIdx.x] = leftover_numbits;
			leftover_value_slice[blockIdx.x] = value;
		}
		
	}
}