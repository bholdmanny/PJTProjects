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

#include <cutil_inline.h>
#include <cutil_inline_runtime.h>
#include <cutil.h>

inline __device__ void Dct4x4TransformAndQuantize_kernel(short *Diff,short *Quant_tables,int &QAdd,int &Quant_Shift,int &tx,int &ty,int &tz);

inline __device__ void DQuantAndITransform_kernel(short *dct_coef,unsigned char *pred,short *DQuant_tables,int &DQuant_Shift,int &tx,int &ty,int &tz);

inline __device__ int CalcCoefCostsLumablk_kernel(short *dct_coef,int *ZigZag,int *CalcCoefCosts);

__global__ void pframe_inter_resudial_coding_luma_kernel(
															unsigned char *dev_input,
															unsigned char *dev_pred,
															int           in_stride,
															unsigned char *dev_recon,
															int           out_stride,
															short         *dev_dct_coefs,
															short		  *quant_tab,
															short         *d_quant_tab,
															S_QP_DATA     *pQpData,
															int *dev_ZigZag,
															int *dev_CoeffCosts
															);


__global__ void pframe_intra_resudial_coding_luma_kernel(
															unsigned char *dev_input,
															unsigned char *dev_pred,
															int           in_stride,
															unsigned char *dev_recon,
															int           out_stride,
															S_BLK_MB_INFO *dev_blk_mb_info,
															short         *dev_dct_coefs,
															short		  *dev_dc_coefs,
															short		  *quant_tab,
															short         *d_quant_tab,
															S_QP_DATA     *pQpData,
															int avg_sad,
															int intra_lambda_fact,
															int num_mb_hor,
															int num_mb_ver,
															int slice_num
															);
