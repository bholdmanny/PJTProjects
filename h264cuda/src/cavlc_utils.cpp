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
#include "../inc/encoder_tables.h"
#include "../inc/cavlc.h"
#include "../inc/encoder_context.h"
#include "../inc/cavlc_data.h"


void concatenate_and_duplicate_cavlc_texture_tables(int *p_tables_buf);


/////////////////////////////////////////////////////////////////////
//

E_ERR init_cavlc_context(encoder_context_t *p_enc)

//
// Description:     Allocate and initialize CAVLC-related buffers
//  
/////////////////////////////////////////////////////////////////////
{
    int num_mbs;
    E_ERR err = ERR_SUCCESS;

    num_mbs = (p_enc->width / MB_WIDTH) * (p_enc->height / MB_HEIGHT);

    if (!err)
    {
        //generate_coef_index_for_cavlc (p_enc->cavlc_ctxt.p_index, &(p_enc->transform_coefs), MAX_CAVLC_STRIP_SIZE);   

        p_enc->cavlc_ctxt.block_context_param_buf = (int *)malloc(sizeof(int) * 16);
        if (p_enc->cavlc_ctxt.block_context_param_buf == NULL)
        {
            printf("Couldn't allocate memory for CAVLC block context parameter buffer!\n");
            err = ERR_MEM;
        }
    }


    if (!err)
    {
          {
              p_enc->cavlc_ctxt.p_coef_zigzag     = (S_TEXTURE_BLOCK_PAIRS *)malloc((ROUND_TWO_MB_PER_LANE(num_mbs) * BLOCK_PAIRS_PER_MB * sizeof(S_TEXTURE_BLOCK_PAIRS)));
          }
        if (p_enc->cavlc_ctxt.p_coef_zigzag == NULL)
        {
            printf("Couldn't allocate memory for CAVLC zigzag coefficients!\n");
            err = ERR_MEM;
        }
        else
        {
         memset(p_enc->cavlc_ctxt.p_coef_zigzag, 0, (ROUND_TWO_MB_PER_LANE(num_mbs) * BLOCK_PAIRS_PER_MB * sizeof(S_TEXTURE_BLOCK_PAIRS)));
        }
    }

    if (!err)
    {
        {
            p_enc->cavlc_ctxt.p_context = (S_CAVLC_CONTEXT *)malloc((ROUND_TWO_MB_PER_LANE(num_mbs) * BLOCK_PAIRS_PER_MB * sizeof(S_CAVLC_CONTEXT)));
        }

        if (p_enc->cavlc_ctxt.p_context == NULL)
        {
            printf("Couldn't allocate memory for CAVLC texture context!\n");
            err = ERR_MEM;
        }
        else
        {
         memset(p_enc->cavlc_ctxt.p_context, 0 , (ROUND_TWO_MB_PER_LANE(num_mbs) * BLOCK_PAIRS_PER_MB * sizeof(S_CAVLC_CONTEXT)));
        }
    }

    if (!err)
    {
        // cavlc_gen_bits related storage
        if(p_enc->s_nonpersist_individual_mem.is_app_mem == 1)
        {
           p_enc->cavlc_ctxt.p_mb_bits     = (unsigned int *)((unsigned long)p_enc->s_nonpersist_individual_mem.p_mem + p_enc->scratch_mem_info.mem_offset_5);
       
        }
        else
        {
            p_enc->cavlc_ctxt.p_mb_bits = (unsigned int *)malloc((ROUND_TWO_MB_PER_LANE(num_mbs) * MAX_BYTES_PER_MB)); 
        } 
        if (p_enc->cavlc_ctxt.p_mb_bits == NULL)
        {
            printf("Can not allocate memory for cavlc bits!\n");
            err = ERR_MEM;
        }
    }

    if (!err)
    {
        // Scalar params
        p_enc->cavlc_ctxt.param_buf = (int *)malloc(sizeof(int) * 16);
        if (p_enc->cavlc_ctxt.param_buf == NULL)
        {
            printf("Couldn't allocate memory for CAVLC parameter buffer!\n");
            err = ERR_MEM;
        }
    }



    if (!err)
    {
    	   // Scalar params
        p_enc->cavlc_ctxt.p_cavlc_leftover_info = (int *)malloc(sizeof(int) * 16 * 6);
        if (p_enc->cavlc_ctxt.p_cavlc_leftover_info == NULL)
        {
    	     printf("Couldn't allocate memory for CAVLC Mb state info parameter buffer!\n");
                err = ERR_MEM;
        }
                
        // Duplicated VLC tables
        p_enc->cavlc_ctxt.p_tables_buf = (int *)malloc(sizeof(int) * TEXTURE_UINT16x2_TABLE_LANE_SIZE);
        if (p_enc->cavlc_ctxt.p_tables_buf == NULL)
        {
            printf("Couldn't allocate memory for CAVLC tables!\n");
            err = ERR_MEM;
        }
    }

    if (!err)
    {
        concatenate_and_duplicate_cavlc_texture_tables(p_enc->cavlc_ctxt.p_tables_buf);
    }

    return err;
}

/////////////////////////////////////////////////////////////////////
//

void free_cavlc_context(encoder_context_t *p_enc)

//
/////////////////////////////////////////////////////////////////////
{
    //free(p_enc->cavlc_ctxt.p_index);
    free(p_enc->cavlc_ctxt.block_context_param_buf);
    if(p_enc->s_nonpersist_individual_mem.is_app_mem == 0)
    {

        free(p_enc->cavlc_ctxt.p_mb_bits);
    }
    free(p_enc->cavlc_ctxt.p_coef_zigzag);
    free(p_enc->cavlc_ctxt.p_context);
    free(p_enc->cavlc_ctxt.param_buf);
    free(p_enc->cavlc_ctxt.p_tables_buf);


    free(p_enc->cavlc_ctxt.p_cavlc_leftover_info);
}

/////////////////////////////////////////////////////////////////////
//

void concatenate_and_duplicate_cavlc_texture_tables(int *p_tables_buf)

//
// Description: Copy all int16x2 texture tables into one contiguous
//              area to reduce the number of stream arguments.  Also,
//              duplicate the table while we copy it.
//
/////////////////////////////////////////////////////////////////////
{
    int i, j;

    // Copy all int16x2 texture tables into one contiguous area to reduce the number of stream arguments.
    // Also, duplicate the table while we copy it
    j = 0;
    for (i = 0; i < COEFF_TOKEN_TABLE_SIZE/4; i++)
    {
        p_tables_buf[j++] = ((int *)CoeffTokenTable)[i];
    }
    for (i = 0; i < COEFF_TOKEN_CHROMA_DC_TABLE_SIZE/4; i++)
    {
        p_tables_buf[j++] = ((int *)CoeffTokenChromaDCTable)[i];
    }
    for (i = 0; i < RUN_TABLE_SIZE/4; i++)
    {
        p_tables_buf[j++] = ((int *)RunTable)[i];
    }
    for (i = 0; i < TOTAL_ZEROS_TABLE_SIZE/4; i++)
    {
        p_tables_buf[j++] = ((int *)TotalZerosTable)[i];
    }
    for (i = 0; i < TOTAL_ZEROS_CHROMA_DC_TABLE_SIZE/4; i++)
    {
        p_tables_buf[j++] = ((int *)TotalZerosChromaDCTable)[i];
    }
}

void print_frame_stats(CUVE264B_FRAME_ENCODE_INFO *p_stats, double psnr_y, double psnr_u, double psnr_v, int frame_num, int frame_start)
{
    if (frame_num == frame_start) printf("\n");

    if (((frame_num % 50) == 0)||(frame_num == frame_start))
    {
        printf("|-------|-------|-------|-------|-----------|----|-------|-------|-------|-------|-------|\n");
        printf("| %5s | %5s | %5s | %5s | %9s | %2s | %5s | %5s | %5s | %5s | %5s |\n", "Frame",  "Frame", "Frame", "# of",    "Total", " ",  "PSNR", "PSNR", "PSNR", "Intra", "Enc.");
        printf("| %5s | %5s | %5s | %5s | %9s | %2s | %5s | %5s | %5s | %5s | %5s |\n", "#", "Index", "Type" , "Slice", "Bits",  "QP", "Y",    "U",    "V",    "MBs",   "MBs");
        printf("|-------|-------|-------|-------|-----------|----|-------|-------|-------|-------|-------|\n");
    }

    printf ("| %5d | %5d | %5d | %5d | %9d | %2d | %5.2f | %5.2f | %5.2f | %5d | %5d |\n",
                frame_num,
                p_stats->frame_number,
                p_stats->frame_type,        // YL- translate this to a name
                p_stats->num_slices,
                p_stats->num_bits,
                p_stats->qp_used,
                psnr_y,
                psnr_u,
                psnr_v,
                p_stats->num_intra_mbs,
                p_stats->num_encoded_mbs
                );
}

void print_enc_stats(
    double total_psnr_y, double total_psnr_u, double total_psnr_v,
    int total_bits, int total_me_cands,
    int qp_sum, int min_qp, int max_qp,
    int num_frames)
{
    printf("\n");
    printf("------------------ Average data all frames  ------------------------------\n");
    printf(" SNR Y(dB)                         : %5.2f\n", total_psnr_y / num_frames);
    printf(" SNR U(dB)                         : %5.2f\n", total_psnr_u / num_frames);
    printf(" SNR V(dB)                         : %5.2f\n", total_psnr_v / num_frames);
    printf(" Total bits                        : %d\n",    total_bits);
    printf(" Bit rate (kbit/s)  @ 30 Hz        : %5.2f\n", (double)total_bits * 30 / num_frames / 1000);
    printf(" Average ME Candidates             : %5.2f\n", (float)total_me_cands / ((num_frames-1)* 256)); // stats assume only 1 I-frame
    printf("--------------------------------------------------------------------------\n");
    printf(" Total frames                      : %d\n", num_frames);
    printf(" Average frame QP                  : %.2f\n", (double) qp_sum / num_frames);
    printf(" Minimum frame QP                  : %d\n", min_qp);
    printf(" Maximum frame QP                  : %d\n", max_qp);
    printf("\n");
}
