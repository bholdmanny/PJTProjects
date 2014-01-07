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


#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../inc/h264_types.h"
#include "../inc/cuve264b.h"
#include "../inc/encoder_context.h"
#include "../inc/output.h"
#include "../inc/h264_common.h"
#include "../inc/const_defines.h"
#include "../inc/cuve264b_utils.h"

//--------------------------------------------------------------------------------------
void copy_and_pad_frame_for_encode 
//--------------------------------------------------------------------------------------
(
    CUVE264B_IMAGE_SPECS *p_input_image,
    yuv_frame_t *p_frame
)
{
    unsigned char *p_src = NULL;
    unsigned char *p_dst = NULL;

    int width_to_copy;
    int width_to_pad;
    int height_to_copy;
    int height_to_pad;
    int src_stride;
    int dst_stride;
    int i, j;

    assert (p_frame->width >= p_frame->image_width);
    assert (p_frame->height >= p_frame->image_height);

    width_to_copy  = min (p_input_image->width, p_frame->image_width);
    width_to_pad   = p_frame->width - width_to_copy;

    height_to_copy = min (p_input_image->height, p_frame->image_height);
    height_to_pad  = p_frame->height - height_to_copy;

    src_stride = p_input_image->buffer.buf_width;
    dst_stride = p_frame->width;

    for (j = 0; j < 3; j++)
    {// three buffers y, u, and v
        switch (j)
        {
        case 0:
            p_src = p_input_image->buffer.p_y_buffer + p_input_image->x_offset;
            p_dst = p_frame->y;
            break;
        case 1:
            p_src = p_input_image->buffer.p_u_buffer + p_input_image->x_offset / 2;
            p_dst = p_frame->u;
            break;
        case 2:
            p_src = p_input_image->buffer.p_v_buffer + p_input_image->x_offset / 2;
            p_dst = p_frame->v;
            break;
        default:
            break;
        }

        // copy buffer data and pad horizontal
        for (i = 0; i < height_to_copy; i++)
        {
            memcpy (p_dst, p_src, width_to_copy);
            memset (&p_dst[width_to_copy], p_src[width_to_copy - 1], width_to_pad);
            p_src += src_stride;
            p_dst += dst_stride;
        }

        // pad the bottom of the dstination buffer
        p_src = p_dst - dst_stride;
       
        for (i = 0; i < height_to_pad; i++)
        {
            memcpy (p_dst, p_src, dst_stride);
            p_dst += dst_stride;
        }
        
        if (j == 0)
        {
            width_to_copy  /= 2;
            width_to_pad   /= 2;
            height_to_copy /= 2;
            height_to_pad  /= 2;
            src_stride     /= 2;
            dst_stride     /= 2;
        }
    }
}

//--------------------------------------------------------------------------------------
CUVE264B_ERROR init_output_bitstream_buffer
//--------------------------------------------------------------------------------------
(
    encoder_context_t *p_enc, 
    CUVE264B_BITSTREAM_BUFFER *p_output_buffer
)
{
    unsigned char *p_buffer;
    int buffer_size;
    int num_mbs;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;

    assert(p_enc);

    num_mbs = (p_enc->width / MB_WIDTH) * (p_enc->height / MB_HEIGHT);
    
    /* Calc estimated buffer size based on resolution */
    p_enc->estimated_output_buffer_size = num_mbs * (MAX_BYTES_PER_MB >> 1) ;

    if ((p_output_buffer == NULL))
    {
        buffer_size = p_enc->estimated_output_buffer_size;
        // allocate a buffer if necessary
        if (p_enc->own_output_buffer == 0)
        {
            p_buffer = (unsigned char *)malloc (buffer_size);
            if (p_buffer == NULL)
            {
                printf("Not enough memory for bs buffer allocation \n");
                err = CUVE264B_ERR_MEM;
            }
            else
            {
                p_enc->own_output_buffer = 1;
            }
        }
        else
        {
            p_buffer = p_enc->bitstream.p_buffer;
            if (p_enc->bitstream.buffer_size != buffer_size)
            {
                free (p_buffer);
                p_buffer = (unsigned char *)malloc (buffer_size);
                if (p_buffer == NULL)
                {
                    printf("Not enough memory for bs buffer allocation \n");
                    err = CUVE264B_ERR_MEM;
                }
            }
        }
    }
    else
    {
        if (p_output_buffer->total_num_bytes < p_enc->estimated_output_buffer_size)
        {
			;
        }
        p_buffer = p_output_buffer->p_buffer;
        buffer_size = p_output_buffer->total_num_bytes;
    }

    if (!err)
    {
        p_enc->bitstream.buffer_size = buffer_size;
        p_enc->bitstream.p_buffer = p_buffer;
    }

    return err;
}

//---------------------------------------------------------------------
int check_input_buffer_conformance_to_encoder (
// utility function to check whether a frame buffer allocated by user is 
// conformance to size, stride, alignment requirement
    CUVE264B_IMAGE_SPECS  *p_input_image,
    encoder_context_t     *p_enc
)
{
    int conform = 1;
    
    if ((p_input_image->buffer.buf_width < p_enc->width) || (p_input_image->buffer.buf_height < p_enc->height))
    { // in this case, the input image needs pading for encode. 
        conform = 0;
    }
    return conform;
}

//---------------------------------------------------------------------
void calc_psnr(yuv_frame_t *frame0, yuv_frame_t *frame1, 
	double *res_snr_y, double *res_snr_u, double *res_snr_v)
{
	const double CALC_SNR_MIN_MSE   = 0.00001;
	const double CALC_SNR_MAX_VALUE = 48.0;
	int x, y;
	double mse;
	int size;
	unsigned char *curr0;
	unsigned char *curr1;
	int aux;
	int width, height;
	int width1;

    assert(frame0->width == frame1->image_width && frame0->height == frame1->image_height);

	mse = 0.;
	width = frame0->width;
	height = frame0->height;
    width1 = frame1->width;
	size = width * height;
	curr0 = frame0->y;
	curr1 = frame1->y + REFERENCE_FRAME_PAD_AMT * frame1->width + REFERENCE_FRAME_PAD_AMT;
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			aux = curr0[x] - curr1[x];
			mse += aux*aux;
		}
		curr0 += width;
		curr1 += width1;
	}
	if (mse < CALC_SNR_MIN_MSE)
	{
		*res_snr_y = CALC_SNR_MAX_VALUE;
	}
	else
	{
		mse /= size;
		*res_snr_y = 10.* log10((255.*255.)/mse);
	}

	mse = 0.;
	width = frame0->width/2;
	height = frame0->height/2;
    width1 = frame1->width/2;
	size = width * height;
	curr0 = frame0->u;
	curr1 = frame1->u + (REFERENCE_FRAME_PAD_AMT/2 * frame1->width/2) + (REFERENCE_FRAME_PAD_AMT/2);
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			aux = curr0[x] - curr1[x];
			mse += aux*aux;
		}
		curr0 += width;
		curr1 += width1;
	}
	if (mse < CALC_SNR_MIN_MSE)
	{
		*res_snr_u = CALC_SNR_MAX_VALUE;
	}
	else
	{
		mse /= size;
		*res_snr_u = 10.* log10((255.*255.)/mse);
	}

	mse = 0.;
	width = frame0->width/2;
	height = frame0->height/2;
    width1 = frame1->width/2;
	size = width * height;
	curr0 = frame0->v;
	curr1 = frame1->v + (REFERENCE_FRAME_PAD_AMT/2 * frame1->width/2) + (REFERENCE_FRAME_PAD_AMT/2);
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			aux = curr0[x] - curr1[x];
			mse += aux*aux;
		}
		curr0 += width;
		curr1 += width1;
	}
	if (mse < CALC_SNR_MIN_MSE)
	{
		*res_snr_v = CALC_SNR_MAX_VALUE;
	}
	else
	{
		mse /= size;
		*res_snr_v = 10.* log10((255.*255.)/mse);
	}
}
