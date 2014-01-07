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
#include <assert.h>

#include "../inc/h264_types.h"
#include "../inc/h264_common.h"
#include "../inc/mem_alloc_free.h"

/**
 * Memory allocation (internal analog of the malloc(...))
 * @param size Bytes to allocate
 * @returns Pointer to the allocated memory,
 */
void *getmem_1d_void(int size)
{
    unsigned char *ptrret = NULL;
    unsigned char *ptr = (unsigned char *)malloc(size + 4 + 31);
    if (ptr == NULL)
    {
        printf("Allocation failure in getmem_1d_void()");
    }
    else
    {
        ptrret = ptr + 4;
        ptrret = (unsigned char *)((unsigned long)(ptrret+31)&~31);
        *(unsigned int *)(ptrret-4) = (unsigned long)ptr;
    }

    return ptrret;
}


/**
 * Memory freeing (internal analog of the free(...))
 * @param ptr Previously allocated memory block to be freed
 */
void free_1D(void *ptr)
{
    if (ptr != NULL)
    {
        unsigned int *realptr = (unsigned int *)ptr;
        free((void*)(*(realptr-1)));
    }
}

E_ERR alloc_yuv_frame(
    yuv_frame_t *p_frame,
    int width,
    int height,
    int image_width,
    int image_height
    )
{
    int width_cr = width/2;
    int height_cr = height/2;
    int size_y = width * height;
    int size_cr = width_cr * height_cr;
    unsigned char *p_yuv;
    E_ERR err = ERR_SUCCESS;

    assert (width >= image_width);
    assert (height >= image_height);

    p_yuv = (unsigned char *) getmem_1d_void(size_y + size_cr * 2);

    if (p_yuv == NULL)
    {
        printf("alloc_yuv_frame(): error allocating memory\n");
        err = ERR_MEM;
    }

    if (!err)
    {
        p_frame->y = p_yuv;
        p_frame->u = p_yuv + size_y;
        p_frame->v = p_yuv + size_y + size_cr;
    
        p_frame->buffer_owner = 1;
    
        p_frame->width = width;
        p_frame->height = height;
        p_frame->image_width = image_width;
        p_frame->image_height = image_height;
    }

    return err;
}

void alloc_empty_yuv_frame(
    yuv_frame_t *frame,
    int width,
    int height,
    int image_width,
    int image_height
    )
{
    frame->y = NULL;
    frame->u = NULL;
    frame->v = NULL;

    frame->buffer_owner = 0;

	frame->width = width;
	frame->height = height;
	frame->image_width = image_width;
	frame->image_height = image_height;
}

void free_yuv_frame(yuv_frame_t * frame)
{
    if (frame->buffer_owner)
    {
	    free_1D(frame->y);
    }
	frame->y = NULL;
    frame->u = NULL;
    frame->v = NULL;
}
