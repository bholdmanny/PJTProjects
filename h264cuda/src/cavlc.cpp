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

#include "../inc/cavlc_data.h"
#include "../inc/cavlc.h"
#include "../inc/h264_common.h"

#include "../inc/const_defines.h"
#include "../inc/entropy_data.h"
#include "../inc/encoder_context.h"
#include "../inc/output.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void cavlc_put_bits(
                        unsigned int *p_bits,
                        int          num_words,
                        bitstream_t  *p_bitstream
                        )
{
    int i;

    for (i = 0; i < num_words; i++) {
        put_bits(p_bitstream, 32, p_bits[i]);
    }
}


void cavlc_put_bits_ref(
                        unsigned int *p_bits,
                        int          num_words,
                        bitstream_t  *p_bitstream
                        )
{
    int i;

    for (i = 0; i < num_words; i++) {
        put_bits(p_bitstream, 32, p_bits[i]);
    }
}

// Returns the number of bits written to the bitstream.
int write_last_skip_count
(
 int num_skip,
 bitstream_t *p_bitstream
 )
{
    int size = 0;
    if (num_skip > 0) {
        size = write_unsigned_uvlc(p_bitstream, num_skip);
    }
    return (size);
}

