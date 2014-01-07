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


#ifndef _OUTPUT_H
#define _OUTPUT_H

#include "entropy_data.h"

void put_24_or_less_bits(bitstream_t *p_bitstream, unsigned int num_of_bits, unsigned int bits);
void put_bits(bitstream_t *p_bitstream, unsigned int num_of_bits, unsigned int bits);
void put_raw_bits(bitstream_t *p_bitstream, unsigned int num_of_bits, unsigned int bits);
unsigned int num_bytes_in_bitstream(bitstream_t *p_bitstream);
void output_stream_init(bitstream_t *p_bitstream);
int output_stream_done(bitstream_t *p_bitstream);
void byte_align_bitstream (bitstream_t *p_bitstream);
unsigned int write_unsigned_uvlc(bitstream_t *p_bitstream, unsigned int value);
unsigned int write_unsigned_uvlc_big(bitstream_t *p_bitstream, unsigned int value);
unsigned int write_signed_uvlc(bitstream_t *p_bitstream, int value);
    
#endif

