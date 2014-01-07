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
#include <stdlib.h>
#include "../inc/entropy_data.h"
#include "../inc/output.h"

// The following function should not be called directly from outside of this file
void put_byte_avoid_start_code(bitstream_t *p_bitstream, int next_byte)
{
    // make sure we don't write over the end of the buffer
    assert ((unsigned int)(p_bitstream->p_buffer_curr - p_bitstream->p_buffer) < (p_bitstream->buffer_size - 2) );

    if ((p_bitstream->zero_count == 2) && !(next_byte & 0xFC))
    {
        *(p_bitstream->p_buffer_curr)++ = 0x03;
        p_bitstream->zero_count = 0;
    }
    if (next_byte == 0)
    {
        p_bitstream->zero_count++;
    }
    else
    {
        p_bitstream->zero_count = 0;
    }
    *(p_bitstream->p_buffer_curr)++ = (unsigned char)(next_byte);
}

// The following function should not be called directly from outside of this file
void put_raw_byte (bitstream_t *p_bitstream, int next_byte)
{
    // make sure we don't write over the end of the buffer
    assert ((unsigned int)(p_bitstream->p_buffer_curr - p_bitstream->p_buffer) < (p_bitstream->buffer_size - 2) );

    if (next_byte == 0)
    {
        p_bitstream->zero_count++;
    }
    else
    {
        p_bitstream->zero_count = 0;
    }
    *(p_bitstream->p_buffer_curr)++ = (unsigned char)(next_byte);
}



void put_24_or_less_bits(bitstream_t *p_bitstream, unsigned int num_of_bits, unsigned int bits) 
{
    assert(num_of_bits <= 24);
    p_bitstream->bitpos -= num_of_bits; //bitpos is number of free bits in 32-bit buffer
	p_bitstream->bitbuf |= (bits << (p_bitstream->bitpos));
	while (p_bitstream->bitpos <= 24)
    {
		put_byte_avoid_start_code (p_bitstream, (p_bitstream->bitbuf >> 24));
		p_bitstream->bitbuf <<= 8;
		p_bitstream->bitpos += 8;
	}
}

void put_bits(bitstream_t *p_bitstream, unsigned int num_of_bits, unsigned int bits)
{
    if (num_of_bits > 24) 
	{
        put_24_or_less_bits(p_bitstream, num_of_bits - 24, bits >> 24);
        put_24_or_less_bits(p_bitstream, 24, bits & 0x00ffffff);
    } 
	else 
	{
        put_24_or_less_bits(p_bitstream, num_of_bits, bits);
    }
}


void put_raw_bits(bitstream_t *p_bitstream, unsigned int num_of_bits, unsigned int bits) 
{
    p_bitstream->bitpos -= num_of_bits; //bitpos is number of free bits in 32-bit buffer
	p_bitstream->bitbuf |= (bits << (p_bitstream->bitpos));
	while (p_bitstream->bitpos <= 24)
    {
		put_raw_byte(p_bitstream, (p_bitstream->bitbuf >> 24));
		p_bitstream->bitbuf <<= 8;
		p_bitstream->bitpos += 8;
	}
}



//////////////////////////////////////////////////////////////////////////
//
unsigned int num_bytes_in_bitstream(bitstream_t *p_bitstream)
//
//////////////////////////////////////////////////////////////////////////
{
    return (p_bitstream->p_buffer_curr - p_bitstream->p_buffer);
}

//////////////////////////////////////////////////////////////////////////
//
void output_stream_init(bitstream_t *p_bitstream)
//
// returns nothing
//////////////////////////////////////////////////////////////////////////
{
    assert (p_bitstream->p_buffer != NULL);
    assert (p_bitstream->buffer_size != 0);

    p_bitstream->p_buffer_curr = p_bitstream->p_buffer;
    p_bitstream->bitbuf        = 0;
    p_bitstream->bitpos        = 32; //number of free bits
    p_bitstream->zero_count    = 0;
}

//////////////////////////////////////////////////////////////////////////
//
int output_stream_done(bitstream_t *p_bitstream)
//
// return number of bytes
//////////////////////////////////////////////////////////////////////////
{
	if (p_bitstream->bitpos < 32) // num_of_free_bits < 32 
    {
		put_byte_avoid_start_code (p_bitstream, (p_bitstream->bitbuf >> 24));
		p_bitstream->bitpos = 32;
	}
	return (p_bitstream->p_buffer_curr - p_bitstream->p_buffer);
}


//////////////////////////////////////////////////////////////////////////
void byte_align_bitstream (bitstream_t *p_bitstream)
{
	int nbits = p_bitstream->bitpos & 7;
	if (nbits)
    {
        put_bits(p_bitstream, nbits, 0);
    }
}


//////////////////////////////////////////////////////////////////////////
//
unsigned int write_unsigned_uvlc(bitstream_t *p_bitstream, unsigned int value)
//
// Writes unsigned value using UVLC
//////////////////////////////////////////////////////////////////////////
{
    unsigned int size = 0;
    unsigned int u = (value + 1) >> 1;

    for (; u != 0; u >>= 1, size++);

    assert(size < 13);

    size = size * 2 + 1;

    put_bits(p_bitstream, size, value + 1);

    return size;
}

unsigned int write_unsigned_uvlc_big(bitstream_t *p_bitstream, unsigned int value)
{
    unsigned int size = 0;
    unsigned int u = (value + 1) >> 1;

    for (; u != 0; u >>= 1, size++);

    assert(size < 24);

    put_bits(p_bitstream, size, 0);
    put_bits(p_bitstream, size + 1, value + 1);

    return size;
}

unsigned int write_signed_uvlc(bitstream_t *p_bitstream, int value)
{
    unsigned int size = 0;
    unsigned int sign = (value <= 0) ? ((value = -value), 1) : 0;
    unsigned int u = value;

    for (; u != 0; u >>= 1, size++);

    assert(size < 12);

    u = 1 << size;
    size = size * 2 + 1;

    put_bits(p_bitstream, size, u | (value * 2 + sign - u));

    return size;
}
