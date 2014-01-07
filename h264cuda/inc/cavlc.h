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


#ifndef _CAVLC_HPP
#define _CAVLC_HPP

#include "encoder_context.h"
#include "cavlc_data.h"

void cavlc_put_bits(
                        unsigned int *p_bits,
                        int          num_words,
                        bitstream_t  *p_bitstream
                        );
int write_last_skip_count(
                          int num_skip,
                          struct bitstream_t *p_bitstream
                          );

E_ERR init_cavlc_context(encoder_context_t *p_enc);
void free_cavlc_context(encoder_context_t *p_enc);

#endif   //

