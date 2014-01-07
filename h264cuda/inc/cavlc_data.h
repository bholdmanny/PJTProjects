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


#ifndef _CAVLC_DATA_HPP_
#define _CAVLC_DATA_HPP_

#include <assert.h>

#include "const_defines.h"
// Need this to include S_MV definition
#include "mb_info.h"


#define BLOCK_PAIRS_PER_MB 16
#define HEAD_COED_PER_I_MB 8
#define STREAM_PROCESSORS  16
#define CAVLC_BLOCKS_PER_MB     32

#define MAKE_DIV_SIZE(size)         ((((size) + (16 - 1)) / (16)) * (16))
#define ROUND_TWO_MB_PER_LANE(size)         ((((size) + ((16*2) - 1)) / (16* 2)) * (16 * 2))

//---------------------------------------------------------------------
//  Typedefs
//---------------------------------------------------------------------


// DC and DCT coefficients in zig-zag order.  There are 16
// block-pairs.  Each block pair consists of 16 coefficients for two
// 4x4 blocks.  But the block index is the minor index.  When typecast
// as an int, these will match the int16x2 variables used in the
// stream code.  That's why the minor index is the block index---to be
// able to validate the outputs of the stream and reference code
// easily.

typedef short S_TEXTURE_BLOCK_PAIRS[BLK_SIZE][2];


// There is one copy of this structure below for each 4x4 block.  So,
// of course, DeltaQuant and NumSkipMB will be exactly the same for
// each copy of this struct for the same macroblock, but DeltaMV and
// Subtype won't necessarily be the same.

typedef struct {

    // These fields hold information about each macroblock

    // This contains Type, SubType, IntraChromaMode, and RefFrameIdx.
    // SubType is also used for encoding the prediction direction (or
    // mode) used by this 4x4 block if the macroblock was encoded
    // using Intra-4x4 prediction.  If the direction is equal to the
    // most probable direction (based on the minimum direction # of
    // its neighbors chosen directions), then it will be set to -1.
    // Otherwise it will be set to 0-7, which is the value that will
    // get signalled directly in the bitstream.  If the block was not
    // coded using Intra-4x4 prediction, then this will be set to -2.
#define HDR_CONTEXT_TYPE             0
#define HDR_CONTEXT_SUBTYPE          1
#define HDR_CONTEXT_INTRACHROMAMODE  2
#define HDR_CONTEXT_REF_IDX          3
    char MBModes[4];

    // DELTAQUANT: Difference from previous macroblock's value of QP.
    // For the first macroblock, this is the difference from the value
    // of QP specified in the Slice header.
    // NUMSKIPMB: This variable indicates how many consecutive
    // macroblocks before this one are "skipped" (i.e., not encoded).
#define HDR_CONTEXT_CBP              0
#define HDR_CONTEXT_DELTA_QUANT      1
#define HDR_CONTEXT_NUMSKIPMB        2
    unsigned int Misc;

    // Differential motion vector, which is the difference between the
    // motion vector for a partition and the predicted motion vector
    // for that same partition.  There is one per 4x4 block.  If
    // multiple 4x4 blocks are in the same macroblock partition, this
    // variable will be the same for all of those 4x4 blocks.  The low
    // element is X, and the high element is Y.
    short DeltaMV[2];

    // These fields hold information about each 4x4 texture block.

    // This pair of shorts indicates the block sizes of each block in
    // a 4x4 block pair.  A size of 0 indicates that there are no
    // valid coefficients.  All coefficients will be in the first
    // BlockSize locations out of the total 16 locations allocated for
    // each coefficient block.  For instance, the ChromaDC block only
    // contains 4 valid values out of the possible 16.  So this
    // variable would be set to {4, 4} for the lane that handles the
    // two ChromaDC blocks.
    short BlockSize[2];

    // This is the same name as used in the standard.  It basically is
    // a predictor for the number of non-zero coefficients in a 4x4
    // block.
    short nC[2];

} S_CAVLC_CONTEXT;



// This block contains syntax elements that are to be coded into the
// bitstream for a texture block pair.

typedef struct { 

    // Total number of non-zero coefficients in the block.  This is
    // one field in coeff_token.  If this is zero, then coeff_token is
    // coded, but the rest of the 4x4 block coefficient data need not
    // be coded (i.e., TotalZeros, TrailingOnes, Levels, Runs, etc.).
    // If this is -1, then even coeff_token need not be coded.
    short NumCoeff[2];

    // This is the other field that makes up coeff_token.  A value of
    // -1 means it does not need to be encoded in the bitstream.
    short NumTrailingOnes[2];

    // The context (based on nC) for encoding coeff_token.  A value of
    // -1 means it does not need to be encoded in the bitstream.
    short ContextCoeffToken[2];

    // One bit per trailing one.  '1' means negative, '0' means
    // positive.  The LSB corresponds to the trailing one whose
    // coefficient index inthe zig-zag ordered block is closest to 0.
    short TrailingOnesSigns[2];

    // The total number of zeros in a block before the last non-zero
    // coefficient in the zig-zag ordered 4x4 or 2x2 block.  A value
    // of -1 means it doesn't need to be encoded.
    short TotalZeros[2];

} S_TEXTURE_SYMBOLS;


// This holds the non trailing-ones non-zero coefficients in the 4x4
// block.  The coefficients are in REVERSE order---that is the last
// non-zero coefficient is in location "0", the next-to-last non-zero
// coefficient is in location "1", and so on.  A suffix length of -1
// indicates an invalid coefficient.

typedef struct S_LEVEL_SYMBOLS_T {
    short suffix_length[2];
    short coeff[2];
} S_LEVEL_SYMBOLS[BLK_SIZE];


// This is for the suffix_length==0 level, if one exists

typedef struct S_LEVEL_SUFFIX_LENGTH0_SYMBOL_T {
    short valid[2];
    short coeff[2];
} S_LEVEL_SUFFIX_LENGTH0_SYMBOL;


typedef struct S_RUN_SYMBOLS_T {
    char run[4];
    char zeros_left[4];
} S_RUN_SYMBOLS[BLK_SIZE/4];


// The S_CODES structure (below) holds the codes for texture blocks
// and mb headers.  Each struct contains the length and value for two
// codes, one in each half of an int16x2.  For a macroblock, each lane
// contains CODES_PAIRS_PER_LANE copies of this structure.  (See
// const_defines.h for detailed description of what syntax elements
// are assigned to each of the CODE_PAIRS_PER_LANE entries).

// The details on code indices presented above apply to this data
// structure.
typedef struct S_CODES_T {
    short length[2];
    unsigned short value[2];
} S_CODES[CODE_PAIRS_PER_LANE];

// Useful within c version of kernels for function return values
typedef struct SINGLE_CODE_T {
    short length;
    unsigned short value;
} SINGLE_CODE;


// DC and DCT coefficients in zig-zag order.  There are 16
// block-pairs.  Each block pair consists of 16 coefficients for two
// 4x4 blocks.  But the block index is the minor index.  When typecast
// as an int, these will match the int16x2 variables used in the
// stream code.  That's why the minor index is the block index---to be
// able to validate the outputs of the stream and reference code
// easily.

typedef short S_TEXTURE_BLOCK[BLK_SIZE];


// There is one copy of this structure below for each 4x4 block.  So,
// of course, DeltaQuant and NumSkipMB will be exactly the same for
// each copy of this struct for the same macroblock, but DeltaMV and
// Subtype won't necessarily be the same.

typedef struct {

    // These fields hold information about each macroblock

    // This contains Type, SubType, IntraChromaMode, and RefFrameIdx.
    // SubType is also used for encoding the prediction direction (or
    // mode) used by this 4x4 block if the macroblock was encoded
    // using Intra-4x4 prediction.  If the direction is equal to the
    // most probable direction (based on the minimum direction # of
    // its neighbors chosen directions), then it will be set to -1.
    // Otherwise it will be set to 0-7, which is the value that will
    // get signalled directly in the bitstream.  If the block was not
    // coded using Intra-4x4 prediction, then this will be set to -2.
#define HDR_CONTEXT_TYPE             0
#define HDR_CONTEXT_SUBTYPE          1
#define HDR_CONTEXT_INTRACHROMAMODE  2
#define HDR_CONTEXT_REF_IDX          3
    char MBModes[4];

    // DELTAQUANT: Difference from previous macroblock's value of QP.
    // For the first macroblock, this is the difference from the value
    // of QP specified in the Slice header.
    // NUMSKIPMB: This variable indicates how many consecutive
    // macroblocks before this one are "skipped" (i.e., not encoded).
#define HDR_CONTEXT_CBP              0
#define HDR_CONTEXT_DELTA_QUANT      1
#define HDR_CONTEXT_NUMSKIPMB        2
    unsigned int Misc;

    // Differential motion vector, which is the difference between the
    // motion vector for a partition and the predicted motion vector
    // for that same partition.  There is one per 4x4 block.  If
    // multiple 4x4 blocks are in the same macroblock partition, this
    // variable will be the same for all of those 4x4 blocks.  The low
    // element is X, and the high element is Y.
    short DeltaMV[2];

    // These fields hold information about each 4x4 texture block.

    // This pair of shorts indicates the block sizes of each block in
    // a 4x4 block pair.  A size of 0 indicates that there are no
    // valid coefficients.  All coefficients will be in the first
    // BlockSize locations out of the total 16 locations allocated for
    // each coefficient block.  For instance, the ChromaDC block only
    // contains 4 valid values out of the possible 16.  So this
    // variable would be set to {4, 4} for the lane that handles the
    // two ChromaDC blocks.
    short BlockSize;

    // This is the same name as used in the standard.  It basically is
    // a predictor for the number of non-zero coefficients in a 4x4
    // block.
    short nC;

} S_CAVLC_CONTEXT_BLOCK;

typedef struct {
    short BlockSize;
    short nC;

} S_CAVLC_CONTEXT_DC_CHROMA;

// This block contains syntax elements that are to be coded into the
// bitstream for a texture block pair.

typedef struct { 

    // Total number of non-zero coefficients in the block.  This is
    // one field in coeff_token.  If this is zero, then coeff_token is
    // coded, but the rest of the 4x4 block coefficient data need not
    // be coded (i.e., TotalZeros, TrailingOnes, Levels, Runs, etc.).
    // If this is -1, then even coeff_token need not be coded.
    short NumCoeff;

    // This is the other field that makes up coeff_token.  A value of
    // -1 means it does not need to be encoded in the bitstream.
    short NumTrailingOnes;

    // The context (based on nC) for encoding coeff_token.  A value of
    // -1 means it does not need to be encoded in the bitstream.
    short ContextCoeffToken;

    // One bit per trailing one.  '1' means negative, '0' means
    // positive.  The LSB corresponds to the trailing one whose
    // coefficient index inthe zig-zag ordered block is closest to 0.
    short TrailingOnesSigns;

    // The total number of zeros in a block before the last non-zero
    // coefficient in the zig-zag ordered 4x4 or 2x2 block.  A value
    // of -1 means it doesn't need to be encoded.
    short TotalZeros;

} S_TEXTURE_SYMBOLS_BLOCK;


// This holds the non trailing-ones non-zero coefficients in the 4x4
// block.  The coefficients are in REVERSE order---that is the last
// non-zero coefficient is in location "0", the next-to-last non-zero
// coefficient is in location "1", and so on.  A suffix length of -1
// indicates an invalid coefficient.

typedef struct S_LEVEL_SYMBOLS_S {
    short suffix_length;
    short coeff;
} S_LEVEL_SYMBOLS_BLOCK;


// This is for the suffix_length==0 level, if one exists

typedef struct S_LEVEL_SUFFIX_LENGTH0_SYMBOL_S {
    short valid;
    short coeff;
} S_LEVEL_SUFFIX_LENGTH0_SYMBOL_BLOCK;


typedef struct S_RUN_SYMBOLS_S {
    char run;
    char zeros_left;
} S_RUN_SYMBOLS_BLOCK;

// The details on code indices presented above apply to this data
// structure.
typedef struct S_CODES_S {
    short length;
    unsigned short value;
} S_CODES_BLOCK[CODE_PAIRS_PER_LANE];



//---------------------------------------------------------------------
//  Function prototypes
//---------------------------------------------------------------------
void concatenate_codes_two_code(
                           SINGLE_CODE *code1,      // will be updated    
                           SINGLE_CODE code2    // will be cat'd to code1
                           );
void concatenate_codes_ref(
                           struct S_CODES_T *code1,      // will be updated
                           int j,                 // which block
                           SINGLE_CODE code2    // will be cat'd to code1
                           );

void concatenate_codes_s_codes_s(
                           struct S_CODES_S *code1,      // will be updated
                           SINGLE_CODE code2    // will be cat'd to code1
                           );

#endif 

