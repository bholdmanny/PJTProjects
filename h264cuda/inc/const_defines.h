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

#ifndef _CONST_DEFINES_H
#define _CONST_DEFINES_H

// some useful constants
#define LARGE_NUMBER  0x7fffffff

#define SP_FALSE    0
#define SP_TRUE     0xffffffff


/////////////////////////////////////////////////////////////////////////
//
// H.264 related general constants
//
/////////////////////////////////////////////////////////////////////////

// Macroblock constant
#define BLOCKS_PER_MB              16                           // number of 4x4 blocks in a luma macroblock
#define BLOCKS_PER_MB_C            4                            // number of 4x4 blocks in a chroma macroblock
#define BLOCKS_PER_MB_ROW          4                            // number of 4x4 blocks in per row in a luma macroblock

#define BLK_WIDTH                  4                            // width of a 4x4 block
#define BLK_HEIGHT                 4                            // height of a 4x4 block
#define BLK_SIZE                   (BLK_WIDTH * BLK_HEIGHT)     // number of pixels of a 4x4 block

#define MB_WIDTH                   16                           // width of a luma macroblock
#define MB_HEIGHT                  16                           // height of a luma macroblock
#define MB_TOTAL_SIZE              (MB_WIDTH * MB_HEIGHT)       // total nubmer of pixels in a luma macroblock
#define MB_WIDTH_C                 8                            // width of a chroma macroblock
#define MB_HEIGHT_C                8                            // height of a chroma macroblock
#define MB_TOTAL_SIZE_C            (MB_WIDTH_C * MB_HEIGHT_C)   // total number of pixels in a chroma macroblock

#define NUM_DC_COEFS_PER_MB        (BLOCKS_PER_MB)
#define NUM_DC_COEFS_PER_MB_C      (BLOCKS_PER_MB_C)

// Max value of quantization parameter
#define MIN_QP           0
#define MAX_QP          51
#define NUM_QP          52
#define MAX_NUM_QPS      4

#define MAX_NUM_ROIS     8

#define D1_STRIP_SIZE    45
// Threshold for Intra/Inter mode decision

#define AVG_MIN_THRESHOLD       1200

// Default rate control parameters
#define DEFAULT_IFRAME_QP 30
#define DEFAULT_PFRAME_QP 30
#define DEFAULT_MIN_QP  14
#define DEFAULT_MAX_QP  51

#define DEFAULT_BITRATE     1000000
#define DEFAULT_FRAMERATE   30
#define DEFAULT_IFRAME_RATE 300

#define IDXA_PAD        12
#define IDXB_PAD         3
/////////////////////////////////////////////////////////////////////////
//
// For Macroblock Info Data Structure
//
/////////////////////////////////////////////////////////////////////////

// Defines for R_BLK_MB_INFO::Loc and S_BLK_MB_INFO::Loc
#define LOC_BLK_LEFT_PICTURE_EDGE   0x0001
#define LOC_MB_LEFT_PICTURE_EDGE    0x0010
#define LOC_BLK_TOP_PICTURE_EDGE    0x0002
#define LOC_MB_TOP_PICTURE_EDGE     0x0020
#define LOC_BLK_RIGHT_PICTURE_EDGE  0x0004
#define LOC_MB_RIGHT_PICTURE_EDGE   0x0040
#define LOC_BLK_BOTTOM_PICTURE_EDGE 0x0008
#define LOC_MB_BOTTOM_PICTURE_EDGE  0x0080
// Note, that these below will only be set after slice boundaries have
// been determined.
#define LOC_BLK_LEFT_SLICE_EDGE     0x0100
#define LOC_MB_LEFT_SLICE_EDGE      0x1000
#define LOC_BLK_TOP_SLICE_EDGE      0x0200
#define LOC_MB_TOP_SLICE_EDGE       0x2000
// Note, that these are not being used currently
#define LOC_BLK_RIGHT_SLICE_EDGE    0x0400
#define LOC_MB_RIGHT_SLICE_EDGE     0x4000
#define LOC_BLK_BOTTOM_SLICE_EDGE   0x0800
#define LOC_MB_BOTTOM_SLICE_EDGE    0x8000

#define LOC_BLK_LEFT_EDGE           0x0101
#define LOC_BLK_TOP_EDGE            0x0202
#define LOC_MB_LEFT_EDGE            0x1010
#define LOC_MB_TOP_EDGE             0x2020
#define LOC_MB_RIGHT_EDGE           (LOC_MB_RIGHT_PICTURE_EDGE | LOC_MB_RIGHT_SLICE_EDGE)


#define LOC_BLK_LEFT_EDGE_16x2      0x01010101
#define LOC_BLK_TOP_EDGE_16x2       0x02020202

#define LOC_BLK_LEFT_OR_TOP_PICTURE_EDGE 0x0003
#define LOC_BLK_LEFT_OR_TOP_SLICE_EDGE   0x0300
#define LOC_MB_LEFT_OR_TOP_PICTURE_EDGE  0x0030
#define LOC_MB_LEFT_OR_TOP_SLICE_EDGE    0x3000

#define LOC_BLK_TOP_OR_RIGHT_PICTURE_EDGE 0x0006

#define LOC_BLK_LEFT_OR_TOP_EDGE    0x0303
#define LOC_MB_LEFT_OR_TOP_EDGE     0x3030

// reference padding for prediction
#define REFERENCE_FRAME_PAD_AMT    ( 16 )
// Frame offsets
#define INPUT_FRAME_X_OFFSET        0   // luma
#define INPUT_FRAME_Y_OFFSET        0   // luma
#define RECON_FRAME_X_OFFSET        (REFERENCE_FRAME_PAD_AMT)     // luma
#define RECON_FRAME_Y_OFFSET        (REFERENCE_FRAME_PAD_AMT)     // luma

#define INPUT_FRAME_X_OFFSET_C      0   // chroma
#define INPUT_FRAME_Y_OFFSET_C      0   // chroma

#define RECON_FRAME_X_OFFSET_C      ((RECON_FRAME_X_OFFSET)>>1)   // chroma
#define RECON_FRAME_Y_OFFSET_C      ((RECON_FRAME_Y_OFFSET)>>1)   // chroma


// Values for R_BLK_MB_INFO::Type.
#define INTER_LARGE_BLOCKS_MB_TYPE  0
#define INTER_SMALL_BLOCKS_MB_TYPE  1
#define INTRA_LARGE_BLOCKS_MB_TYPE  2
#define INTRA_SMALL_BLOCKS_MB_TYPE  3
#define INTRA_IPCM_MB_TYPE          4
#define UNDEFINED_MB_TYPE           5

#define vINTER_LARGE_BLOCKS_MB_TYPE  (vec int)0
#define vINTER_SMALL_BLOCKS_MB_TYPE  (vec int)1
#define vINTRA_LARGE_BLOCKS_MB_TYPE  (vec int)2
#define vINTRA_SMALL_BLOCKS_MB_TYPE  (vec int)3
#define vINTRA_IPCM_MB_TYPE          (vec int)4
#define vUNDEFINED_MB_TYPE           (vec int)5

// Values for R_BLK_MB_INFO::SubType if R_BLK_MB_INFO::Type is equal
// to INTRA_LARGE_BLOCKS_MB_TYPE (i.e., these are the Intra-16x16
// prediction modes).  DO NOT CHANGE THE ORDER OR VALUE OF THESE
// DEFINES.
#define L_PRED_VERTICAL      0
#define L_PRED_HORIZONTAL    1
#define L_PRED_DC            2
#define L_PRED_PLANE         3
#define PRED_UNKNOWN        -1      // This number has to be a negative number.


// Values for R_BLK_MB_INFO::SubType if R_BLK_MB_INFO::Type is equal
// to INTRA_SMALL_BLOCKS_MB_TYPE (i.e., these are the Intra-4x4
// prediction modes).  DO NOT CHANGE THE ORDER OR VALUE OF THESE
// DEFINES.
#define S_PRED_VERT             0
#define S_PRED_HOR              1
#define S_PRED_DC               2
#define S_PRED_DIAG_DOWN_LEFT   3
#define S_PRED_DIAG_DOWN_RIGHT  4
#define S_PRED_VERT_RIGHT       5
#define S_PRED_HOR_DOWN         6
#define S_PRED_VERT_LEFT        7
#define S_PRED_HOR_UP           8
#define S_PRED_INVALID_MODE     -1    // This number has to be negative

// Values for R_BLK_MB_INFO::IntraChromaMode if R_BLK_MB_INFO::Type is equal
// to INTRA_SMALL_BLOCKS_MB_TYPE or INTRA_LARGE_BLOCKS_MB.
// DO NOT CHANGE THE ORDER OR VALUE OF THESE DEFINES.
#define C_PRED_DC           0
#define C_PRED_HORIZONTAL   1
#define C_PRED_VERTICAL     2
#define C_PRED_PLANE        3
// #define PRED_UNKNOWN  -1 see above

// Values for R_BLK_MB_INFO::SubType if R_BLK_MB_INFO::Type is equal
// to INTER_LARGE_BLOCKS_MB_TYPE or INTER_SMALL_BLOCKS_MB_TYPE (i.e.,
// these are the Inter prediction block shapes).  For clarification,
// the SINGLE_BLOCK_SUBDIV_TYPE refers to 16x16 and 8x8 blocks, the
// HOR_SUBDIV_TYPE refers to 16x8 and 8x4 blocks, the VERT_SUBDIV_TYPE
// refers to 8x16 and 4x8 blocks, and the HOR_VERT_SUBDIV_TYPE refers
// to 8x8 and 4x4 blocks.
#define	SINGLE_BLOCK_SUBDIV_TYPE  0
#define	HOR_SUBDIV_TYPE           1
#define	VERT_SUBDIV_TYPE          2
#define	HOR_VERT_SUBDIV_TYPE      3
#define	UNDEFINED_SUBDIV_TYPE     4

#define	vSINGLE_BLOCK_SUBDIV_TYPE  (vec int)0
#define	vHOR_SUBDIV_TYPE           (vec int)1
#define	vVERT_SUBDIV_TYPE          (vec int)2
#define	vHOR_VERT_SUBDIV_TYPE      (vec int)3
#define	vUNDEFINED_SUBDIV_TYPE     (vec int)4

#define CODE_PAIRS_PER_LANE        26

// clusters 0--13 contain codes for texture blocks (see comments for
// BLOCK_PAIRS_PER_MB to see the assignment of texture blocks to
// clusters).  Within each lane, there are CODES_PER_LANE/2 codes
// assigned to each texture block (2 blocks per lane).  These codes
// are assigned in the following order to the elements within a
// texture block:
//
//   0  -  CoeffToken , Trailing-Ones Signs
//   1  -  Level Code 0
//           :
//           :
//   16 -  Level Code 15
//   17 -  TotalZeros
//   18 -  Run Code 0 , Run Code 1
//           :
//           :
//   24 -  Run Code 12 , Run Code 13
//   25 -  Run Code 14 , Run Code 15
//
// Where more than one element is indicated for an entry and separated
// by a comma, those two elements are "packed" together such that the
// NumBits field indicates the total bits for both elements combined,
// and the Value field contains the bits from both elements
// concatenated in the correct order.

#define COEFFTOKEN_T1SIGN_CODE_IDX  0
#define FIRST_LEVEL_CODE_IDX        1
#define TOTALZEROS_CODE_IDX        17
#define FIRST_RUN_CODE_IDX         18



#define MBTYPE_CODE_IDX          1
#define ILUMA4X4_MODE_CODE_IDX   2
#define ICHROMA_MODE_CODE_IDX    3


#define MBSKIPRUN_CODE_IDX       0
#define FIRST_SUBDIV_CODE_IDX    4
#define FIRST_REFFRAME_CODE_IDX  6
#define FIRST_DMV_CODE_IDX       8
#define CBP_CODE_IDX            16
#define DQUANT_CODE_IDX         17



// Number of bytes in the output stream for each macroblock.  Must be
// a multiple of 4.  Worst case should be 384, if PCM blocks are supported.
#define MAX_BYTES_PER_MB      512
#define MAX_NUM_MBS_X         60    //120       
#define MAX_NUM_MBS_X_1080    120       //(1920/16)

#define FRAC_MUL            (1 << 20) //Q.20 format

/* ME Modes */
#define ME_MODE_MINUS_ONE       -1
#define ME_MODE_ZERO            0
#define ME_MODE_FIVE            5
#define ME_MODE_TEN             10
#define ME_MODE_TWENTY          20
#define ME_MODE_THIRTY          30

// Definition for rate control
#define BIG_MV_DEF           32

// Multi Channel Definitions

#define NUM_CHANNELS        16

// Level information
#define NUM_LEVELS 16
#define MAX_FPS 60

#define MAX_FILTER_COEFFS       5
#define MAX_NUM_FILTER_SETS     5

// Level information
#define NUM_LEVELS 16
#define MAX_FPS 60

/////////////////////////////////////////////////////////////////////////
//
// General constants
//
/////////////////////////////////////////////////////////////////////////

// some useful constants
#define LARGE_NUMBER  0x7fffffff

#define SP_FALSE    0
#define SP_TRUE     0xffffffff
#define NEW_LOW_RES

/////////////////////////////////////////////////////////////////////////
//
// H.264 related general constants
//
/////////////////////////////////////////////////////////////////////////
#define AVG_QP_H264                32
#define AVG_QP_MPEG4               20
#define SAD_RATIO_MULT_CONST       10

// Macroblock constant
#define BLOCKS_PER_MB              16                           // number of 4x4 blocks in a luma macroblock
#define BLOCKS_PER_MB_C            4                            // number of 4x4 blocks in a chroma macroblock
#define BLOCKS_PER_MB_ROW          4                            // number of 4x4 blocks in per row in a luma macroblock

#define BLK_WIDTH                  4                            // width of a 4x4 block
#define BLK_HEIGHT                 4                            // height of a 4x4 block
#define BLK_SIZE                   (BLK_WIDTH * BLK_HEIGHT)     // number of pixels of a 4x4 block

#define MB_WIDTH                   16                           // width of a luma macroblock
#define MB_HEIGHT                  16                           // height of a luma macroblock
#define MB_TOTAL_SIZE              (MB_WIDTH * MB_HEIGHT)       // total nubmer of pixels in a luma macroblock
#define MB_WIDTH_C                 8                            // width of a chroma macroblock
#define MB_HEIGHT_C                8                            // height of a chroma macroblock
#define MB_TOTAL_SIZE_C            (MB_WIDTH_C * MB_HEIGHT_C)   // total number of pixels in a chroma macroblock
#define MB_SIZE                    16    // size in each direction

#define HRMB_SIZE 8
#define HLFRES_DEC_RATIO (MB_SIZE/HRMB_SIZE)

// Maximum search range
#ifdef NEW_LOW_RES
#define MAX_SEARCH_RANGE_X 128
#define MAX_SEARCH_RANGE_Y 96

#endif

#define MAX_HALF_RES_SEARCH_X 16
#define MAX_HALF_RES_SEARCH_Y 16
#define MIN_SEARCH_RANGE_X 16
#define MIN_SEARCH_RANGE_Y 16

#define HR_SEARCH_RANGE_X 16 // This is defined in integer, 1/2 in half res
#define HR_SEARCH_RANGE_Y 16 // This is defined in integer, 1/2 in half res
#define HR_SEARCH_SIZE HRMB_SIZE
#define HR_ZERO_BIAS 20
#define HR_WEIGHT 1

#define LRMB_SIZE 4
#define QRMB_SIZE 4
#define QR_SEARCH_SIZE QRMB_SIZE
#define LOWRES_DEC_RATIO (MB_SIZE/QRMB_SIZE)
#define QR_ZERO_BIAS 10
#define QR_WEIGHT 1
// reference padding for prediction
#define REFERENCE_FRAME_PAD_AMT    ( 16 )

#define DEFAULT_SEARCH_RANGE_X 32
#define DEFAULT_SEARCH_RANGE_Y 32

#define ME_NAME_SIZE   40
#define ROUND_CONSTANT 0x00020002p2
#define ROW_STRIP_DECIMATE 4


/////////////////////////////////////////
// MACROS for NEW LOW RES
/////////////////////////////////////////
#ifdef NEW_LOW_RES
#define LOWRESCOST // Modulate cost -> new low res cost
#define	MAXMBCOLS				(120)// for fullHD
#define MAX_SEARCH_RANGEMB_X	(MAX_SEARCH_RANGE_X>>4)	// (=64integerpixels)Max Search Range in MBs required in  horz direction for QR search
#define MAX_SEARCH_RANGEMB_Y	(MAX_SEARCH_RANGE_Y>>4)	// (=32integerpixels)Max Search Range in MBs required in  vert direction for QR search
#define MAX_SUB_SEC_COL			(6)// Max Sub sec size in Cols (for D1 size)
#define MAX_SUB_SEC_ROW			(4)	// Max Sub sec size in Rows (for D1 size)
#define HALF_RES_SEARCH_RANGE_X (8) // In integer pixels
#define HALF_RES_SEARCH_RANGE_Y (8) // In integer pixels
#define HALF_RES_SEARCH_RANGE_Y_LESS (4) // In integer pixels, lesser HR Y search for better performance/speed in HD/CIF resolutions
#define MAX_REF_SUB_SEC_AREA	(306)//(for 2*9=18)calculated from std resoultions' subsection sizes
#define MAX_HORZ_WORDS_PER_MV	(5) // Assuming HR data to be half word aligned, and search fixed at +/-4 
									// we will fix load to 4 (1+2+1)words for search in X direction.
// Maximum frame size expected
#define MAX_FRAME_WIDTH			(1920)
#define MAX_FRAME_HEIGHT		(1152)// round up for 1088 to have equal row subsectioning
//The maximum MVs generated is for HD size(calc for HD 1920x1088 = (15*3)*6*2)
#define SIZE_OF_MVINFO			(2)//One word each for X,Y and 16bits for SAD,per macroblock,

#define TEXTURE_UINT16x2_TABLE_LANE_SIZE         ( 520U )
#endif

#endif 

