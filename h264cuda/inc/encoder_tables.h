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


#ifndef _ENCODER_TABLES_H
#define _ENCODER_TABLES_H

#include "const_defines.h"

extern const int QP2QUANT_NEW[];

extern const int NeighborTables[8][16];

extern const int VECTORS_COST_MINUS_127[5][256];
extern const int *VECTORS_COST_MINUS_127_4x4;
extern const int *VECTORS_COST_MINUS_127_4x8;
extern const int *VECTORS_COST_MINUS_127_8x8;
extern const int *VECTORS_COST_MINUS_127_8x16;
extern const int *VECTORS_COST_MINUS_127_16x16;
extern const int MODE_COST4567[3][52];

/////////////////////////////////////
// Tables for transform

extern short QuantTable[6][16];
extern short DQuantTable[6][16];
extern const int FValIntra[];
extern const int FValInter[];


/////////////////////////////////////
// Tables for QP-related calulations

extern const unsigned char QP_TO_CHROMA_MAPPING[NUM_QP];
extern const unsigned char div_6[];
extern const unsigned char mod_6[];


/////////////////////////////////////
// The following two tables are for the loop filter kernel code.

// This table consists of four tables of length NUM_QP, all indexed by
// indexA in the standard
//   Table 0: a (alpha)
//   Table 1: tC0 for bS=1
//   Table 2: tC0 for bS=2
//   Table 3: tC0 for bS=3

extern const unsigned char IndexATable[NUM_QP][4];
// Table for B (beta)
extern const unsigned char IndexBTable[NUM_QP];
/////////////////////////////////////
// The following three tables are for the loop filter reference code.
extern const unsigned char ALPHA_TABLE[NUM_QP];
extern const unsigned char BETA_TABLE[NUM_QP];
extern const unsigned char CLIP_TAB[NUM_QP][5];


/////////////////////////////////////
// For zig-zag scanning
extern const unsigned int ZigZagScan[16];
extern const unsigned int BlockScan[16];


/////////////////////////////////////
// CAVLC Tables
#define COEFF_TOKEN_TABLE_SIZE 3*4*17*2
extern const unsigned char CoeffTokenTable[3][4][17][2];

#define COEFF_TOKEN_CHROMA_DC_TABLE_SIZE 4*5*2
extern const unsigned char CoeffTokenChromaDCTable[4][5][2];

// This table only has 43 valid entries, but the size has been rounded
// up to fit inside a 32-bit granularity.
#define RUN_TABLE_SIZE 44*2
extern const unsigned char RunTable[44][2];		
extern const unsigned int RunIndexTable[7];		

#define TOTAL_ZEROS_TABLE_SIZE 15*16*2
extern const unsigned char TotalZerosTable[15][16][2];
#define TOTAL_ZEROS_CHROMA_DC_TABLE_SIZE 3*4*2
extern const unsigned char TotalZerosChromaDCTable[3][4][2]; 

#define CBP_TABLE_SIZE 48*2
extern const unsigned char CBPTable[48][2];

#endif

