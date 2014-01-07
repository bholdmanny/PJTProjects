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


#ifndef _MB_INFO_H
#define _MB_INFO_H


typedef struct S_MV {
    int x;
    int y;
} S_MV;

typedef struct S_BLK_MB_INFO {
    int SliceNum;
    int Type;
    int SubType;
	int Pred_mode;
    int IntraChromaMode;
    int QP;
    int CBP;
    int TotalCoeffLuma;
    int TotalCoeffChroma;
    int MinSAD;
    int Loc;
    struct S_MV MV;
    //pos_int8x4_t OffsetMV;
    int RefFrameIdx;
} S_BLK_MB_INFO;


// Macros used by kernels to access mb_info as individual words
// instead of full record.
#define MBINFO_OFFSET_MBMODES   0
#define MBINFO_OFFSET_NUMCOEF   1
#define MBINFO_OFFSET_MISC      2
#define MBINFO_OFFSET_LOC       3
#define MBINFO_SIZE_IN_WORDS    4

// !!!!! CAUTION: Please modify macros above before changing this structure!!!!!
typedef struct S_BLK_MB_INFO_COMPRESSED {
    
#define MBINFO_MODES_TYPE             0
#define MBINFO_MODES_SUBTYPE          1
#define MBINFO_MODES_INTRACHROMAMODE  2
#define MBINFO_MODES_REF_IDX          3
    
    unsigned int MBModes;

#define MBINFO_TOTAL_COEFF_LUMA_MASK    0xff000000      // 8 bits
#define MBINFO_TOTAL_COEFF_LUMA_SHIFT   (24)              
#define MBINFO_TOTAL_COEFF_CHROMA_MASK  0x00fc0000      // 6 bits
#define MBINFO_TOTAL_COEFF_CHROMA_SHIFT (18)
#define MB_INFO_MIN_SAD_MASK            0x0003ffff      // 18 bits

    unsigned int NumCoefsAndSAD;

#define MBINFO_MV_X                   0
#define MBINFO_MV_Y                   1

    unsigned int MV;

#define MBINFO_MISC_CBP               0
#define MBINFO_MISC_QP                1
#define MBINFO_MISC_SLICENUM          2

    unsigned int Misc;

    unsigned int Loc;

} S_BLK_MB_INFO_COMPRESSED;

//---------------------------------------------------------------------
//
//  Public Prototypes
//
//---------------------------------------------------------------------
void InitMBInfo(
                S_BLK_MB_INFO *pBlkMBInfo,
                int CurrMBx, int CurrMBy,
                int NumMBHor, int NumMBVer,
				int Slice_num
                );
void InitMBInfoCompressed(
			  S_BLK_MB_INFO_COMPRESSED *pBlkMBInfo,
			  int CurrMBx, int CurrMBy,
			  int NumMBHor, int NumMBVer
			  );


#endif  
