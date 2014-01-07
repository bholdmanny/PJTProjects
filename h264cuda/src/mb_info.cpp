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
#include "../inc/mb_info.h"
#include "../inc/const_defines.h"

//--------------------------------------------------------------------
//
// FunctionName:   InitMBInfo()
//
// Description:
//     Initialize MBInfo for from VSofts data structure.  If
//     If InitForCAVLC is true, this function should be called after
//     residual coding and inter/intra mode decision, and before
//     CAVLC.  This function handles a single macroblock.
// Input:
// Output:
// Return Value:
//--------------------------------------------------------------------

void InitMBInfo(
                S_BLK_MB_INFO *pBlkMBInfo,
                int CurrMBx, int CurrMBy,
                int NumMBHor, int NumMBVer,
				int Slice_num
                )
{
    int k;
    int Loc, MBTopLoc, MBLeftLoc, MBBottomLoc, MBRightLoc;
    int BlkTopLoc, BlkLeftLoc, BlkBottomLoc, BlkRightLoc;

    MBTopLoc = MBLeftLoc = MBBottomLoc = MBRightLoc = 0;
    BlkTopLoc = BlkLeftLoc = BlkBottomLoc = BlkRightLoc = 0;
    if (CurrMBy == 0) {
        MBTopLoc  |= LOC_MB_TOP_PICTURE_EDGE | LOC_MB_TOP_SLICE_EDGE;
        BlkTopLoc |= LOC_BLK_TOP_PICTURE_EDGE | LOC_BLK_TOP_SLICE_EDGE;
    }
    if ((CurrMBx == 0)) {
        MBLeftLoc  |= LOC_MB_LEFT_PICTURE_EDGE | LOC_MB_LEFT_SLICE_EDGE;
        BlkLeftLoc |= LOC_BLK_LEFT_PICTURE_EDGE | LOC_BLK_LEFT_SLICE_EDGE;
    }
    if ((CurrMBy == NumMBVer - 1)) {
        MBBottomLoc  |= LOC_MB_BOTTOM_PICTURE_EDGE | LOC_MB_BOTTOM_SLICE_EDGE;
        BlkBottomLoc |= LOC_BLK_BOTTOM_PICTURE_EDGE | LOC_BLK_BOTTOM_SLICE_EDGE;
    }
    if ((CurrMBx == NumMBHor - 1)) {
        MBRightLoc  |= LOC_MB_RIGHT_PICTURE_EDGE | LOC_MB_RIGHT_SLICE_EDGE;
        BlkRightLoc |= LOC_BLK_RIGHT_PICTURE_EDGE | LOC_BLK_RIGHT_SLICE_EDGE;
    }

    // Loop over 4x4 blocks in a MB in raster order
    for (k=0; k<BLOCKS_PER_MB; k++) {
        // Check top
        Loc = MBTopLoc | (((k & 0xc) == 0) ? BlkTopLoc : 0);
        // Check bottom
        Loc |= MBBottomLoc | (((k & 0xc) == 0xc) ? BlkBottomLoc : 0);
        // Check left
        Loc |= MBLeftLoc | (((k & 0x3) == 0) ? BlkLeftLoc : 0);
        // Check right
        Loc |= MBRightLoc | (((k & 0x3) == 0x3) ? BlkRightLoc : 0);

        pBlkMBInfo->Loc       = Loc;

        //pBlkMBInfo->SliceNum  = 0;
		pBlkMBInfo->SliceNum  = Slice_num;

        pBlkMBInfo++;
    }
}


void InitMBInfoCompressed(
			  S_BLK_MB_INFO_COMPRESSED *pBlkMBInfo,
			  int CurrMBx, int CurrMBy,
			  int NumMBHor, int NumMBVer
			  )
{
    int k;
    int Loc, MBTopLoc, MBLeftLoc, MBBottomLoc, MBRightLoc;
    int BlkTopLoc, BlkLeftLoc, BlkBottomLoc, BlkRightLoc;

    MBTopLoc = MBLeftLoc = MBBottomLoc = MBRightLoc = 0;
    BlkTopLoc = BlkLeftLoc = BlkBottomLoc = BlkRightLoc = 0;
    if (CurrMBy == 0) {
        MBTopLoc  |= LOC_MB_TOP_PICTURE_EDGE | LOC_MB_TOP_SLICE_EDGE;
        BlkTopLoc |= LOC_BLK_TOP_PICTURE_EDGE | LOC_BLK_TOP_SLICE_EDGE;
    }
    if ((CurrMBx == 0)) {
        MBLeftLoc  |= LOC_MB_LEFT_PICTURE_EDGE | LOC_MB_LEFT_SLICE_EDGE;
        BlkLeftLoc |= LOC_BLK_LEFT_PICTURE_EDGE | LOC_BLK_LEFT_SLICE_EDGE;
    }
    if ((CurrMBy == NumMBVer - 1)) {
        MBBottomLoc  |= LOC_MB_BOTTOM_PICTURE_EDGE | LOC_MB_BOTTOM_SLICE_EDGE;
        BlkBottomLoc |= LOC_BLK_BOTTOM_PICTURE_EDGE | LOC_BLK_BOTTOM_SLICE_EDGE;
    }
    if ((CurrMBx == NumMBHor - 1)) {
        MBRightLoc  |= LOC_MB_RIGHT_PICTURE_EDGE | LOC_MB_RIGHT_SLICE_EDGE;
        BlkRightLoc |= LOC_BLK_RIGHT_PICTURE_EDGE | LOC_BLK_RIGHT_SLICE_EDGE;
    }

    // Loop over 4x4 blocks in a MB in raster order
    for (k=0; k<BLOCKS_PER_MB; k++) {
        // Check top
        Loc = MBTopLoc | (((k & 0xc) == 0) ? BlkTopLoc : 0);
        // Check bottom
        Loc |= MBBottomLoc | (((k & 0xc) == 0xc) ? BlkBottomLoc : 0);
        // Check left
        Loc |= MBLeftLoc | (((k & 0x3) == 0) ? BlkLeftLoc : 0);
        // Check right
        Loc |= MBRightLoc | (((k & 0x3) == 0x3) ? BlkRightLoc : 0);

        pBlkMBInfo->Loc       = Loc;

        pBlkMBInfo->MBModes  = 0;
        pBlkMBInfo->Misc  = 0;
        pBlkMBInfo->NumCoefsAndSAD  = 0;
        //pBlkMBInfo->MV  = 0;

        pBlkMBInfo++;
    }
}

