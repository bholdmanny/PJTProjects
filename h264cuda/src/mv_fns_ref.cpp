/* ##################################################################### */ 
/*                                                                       */ 
/* Notice:   COPYRIGHT (C) STREAM PROCESSORS, INC. 2007-2008                  */ 
/*           THIS PROGRAM IS PROVIDED UNDER THE TERMS OF THE SPI         */ 
/*           END-USER LICENSE AGREEMENT (EULA). THE PROGRAM MAY ONLY     */ 
/*           BE USED IN A MANNER EXPLICITLY SPECIFIED IN THE EULA,       */ 
/*           WHICH INCLUDES LIMITATIONS ON COPYING, MODIFYING,           */ 
/*           REDISTRIBUTION AND WARANTIES. UNAUTHORIZED USE OF THIS      */ 
/*           PROGRAM IS SCTRICTLY PROHIBITED. YOU MAY OBTAIN A COPY OF   */ 
/*           THE EULA FROM WWW.STREAMPROCESSORS.COM.                     */ 
/*                                                                       */ 
/* ##################################################################### */ 


//--------------------------------------------------------------------
//  $File: //depot/main/software/apps/spve264b_1.6.0/src/mv_fns_ref.c $
//  $Revision: #1 $
//  $DateTime: 2009/05/18 04:09:53 $
//
//  Description:
//    Common functions for processing motion vectors
//--------------------------------------------------------------------

#include "../inc/mb_info.h"
#include "../inc/const_defines.h"


//====================================================================
int Median3Ref
//====================================================================
// This function returns the median value of its three signed 32-bit
// integer arguments.
//--------------------------------------------------------------------
(
 int X, int Y, int Z
 )
//--------------------------------------------------------------------
{
    int MinXY, MaxXY, MedianTmp, MedianXYZ;
    MinXY = (X < Y) ? X : Y;
    MaxXY = (X > Y) ? X : Y;
    MedianTmp = (MaxXY < Z) ? MaxXY : Z;
    MedianXYZ = (MedianTmp > MinXY) ? MedianTmp : MinXY;
    return(MedianXYZ);
}



//====================================================================
void CalcPredictedMVRef
// ===================================================================
// This function returns the predicted motion vector for this
// partition, and also returns a boolean that indicates whether this
// macroblock can be skipped.  If a block is skipped, then the
// predicted motion vector returns is guaranteed to be the same as
// this partition's motion vector.
// -------------------------------------------------------------------
(
 ////////////////////////////////////////
 // Inputs
 S_BLK_MB_INFO *MB_INFO_In,
 S_BLK_MB_INFO *MB_INFO_LeftIn,
 S_BLK_MB_INFO *MB_INFO_TopLeftIn,
 S_BLK_MB_INFO *MB_INFO_TopIn,
 S_BLK_MB_INFO *MB_INFO_TopRightIn,

 int CalculatePerms,
 int PermGetA,
 int PermGetB,
 int PermGetC,
 int PermGetD,

 // 4x4 block index (i.e., cluster number)
 int i,

 //////////////////////////////////////// 
 // Outputs
 S_MV* PredMV,
 int* SkipBlock
 )
//--------------------------------------------------------------------
{
    S_MV ZeroMV;
    int MyXPos, MyYPos, PartWidth, PartHeight, PartLeftXPos, PartTopYPos, PartRightXPos, XPosLeftNeighbor, XPosRightNeighbor, YPosTopNeighbor;
    int XPosLeftNeighborCol, XPosRightNeighborCol, PartTopYPosRow, YPosTopNeighborRow;
    int CisNotDecodedYet;
    int PermGetTopLeftPart, PermGetTopRightPart, TopLeftPartLoc, TopRightPartLoc, Avalid, Bvalid, Cvalid, Dvalid;
    int RefA_diff, RefB_diff, RefC_diff, OnlyRefA_same, OnlyRefB_same, OnlyRefC_same, OnlyOneRefSame;
    int SkipSpecialCondition;
    S_BLK_MB_INFO *Blk, *BlkA, *BlkB, *BlkC, *BlkD;
    S_MV SkipPredMV;

    // Declare the neighboring motion vectors and reference frame
    // indices, as well as predicted motion vector
    S_MV MVA, MVB, MVC;
    int RefA, RefB, RefC;

    ZeroMV.x = 0;
    ZeroMV.y = 0;

    // Figure out the size of the partition we are in.  The size is
    // given in terms of 4 pixel increments.  So a value of 1
    // indicates a width/height of 4.
    MyXPos = i & 0x3;
    MyYPos = i >> 2;
    PartWidth = (((MB_INFO_In[i].SubType == SINGLE_BLOCK_SUBDIV_TYPE)
                      || (MB_INFO_In[i].SubType == HOR_SUBDIV_TYPE))
                     ? 4 : 2);
    PartHeight = (((MB_INFO_In[i].SubType == SINGLE_BLOCK_SUBDIV_TYPE)
                       || (MB_INFO_In[i].SubType == VERT_SUBDIV_TYPE))
                      ? 4 : 2);
    if (MB_INFO_In[i].Type == INTER_SMALL_BLOCKS_MB_TYPE) {
        PartWidth >>= 1;
        PartHeight >>= 1;
    }

    // Now get the coordinates of the top-left and top-right pixel
    // within the partition that we are a part of.
    PartLeftXPos = (MyXPos / PartWidth) * PartWidth;
    PartTopYPos = (MyYPos / PartHeight) * PartHeight;
    PartRightXPos = PartLeftXPos + PartWidth - 1;

    // Get neighboring block coordinates
    XPosLeftNeighbor = PartLeftXPos - 1;
    XPosRightNeighbor = PartRightXPos + 1;
    YPosTopNeighbor = PartTopYPos - 1;

    // Convert the coordinates into positive macroblock indices (i.e.,
    // -1 would really be index 3 into MB_INFO_LeftIn).
    XPosLeftNeighborCol = XPosLeftNeighbor & 3;
    XPosRightNeighborCol = XPosRightNeighbor & 3;
    PartTopYPosRow = PartTopYPos << 2;
    YPosTopNeighborRow = (YPosTopNeighbor & 3) << 2;

    // The Perms are really macroblock indices.
    if (CalculatePerms) {
        PermGetA = XPosLeftNeighborCol + PartTopYPosRow;
        PermGetB = PartLeftXPos + YPosTopNeighborRow;
        PermGetC = XPosRightNeighborCol + YPosTopNeighborRow;
        PermGetD = XPosLeftNeighborCol + YPosTopNeighborRow;
    }

    // Detect the case when the C neighbor refers to a macroblock
    // partition that hasn't been decoded yet
    CisNotDecodedYet = (((XPosRightNeighborCol == 0)
                              && (YPosTopNeighborRow < 0xc))
                             || ((XPosRightNeighborCol == 2)
                                 && ((YPosTopNeighborRow & 4) == 0)));

    // In the maps below, double lines indicate Macroblock
    // boundaries, and single lines indicate 4x4 block boundaries.
    //
    //     Cluster Map for Neighbor A           Cluster Map for Neighbor B        
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  |    ||    |    |    |    ||    |    |    || 12 | 13 | 14 | 15 ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  =================================    =================================
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  |  3 ||  0 |  1 |  2 |    ||    |    |    ||  0 |  1 |  2 |  3 ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  |  7 ||  4 |  5 |  6 |    ||    |    |    ||  4 |  5 |  6 |  7 ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  | 11 ||  8 |  9 | 10 |    ||    |    |    ||  8 |  9 | 10 | 11 ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||         |    ||    |    |    |    ||     
    //  | 15 || 12 | 13 | 14 |    ||         |    ||    |    |    |    ||     
    //  |    ||    |    |    |    ||         |    ||    |    |    |    ||     
    //  ============================         ============================     
    //
    //     Cluster Map for Neighbor C           Cluster Map for Neighbor D        
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  |    ||    | 13 | 14 | 15 || 12 |    | 15 || 12 | 13 | 14 |    ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  =================================    =================================
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  |    ||    |  1 |  2 |  3 ||  0 |    |  3 ||  0 |  1 |  2 |    ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  |    ||    |  5 |  6 |  7 ||  4 |    |  7 ||  4 |  5 |  6 |    ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  |    ||    |  9 | 10 | 11 ||  8 |    | 11 ||  8 |  9 | 10 |    ||    |
    //  |    ||    |    |    |    ||    |    |    ||    |    |    |    ||    |
    //  ---------------------------------    ---------------------------------
    //  |    ||    |    |    |    ||         |    ||    |    |    |    ||     
    //  |    ||    |    |    |    ||         |    ||    |    |    |    ||     
    //  |    ||    |    |    |    ||         |    ||    |    |    |    ||     
    //  ============================         ============================     

    // Convert the Perms into pointers to actual data
    Blk = &MB_INFO_In[i];
    BlkA = ((PermGetA & 3) == 3) ? &MB_INFO_LeftIn[PermGetA] : &MB_INFO_In[PermGetA];
    BlkB = ((PermGetB >> 2) == 3) ? &MB_INFO_TopIn[PermGetB] : &MB_INFO_In[PermGetB];
    BlkC = (((PermGetC >> 2) == 3)
            ? (((PermGetC & 3) == 0)
               ? &MB_INFO_TopRightIn[PermGetC] : &MB_INFO_TopIn[PermGetC])
            : &MB_INFO_In[PermGetC]);
    BlkD = ((((PermGetD >> 2) == 3) && ((PermGetD & 3) == 3))
            ? &MB_INFO_TopLeftIn[PermGetD]
            : (((PermGetD >> 2) == 3)
               ? &MB_INFO_TopIn[PermGetD]
               : (((PermGetD & 3) == 3)
                  ? &MB_INFO_LeftIn[PermGetD]
                  : &MB_INFO_In[PermGetD])));

    // Now we have all the neighboring context, The next step is to
    // figure out which of the neighboring MVs we just got are
    // actually valid (or in other words, which are invalid because
    // the macroblock partitions they came from are invalid).

    // These indices are for the top-left and top-right 4x4 block
    // within the SAME partition as this cluster's 4x4 block
    PermGetTopLeftPart = PartLeftXPos + (PartTopYPos << 2);
    PermGetTopRightPart = PartRightXPos + (PartTopYPos << 2);
    // Get location info for this partition
    TopLeftPartLoc = MB_INFO_In[PermGetTopLeftPart].Loc;
    TopRightPartLoc = MB_INFO_In[PermGetTopRightPart].Loc;

    Avalid = ((TopLeftPartLoc & LOC_BLK_LEFT_EDGE) == 0);
    Bvalid = ((TopLeftPartLoc & LOC_BLK_TOP_EDGE) == 0);
    Cvalid = !(((TopRightPartLoc & LOC_BLK_TOP_PICTURE_EDGE) != 0)
                    || ((TopRightPartLoc & LOC_BLK_RIGHT_PICTURE_EDGE) != 0)
                    || (Blk->SliceNum != BlkC->SliceNum));
    Dvalid = !(((TopLeftPartLoc & LOC_BLK_LEFT_OR_TOP_PICTURE_EDGE) != 0)
                    || (Blk->SliceNum != BlkD->SliceNum));

    Cvalid = Cvalid && !CisNotDecodedYet;

    // According to Point 3 in Section 8.4.1.3.2, if C is not
    // available, we should use D instead.
    BlkC = Cvalid ? BlkC : BlkD;
    Cvalid = Cvalid || Dvalid;

    // If a neighbor is not available, then set both components of the
    // MV to zero, and Ref to -1 for that neighbor.
    MVA = Avalid ? BlkA->MV : ZeroMV;
    RefA = Avalid ? BlkA->RefFrameIdx : -1;
    MVB = Bvalid ? BlkB->MV : ZeroMV;
    RefB = Bvalid ? BlkB->RefFrameIdx :  -1;
    MVC = Cvalid ? BlkC->MV : ZeroMV;
    RefC = Cvalid ? BlkC->RefFrameIdx : -1;

    // Precalculate which neighbors have the same reference frame
    // index as the current partition.
    RefA_diff = (RefA != Blk->RefFrameIdx);
    RefB_diff = (RefB != Blk->RefFrameIdx);
    RefC_diff = (RefC != Blk->RefFrameIdx);
    OnlyRefA_same = !RefA_diff && RefB_diff && RefC_diff;
    OnlyRefB_same = RefA_diff && !RefB_diff && RefC_diff;
    OnlyRefC_same = RefA_diff && RefB_diff && !RefC_diff;
    OnlyOneRefSame = (OnlyRefA_same || OnlyRefB_same || OnlyRefC_same);
            
    // Now we can actually go about deriving the predicted motion
    // vector.  We have to check a series of possible special cases
    // before we evaluate the default median prediction scheme.
    if ((Blk->Type == INTER_LARGE_BLOCKS_MB_TYPE)
        && (Blk->SubType == HOR_SUBDIV_TYPE)
        && (PartTopYPos == 0) && (Blk->RefFrameIdx == RefB)) {
        // Handle directional segementation prediction case 1
        *PredMV = MVB;
    } else if ((Blk->Type == INTER_LARGE_BLOCKS_MB_TYPE)
               && (Blk->SubType == HOR_SUBDIV_TYPE)
               && (PartTopYPos == 2) && (Blk->RefFrameIdx == RefA)) {
        // Handle directional segementation prediction case 2
        *PredMV = MVA;
    } else if ((Blk->Type == INTER_LARGE_BLOCKS_MB_TYPE)
               && (Blk->SubType == VERT_SUBDIV_TYPE)
               && (PartLeftXPos == 0) && (Blk->RefFrameIdx == RefA)) {
        // Handle directional segementation prediction case 3
        *PredMV = MVA;
    } else if ((Blk->Type == INTER_LARGE_BLOCKS_MB_TYPE)
               && (Blk->SubType == VERT_SUBDIV_TYPE)
               && (PartLeftXPos == 2) && (Blk->RefFrameIdx == RefC)) {
        // Handle directional segementation prediction case 4
        *PredMV = MVC;
    } else if (Avalid && !Bvalid && !Cvalid) {
        // Handle the case when B & C are not available, but A is.
        *PredMV = MVA;
    } else if (OnlyOneRefSame) {
        // Handle the case when the reference frame index for only one
        // of the neighbors is equal to the reference frame of the
        // current partition.
        *PredMV = OnlyRefA_same ? MVA : (OnlyRefB_same ? MVB : MVC);
    } else {
        // Now handle the default case, which is the median of the
        // three neighbors.
        (*PredMV).x = Median3Ref(MVA.x, MVB.x, MVC.x);
        (*PredMV).y = Median3Ref(MVA.y, MVB.y, MVC.y);
    }

    // If SkipSpecialCondition holds, then the MV needs to be zero to
    // skip the block, otherwise, the MV must equal PredMV to skip.
    SkipSpecialCondition = (!Avalid || !Bvalid
                            || ((RefA == 0) && (MVA.x == 0) && (MVA.y == 0))
                            || ((RefB == 0) && (MVB.x == 0) && (MVB.y == 0)));
    SkipPredMV = SkipSpecialCondition ? ZeroMV : (*PredMV);
    
    *SkipBlock = 0;

    // Check to see if this block can be skipped.  To be skipped, it
    // has to be a single 16x16 partition whose reference frame index
    // is 0 and with no coded coefficients.
    if ((MB_INFO_In[i].Type == INTER_LARGE_BLOCKS_MB_TYPE)
        && (MB_INFO_In[i].SubType == SINGLE_BLOCK_SUBDIV_TYPE)
        && (MB_INFO_In[i].RefFrameIdx == 0)
        && (MB_INFO_In[i].CBP == 0))
    {
        if ((MB_INFO_In[i].MV.x == SkipPredMV.x)
            && (MB_INFO_In[i].MV.y == SkipPredMV.y))
        {
            // We can skip the block!
            *SkipBlock = 1;
        }
    }
}
