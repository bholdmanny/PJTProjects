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

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "../inc/mb_info.h"
#include "../inc/encoder_tables.h"
#include "../inc/residual_coding.h"

void InitQPDataAndTablesFromQP (
    S_QP_DATA *pQpData,
    short **pQuantTable,
    short **pDQuantTable,
    int QpValue,            // QP value for luma
    int IsIDRFrame,
    int IsLuma              // Flag
    )
{
    int MappedQpValue;
    int QpDivBySix, QpModSix;

    MappedQpValue = IsLuma == 0 ? QP_TO_CHROMA_MAPPING[QpValue] : QpValue;

    QpDivBySix = div_6[MappedQpValue];
    QpModSix   = mod_6[MappedQpValue];

    *pQuantTable  = (short *)&QuantTable[QpModSix][0];
    *pDQuantTable = (short *)&DQuantTable[QpModSix][0];

    if(!IsIDRFrame)
    {
        pQpData->QpVal       = QpValue;
        pQpData->QuantAdd    = FValInter[QpDivBySix];
        pQpData->QuantAddDC  = FValIntra[QpDivBySix];
        pQpData->QuantShift  = 15 + QpDivBySix;
        pQpData->DQuantShift = QpDivBySix;
        pQpData->PredPenalty = 4 * QP2QUANT_NEW[QpValue];
    }
    else
    {
        pQpData->QpVal       = QpValue;
        pQpData->QuantAdd    = FValIntra[QpDivBySix];
        pQpData->QuantAddDC  = FValIntra[QpDivBySix];
        pQpData->QuantShift  = 15 + QpDivBySix;
        pQpData->DQuantShift = QpDivBySix;
        pQpData->PredPenalty = 4 * QP2QUANT_NEW[QpValue];
    }
}