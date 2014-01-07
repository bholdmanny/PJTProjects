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

//better to have a common header for ME
#define MBINFO_OFFSET_MBMODES   0
#define MBINFO_OFFSET_NUMCOEF   1
#define MBINFO_OFFSET_MV        2

#define NUM_SEARCH_LOCATIONS  10

#define ALIGN_MEM_32BYTES(x)  ((x+31)&~31)

#define MAXMBNUMX 120 
#define MAXMBNUMX_STRIP 112   // strip size for strip based processing for me 20 and  32
#define TARGET_INTEGER_STRIP_SIZE (MAXMBNUMX)
#define MAXSS  ROUNDUP_STREAM_PROCESSORS(TARGET_INTEGER_STRIP_SIZE)
#define MVSS   (ROUNDUP_STREAM_PROCESSORS(MAXSS) + 16) // MVMAP STRIP SIZE + padded for overwrite tools bug

#define  ROUNDUP_STREAM_PROCESSORS(x) (-( (-(int)(x)) & -((int)16) ) )
#define NUM_SRCHPTS_XDIR  9
#define NUM_SRCHPTS_YDIR  7
#define NUM_REFLINES_LOADED (16 + (NUM_SRCHPTS_YDIR - 1))

