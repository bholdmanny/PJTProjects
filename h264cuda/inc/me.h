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



#ifndef __CUVME_H__
#define __CUVME_H__

#define MAX_FORW_REF_FRAMES 1
#define MAX_BACK_REF_FRAMES 1
#define MAX_MV_REPLICATE 16
typedef void *CUVME_HANDLE;
typedef enum CUVME_ERROR {
    CUVME_ERR_SUCCESS = 0, // no error
    CUVME_ERR_FATAL_SYSTEM, // DSP MIPS or DPU state is corrupted; reload DSP MIPS
    CUVME_ERR_FATAL_TASK, // task state is corrupted; stop task and restart
    CUVME_ERR_FATAL_ENCODER, // ME state is corrupted; close encoder and reopen
    CUVME_ERR_FATAL_INTERNAL, // ME internal state is corrupted; close ME and reopen
    CUVME_ERR_MEM, // insufficient heap memory; allocate more memory to task
    CUVME_ERR_ARG, // incorrect argument; check args (esp. for NULL pointers)
    CUVME_ERR_FAILURE = -1 // generic failure
} CUVME_ERROR;

typedef enum CUVME_APP_TYPE {
    NONE = 0,  // No standard codec used
    H264
} CUVME_APP_TYPE;

typedef enum CUVME_SRC_FORMAT_TYPE {
    RASTER_ORDER = 0,
    BLOCK_FLAT // In this each 4x4 in a MB is in contigous memory. 4x4s in a MB
               // are stored in a encode/decode order
}CUVME_SRC_FORMAT_TYPE;

typedef enum CUVME_ME_REF_DIR {
    CUVME_FORWARD_REF = 0, // indicates forward reference frame used in ME
	CUVME_BACKWARD_REF,     //  indicates backward reference frame used in ME
	CUVME_BI_DIRECTIONAL_REF  //  indicates both forward and backward reference frames used in ME - currently not supported.
} CUVME_ME_REF_DIR; 

// CUVME types
typedef enum CUVME_TYPE {
    CUVME_CudaC = 0,    
    CUVME_CREF            
} CUVME_TYPE;

typedef struct CUVME_MV_RESULTS {
    short MV_X; // horizontal motion vector
    short MV_Y; // vertical motion vector
} CUVME_MV_RESULTS;

typedef struct CUVME_MB_INFO {
    unsigned char reserved;
    unsigned char MBType;
    unsigned short SAD;
} CUVME_MB_INFO;

typedef struct CUVME_Y_BUFFER {
    unsigned char * y; // pointer to the buffer
    int buffer_width; // width of allocated buffer, includes padding
    int buffer_height; // height of allocated buffer, includes padding
    int active_width; // width of active piture
    int active_height; // height of active picture
    int offset_x; // horizontal offset of active picture from left
    int offset_y; // vertical offset of active picture from top
} CUVME_Y_BUFFER;

typedef struct CUVME_MB_CHARAC {
	unsigned char mean;
	unsigned char variance;
	unsigned short reserved;
} CUVME_MB_CHARAC;

typedef struct CUVME_FRAME_CHARAC {
	int mean;  // average of means of all MBS in the frame;  mean  of a MB = average of all pixels in the MB.
	int variance; // average of variance of all MBs in the frame.
} CUVME_FRAME_CHARAC;

// As of now maximum forward and backward reference frames supported are 1.
typedef struct CUVME_REFERENCE_USED_INFO {
	int 				fwd_ref_frame_num;  // forward reference frame number used as reference 
	int 				bwd_ref_frame_num; // backward reference frame number used as reference
	CUVME_ME_REF_DIR 	direction_ref;  // direction of reference : 0 - forward reference used; 1 - backward reference used; 2 - both forward and backward reference used.
						// As of now "direction_ref" can take only 0 or 1.
} CUVME_REFERENCE_USED_INFO;

// cuvme_close closes the ME with the given handle.
CUVME_ERROR cuvme_close (CUVME_HANDLE handle);



//cuvme_free frees all the memory allocated by the ME library.
CUVME_ERROR cuvme_free (CUVME_HANDLE handle);



// cuvme_get_avg_var gets the average variance for the MBs of a frame.
CUVME_ERROR cuvme_get_avg_var(CUVME_HANDLE handle,int *avg_var);
// cuvme_init allocates memory required internally by the library
// (to store indices etc). It also allocates memory for the decimated frames
// in case user doesn’t always pass decimated reference pictures.
CUVME_ERROR cuvme_init(CUVME_HANDLE handle, int width, int height, int dec_ref_avail);


// cuvme_open opens an instance of a reference ME. The reference ME is a C version
// of the ME which runs entirely on DSP MIPS.
CUVME_ERROR cuvme_open (CUVME_HANDLE *handle);

// cuvme_set_app_type sets the codec type for the ME. It is mainly used to determine
// the kind of motion compensation needed. Default is NONE. 
// cuvme_set_app_type must precede the cuvme_init API call.
CUVME_ERROR cuvme_set_app_type (CUVME_HANDLE handle, CUVME_APP_TYPE app_type);


// cuvme_set_me_mode sets the mode of the ME. Each mode corresponds to set of different
// tools. Default me_mode is 20. Cuurent modes supported are 0, 20, 22, 30 and 32.
CUVME_ERROR cuvme_set_me_mode (CUVME_HANDLE handle, int me_mode);

// cuvme_set_num_mvs set the number of times motion vectors need to be replicated.
// The result when this number is divided by the number of partitions (depends on me
// mode) should be 1, 4, or 16. max is 16, default is 1
CUVME_ERROR cuvme_set_num_mvs (CUVME_HANDLE handle, int num_replicate);

// cuvme_set_predicted_picture passes the predicted picture pointer where ME library
CUVME_ERROR cuvme_set_predicted_picture(CUVME_HANDLE handle,
                                        CUVME_Y_BUFFER *p_pred_picture);

// cuvme_set_qp sets the QP to be used for MVCost calculation. Default qp value is 32
CUVME_ERROR cuvme_set_qp(CUVME_HANDLE me_handle, int qp);

// cuvme_set_reference_frame passes the references frames to be used by CUVME_search.
// This API should be called at once before every call to CUVME_search.
// ref_frame_num indicates the frame number in multiple reference frames. direction
// indicates whether the frame is forward or a backward reference. 0 indicates
// forward and 1 indicates backward (used only in B frames). The p_ref
// buffer should be padded 16 bytes on all the sides.
CUVME_ERROR cuvme_set_reference_frame (CUVME_HANDLE handle, CUVME_Y_BUFFER *p_ref,
                                      char ref_frame_num, char direction);
								   
// cuvme_set_return_mb_characteristics passes pointer to return the MB level
// characteristics like mean and median which can used for in encoder
// and for video analytics . The size of the buffer p_mb_characs should be: sizeof(CUVME_MB_CHARAC) * (num_mbs_in_picture + STREAM_PROCESSORS), 
//where num_mbs_in_picture is the number of MBs in the picture.
CUVME_ERROR cuvme_set_return_mb_characteristics(CUVME_HANDLE handle,
    CUVME_MB_CHARAC *p_mb_characs);

// cuvme_set_search_range sets the search range of the ME. search_range_x is
// horizontal search range on both left and right and search_range_y is vertical
// search range on both top and bottom.
CUVME_ERROR cuvme_set_search_range (CUVME_HANDLE handle, int search_range_x,
                                    int search_range_y);

#endif 

