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



#ifndef _CUVE264B_H
#define _CUVE264B_H


#define CUVE264B_QP_NO_CHANGE  ( -1 )

//--------------------------------------------------------------------
//  Enums & data structures
//--------------------------------------------------------------------

typedef void *CUVE264B_HANDLE;
typedef enum CUVE264B_ERROR {

    /* No worries, mate */
    CUVE264B_ERR_SUCCESS = 0,

    /* DSP_MIPS or DPU state is fatally corrupted */
    /* corrective action: reload entire DSP_MIPS  */
    CUVE264B_ERR_FATAL_SYSTEM,

    /* task state is fatally corrupted */
    /* corrective action: stop task and restart */
    CUVE264B_ERR_FATAL_TASK,

    /* encoder state is fatally corrupted */
    /* corrective action: close encoder and reopen */
    CUVE264B_ERR_FATAL_ENCODER,

    /* encoder internal state is fatally corrupted */
    /* corrective action: closing encoder and reopening may solve the problem;  */
    /*                    issue can be further diagnosed by analyzing encoder   */
    /*                    source code or contacting the provider of the encoder */
    CUVE264B_ERR_FATAL_INTERNAL,

    /* insufficient memory in the heap */
    /* corrective action: allocate more memory to the task and retry */
    CUVE264B_ERR_MEM,

    /* error in input arguments */
    /* corrective action: check arguments and retry (commonly due to NULL pointers) */
    CUVE264B_ERR_ARG,

    /* this does not include overflows for heap allocation by malloc()          */
    /* corrective action: check arguments and retry (commonly due to NULL pointers) */
    CUVE264B_ERR_OVERFLOW,

    /* generic failure reported by boolean test functions */
    /* corrective action: depends on situation */
    CUVE264B_ERR_FAILURE = -1

} CUVE264B_ERROR;

typedef enum CUVE264B_E_FRAME_TYPE {
    CUVE264B_FTYPE_I = 0,
    CUVE264B_FTYPE_P,
    CUVE264B_NON_REFERENCE_P_FRAME,
    CUVE264B_NOT_SUPPORTED_FRAME,
    CUVE264B_PSEUDO_I_FRAME,		// this indicates the encoder to code it as a P frame with all I mbs
    CUVE264B_DROP_FRAME,				// the current frame should be dropped
    CUVE264B_FTYPE_UNKNOWN
} CUVE264B_E_FRAME_TYPE;


typedef enum CUVE264B_E_RC_MODE {
    CUVE264B_RC_VBR,
    CUVE264B_RC_CBR,
    CUVE264B_RC_CVBR
} CUVE264B_E_RC_MODE;


typedef enum CUVE264B_E_IMG_FORMAT
{  
    CUVE264B_IMG_FMT_YUV_RASTER = 0,
    CUVE264B_IMG_FMT_YUV_GPUBF 
} CUVE264B_E_IMG_FORMAT;

typedef struct CUVE264B_YUV_BUFFER {
    unsigned char *p_y_buffer;	
    unsigned char *p_u_buffer;
    unsigned char *p_v_buffer;
    int buf_width;          // Width of y buffer. u and v buffers shalled be halfed
    int buf_height;         // Height of y buffer. u and v buffers shalled be halfed
} CUVE264B_YUV_BUFFER;

typedef struct CUVE264B_IMAGE_SPECS {
    CUVE264B_YUV_BUFFER buffer;
    int x_offset;	                // X offset into luma source image of region to compress. Chroma is halfed.
    int y_offset;		            // Y offset into luma source image of region to compress. Chroma is halfed.
    int width;	                    // Encoding width of luma component. Chroma is halfed.
    int height;                     // Encoding height of luma component. Chroma is halfed.
} CUVE264B_IMAGE_SPECS;

typedef struct CUVE264B_BITSTREAM_BUFFER {
    int total_num_bytes;
    int used_num_bytes;
    unsigned char *p_buffer;
} CUVE264B_BITSTREAM_BUFFER;

typedef struct CUVE264B_FRAME_ENCODE_INFO {  // general info about the last encoded frame
    int frame_number;
    CUVE264B_E_FRAME_TYPE frame_type;	// iframe, pframe
    int num_slices;
    int num_intra_mbs;
    int num_encoded_mbs;
    int num_bits;
    int header_bits;
    int texture_bits;
    int num_me_cands;
    int qp_used;
    int scene_detected;
} CUVE264B_FRAME_ENCODE_INFO;

//----------------------------------------------------------------------
// The following functions must be called exactly once per encoder
//----------------------------------------------------------------------

CUVE264B_ERROR cuve264b_open(CUVE264B_HANDLE *p_handle,int slice_num);
CUVE264B_ERROR cuve264b_close(CUVE264B_HANDLE handle);


//-----------------------------------------------------------------------
// Following functions must be called once per encoded frame
//-----------------------------------------------------------------------

CUVE264B_ERROR cuve264b_prep_encode_frame
(
 CUVE264B_HANDLE handle,
 CUVE264B_IMAGE_SPECS *p_input_image            // image parameters
 );

CUVE264B_ERROR cuve264b_encode_frame
(
 CUVE264B_HANDLE handle,                        // (pointer to) context of encoder instance
 CUVE264B_BITSTREAM_BUFFER *p_output_buffer     // Input and output
 );

// Set an unique name to the encoder instance
CUVE264B_ERROR cuve264b_set_name (CUVE264B_HANDLE handle, char *name);

// Set input frame format. Default is SPLIB_IMG_FORMAT_YUV420. Encoding is optimized for SPLIB_IMG_FORMAT_YUV_GPUBF
CUVE264B_ERROR cuve264b_set_input_image_format (CUVE264B_HANDLE handle, CUVE264B_E_IMG_FORMAT image_format);


// Function to get the bitstream buffer pointer and size
CUVE264B_ERROR cuve264b_get_output_buffer (CUVE264B_HANDLE handle, CUVE264B_BITSTREAM_BUFFER *p_output_buffer,int slice_num);

// Default interval is set to 0 -- no periodic iframes
CUVE264B_ERROR cuve264b_set_iframe_interval (CUVE264B_HANDLE handle, int interval);

// Set bits per second
CUVE264B_ERROR cuve264b_set_target_bitrate (CUVE264B_HANDLE handle, int bitrate);

// Set desired frame rate per second, default is 30
CUVE264B_ERROR cuve264b_set_target_framerate (CUVE264B_HANDLE handle, int framerate);

// Intra prediction level 0: full, 1:skip all 4x4, Default is 0
CUVE264B_ERROR cuve264b_set_intra_prediction_level (CUVE264B_HANDLE handle, int level);

// set level of motion estimation range 20,22,30,32.  Default is 20.
CUVE264B_ERROR cuve264b_set_motion_estimation_level (CUVE264B_HANDLE handle, int level);

// enable set to non-zero (true) enables loopfilter, false disables loopfilter.
// 0: Disable Loop filter; 1: Enable loop filter 2: Disable loop filter for non-ref frames. Loopfilter is enabled by default
// By default, loopfilter is enabled. 
CUVE264B_ERROR cuve264b_enable_deblocking_loopfilter (CUVE264B_HANDLE handle, int enable);

// Set parameter to adjust strenth of deblocking loopfilter.
// When the offsets are poistive values, loopfilter is stronger. Vice versa.
// Only applies when loopfilter is enabled
CUVE264B_ERROR cuve264b_set_loopfilter_parameters (CUVE264B_HANDLE handle, int alpha_c0_offset, int beta_offset);

// Set rate control option.
CUVE264B_ERROR cuve264b_set_rate_control (CUVE264B_HANDLE handle, CUVE264B_E_RC_MODE rc_mode);


// Set qp values. If not called, default qp values will be used.
// min_qp and max_qp are used when RC_VBR is defined for rate-control
// iframe_qp and pframe_qp are used when RC_CBR is defined for rate-control
// If the any specified value is QP_NO_CHANGE, then leave the value as before (after checking to ensure it is valid)
CUVE264B_ERROR cuve264b_set_qp_values (CUVE264B_HANDLE handle, int min_qp, int max_qp, int iframe_qp, int pframe_qp);

CUVE264B_ERROR cuve264b_get_psnr (CUVE264B_HANDLE handle, double *p_psnr_y, double *p_psnr_u, double *p_psnr_v);

CUVE264B_ERROR cuve264b_get_frame_encode_info (CUVE264B_HANDLE handle, CUVE264B_FRAME_ENCODE_INFO *p_info);

// Reset performance timers
CUVE264B_ERROR cuve264b_reset_timing (CUVE264B_HANDLE handle);

// level 0 is quietest, level 10 is the noisest. 
CUVE264B_ERROR cuve264b_print_timing (CUVE264B_HANDLE handle, int level);

#endif
