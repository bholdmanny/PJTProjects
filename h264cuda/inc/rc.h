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

#ifndef __CUVRC_H__
#define __CUVRC_H__

#include "me.h"

typedef void *CUVRC_HANDLE;

// CUVRC_ERROR enum specifies different errors given by rate control 
typedef enum CUVRC_ERROR {
    CUVRC_ERR_SUCCESS = 0, 		// no error
    CUVRC_ERR_FATAL_SYSTEM, 	// DSP MIPS or DPU state is corrupted; reload DSP MIPS
    CUVRC_ERR_FATAL_TASK, 		// task state is corrupted; stop task and restart
    CUVRC_ERR_FATAL_ENCODER, 	// RC state is corrupted; close encoder and reopen
    CUVRC_ERR_FATAL_INTERNAL, 	// RC internal state is corrupted; close RC and reopen
    CUVRC_ERR_MEM, 				// insufficient heap memory; allocate more memory to task
    CUVRC_ERR_ARG, 				// incorrect argument; check args (esp. for NULL pointers)
    CUVRC_ERR_FAILURE = -1 		// generic failure
} CUVRC_ERROR;

// CUVRC_APP_TYPE enum specifies the application types (types of encoders) accepted by rate control
typedef enum CUVRC_APP_TYPE {
    CUVRC_NONE = 0,  // No standard codec used : default H.264
    CUVRC_H264
} CUVRC_APP_TYPE; 


// CUVRC_FRAME_TYPE enum specifies the different types of frames
typedef enum CUVRC_FRAME_TYPE {
      CUVRC_I_FRAME = 0,  
      CUVRC_P_FRAME,
      CUVRC_NON_REFERENCE_P_FRAME_FORWARD,      // this means that it is a forward non reference p frame                                                                            
     CUVRC_NON_REFERENCE_P_FRAME_BACKWARD,
     CUVRC_PSEUDO_I_FRAME,           			// this indicates the encoder to code it as a P frame with all I mbs
     CUVRC_NON_REFERENCE_B_FRAME_FORWARD,
     CUVRC_NON_REFERENCE_B_FRAME_BACKWARD,
     CUVRC_NON_REFERENCE_B_FRAME_BIDIRECTIONAL,
     CUVRC_REFERENCE_B_FRAME_FORWARD,
     CUVRC_REFERENCE_B_FRAME_BACKWARD,
     CUVRC_REFERENCE_B_FRAME_BIDIRECTIONAL,
     CUVRC_DROP_FRAME,                   		// the current frame should be dropped
     CUVRC_UNKNOWN_FRAME_TYPE    
} CUVRC_FRAME_TYPE;


// CUVRC_BITRATE_TYPE enum identifies the units in which bitrate can be specified
typedef enum CUVRC_BITRATE_TYPE {
	CUVRC_BITRATE_BPS = 0,	// bits per second   
	CUVRC_BITRATE_BPF        // bits per frame
} CUVRC_BITRATE_TYPE; 

// CUVRC_IFRAME_PERIOD_TYPE identifies the units in which I frame period 
// i.e. gop length can be specified
typedef enum CUVRC_IFRAME_PERIOD_TYPE {
    CUVRC_PERIOD_MILLISECONDS = 0,	// gop length in milliseconds   
    CUVRC_PERIOD_FRAMES		        // gop length in terms of number of frames
} CUVRC_IFRAME_PERIOD_TYPE; 


// CUVRC_BITRATE_MODE identifies the different bitrate modes which can be used 
// for encoding a sequence. This decides whether the bitrate for the sequence 
// will be sensitive to the type of motion in the sequence
typedef enum CUVRC_BITRATE_MODE {
    CUVRC_NORMAL_BITRATE = 0,		// Default mode. The bitrate doesn’t vary 
    								// depending on sequence characteristics   	
    CUVRC_MOTION_ADAPTIVE_BITRATE,	// In this mode the bitrate is higher for 
    								// high motion areas. This high bitrate is compensated for, 
    								// by having a lower bitrate in non-high motion areas. 
    								// Over a longer period of time, 
    								// the bitrate is maintained to the specified input value.	        
    CUVRC_PREVIOUS_BITRATE			// This instructs the encoder to maintain the bitrate mode 
    								// specified in the previous call to 
    								// cuvrc_configure_bitrate() API
} CUVRC_BITRATE_MODE; 

// CUVRC_FRAME_FORMAT identifies the chroma format in the input sequence
typedef enum CUVRC_FRAME_FORMAT { 
	CUVRC_FORMAT_YUV420 = 0, 
	CUVRC_FORMAT_YUV422, 
	CUVRC_FORMAT_YUV444
} CUVRC_FRAME_FORMAT;

// CUVRC_QUALITY_MODE identifies different quality modes which 
// can be used for coding a sequence
typedef enum CUVRC_QUALITY_MODE {
	CUVRC_CONSTANT_QUALITY = 0,		// The quality specified at the input is 
									// maintained throughout the sequence. 
									// This is like a VBR mode.
	CUVRC_NORMAL_QUALITY,			// In this mode qality across the frames is varying 
									// depending on the bitrate but is constant 
									// inside a frame.
	CUVRC_TEXTURE_ADAPTIVE_QUALITY,	// Quality varies across the frames depending on 
									// bitrate & inside a frame depending on the 
									// texture(frequency) of the region.
	CUVRC_MOTION_ADAPTIVE_QUALITY,	// Not implemented
	CUVRC_PREVIOUS_QUALITY			// This instructs the encoder the maintain 
									// the quality mode specified in the previous 
									// call to cuvrc_configure_quality() API.
}CUVRC_QUALITY_MODE;

// CUVRC_ROI_QUALITY_LEVEL identifies the different quality levels 
// which s be used for coding the region of interest
typedef enum CUVRC_ROI_QUALITY_LEVEL {
	CUVRC_EXTREMELY_GOOD_QUALITY,
	CUVRC_GOOD_QUALITY,
	CUVRC_POOR_QUALITY,
	CUVRC_EXTREMELY_POOR_QUALITY
}CUVRC_ROI_QUALITY_LEVEL;

// CUVRC_SCENE_CHANGE_BEHAVIOR identifies whether the GOP will be reset 
// after detecting a scene change or not
typedef enum CUVRC_SCENE_CHANGE_BEHAVIOR
{
	CUVRC_SCENE_CHANGE_RESET_GOP = 0,
	CUVRC_SCENE_CHANGE_DO_NOT_RESET_GOP
}CUVRC_SCENE_CHANGE_BEHAVIOR;


// CUVRC_ROI specifies all the parameters required for setting 
// a region of interest 
typedef struct CUVRC_ROI
{
	int x_offset;	// start of ROI : x offset in terms of number of macroblocks
	int y_offset;	// start of ROI : y offset in terms of number of macroblocks
	int width;		// width of ROI in terms of number of macroblocks
	int height;		// height of ROI in terms of number of macroblocks
	CUVRC_ROI_QUALITY_LEVEL quality_level;	// quality level to be used for the ROI
}CUVRC_ROI;


// cuvrc_open opens an instance of a reference RC. 
// The reference RC is a C version of the RC which runs entirely on DSP MIPS. 
CUVRC_ERROR cuvrc_open(CUVRC_HANDLE *handle, CUVRC_APP_TYPE app_type);

// cuvrc_init allocates memory required internally by the library 
CUVRC_ERROR cuvrc_init(CUVRC_HANDLE handle);

// cuvrc_free frees all the memory allocated by the RC library. 
CUVRC_ERROR cuvrc_free(CUVRC_HANDLE handle);

// cuvrc_close closes the RC with the given handle.
CUVRC_ERROR cuvrc_close(CUVRC_HANDLE handle);



// cuvrc_configure_dimensions() configures the width & height of the input
// if the dimensions have changed from the previous dimensions then the memory that depends on the dimensions
// is freed up & reallocated
CUVRC_ERROR cuvrc_configure_dimensions(CUVRC_HANDLE rc_handle, int width, int height, CUVRC_FRAME_FORMAT fmt);

// cuvrc_configure_bitrate() configures all the parameters related to bitrate i.e. 
// bitrate value, bitrate type & bitreate mode. 
// Bitrate type is the unit in which bitrate is specified. 
// Bitrate mode indicates whether the bitrate should be sensitive to the motion in the sequence.
CUVRC_ERROR cuvrc_configure_bitrate(CUVRC_HANDLE rc_handle, int target_bitrate, CUVRC_BITRATE_TYPE bitrate_type, CUVRC_BITRATE_MODE mode);

// cuvrc_configure_iframe_period() sets the I frame period (gop length) 
// & also specifies the units which should be used for I frame period. 
CUVRC_ERROR cuvrc_configure_iframe_period(CUVRC_HANDLE rc_handle, int iframe_period, CUVRC_IFRAME_PERIOD_TYPE iframe_period_type);

CUVRC_ERROR cuvrc_set_previous_frame_actual_bits(CUVRC_HANDLE rc_handle, int num_bits);

// cuvrc_set_previous_frame_intra_mbs() is used by the encoder to set the number of macroblocks 
// that were coded as intra in the last encoded frame
CUVRC_ERROR cuvrc_set_previous_frame_intra_mbs(CUVRC_HANDLE rc_handle, int num_intra_mbs);

// cuvrc_set_previous_frame_skipped_mbs() is used by the encoder to set the number of macroblocks 
// that were skipped in the last encoded frame
CUVRC_ERROR cuvrc_set_previous_frame_skipped_mbs(CUVRC_HANDLE rc_handle, int num_skipped_mbs);

// cuvrc_set_previous_frame_num_large_mvs() is used by the encoder to set the number of motion vectors
// which have an amplitude higher than a certain threshhold in the last encoded frame
// the threshhold for the large mvs is decided by the encoder
CUVRC_ERROR cuvrc_set_previous_frame_num_large_mvs(CUVRC_HANDLE rc_handle, int num_large_mvs);

// cuvrc_set_previous_frame_avg_sad() is used by the encoder to set total error in the last encoded frame
// in terms of average sad per macroblock 
CUVRC_ERROR cuvrc_set_previous_frame_avg_sad(CUVRC_HANDLE rc_handle, int avg_sad);


CUVRC_ERROR cuvrc_set_mb_level_pointers(CUVRC_HANDLE rc_handle, int *p_mb_qp, CUVME_MB_CHARAC *p_mb_characs, int *qp_list);

// cuvrc_set_roi() sets the region of interest for current frame 
// by specifiying the required data 
CUVRC_ERROR cuvrc_set_roi(CUVRC_HANDLE rc_handle, CUVRC_ROI roi);

// cuvrc_set_avg_var() sets the average variance per macroblock of the current picture
CUVRC_ERROR cuvrc_set_avg_var(CUVRC_HANDLE rc_handle, int avg_var);



// cuvrc_get_current_frame_type() gives the frame type of the current frame in pointer "*type"
CUVRC_ERROR cuvrc_get_current_frame_type(CUVRC_HANDLE rc_handle, CUVRC_FRAME_TYPE *type);

// cuvrc_get_next_frame_type() gives the frame type of the next frame in pointer "*type"
// this might not be the same as current frame type for next frame when a scene cut is detected 
CUVRC_ERROR cuvrc_get_next_frame_type(CUVRC_HANDLE rc_handle, CUVRC_FRAME_TYPE *type);

// cuvrc_get_avg_frame_qp() gives one average qp value of the frame in pointer "*avg_qp"
CUVRC_ERROR cuvrc_get_avg_frame_qp(CUVRC_HANDLE rc_handle, int *avg_qp);


//rate control main processing function
CUVRC_ERROR cuvrc_process_frame(CUVRC_HANDLE handle, int scene_cut);

CUVRC_ERROR cuvrc_set_frame_rate(CUVRC_HANDLE rc_handle, int frame_rate);
#endif 
