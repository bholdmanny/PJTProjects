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


#ifndef _RC_COMMON_H_
#define _RC_COMMON_H_

#include "rc.h"
#include"cuve264b.h"

#define RC_LARGE_NUMBER  0x7fffffff
#define MAX_NUM_MBS_HD	 8160
#define NUM_BLKS_IN_MB	 16

#define TRUE  1
#define FALSE 0

#define MB_WIDTH		16
#define MB_HEIGHT		16

// Max value of quantization parameter
#define MIN_QP           0
#define MAX_QP          51
#define NUM_QP          52

/// difference in QP for Iframes: usually -2
#define IFRAMEDIFF_CBR      2
#define IFRAMEDIFF_CVBR     0

#define HIGH_MOTION_SCALE	2.00
#define NORMAL_SCALE		1.00
#define LOWER_BITRATE_SCALE 0.75


#define VBV_BUF_RATIO       4
#define VBV_BUF_RATIO_MIN   2
///	WATERMARK = percentage fullness we strive for
#define HRB_WATERMARK       0.5

// Default rate control parameters
#define DEFAULT_IFRAME_QP 30
#define DEFAULT_PFRAME_QP 30
#define DEFAULT_MIN_QP  14
#define DEFAULT_MAX_QP  51

#define DEFAULT_BITRATE     1000000
#define DEFAULT_FRAMERATE   30
#define DEFAULT_BITRATE_TYPE	CUVRC_BITRATE_BPF
#define DEFAULT_IFRAME_RATE 300
#define DEFAULT_IFRAME_PERIOD_TYPE	CUVRC_PERIOD_FRAMES
/// nominal ratio of Iframe to Pframe size, for calculations.  Perhaps this should be feedback-adjusted?
#define DEFAULT_IFRAMERATIO 8.0
#define DEFAULT_P_2_NON_REF_P_RATIO	1.25
#define DEFAULT_ABR_SCALE	2
#define DEFAULT_AVG_SAD_THRESHOLD	1000
#define DEFAULT_WAIT_STATE_COUNT	6

// changing is to have min qp as 3
// this will lean to minimum initial qp as 8
// this is becs mpeg4 cannot handle huge number of bits generated for a particular frame
static const int quant_range[32][2] = { 
	{5, 25}, {5, 25}, {5, 18}, {5, 18}, // 0, .., 192
	{4, 20}, {4, 20}, {4, 20}, {3, 16}, // 256, .., 448
	{3, 14}, {3, 14}, {3, 14}, {3, 14}, // 512, ...
	{3, 12}, {3, 12}, {3, 12}, {3, 12}, // 768, ...
	{3, 10}, {3, 10}, {3, 10}, {3, 10}, // 1024, ...
	{3, 8}, {3, 8}, {3, 8}, {3, 8}, // 1280, ...
	{3, 8}, {3, 8}, {3, 8}, {3, 8}, // 1536, ...
	{3, 8}, {3, 8}, {3, 8}, {3, 8} // 1792, ...
};

extern const int qp_offsets[41];
extern const float qscale[52];

#define RC_FR_COMPUTER_BUF_SIZE     32 // Must be 2^x
#define RC_BR_COMPUTER_BUF_SIZE     512 // Must be power of 2. 512 is roughly 17 seconds on 30 fps

#define MIN_SIZE_NUM_BITS_BUFFER    15
#define MAX_SIZE_NUM_BITS_BUFFER    75

#define DEFAULT_EXTREMELY_GOOD_QUALITY_MAX_QP		26
#define DEFAULT_GOOD_QUALITY_MAX_QP					32
#define DEFAULT_POOR_QUALITY_MIN_QP					40
#define DEFAULT_EXTREMELY_POOR_QUALITY_MIN_QP		46


#undef min
#define min(x,y) ((x<y)?(x):(y))
#undef max
#define max(x,y) ((x>y)?(x):(y))

#define clip_3(x, max, min) ((x)>max ? max : ((x)<min ? min : (x)))

typedef enum RC_MODE {
    RC_MODE_VBR,
    RC_MODE_CBR,
    RC_MODE_CVBR
} RC_MODE;

typedef enum MOTION_STATE
{
	NORMAL_MOTION,
	NORMAL_TO_HIGH_MOTION_WAIT,
	HIGH_MOTION,
	HIGH_TO_NORMAL_MOTION_WAIT,
	LOWER_BITRATE_AFTER_HIGH_MOTION,
	LOWER_BITRATE_TO_HIGH_MOTION_WAIT
}MOTION_STATE;

typedef enum GOP_STATE
{
	GOP_WITH_NO_SCENE_CHANGE_ERROR,
	GOP_WITH_SCENE_CHANGE,
	FIRST_GOP_AFTER_SCENE_CHANGE,
	SECOND_GOP_AFTER_SCENE_CHANGE		
}GOP_STATE;

typedef enum FSM_STATUS
{
	CONTINUE,
	EXIT
}FSM_STATUS;

typedef enum MOTION_TYPE
{
	NORMAL,
	HIGH
}MOTION_TYPE;

typedef enum MOTION_TRANSITION_PHASE
{
	NORMAL_MOTION_TO_HIGH_MOTION,
	HIGH_MOTION_TO_LOW_BITRATE,
	LOW_BITRATE_TO_HIGH_MOTION,
	LOW_BITRATE_TO_NORMAL_MOTION	
}MOTION_TRANSITION_PHASE;

typedef enum FRAME_STATE
{
	I_FRAME,
	PREVIOUS_GOP_B_FRAME,
	P_FRAME,
	B_FRAME,
	PSEUDO_I_FRAME
}FRAME_STATE;

static const int qp_steps[4] = {-6, 7, -6, -3};
 
typedef struct RC_FR_COMPUTER       // This is a circular buffer to keep a running average of frame rate
{
    unsigned int buf_size;
    unsigned int buf_mask;
    unsigned int cur_pos;
    unsigned int frame_time[RC_FR_COMPUTER_BUF_SIZE];        // microseconds
    unsigned int num_frames;    // number of frames over which average is taken
    unsigned int time_low;
    unsigned int time_high;
    unsigned int total_time;
    unsigned long long prev_time;		// time in microseconds
    unsigned long long cur_time;			// time in microseconds
    unsigned int new_timer;
    unsigned long long iframe_time;			// last i frame timestamp
    int iframe_time_adjustment;
    unsigned long long frame_timestamp[RC_FR_COMPUTER_BUF_SIZE];	// timestamps of frames in display order
    int read_pos;	// read position in "frame_timestamp" array
    int write_pos;	// write position in "frame_timestamp" array
} RC_FR_COMPUTER;

typedef struct RC_IFRAME_STAT
{
    int qp;
    int target_bits;
    int actual_bits;
    int HRB_fullness_after_iframe;
} RC_IFRAME_STAT;

typedef struct RC_BITRATE_STAT
{
    unsigned int buf_size;
    unsigned int gop_size;
    unsigned int cur_pos;
    int 		 bits_error_per_frame[MAX_SIZE_NUM_BITS_BUFFER];
    int 		 total_error;
    int          previous_accumulated_error;    
}RC_BITRATE_STAT;

typedef struct FSM_PARAMS
{
	MOTION_STATE current_state;
	MOTION_STATE next_state;
	FSM_STATUS	 state_status;
	int			 wait_state_counter;
	int			 reset_HRB_fullness;
}FSM_PARAMS;

// this fsm is used to distribute the scene change error 
// appropriately in the gops that follow the scene change
typedef struct DISTRIBUTE_ERROR_FSM_PARAMS
{
	GOP_STATE current_state;
	GOP_STATE next_state;
	FSM_STATUS state_status;
}DISTRIBUTE_ERROR_FSM_PARAMS;

// this structure is used to store the original values set through the APIs
// this is required since actual values keep changing for motion adaptive bitrate mode
typedef struct ORIGINAL_PARAMS
{
	int target_bps;
	int iframe_bits;
	int pframe_bits;
	int non_ref_frame_bits;
	int max_qp;
	int min_qp;
}ORIGINAL_PARAMS;

typedef struct ROI_CONTEXT
{
	int rois_present_flag;
	int extremely_good_quality_rois_count;
	int good_quality_rois_count;
	int poor_quality_rois_count;
	int extremely_poor_quality_rois_count;
	CUVRC_ROI *extremely_good_quality_rois;
	CUVRC_ROI *good_quality_rois;
	CUVRC_ROI *poor_quality_rois;
	CUVRC_ROI *extremely_poor_quality_rois;
	int extremely_good_quality_max_qp;
	int good_quality_max_qp;
	int poor_quality_min_qp;
	int extremely_poor_quality_min_qp;
}ROI_CONTEXT;

typedef struct ABR_CONTEXT
{
	int scale;
	float lower_br_scale;
	float higher_br_scale;
}ABR_CONTEXT;

// this fsm is used to decide the frame type of current & next frame
typedef struct FRAME_TYPE_FSM_PARAMS
{
	FRAME_STATE current_state;
	FRAME_STATE next_state;
	FSM_STATUS state_status;
	int is_prev_ref_frame;
}FRAME_TYPE_FSM_PARAMS;

typedef struct INPUT_CONFIGURATION
{
	CUVRC_APP_TYPE prime_format;
	int im_width;
	int im_height;
	int target_bitrate;
	CUVRC_BITRATE_TYPE bitrate_type;
	CUVRC_BITRATE_MODE bitrate_mode;
	CUVRC_QUALITY_MODE quality_mode;
	int iframe_period;
	CUVRC_IFRAME_PERIOD_TYPE iframe_period_type;
	int non_ref_pframe_interval;
	int bframe_interval;
	int target_fps;
}INPUT_CONFIGURATION;

typedef struct PREVIOUS_CONTEXT
{
	int iframe_bits;
	int pframe_bits;
	int non_ref_frame_bits;
	int scaled_non_ref_frame_bits;
	int last_qp;
	int last_actual_bits;
}PREVIOUS_CONTEXT;


typedef struct RC_REF_FRAME_STAT
{
    int qp;
    int target_bits;
    int actual_bits; 
    CUVRC_FRAME_TYPE frame_type;
    int iframe_flag;
} RC_REF_FRAME_STAT;

typedef struct RC_STATE {
	int prime_format;			// codec type
    int need_reset;
    int need_full_reset;
    float ifrate;
    unsigned int ifrate_uint;
    int iframe_period;
    CUVRC_IFRAME_PERIOD_TYPE iframe_period_type;
    int gop_num;
    int num_mbs;
    int iframe_bits;
    int pframe_bits;   
    int non_ref_frame_bits;
    int sum_p_qp_in_gop;
    unsigned int avg_p_qp;
    int scale_avg_p_qp;	// if there was a change in motion state then the avg p qp
    					// that is calculated from previous gop needs to be scaled
    unsigned int current_gop_size;
    unsigned int curr_gop_frames_after_last_iframe; // this will not be same as "current_gop_size"
    									   // when a scene change occurs & we do not reset the gop
    int prev_gop_bit_error;
    float iframeratio;
    float if_persec;
    int large_mv_thresh;
    int imb_thresh;
    int target_bitrate;
    CUVRC_BITRATE_TYPE bitrate_type;
    CUVRC_BITRATE_MODE bitrate_mode;
    int target_bps;  // bits per second
    int target_bpf;  // bits per frame
    float target_fps;
    float actual_fps;
    int frame_qp;
    int actualQP;
    int first_p_qp_changed;		// this flag indicates that qp of the first P frame
    							// after I frame was incremented by 1 to produce 
    							// sufficient number of bits
    int *p_mb_qp;   
    CUVME_MB_CHARAC *p_mb_characs;
    int *p_qp_list;
    int im_width;
    int im_height;
    int width_in_mbs;
    int height_in_mbs;
    int HRB_size;
    int HRB_fullness;
    int HRB_max;
    int HRB_min;
    signed long long high_motion_error;		// this accumulates the error introduced due to 
    							// change on target bitrate for high motion areas
    int original_qp;
    int iframe_flag;
    int force_iframe_flag;
    CUVRC_FRAME_TYPE curr_frame_type;
    CUVRC_FRAME_TYPE next_frame_type;
    int start_of_gop_flag; // this will tell if this frame marks the start of a gop
    					   // this will differentiate between the i frames which are 
    					   // at the start of a GOP & the ones which are in the middle of it
    int force_qp_flag;
    int max_qp;
    int min_qp;
    CUVRC_QUALITY_MODE quality_mode;    
    int target_bits;
    int avg_sad;
    //int header_bits;
    //int texture_bits;
    int num_bits;
    int total_p_num_bits;
    int num_imbs;   //number of intra MBs
    int num_enc_mbs; // number of encoded mbs
    int num_big_mvs; // number of mbs that has mv exceeds threshold
    int num_skipped_mbs; // number of mbs that were skipped
    int last_iframe_flag;
    int last_qp;
    int last_sad;
    int last_frame_type;
    int last_p_qp;
    int is_initial_qp;
    int i_p_frame_diff;		// differnce between  qp values of I & P frame
    int scene_change;
    int scene_change_error;	// this accumulates error introduced due to 
    						// coding the frame with scene change as an I frame
    CUVRC_SCENE_CHANGE_BEHAVIOR gop_reset_flag;
    RC_IFRAME_STAT iframe_stat;
    RC_FR_COMPUTER fr_computer;
    RC_BITRATE_STAT bitrate_stat;
    RC_MODE mode;
    int frame_num;
    FSM_PARAMS fsm;
    ORIGINAL_PARAMS original_values;
    MOTION_TYPE current_frame_motion_type;
    ROI_CONTEXT roi_info;
    int ratefactor; //for mpeg4
    float last_ratio; //for mpeg4
    int bCif;	//for mpeg4    
    //S_PERF_TIMERS timers;
    int avg_var;
    int valid_avg_var;
    ABR_CONTEXT abr_params;
    DISTRIBUTE_ERROR_FSM_PARAMS scene_change_fsm;
    unsigned int non_ref_pframe_interval;
    float ref_p_frames_per_sec;
    float non_ref_frames_per_sec;
    int non_ref_p_count;
    int scene_change_postponed;
    int current_gop_scene_change_error;
    int bframe_interval;
    int bframe_count;
    int num_prev_gop_bframes;
    float prev_gop_brames_time;
    int prev_gop_bframes_count;
    int next_frame_offset;
    FRAME_TYPE_FSM_PARAMS frame_type_fsm;
    INPUT_CONFIGURATION input_config;
    int reset_postponed;
    PREVIOUS_CONTEXT prev;
    int is_bframe_from_prev_gop;
    unsigned int frames_after_last_iframe;
    int non_ref_frame_interval;
    int use_prev_bits;
    int non_ref_frame_flag;
    int last_non_ref_frame_flag;
    int is_backward_prediction;
    float p_2_non_ref_p_ratio;
    int total_ref_p_num_bits;
    int total_non_ref_num_bits;
    RC_REF_FRAME_STAT last_ref_frame_stat;
    int unmodified_HRB_fullness;
    int acc_err_from_last_ref_frame;
    int min_frame_qp;
    int max_frame_qp;
    int force_iframe_postponed;
} RC_STATE;

#endif 
