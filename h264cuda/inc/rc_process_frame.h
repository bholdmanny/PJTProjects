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


#ifndef _RC_PROCESS_FRAME_H_
#define _RC_PROCESS_FRAME_H_

void accumulated_error_to_qp(RC_STATE* rcs, int base_qp, int new_target_qp, int actual_bits);

void set_targets(RC_STATE* rcs, float scale);
void scale_qp(RC_STATE* rcs, MOTION_TRANSITION_PHASE transition_phase);
void modify_iframeratio(RC_STATE *rcs);
signed long long update_high_motion_error(RC_STATE *rcs);

void rate_control_frame_level (RC_STATE* rcs);

#endif 
