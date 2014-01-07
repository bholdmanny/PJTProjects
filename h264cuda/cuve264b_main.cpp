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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "inc/cuve264b_utils.h"
#include "inc/cuve264b.h"

#define __int64 long long

char   g_input_filename[100];
char   *g_output_filename;
char   *g_rec_image_filename;


int main(int argc, char *argv[])

{

	 FILE *input_file = NULL;
    FILE *output_file = NULL;
    int frame_num, num_bytes_per_frame, read_bytes;
    int image_size;
    int bf_image_size;
    int EncodedBytes = 0;
    unsigned char *p_input_data;
    unsigned char *p_input_data_bf;
	clock_t start,end,start1,end1;
    char type[4]; //flag of code type
    int max_memory_usage = 0;
	int read_file_time = 0;
	int pre_process_time = 0;

    // user stuff
    SESSION_PARAMS session;

    // io buffers
    CUVE264B_IMAGE_SPECS input_image;
    CUVE264B_IMAGE_SPECS input_image_rast;
    CUVE264B_BITSTREAM_BUFFER output_buffer[2];
    CUVE264B_IMAGE_SPECS rec_image[2];
    CUVE264B_FRAME_ENCODE_INFO stats;

    double psnr_y, psnr_u, psnr_v;
    double total_psnr_y, total_psnr_u, total_psnr_v;
    int total_bits;
    int total_me_cands;
    int qp_sum, min_qp, max_qp;
    int i;

    int IFrate=2000000000;
	total_psnr_y = 0;
    total_psnr_u = 0;
    total_psnr_v = 0;
    total_bits = 0;
    total_me_cands = 0;
    qp_sum = 0;
    min_qp = 52;
    max_qp = 0;
    stats.scene_detected = 0;
    
    CUVE264B_HANDLE p_encoder = NULL;
    CUVE264B_ERROR err = CUVE264B_ERR_SUCCESS;

    int byte_idx=0;
    int found_error = 0;
    int performed_compares = 0;
    int entropy_mode = 0;

    for (i= 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "-i") ==  0))
        {
           //input file
           if(++i < argc)
	     {
	         strcpy(g_input_filename,argv[i]);
	     }
          else
	     {
		 printf("\n Error: Wrong set of arguments : %s %s.\n", argv[i-1], argv[i]);
                 return -1;
	     }
        }
    }
    g_output_filename = "H264.out";
    g_rec_image_filename = "test_rec.yuv";
   
   // open input file

    input_file = fopen(g_input_filename, "rb");
    if(input_file == NULL){
        printf("Can't find input file %s\n", g_input_filename);
        return -1;
    }
    //open output file
    if (g_output_filename[0] != 0) 
    {
        output_file = fopen(g_output_filename, "wb");
        if(output_file == NULL){
            printf("Can't open output file %s\n", g_output_filename);
            return -1;
        }
    }

    session.ref = 1;
    session.intra4x4_mode = 0;        //"0: full, 1: skip all 4x4. Default is 1.",                                                    
    session.TestMode= 0;              //"Test mode. 0: don't enter test mode; 1: don't generate deterministic data (fastest performance); 2: generate linear input."
    session.PrintStats = 1;           //"Statistics collection and display level: 0 (none) to 2 (maximum) Default is 0.",                     
    session.PrintTiming = 1;          //"Time information collection and display level: 0 (none) to 10 (maximum). Default is 0.",              
    session.DisableLoopFilter=0;      //"Disable loop filter. Loopfilter is enabled by default",                                              
    session.FrameHeight = 1080;       // "Height of the input image. Default is set to be the height of encoding.",       
    session.FrameWidth  =1920;        //"Width of the input image. Default is set to be the width of encoding.",        
    session.Height  =1080 ;           //"Height of encoding.",                  
    session.Width  =1920;             //"Width of encoding.",        
    session.Xoffset =0;               //"X offset of input image. Default is 0",                                                
    session.Yoffset =0;               // "Y offset of input image. Default is 0",
    session.QP  = 30;                 //"p-frame qp (ignored if bitrate != 0).  Default is 20",                                       
    session.QPI =30;                  //"i-frame qp (ignored if bitrate != 0).  Default is the same as p-frame qp",    
    session.IFrate =20;                //"Number of frames between each I frame",                                                      
    session.Fps = 30;                 //"Frame rate. Default is 30 frames per second.",                                     
    session.frame_start=0;            //"Index of frame to start coding. Default is0",   
    //session.frame_end =3;          //"Index of frame to end encoding.  Default is 1",                                                      
    session.trace_start= 0;           //"Index of frame on which to start recording a trace for profiling. Default is 0",                    
    session.trace_end =1;           //"Index of frame on which to end recording a trace for profiling.  Default is 1",                    
    session.rc_mode =0;               //"Rate control option. Rate-control mode. 0 - VBR, 1 - CBR. Default is 0",                      
    session.min_QP = 1;
    session.max_QP = 51;
    session.target_bitrate =6000000;  //"Target bitrate (overrides -qp_i and -qp)",                                                           
    session.me = 20;                  //"Motion estimation level. Available levels are: \"20\", and \"30\". Default is 20.",;
    session.AlphaC0Offset=0;          //"Loop filter parameter: alpha_c0_offset. Range from 0 to 51. Default is 0.",                          
    session.BetaOffset =0;            //"Loop filter parameter: beta_offset. Range from 0 to 51. Default is 0.",                           
    session.debug_level=0;            //"Debug message level. Default is 0.",                                                                
    session.input_format=0;           //"0: raster, 1: block flattened. Default is 0.",                                                      
    session.mem_leak_test=0;          //"0: No test of memory leak, 1: Test memory leak at open and close, 2: More detailed memory leak. Default is 0.",  
	session.slice_num = 34;
    
    int i_frame_total = 0;
	if( !fseek( input_file, 0, SEEK_END ) )
    {
        unsigned __int64 i_size = ftell( input_file );
		fseek( input_file, 0, SEEK_SET );
        i_frame_total = (int)(i_size / ( session.FrameHeight * session.FrameWidth * 3 / 2 ));
    }                                                                            
    session.frame_end =i_frame_total;         //"Index of frame to end encoding.  Default is 1", 
	//session.frame_end = 3;
    //if (session.IFrate == 0) 
    //{
       // IFrate = session.frame_end;
    //}
    //else
    //{
        IFrate = session.IFrate;
    //}

    if (session.FrameWidth==0)
    {
        session.FrameWidth=session.Width;
    }
    if (session.FrameHeight==0)
    {
        session.FrameHeight=session.Height;
    }
    
    image_size = session.FrameWidth * session.FrameHeight;       // YUV12: 50% extra chroma data
    bf_image_size = session.Width * session.Height;
    
    // memory alloction for input images
    p_input_data = (unsigned char *)malloc ((session.Width+64)*(session.Height+32)*3/2);
    if (session.input_format == 0)
    {
        // raster format
        input_image.buffer.p_y_buffer = p_input_data;
        input_image.buffer.p_u_buffer = input_image.buffer.p_y_buffer + image_size;
        input_image.buffer.p_v_buffer = input_image.buffer.p_u_buffer + image_size / 4;
        input_image.buffer.buf_width  = session.FrameWidth;
        input_image.buffer.buf_height = session.FrameHeight;
        input_image.x_offset = session.Xoffset;
        input_image.y_offset = session.Yoffset;
        input_image.width  = session.Width;
        input_image.height = session.Height;
    }
		start = clock(); 
        err = cuve264b_open (&p_encoder,session.slice_num);
        CHK_ERR(err, "open encoder");
        err = cuve264b_set_name (p_encoder, "GPU H.264 encoder");
        CHK_ERR(err, "set name");
 
		assert(p_encoder);
   
    if (session.PrintTiming) 
    {
        err = cuve264b_reset_timing (p_encoder);
        CHK_ERR(err, "reset timing");
    }
    //输入文件格式，raster模式
    cuve264b_set_input_image_format (p_encoder, (CUVE264B_E_IMG_FORMAT)0); 

    err = cuve264b_set_rate_control (p_encoder, (CUVE264B_E_RC_MODE)session.rc_mode);
    CHK_ERR(err, "set rate control");
    err = cuve264b_set_target_bitrate (p_encoder, session.target_bitrate);
    CHK_ERR(err, "set target bitrate");
    err = cuve264b_set_target_framerate (p_encoder, session.Fps);
    CHK_ERR(err, "set target framerate");

    err = cuve264b_set_qp_values (p_encoder, session.min_QP, session.max_QP, session.QPI, session.QP);
    CHK_ERR(err, "set qp values");
    err = cuve264b_set_iframe_interval (p_encoder, session.IFrate);
    CHK_ERR(err, "set iframe interval");
    err = cuve264b_set_intra_prediction_level (p_encoder, session.intra4x4_mode);
    CHK_ERR(err, "set intra prediction level");
    err = cuve264b_set_motion_estimation_level (p_encoder, session.me);
    CHK_ERR(err, "set motion estimation level");
    err = cuve264b_set_loopfilter_parameters (p_encoder, session.AlphaC0Offset, session.BetaOffset);
    CHK_ERR(err, "set loop filter parameters");
    err = cuve264b_set_iframe_interval (p_encoder, IFrate);
    CHK_ERR(err, "set iframe interval");

    if (session.DisableLoopFilter)
    {
        err = cuve264b_enable_deblocking_loopfilter (p_encoder, 0);
        CHK_ERR(err, "enable deblocking filter");
  
    }

   /// mainloop
	num_bytes_per_frame = (image_size + (image_size / 2));
    for (frame_num = 0; frame_num < session.frame_end; frame_num++) 
    { 
       
        
       start1 = clock();
	  // input image data from .yuv file
        read_bytes = fread (p_input_data, 1, num_bytes_per_frame, input_file);
		end1 = clock();
		read_file_time +=  end1-start1;
      
        if(frame_num >= session.frame_start) 
        {
			start1 = clock();
            err = cuve264b_prep_encode_frame (p_encoder, &input_image);
            CHK_ERR(err, "prep encode frame");
			end1 = clock();
			pre_process_time +=  end1-start1;

            err = cuve264b_encode_frame (p_encoder, NULL);   //编码处理

            CHK_ERR(err, "encode frame");

			start1 = clock();
			for(int i = 0; i < session.slice_num; i++ )
			{
				err = cuve264b_get_output_buffer (p_encoder, &output_buffer[0],i);
				CHK_ERR(err, "get output buffer");

				EncodedBytes = output_buffer[0].used_num_bytes;
				if (output_file) 
				{
					fwrite(output_buffer[0].p_buffer, 1, EncodedBytes, output_file);
				}
			}
			
            err = cuve264b_get_frame_encode_info (p_encoder, &stats);
            CHK_ERR(err, "get frame encode info");

            if (session.PrintStats >= 1) 
            {                
                    err = cuve264b_get_psnr (p_encoder, &psnr_y, &psnr_u, &psnr_v);
                    CHK_ERR(err, "get psnr");
                    if (session.PrintStats >= 2) {
                        print_frame_stats (&stats, psnr_y, psnr_u, psnr_v, frame_num, session.frame_start);
                    }           
                    total_psnr_y += psnr_y;
                    total_psnr_u += psnr_u;
                    total_psnr_v += psnr_v;                
            }
            
            total_bits +=  stats.num_bits;
            total_me_cands += stats.num_me_cands;
            qp_sum += stats.qp_used;
            min_qp = stats.qp_used < min_qp ? stats.qp_used : min_qp;
            max_qp = stats.qp_used > max_qp ? stats.qp_used : max_qp;

			end1 = clock();
			read_file_time +=  end1-start1;
        }    
    }
    end = clock();
	printf("total time is %d\n",end-start);
	printf("read_write time is %d\n",read_file_time);
	printf("pre_process time is %d\n",pre_process_time);
    if (session.PrintStats >= 1) 
    {
        print_enc_stats (total_psnr_y, total_psnr_u, total_psnr_v, total_bits,
                         total_me_cands, qp_sum, min_qp, max_qp, (session.frame_end-session.frame_start));
    }

    if (session.PrintTiming) 
    {
        err = cuve264b_print_timing(p_encoder, session.PrintTiming);
        CHK_ERR(err, "print timing");

    }

    err = cuve264b_close (p_encoder);
    CHK_ERR(err, "close encoder");
    if (output_file)
    {
        fclose(output_file);
    }
    if (input_file)
    {
        fclose(input_file);
    }

    free(p_input_data);
    return found_error;
}
