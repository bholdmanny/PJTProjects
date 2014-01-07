CFLAGS= -O0 -g
CPPFLAGS = $(CFLAGS)
# remove debug ref
OBJS = cavlc.o cavlc_header_codes.o cavlc_utils.o cuve264b.o cuve264b_utils.o deblock.o encode_frame.o encoder_context.o encoder_main.o encoder_tables.o mb_info.o me_context.o me_main.o mem_alloc_free.o mv_fns_ref.o output.o rc_internal_settings.o rc_main.o rc_mem_alloc_free.o rc_process_frame.o rc_utils.o

foo: $(OBJS)

clean:
	rm *.o
