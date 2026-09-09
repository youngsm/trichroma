"""Generated legacy FP32 wire primitive; see adjacent provenance JSON."""

ASM = r"""{

	.reg .pred 	%wire_p<79>;
	.reg .f32 	%wire_f<385>;
	.reg .b32 	%wire_r<106>;
	.reg .b64 	%wire_rd<48>;


	mov.u64 %wire_rd7, $1;
	mov.u64 %wire_rd8, $2;
	mov.u64 %wire_rd9, $3;
	mov.u64 %wire_rd10, $4;
	mov.u64 %wire_rd15, $9;
	mov.u32 %wire_r50, $10;
	mov.u32 %wire_r51, $11;
	mov.u32 %wire_r52, $12;
	setp.eq.s32 	%wire_p1, %wire_r52, 0;
	@%wire_p1 bra 	WIRE_L__BB6_64;

	cvta.to.global.u64 	%wire_rd16, %wire_rd15;
	cvta.to.global.u64 	%wire_rd17, %wire_rd7;
	mul.wide.s32 	%wire_rd18, %wire_r51, 12;
	add.s64 	%wire_rd19, %wire_rd17, %wire_rd18;
	ld.global.f32 	%wire_f1, [%wire_rd19];
	ld.global.f32 	%wire_f2, [%wire_rd19+4];
	ld.global.f32 	%wire_f3, [%wire_rd19+8];
	cvta.to.global.u64 	%wire_rd20, %wire_rd8;
	add.s64 	%wire_rd21, %wire_rd20, %wire_rd18;
	ld.global.f32 	%wire_f4, [%wire_rd21];
	ld.global.f32 	%wire_f5, [%wire_rd21+4];
	ld.global.f32 	%wire_f6, [%wire_rd21+8];
	mul.wide.s32 	%wire_rd22, %wire_r51, 4;
	add.s64 	%wire_rd2, %wire_rd16, %wire_rd22;
	ld.global.u32 	%wire_r53, [%wire_rd2];
	setp.eq.s32 	%wire_p2, %wire_r53, -1;
	cvta.to.global.u64 	%wire_rd23, %wire_rd10;
	add.s64 	%wire_rd3, %wire_rd23, %wire_rd22;
	mov.f32 	%wire_f376, 0f7149F2CA;
	mov.f32 	%wire_f324, %wire_f376;
	@%wire_p2 bra 	WIRE_L__BB6_3;

	ld.global.f32 	%wire_f324, [%wire_rd3];

WIRE_L__BB6_3:
	setp.eq.s64 	%wire_p3, %wire_rd9, 0;
	setp.lt.s32 	%wire_p4, %wire_r50, 1;
	mov.u32 	%wire_r100, -1;
	mov.f32 	%wire_f372, 0f00000000;
	or.pred  	%wire_p5, %wire_p3, %wire_p4;
	mov.f32 	%wire_f373, %wire_f372;
	mov.f32 	%wire_f374, %wire_f372;
	mov.f32 	%wire_f375, %wire_f372;
	mov.u32 	%wire_r101, %wire_r100;
	mov.u32 	%wire_r102, %wire_r100;
	@%wire_p5 bra 	WIRE_L__BB6_60;

	mov.f32 	%wire_f375, 0f00000000;
	mov.u32 	%wire_r77, 0;
	mov.f32 	%wire_f376, 0f7149F2CA;
	mov.u32 	%wire_r102, -1;
	mov.u32 	%wire_r101, %wire_r102;
	mov.u32 	%wire_r100, %wire_r102;
	mov.f32 	%wire_f374, %wire_f375;
	mov.f32 	%wire_f373, %wire_f375;
	mov.f32 	%wire_f372, %wire_f375;

WIRE_L__BB6_5:
	cvt.s64.s32 	%wire_rd5, %wire_r77;
	mul.wide.s32 	%wire_rd24, %wire_r77, 124;
	add.s64 	%wire_rd25, %wire_rd9, %wire_rd24;
	setp.eq.s64 	%wire_p6, %wire_rd25, 0;
	@%wire_p6 bra 	WIRE_L__BB6_59;

	cvta.to.global.u64 	%wire_rd47, %wire_rd9;
	mul.lo.s64 	%wire_rd26, %wire_rd5, 124;
	add.s64 	%wire_rd27, %wire_rd47, %wire_rd26;
	add.s64 	%wire_rd6, %wire_rd27, 80;
	ld.global.f32 	%wire_f14, [%wire_rd27+80];
	ld.global.f32 	%wire_f15, [%wire_rd27+84];
	ld.global.f32 	%wire_f16, [%wire_rd27+88];
	ld.global.f32 	%wire_f17, [%wire_rd27+92];
	ld.global.f32 	%wire_f18, [%wire_rd27+96];
	ld.global.f32 	%wire_f19, [%wire_rd27+100];
	ld.global.f32 	%wire_f172, [%wire_rd27];
	sub.ftz.f32 	%wire_f20, %wire_f1, %wire_f172;
	ld.global.f32 	%wire_f173, [%wire_rd27+4];
	sub.ftz.f32 	%wire_f21, %wire_f2, %wire_f173;
	ld.global.f32 	%wire_f174, [%wire_rd27+8];
	sub.ftz.f32 	%wire_f22, %wire_f3, %wire_f174;
	ld.global.f32 	%wire_f23, [%wire_rd27+104];
	ld.global.f32 	%wire_f24, [%wire_rd27+108];
	mul.ftz.f32 	%wire_f175, %wire_f5, %wire_f24;
	fma.rn.ftz.f32 	%wire_f176, %wire_f4, %wire_f23, %wire_f175;
	ld.global.f32 	%wire_f25, [%wire_rd27+112];
	fma.rn.ftz.f32 	%wire_f26, %wire_f6, %wire_f25, %wire_f176;
	mul.ftz.f32 	%wire_f177, %wire_f24, %wire_f21;
	fma.rn.ftz.f32 	%wire_f178, %wire_f23, %wire_f20, %wire_f177;
	fma.rn.ftz.f32 	%wire_f27, %wire_f25, %wire_f22, %wire_f178;
	abs.ftz.f32 	%wire_f28, %wire_f27;
	ld.global.f32 	%wire_f29, [%wire_rd27+40];
	add.ftz.f32 	%wire_f179, %wire_f29, 0f3C23D70A;
	setp.leu.ftz.f32 	%wire_p7, %wire_f28, %wire_f179;
	@%wire_p7 bra 	WIRE_L__BB6_9;

	mul.ftz.f32 	%wire_f180, %wire_f26, %wire_f27;
	setp.gt.ftz.f32 	%wire_p8, %wire_f180, 0f00000000;
	@%wire_p8 bra 	WIRE_L__BB6_59;

	neg.ftz.f32 	%wire_f181, %wire_f27;
	div.approx.ftz.f32 	%wire_f182, %wire_f181, %wire_f26;
	add.ftz.f32 	%wire_f183, %wire_f324, %wire_f29;
	setp.gt.ftz.f32 	%wire_p9, %wire_f182, %wire_f183;
	@%wire_p9 bra 	WIRE_L__BB6_59;

WIRE_L__BB6_9:
	mul.ftz.f32 	%wire_f184, %wire_f5, %wire_f15;
	fma.rn.ftz.f32 	%wire_f185, %wire_f4, %wire_f14, %wire_f184;
	fma.rn.ftz.f32 	%wire_f30, %wire_f6, %wire_f16, %wire_f185;
	mul.ftz.f32 	%wire_f186, %wire_f5, %wire_f18;
	fma.rn.ftz.f32 	%wire_f187, %wire_f4, %wire_f17, %wire_f186;
	fma.rn.ftz.f32 	%wire_f31, %wire_f6, %wire_f19, %wire_f187;
	mul.ftz.f32 	%wire_f188, %wire_f15, %wire_f21;
	fma.rn.ftz.f32 	%wire_f189, %wire_f14, %wire_f20, %wire_f188;
	fma.rn.ftz.f32 	%wire_f32, %wire_f16, %wire_f22, %wire_f189;
	mul.ftz.f32 	%wire_f190, %wire_f18, %wire_f21;
	fma.rn.ftz.f32 	%wire_f191, %wire_f17, %wire_f20, %wire_f190;
	fma.rn.ftz.f32 	%wire_f192, %wire_f19, %wire_f22, %wire_f191;
	ld.global.f32 	%wire_f193, [%wire_rd6+-20];
	sub.ftz.f32 	%wire_f33, %wire_f192, %wire_f193;
	abs.ftz.f32 	%wire_f194, %wire_f30;
	setp.lt.ftz.f32 	%wire_p10, %wire_f194, 0f33D6BF95;
	ld.global.f32 	%wire_f34, [%wire_rd6+-36];
	@%wire_p10 bra 	WIRE_L__BB6_11;
	bra.uni 	WIRE_L__BB6_10;

WIRE_L__BB6_11:
	setp.lt.ftz.f32 	%wire_p13, %wire_f32, %wire_f34;
	@%wire_p13 bra 	WIRE_L__BB6_59;

	ld.global.f32 	%wire_f366, [%wire_rd6+-32];
	setp.gt.ftz.f32 	%wire_p14, %wire_f32, %wire_f366;
	mov.f32 	%wire_f332, 0f7149F2CA;
	mov.f32 	%wire_f331, 0fF149F2CA;
	@%wire_p14 bra 	WIRE_L__BB6_59;
	bra.uni 	WIRE_L__BB6_13;

WIRE_L__BB6_10:
	sub.ftz.f32 	%wire_f195, %wire_f34, %wire_f32;
	div.approx.ftz.f32 	%wire_f196, %wire_f195, %wire_f30;
	ld.global.f32 	%wire_f366, [%wire_rd6+-32];
	sub.ftz.f32 	%wire_f197, %wire_f366, %wire_f32;
	div.approx.ftz.f32 	%wire_f198, %wire_f197, %wire_f30;
	setp.gt.ftz.f32 	%wire_p11, %wire_f196, %wire_f198;
	selp.f32 	%wire_f199, %wire_f198, %wire_f196, %wire_p11;
	selp.f32 	%wire_f200, %wire_f196, %wire_f198, %wire_p11;
	max.ftz.f32 	%wire_f331, %wire_f199, 0fF149F2CA;
	min.ftz.f32 	%wire_f332, %wire_f200, 0f7149F2CA;
	setp.gt.ftz.f32 	%wire_p12, %wire_f331, %wire_f332;
	@%wire_p12 bra 	WIRE_L__BB6_59;

WIRE_L__BB6_13:
	ld.global.f32 	%wire_f42, [%wire_rd6+-44];
	setp.eq.ftz.f32 	%wire_p15, %wire_f42, 0f00000000;
	mov.f32 	%wire_f333, 0f00000000;
	@%wire_p15 bra 	WIRE_L__BB6_15;

	rcp.approx.ftz.f32 	%wire_f333, %wire_f42;

WIRE_L__BB6_15:
	add.ftz.f32 	%wire_f45, %wire_f29, %wire_f29;
	fma.rn.ftz.f32 	%wire_f46, %wire_f45, 0f3F000000, 0f3727C5AC;
	mul.rn.ftz.f32 	%wire_f204, %wire_f31, %wire_f31;
	fma.rn.ftz.f32 	%wire_f47, %wire_f26, %wire_f26, %wire_f204;
	ld.global.u32 	%wire_r79, [%wire_rd6+40];
	ld.global.u32 	%wire_r78, [%wire_rd6+36];
	setp.gt.s32 	%wire_p16, %wire_r78, %wire_r79;
	@%wire_p16 bra 	WIRE_L__BB6_24;
	bra.uni 	WIRE_L__BB6_16;

WIRE_L__BB6_24:
	mul.ftz.f32 	%wire_f58, %wire_f29, %wire_f29;
	setp.gt.s32 	%wire_p27, %wire_r78, %wire_r79;
	@%wire_p27 bra 	WIRE_L__BB6_59;

	mul.ftz.f32 	%wire_f59, %wire_f26, %wire_f27;
	mul.ftz.f32 	%wire_f61, %wire_f58, 0f358637BD;
	add.s32 	%wire_r64, %wire_r79, 1;
	sub.s32 	%wire_r65, %wire_r64, %wire_r78;
	and.b32  	%wire_r66, %wire_r65, 1;
	setp.eq.b32 	%wire_p28, %wire_r66, 1;
	mov.pred 	%wire_p29, 0;
	xor.pred  	%wire_p30, %wire_p28, %wire_p29;
	not.pred 	%wire_p31, %wire_p30;
	mov.u32 	%wire_r90, %wire_r78;
	@%wire_p31 bra 	WIRE_L__BB6_37;

	mul.ftz.f32 	%wire_f313, %wire_f27, %wire_f27;
	cvt.rn.f32.s32 	%wire_f231, %wire_r78;
	mul.ftz.f32 	%wire_f232, %wire_f42, %wire_f231;
	sub.ftz.f32 	%wire_f62, %wire_f33, %wire_f232;
	fma.rn.ftz.f32 	%wire_f63, %wire_f31, %wire_f62, %wire_f59;
	fma.rn.ftz.f32 	%wire_f64, %wire_f62, %wire_f62, %wire_f313;
	sub.ftz.f32 	%wire_f233, %wire_f64, %wire_f58;
	mul.ftz.f32 	%wire_f234, %wire_f63, %wire_f63;
	mul.ftz.f32 	%wire_f235, %wire_f47, %wire_f233;
	sub.ftz.f32 	%wire_f65, %wire_f234, %wire_f235;
	setp.lt.ftz.f32 	%wire_p32, %wire_f65, 0f00000000;
	@%wire_p32 bra 	WIRE_L__BB6_36;

	sqrt.approx.ftz.f32 	%wire_f236, %wire_f65;
	neg.ftz.f32 	%wire_f237, %wire_f63;
	sub.ftz.f32 	%wire_f238, %wire_f237, %wire_f236;
	div.approx.ftz.f32 	%wire_f337, %wire_f238, %wire_f47;
	sub.ftz.f32 	%wire_f239, %wire_f236, %wire_f63;
	div.approx.ftz.f32 	%wire_f67, %wire_f239, %wire_f47;
	mov.f32 	%wire_f240, 0f2B8CBCCC;
	max.ftz.f32 	%wire_f68, %wire_f240, %wire_f61;
	add.ftz.f32 	%wire_f241, %wire_f58, %wire_f68;
	setp.gt.ftz.f32 	%wire_p33, %wire_f64, %wire_f241;
	@%wire_p33 bra 	WIRE_L__BB6_30;
	bra.uni 	WIRE_L__BB6_28;

WIRE_L__BB6_30:
	setp.le.ftz.f32 	%wire_p36, %wire_f337, 0f38D1B717;
	@%wire_p36 bra 	WIRE_L__BB6_36;
	bra.uni 	WIRE_L__BB6_31;

WIRE_L__BB6_16:
	mov.f32 	%wire_f205, 0f38D1B717;
	max.ftz.f32 	%wire_f334, %wire_f331, %wire_f205;
	setp.lt.ftz.f32 	%wire_p17, %wire_f324, %wire_f332;
	selp.f32 	%wire_f336, %wire_f324, %wire_f332, %wire_p17;
	abs.ftz.f32 	%wire_f50, %wire_f26;
	setp.gt.ftz.f32 	%wire_p18, %wire_f50, 0f33D6BF95;
	@%wire_p18 bra 	WIRE_L__BB6_18;
	bra.uni 	WIRE_L__BB6_17;

WIRE_L__BB6_18:
	neg.ftz.f32 	%wire_f206, %wire_f46;
	sub.ftz.f32 	%wire_f207, %wire_f206, %wire_f27;
	div.approx.ftz.f32 	%wire_f208, %wire_f207, %wire_f26;
	sub.ftz.f32 	%wire_f209, %wire_f46, %wire_f27;
	div.approx.ftz.f32 	%wire_f210, %wire_f209, %wire_f26;
	setp.gt.ftz.f32 	%wire_p20, %wire_f208, %wire_f210;
	selp.f32 	%wire_f211, %wire_f210, %wire_f208, %wire_p20;
	selp.f32 	%wire_f212, %wire_f208, %wire_f210, %wire_p20;
	max.ftz.f32 	%wire_f334, %wire_f334, %wire_f211;
	min.ftz.f32 	%wire_f336, %wire_f336, %wire_f212;
	bra.uni 	WIRE_L__BB6_19;

WIRE_L__BB6_17:
	abs.ftz.f32 	%wire_f316, %wire_f27;
	setp.gt.ftz.f32 	%wire_p19, %wire_f316, %wire_f46;
	@%wire_p19 bra 	WIRE_L__BB6_59;

WIRE_L__BB6_19:
	setp.lt.ftz.f32 	%wire_p21, %wire_f336, %wire_f334;
	@%wire_p21 bra 	WIRE_L__BB6_59;

	setp.gtu.ftz.f32 	%wire_p22, %wire_f50, 0f33D6BF95;
	@%wire_p22 bra 	WIRE_L__BB6_23;

	abs.ftz.f32 	%wire_f55, %wire_f31;
	setp.leu.ftz.f32 	%wire_p23, %wire_f55, 0f33D6BF95;
	@%wire_p23 bra 	WIRE_L__BB6_23;

	add.ftz.f32 	%wire_f317, %wire_f29, %wire_f29;
	add.ftz.f32 	%wire_f213, %wire_f42, %wire_f317;
	div.approx.ftz.f32 	%wire_f214, %wire_f213, %wire_f55;
	add.ftz.f32 	%wire_f215, %wire_f334, %wire_f214;
	min.ftz.f32 	%wire_f336, %wire_f336, %wire_f215;

WIRE_L__BB6_23:
	fma.rn.ftz.f32 	%wire_f216, %wire_f31, %wire_f334, %wire_f33;
	fma.rn.ftz.f32 	%wire_f217, %wire_f31, %wire_f336, %wire_f33;
	min.ftz.f32 	%wire_f218, %wire_f216, %wire_f217;
	sub.ftz.f32 	%wire_f219, %wire_f218, %wire_f46;
	max.ftz.f32 	%wire_f220, %wire_f216, %wire_f217;
	add.ftz.f32 	%wire_f221, %wire_f46, %wire_f220;
	sub.ftz.f32 	%wire_f222, %wire_f33, %wire_f46;
	setp.lt.ftz.f32 	%wire_p24, %wire_f222, %wire_f219;
	selp.f32 	%wire_f223, %wire_f222, %wire_f219, %wire_p24;
	add.ftz.f32 	%wire_f224, %wire_f33, %wire_f46;
	setp.gt.ftz.f32 	%wire_p25, %wire_f224, %wire_f221;
	selp.f32 	%wire_f225, %wire_f224, %wire_f221, %wire_p25;
	mul.ftz.f32 	%wire_f226, %wire_f333, %wire_f223;
	cvt.rmi.ftz.f32.f32 	%wire_f227, %wire_f226;
	cvt.rzi.ftz.s32.f32 	%wire_r61, %wire_f227;
	mul.ftz.f32 	%wire_f228, %wire_f333, %wire_f225;
	cvt.rpi.ftz.f32.f32 	%wire_f229, %wire_f228;
	cvt.rzi.ftz.s32.f32 	%wire_r62, %wire_f229;
	max.s32 	%wire_r78, %wire_r61, %wire_r78;
	min.s32 	%wire_r79, %wire_r62, %wire_r79;
	setp.gt.s32 	%wire_p26, %wire_r78, %wire_r79;
	@%wire_p26 bra 	WIRE_L__BB6_59;
	bra.uni 	WIRE_L__BB6_24;

WIRE_L__BB6_28:
	sub.ftz.f32 	%wire_f243, %wire_f58, %wire_f68;
	setp.geu.ftz.f32 	%wire_p34, %wire_f64, %wire_f243;
	mov.f32 	%wire_f337, 0f38D1B717;
	@%wire_p34 bra 	WIRE_L__BB6_31;

	setp.gtu.ftz.f32 	%wire_p35, %wire_f67, 0f38D1B717;
	mov.f32 	%wire_f337, %wire_f67;
	@%wire_p35 bra 	WIRE_L__BB6_31;
	bra.uni 	WIRE_L__BB6_36;

WIRE_L__BB6_31:
	fma.rn.ftz.f32 	%wire_f70, %wire_f30, %wire_f337, %wire_f32;
	setp.lt.ftz.f32 	%wire_p37, %wire_f70, %wire_f34;
	@%wire_p37 bra 	WIRE_L__BB6_36;

	setp.gt.ftz.f32 	%wire_p38, %wire_f70, %wire_f366;
	setp.ge.ftz.f32 	%wire_p39, %wire_f337, %wire_f376;
	or.pred  	%wire_p40, %wire_p39, %wire_p38;
	@%wire_p40 bra 	WIRE_L__BB6_36;

	setp.lt.ftz.f32 	%wire_p41, %wire_f337, %wire_f331;
	setp.gt.ftz.f32 	%wire_p42, %wire_f337, %wire_f332;
	or.pred  	%wire_p43, %wire_p41, %wire_p42;
	@%wire_p43 bra 	WIRE_L__BB6_36;

	cvt.rn.f32.s32 	%wire_f320, %wire_r78;
	mul.ftz.f32 	%wire_f319, %wire_f42, %wire_f320;
	sub.ftz.f32 	%wire_f318, %wire_f33, %wire_f319;
	fma.rn.ftz.f32 	%wire_f71, %wire_f31, %wire_f337, %wire_f318;
	fma.rn.ftz.f32 	%wire_f72, %wire_f26, %wire_f337, %wire_f27;
	mul.ftz.f32 	%wire_f244, %wire_f72, %wire_f72;
	fma.rn.ftz.f32 	%wire_f245, %wire_f71, %wire_f71, %wire_f244;
	sqrt.approx.ftz.f32 	%wire_f73, %wire_f245;
	setp.le.ftz.f32 	%wire_p44, %wire_f73, 0f00000000;
	@%wire_p44 bra 	WIRE_L__BB6_36;

	rcp.approx.ftz.f32 	%wire_f246, %wire_f73;
	mul.ftz.f32 	%wire_f247, %wire_f71, %wire_f246;
	mul.ftz.f32 	%wire_f248, %wire_f72, %wire_f246;
	mul.ftz.f32 	%wire_f249, %wire_f23, %wire_f248;
	fma.rn.ftz.f32 	%wire_f372, %wire_f17, %wire_f247, %wire_f249;
	mul.ftz.f32 	%wire_f250, %wire_f24, %wire_f248;
	fma.rn.ftz.f32 	%wire_f373, %wire_f18, %wire_f247, %wire_f250;
	mul.ftz.f32 	%wire_f251, %wire_f25, %wire_f248;
	fma.rn.ftz.f32 	%wire_f374, %wire_f19, %wire_f247, %wire_f251;
	mul.ftz.f32 	%wire_f252, %wire_f4, %wire_f372;
	mul.ftz.f32 	%wire_f253, %wire_f5, %wire_f373;
	neg.ftz.f32 	%wire_f254, %wire_f253;
	sub.ftz.f32 	%wire_f255, %wire_f254, %wire_f252;
	mul.ftz.f32 	%wire_f256, %wire_f6, %wire_f374;
	sub.ftz.f32 	%wire_f375, %wire_f255, %wire_f256;
	ld.global.u32 	%wire_r102, [%wire_rd6+-16];
	ld.global.u32 	%wire_r101, [%wire_rd6+-8];
	ld.global.u32 	%wire_r100, [%wire_rd6+-12];
	mov.f32 	%wire_f376, %wire_f337;

WIRE_L__BB6_36:
	add.s32 	%wire_r90, %wire_r78, 1;

WIRE_L__BB6_37:
	setp.eq.s32 	%wire_p45, %wire_r79, %wire_r78;
	@%wire_p45 bra 	WIRE_L__BB6_59;

WIRE_L__BB6_38:
	mul.ftz.f32 	%wire_f314, %wire_f27, %wire_f27;
	cvt.rn.f32.s32 	%wire_f257, %wire_r90;
	mul.ftz.f32 	%wire_f258, %wire_f42, %wire_f257;
	sub.ftz.f32 	%wire_f99, %wire_f33, %wire_f258;
	fma.rn.ftz.f32 	%wire_f100, %wire_f31, %wire_f99, %wire_f59;
	fma.rn.ftz.f32 	%wire_f101, %wire_f99, %wire_f99, %wire_f314;
	sub.ftz.f32 	%wire_f259, %wire_f101, %wire_f58;
	mul.ftz.f32 	%wire_f260, %wire_f100, %wire_f100;
	mul.ftz.f32 	%wire_f261, %wire_f47, %wire_f259;
	sub.ftz.f32 	%wire_f102, %wire_f260, %wire_f261;
	setp.lt.ftz.f32 	%wire_p46, %wire_f102, 0f00000000;
	@%wire_p46 bra 	WIRE_L__BB6_48;

	sqrt.approx.ftz.f32 	%wire_f262, %wire_f102;
	neg.ftz.f32 	%wire_f263, %wire_f100;
	sub.ftz.f32 	%wire_f264, %wire_f263, %wire_f262;
	div.approx.ftz.f32 	%wire_f359, %wire_f264, %wire_f47;
	sub.ftz.f32 	%wire_f265, %wire_f262, %wire_f100;
	div.approx.ftz.f32 	%wire_f104, %wire_f265, %wire_f47;
	mov.f32 	%wire_f266, 0f2B8CBCCC;
	max.ftz.f32 	%wire_f105, %wire_f266, %wire_f61;
	add.ftz.f32 	%wire_f267, %wire_f58, %wire_f105;
	setp.gt.ftz.f32 	%wire_p47, %wire_f101, %wire_f267;
	@%wire_p47 bra 	WIRE_L__BB6_42;
	bra.uni 	WIRE_L__BB6_40;

WIRE_L__BB6_42:
	setp.le.ftz.f32 	%wire_p50, %wire_f359, 0f38D1B717;
	@%wire_p50 bra 	WIRE_L__BB6_48;
	bra.uni 	WIRE_L__BB6_43;

WIRE_L__BB6_40:
	sub.ftz.f32 	%wire_f269, %wire_f58, %wire_f105;
	setp.geu.ftz.f32 	%wire_p48, %wire_f101, %wire_f269;
	mov.f32 	%wire_f359, 0f38D1B717;
	@%wire_p48 bra 	WIRE_L__BB6_43;

	setp.gtu.ftz.f32 	%wire_p49, %wire_f104, 0f38D1B717;
	mov.f32 	%wire_f359, %wire_f104;
	@%wire_p49 bra 	WIRE_L__BB6_43;
	bra.uni 	WIRE_L__BB6_48;

WIRE_L__BB6_43:
	fma.rn.ftz.f32 	%wire_f107, %wire_f30, %wire_f359, %wire_f32;
	setp.lt.ftz.f32 	%wire_p51, %wire_f107, %wire_f34;
	@%wire_p51 bra 	WIRE_L__BB6_48;

	setp.gt.ftz.f32 	%wire_p52, %wire_f107, %wire_f366;
	setp.ge.ftz.f32 	%wire_p53, %wire_f359, %wire_f376;
	or.pred  	%wire_p54, %wire_p53, %wire_p52;
	@%wire_p54 bra 	WIRE_L__BB6_48;

	setp.lt.ftz.f32 	%wire_p55, %wire_f359, %wire_f331;
	setp.gt.ftz.f32 	%wire_p56, %wire_f359, %wire_f332;
	or.pred  	%wire_p57, %wire_p55, %wire_p56;
	@%wire_p57 bra 	WIRE_L__BB6_48;

	cvt.rn.f32.s32 	%wire_f323, %wire_r90;
	mul.ftz.f32 	%wire_f322, %wire_f42, %wire_f323;
	sub.ftz.f32 	%wire_f321, %wire_f33, %wire_f322;
	fma.rn.ftz.f32 	%wire_f108, %wire_f31, %wire_f359, %wire_f321;
	fma.rn.ftz.f32 	%wire_f109, %wire_f26, %wire_f359, %wire_f27;
	mul.ftz.f32 	%wire_f270, %wire_f109, %wire_f109;
	fma.rn.ftz.f32 	%wire_f271, %wire_f108, %wire_f108, %wire_f270;
	sqrt.approx.ftz.f32 	%wire_f110, %wire_f271;
	setp.le.ftz.f32 	%wire_p58, %wire_f110, 0f00000000;
	@%wire_p58 bra 	WIRE_L__BB6_48;

	rcp.approx.ftz.f32 	%wire_f272, %wire_f110;
	mul.ftz.f32 	%wire_f273, %wire_f108, %wire_f272;
	mul.ftz.f32 	%wire_f274, %wire_f109, %wire_f272;
	mul.ftz.f32 	%wire_f275, %wire_f23, %wire_f274;
	fma.rn.ftz.f32 	%wire_f372, %wire_f17, %wire_f273, %wire_f275;
	mul.ftz.f32 	%wire_f276, %wire_f24, %wire_f274;
	fma.rn.ftz.f32 	%wire_f373, %wire_f18, %wire_f273, %wire_f276;
	mul.ftz.f32 	%wire_f277, %wire_f25, %wire_f274;
	fma.rn.ftz.f32 	%wire_f374, %wire_f19, %wire_f273, %wire_f277;
	mul.ftz.f32 	%wire_f278, %wire_f4, %wire_f372;
	mul.ftz.f32 	%wire_f279, %wire_f5, %wire_f373;
	neg.ftz.f32 	%wire_f280, %wire_f279;
	sub.ftz.f32 	%wire_f281, %wire_f280, %wire_f278;
	mul.ftz.f32 	%wire_f282, %wire_f6, %wire_f374;
	sub.ftz.f32 	%wire_f375, %wire_f281, %wire_f282;
	ld.global.u32 	%wire_r102, [%wire_rd6+-16];
	ld.global.u32 	%wire_r101, [%wire_rd6+-8];
	ld.global.u32 	%wire_r100, [%wire_rd6+-12];
	mov.f32 	%wire_f376, %wire_f359;

WIRE_L__BB6_48:
	mul.ftz.f32 	%wire_f315, %wire_f27, %wire_f27;
	add.s32 	%wire_r35, %wire_r90, 1;
	cvt.rn.f32.s32 	%wire_f283, %wire_r35;
	mul.ftz.f32 	%wire_f284, %wire_f42, %wire_f283;
	sub.ftz.f32 	%wire_f120, %wire_f33, %wire_f284;
	fma.rn.ftz.f32 	%wire_f121, %wire_f31, %wire_f120, %wire_f59;
	fma.rn.ftz.f32 	%wire_f122, %wire_f120, %wire_f120, %wire_f315;
	sub.ftz.f32 	%wire_f285, %wire_f122, %wire_f58;
	mul.ftz.f32 	%wire_f286, %wire_f121, %wire_f121;
	mul.ftz.f32 	%wire_f287, %wire_f47, %wire_f285;
	sub.ftz.f32 	%wire_f123, %wire_f286, %wire_f287;
	setp.lt.ftz.f32 	%wire_p59, %wire_f123, 0f00000000;
	@%wire_p59 bra 	WIRE_L__BB6_58;

	sqrt.approx.ftz.f32 	%wire_f288, %wire_f123;
	neg.ftz.f32 	%wire_f289, %wire_f121;
	sub.ftz.f32 	%wire_f290, %wire_f289, %wire_f288;
	div.approx.ftz.f32 	%wire_f365, %wire_f290, %wire_f47;
	sub.ftz.f32 	%wire_f291, %wire_f288, %wire_f121;
	div.approx.ftz.f32 	%wire_f125, %wire_f291, %wire_f47;
	mov.f32 	%wire_f292, 0f2B8CBCCC;
	max.ftz.f32 	%wire_f126, %wire_f292, %wire_f61;
	add.ftz.f32 	%wire_f293, %wire_f58, %wire_f126;
	setp.gt.ftz.f32 	%wire_p60, %wire_f122, %wire_f293;
	@%wire_p60 bra 	WIRE_L__BB6_52;
	bra.uni 	WIRE_L__BB6_50;

WIRE_L__BB6_52:
	setp.le.ftz.f32 	%wire_p63, %wire_f365, 0f38D1B717;
	@%wire_p63 bra 	WIRE_L__BB6_58;
	bra.uni 	WIRE_L__BB6_53;

WIRE_L__BB6_50:
	sub.ftz.f32 	%wire_f295, %wire_f58, %wire_f126;
	setp.geu.ftz.f32 	%wire_p61, %wire_f122, %wire_f295;
	mov.f32 	%wire_f365, 0f38D1B717;
	@%wire_p61 bra 	WIRE_L__BB6_53;

	setp.gtu.ftz.f32 	%wire_p62, %wire_f125, 0f38D1B717;
	mov.f32 	%wire_f365, %wire_f125;
	@%wire_p62 bra 	WIRE_L__BB6_53;
	bra.uni 	WIRE_L__BB6_58;

WIRE_L__BB6_53:
	fma.rn.ftz.f32 	%wire_f128, %wire_f30, %wire_f365, %wire_f32;
	setp.lt.ftz.f32 	%wire_p64, %wire_f128, %wire_f34;
	@%wire_p64 bra 	WIRE_L__BB6_58;

	setp.gt.ftz.f32 	%wire_p65, %wire_f128, %wire_f366;
	setp.ge.ftz.f32 	%wire_p66, %wire_f365, %wire_f376;
	or.pred  	%wire_p67, %wire_p66, %wire_p65;
	@%wire_p67 bra 	WIRE_L__BB6_58;

	setp.lt.ftz.f32 	%wire_p68, %wire_f365, %wire_f331;
	setp.gt.ftz.f32 	%wire_p69, %wire_f365, %wire_f332;
	or.pred  	%wire_p70, %wire_p68, %wire_p69;
	@%wire_p70 bra 	WIRE_L__BB6_58;

	add.s32 	%wire_r72, %wire_r90, 1;
	cvt.rn.f32.s32 	%wire_f312, %wire_r72;
	mul.ftz.f32 	%wire_f311, %wire_f42, %wire_f312;
	sub.ftz.f32 	%wire_f310, %wire_f33, %wire_f311;
	fma.rn.ftz.f32 	%wire_f129, %wire_f31, %wire_f365, %wire_f310;
	fma.rn.ftz.f32 	%wire_f130, %wire_f26, %wire_f365, %wire_f27;
	mul.ftz.f32 	%wire_f296, %wire_f130, %wire_f130;
	fma.rn.ftz.f32 	%wire_f297, %wire_f129, %wire_f129, %wire_f296;
	sqrt.approx.ftz.f32 	%wire_f131, %wire_f297;
	setp.le.ftz.f32 	%wire_p71, %wire_f131, 0f00000000;
	@%wire_p71 bra 	WIRE_L__BB6_58;

	rcp.approx.ftz.f32 	%wire_f298, %wire_f131;
	mul.ftz.f32 	%wire_f299, %wire_f129, %wire_f298;
	mul.ftz.f32 	%wire_f300, %wire_f130, %wire_f298;
	mul.ftz.f32 	%wire_f301, %wire_f23, %wire_f300;
	fma.rn.ftz.f32 	%wire_f372, %wire_f17, %wire_f299, %wire_f301;
	mul.ftz.f32 	%wire_f302, %wire_f24, %wire_f300;
	fma.rn.ftz.f32 	%wire_f373, %wire_f18, %wire_f299, %wire_f302;
	mul.ftz.f32 	%wire_f303, %wire_f25, %wire_f300;
	fma.rn.ftz.f32 	%wire_f374, %wire_f19, %wire_f299, %wire_f303;
	mul.ftz.f32 	%wire_f304, %wire_f4, %wire_f372;
	mul.ftz.f32 	%wire_f305, %wire_f5, %wire_f373;
	neg.ftz.f32 	%wire_f306, %wire_f305;
	sub.ftz.f32 	%wire_f307, %wire_f306, %wire_f304;
	mul.ftz.f32 	%wire_f308, %wire_f6, %wire_f374;
	sub.ftz.f32 	%wire_f375, %wire_f307, %wire_f308;
	ld.global.u32 	%wire_r102, [%wire_rd6+-16];
	ld.global.u32 	%wire_r101, [%wire_rd6+-8];
	ld.global.u32 	%wire_r100, [%wire_rd6+-12];
	mov.f32 	%wire_f376, %wire_f365;

WIRE_L__BB6_58:
	add.s32 	%wire_r73, %wire_r90, 1;
	add.s32 	%wire_r90, %wire_r90, 2;
	setp.lt.s32 	%wire_p72, %wire_r73, %wire_r79;
	@%wire_p72 bra 	WIRE_L__BB6_38;

WIRE_L__BB6_59:
	cvt.u32.u64 	%wire_r67, %wire_rd5;
	add.s32 	%wire_r77, %wire_r67, 1;
	setp.lt.s32 	%wire_p73, %wire_r77, %wire_r50;
	@%wire_p73 bra 	WIRE_L__BB6_5;

WIRE_L__BB6_60:
	add.ftz.f32 	%wire_f309, %wire_f376, 0f358637BD;
	setp.geu.ftz.f32 	%wire_p74, %wire_f309, %wire_f324;
	setp.lt.s32 	%wire_p75, %wire_r102, 0;
	or.pred  	%wire_p76, %wire_p74, %wire_p75;
	@%wire_p76 bra 	WIRE_L__BB6_64;

	st.global.f32 	[%wire_rd3], %wire_f376;
	setp.gt.ftz.f32 	%wire_p77, %wire_f375, 0f00000000;
	@%wire_p77 bra 	WIRE_L__BB6_63;

	neg.ftz.f32 	%wire_f374, %wire_f374;
	neg.ftz.f32 	%wire_f373, %wire_f373;
	neg.ftz.f32 	%wire_f372, %wire_f372;

WIRE_L__BB6_63:
	mov.u32 %wire_r71, $11;
	mov.u64 %wire_rd46, $9;
	mul.wide.s32 	%wire_rd45, %wire_r71, 4;
	cvta.to.global.u64 	%wire_rd44, %wire_rd46;
	add.s64 	%wire_rd43, %wire_rd44, %wire_rd45;
	mov.u64 %wire_rd42, $8;
	mov.u64 %wire_rd41, $7;
	mov.u64 %wire_rd40, $6;
	cvt.s64.s32 	%wire_rd39, %wire_r71;
	mov.u64 %wire_rd38, $5;
	cvta.to.global.u64 	%wire_rd28, %wire_rd38;
	mul.lo.s64 	%wire_rd29, %wire_rd39, 12;
	add.s64 	%wire_rd30, %wire_rd28, %wire_rd29;
	st.global.f32 	[%wire_rd30], %wire_f372;
	st.global.f32 	[%wire_rd30+4], %wire_f373;
	st.global.f32 	[%wire_rd30+8], %wire_f374;
	selp.b32 	%wire_r68, %wire_r100, %wire_r101, %wire_p77;
	cvta.to.global.u64 	%wire_rd31, %wire_rd40;
	shl.b64 	%wire_rd32, %wire_rd39, 2;
	add.s64 	%wire_rd33, %wire_rd31, %wire_rd32;
	st.global.u32 	[%wire_rd33], %wire_r68;
	selp.b32 	%wire_r69, %wire_r101, %wire_r100, %wire_p77;
	cvta.to.global.u64 	%wire_rd34, %wire_rd41;
	add.s64 	%wire_rd35, %wire_rd34, %wire_rd32;
	st.global.u32 	[%wire_rd35], %wire_r69;
	cvta.to.global.u64 	%wire_rd36, %wire_rd42;
	add.s64 	%wire_rd37, %wire_rd36, %wire_rd32;
	st.global.u32 	[%wire_rd37], %wire_r102;
	mov.u32 	%wire_r70, -2;
	st.global.u32 	[%wire_rd43], %wire_r70;

WIRE_L__BB6_64:
	bra LEGACY_WIRE_DONE;


LEGACY_WIRE_DONE:
mov.u32 $0, 0;
}
"""
