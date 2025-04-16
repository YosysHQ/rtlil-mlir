module \top
  wire width 1 \in_from_prev_stall
  wire width 1 \out_stall
  wire width 32 \in_from_prev_pc
  wire width 32 \out_pc
  wire width 32 \in_from_prev_instruction
  wire width 32 \out_instruction
  wire width 1 \in_from_prev_stall$8
  wire width 1 \out_stall$9
  wire width 32 \in_from_prev_pc$10
  wire width 32 \out_pc$11
  wire width 7 \in_from_prev_op
  wire width 7 \out_op
  wire width 32 \in_from_prev_imm
  wire width 32 \out_imm
  wire width 32 \in_from_prev_rs1
  wire width 32 \out_rs1
  wire width 32 \in_from_prev_rs2
  wire width 32 \out_rs2
  wire width 5 \in_from_prev_which_rs2
  wire width 5 \out_which_rs2
  wire width 3 \in_from_prev_alu_op
  wire width 3 \out_alu_op
  wire width 2 \in_from_prev_alu_src1
  wire width 2 \out_alu_src1
  wire width 2 \in_from_prev_alu_src2
  wire width 2 \out_alu_src2
  wire width 1 \in_from_prev_jump_en
  wire width 1 \out_jump_en
  wire width 1 \in_from_prev_branch_en
  wire width 1 \out_branch_en
  wire width 3 \in_from_prev_branch_op
  wire width 3 \out_branch_op
  wire width 1 \in_from_prev_shamt_is_rs2
  wire width 1 \out_shamt_is_rs2
  wire width 1 \in_from_prev_sub
  wire width 1 \out_sub
  wire width 1 \in_from_prev_shift_arithmetic
  wire width 1 \out_shift_arithmetic
  wire width 2 \in_from_prev_mem_mode
  wire width 2 \out_mem_mode
  wire width 3 \in_from_prev_mem_width
  wire width 3 \out_mem_width
  wire width 5 \in_from_prev_which_rd
  wire width 5 \out_which_rd
  wire width 2 \in_from_prev_wrd_sel
  wire width 2 \out_wrd_sel
  wire width 1 \in_from_prev_stall$48
  wire width 1 \out_stall$49
  wire width 7 \in_from_prev_op$50
  wire width 7 \out_op$51
  wire width 32 \in_from_prev_rs2$52
  wire width 32 \out_rs2$53
  wire width 32 \in_from_prev_alu_out
  wire width 32 \out_alu_out
  wire width 32 \in_from_prev_incr_pc
  wire width 32 \out_incr_pc
  wire width 2 \in_from_prev_mem_mode$58
  wire width 2 \out_mem_mode$59
  wire width 3 \in_from_prev_mem_width$60
  wire width 3 \out_mem_width$61
  wire width 5 \in_from_prev_which_rd$62
  wire width 5 \out_which_rd$63
  wire width 2 \in_from_prev_wrd_sel$64
  wire width 2 \out_wrd_sel$65
  wire width 1 \in_from_prev_stall$66
  wire width 1 \out_stall$67
  wire width 7 \in_from_prev_op$68
  wire width 7 \out_op$69
  wire width 32 \in_from_prev_alu_out$70
  wire width 32 \out_alu_out$71
  wire width 32 \in_from_prev_incr_pc$72
  wire width 32 \out_incr_pc$73
  wire width 32 \in_from_prev_mem_read
  wire width 32 \out_mem_read
  wire width 5 \in_from_prev_which_rd$76
  wire width 5 \out_which_rd$77
  wire width 2 \in_from_prev_wrd_sel$78
  wire width 2 \out_wrd_sel$79
  wire width 1 \stall_because_read
  wire width 1 \stall_from_top
  wire width 1 \stall_from_top$82
  wire width 5 \which_rd_from_wb
  wire width 5 \which_rd
  wire width 32 \rd_from_wb
  wire width 32 \rd
  wire width 32 \from_ex
  wire width 32 \from_mem
  wire width 32 \rd$89
  wire width 32 \from_wb
  wire width 1 \ex_is_mem_read
  wire width 1 \true_stall
  wire width 1 \ex_is_jump
  wire width 1 \do_jump
  wire width 1 \stall_from_top_next
  wire width 1 \stall_from_top_next$96
  wire width 1 \stall_from_top_next$97
  wire width 32 \pc
  wire width 32 \new_pc
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 1 $1
  wire width 1 $2
  wire width 1 $3
  wire width 1 $4
  wire width 1 $5
  wire width 33 $6
  wire width 1 $7
  wire width 1 $8
  wire width 1 $9
  wire width 1 $10
  wire width 32 $11
  cell \top.fetch \fetch
    connect \clk \clk [0]
    connect \rst \rst [0]
    connect \true_stall \in_from_prev_stall [0]
    connect \stall_from_top \stall_from_top [0]
    connect \stall_from_top_next \stall_from_top_next [0]
    connect \pc \in_from_prev_pc [31:0]
  end
  cell \top.id \id
    connect \clk \clk [0]
    connect \rst \rst [0]
    connect \ex_is_mem_read \ex_is_mem_read [0]
    connect \ex_is_jump \ex_is_jump [0]
    connect \in_from_prev_stall \in_from_prev_stall [0]
    connect \true_stall \in_from_prev_stall$8 [0]
    connect \out_rs1 \in_from_prev_rs1 [31:0]
    connect \out_rs2 \in_from_prev_rs2 [31:0]
    connect \stall_from_top \stall_from_top$82 [0]
    connect \stall_from_top_next \stall_from_top_next$96 [0]
    connect \in_from_prev_pc \in_from_prev_pc [31:0]
    connect \out_which_rd \in_from_prev_which_rd [4:0]
    connect \stall_because_read \stall_because_read [0]
    connect \out_pc \in_from_prev_pc$10 [31:0]
    connect \out_op \in_from_prev_op [6:0]
    connect \out_mem_width \in_from_prev_mem_width [2:0]
    connect \which_rs2 \in_from_prev_which_rs2 [4:0]
    connect \out_imm \in_from_prev_imm [31:0]
    connect \out_alu_src1 \in_from_prev_alu_src1 [1:0]
    connect \out_alu_src2 \in_from_prev_alu_src2 [1:0]
    connect \out_alu_op \in_from_prev_alu_op [2:0]
    connect \out_mem_mode \in_from_prev_mem_mode [1:0]
    connect \out_wrd_sel \in_from_prev_wrd_sel [1:0]
    connect \out_branch_en \in_from_prev_branch_en [0]
    connect \out_branch_op \in_from_prev_branch_op [2:0]
    connect \out_jump_en \in_from_prev_jump_en [0]
    connect \out_shift_arithmetic \in_from_prev_shift_arithmetic [0]
    connect \out_shamt_is_rs2 \in_from_prev_shamt_is_rs2 [0]
    connect \out_sub \in_from_prev_sub [0]
    connect \from_ex \in_from_prev_alu_out [31:0]
    connect \from_mem \from_mem [31:0]
    connect \which_rd \which_rd_from_wb [4:0]
    connect \rd \rd_from_wb [31:0]
  end
  cell \top.ex \ex
    connect \clk \clk [0]
    connect \rst \rst [0]
    connect \in_from_prev_stall \in_from_prev_stall$8 [0]
    connect \in_from_prev_rs1 \in_from_prev_rs1 [31:0]
    connect \in_from_prev_rs2 \in_from_prev_rs2 [31:0]
    connect \true_stall \in_from_prev_stall$48 [0]
    connect \out_incr_pc \in_from_prev_incr_pc [31:0]
    connect \stall_from_top_next \stall_from_top_next$97 [0]
    connect \in_from_prev_which_rd \in_from_prev_which_rd [4:0]
    connect \in_from_prev_pc \in_from_prev_pc$10 [31:0]
    connect \in_from_prev_op \in_from_prev_op [6:0]
    connect \in_from_prev_mem_width \in_from_prev_mem_width [2:0]
    connect \in_from_prev_which_rs2 \in_from_prev_which_rs2 [4:0]
    connect \in_from_prev_imm \in_from_prev_imm [31:0]
    connect \in_from_prev_alu_src1 \in_from_prev_alu_src1 [1:0]
    connect \in_from_prev_alu_src2 \in_from_prev_alu_src2 [1:0]
    connect \in_from_prev_alu_op \in_from_prev_alu_op [2:0]
    connect \in_from_prev_mem_mode \in_from_prev_mem_mode [1:0]
    connect \in_from_prev_wrd_sel \in_from_prev_wrd_sel [1:0]
    connect \in_from_prev_branch_en \in_from_prev_branch_en [0]
    connect \in_from_prev_branch_op \in_from_prev_branch_op [2:0]
    connect \in_from_prev_jump_en \in_from_prev_jump_en [0]
    connect \in_from_prev_shift_arithmetic \in_from_prev_shift_arithmetic [0]
    connect \in_from_prev_shamt_is_rs2 \in_from_prev_shamt_is_rs2 [0]
    connect \in_from_prev_sub \in_from_prev_sub [0]
    connect \do_jump \do_jump [0]
    connect \out_op \in_from_prev_op$50 [6:0]
    connect \out_rs2 \in_from_prev_rs2$52 [31:0]
    connect \out_mem_mode \in_from_prev_mem_mode$58 [1:0]
    connect \out_mem_width \in_from_prev_mem_width$60 [2:0]
    connect \out_which_rd \in_from_prev_which_rd$62 [4:0]
    connect \out_wrd_sel \in_from_prev_wrd_sel$64 [1:0]
    connect \out_alu_out \in_from_prev_alu_out [31:0]
    connect \new_pc \new_pc [31:0]
  end
  cell \top.mem \mem
    connect \clk \clk [0]
    connect \rst \rst [0]
    connect \in_from_prev_stall \in_from_prev_stall$48 [0]
    connect \in_from_prev_incr_pc \in_from_prev_incr_pc [31:0]
    connect \true_stall \in_from_prev_stall$66 [0]
    connect \in_from_prev_op \in_from_prev_op$50 [6:0]
    connect \in_from_prev_rs2 \in_from_prev_rs2$52 [31:0]
    connect \in_from_prev_mem_mode \in_from_prev_mem_mode$58 [1:0]
    connect \in_from_prev_mem_width \in_from_prev_mem_width$60 [2:0]
    connect \in_from_prev_which_rd \in_from_prev_which_rd$62 [4:0]
    connect \in_from_prev_wrd_sel \in_from_prev_wrd_sel$64 [1:0]
    connect \in_from_prev_alu_out \in_from_prev_alu_out [31:0]
    connect \out_mem_read \in_from_prev_mem_read [31:0]
    connect \out_op \in_from_prev_op$68 [6:0]
    connect \out_alu_out \in_from_prev_alu_out$70 [31:0]
    connect \out_incr_pc \in_from_prev_incr_pc$72 [31:0]
    connect \out_which_rd \in_from_prev_which_rd$76 [4:0]
    connect \out_wrd_sel \in_from_prev_wrd_sel$78 [1:0]
    connect \rd \from_mem [31:0]
  end
  cell \top.wb \wb
    connect \clk \clk [0]
    connect \rst \rst [0]
    connect \in_from_prev_stall \in_from_prev_stall$66 [0]
    connect \in_from_prev_mem_read \in_from_prev_mem_read [31:0]
    connect \in_from_prev_op \in_from_prev_op$68 [6:0]
    connect \in_from_prev_alu_out \in_from_prev_alu_out$70 [31:0]
    connect \in_from_prev_incr_pc \in_from_prev_incr_pc$72 [31:0]
    connect \in_from_prev_which_rd \in_from_prev_which_rd$76 [4:0]
    connect \in_from_prev_wrd_sel \in_from_prev_wrd_sel$78 [1:0]
    connect \which_rd \which_rd_from_wb [4:0]
    connect \rd \rd_from_wb [31:0]
  end
  cell $ne $12
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \stall_because_read [0]
    connect \B {  }
    connect \Y $1
  end
  cell $eq $13
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 1
    connect \A \in_from_prev_mem_mode$58 [1:0]
    connect \B 2'10
    connect \Y $2
  end
  cell $not $14
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \in_from_prev_stall$48 [0]
    connect \Y $3
  end
  cell $and $15
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $2 [0]
    connect \B $3 [0]
    connect \Y \ex_is_mem_read
  end
  cell $not $16
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \in_from_prev_stall$48 [0]
    connect \Y $4
  end
  cell $and $17
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \do_jump [0]
    connect \B $4 [0]
    connect \Y \ex_is_jump
  end
  cell $ne $18
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \stall_because_read [0]
    connect \B {  }
    connect \Y $5
  end
  cell $add $19
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 3
    parameter \Y_WIDTH 33
    connect \A \in_from_prev_pc [31:0]
    connect \B 3'100
    connect \Y $6
  end
  cell $ne $20
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \do_jump [0]
    connect \B {  }
    connect \Y $7
  end
  process $21
    assign \stall_from_top [0] 1'0
    switch $1 [0]
      case 1'1
        assign \stall_from_top [0] 1'1
      case
    end
  end
  process $22
    assign \stall_from_top$82 [0] 1'0
    switch $1 [0]
      case 1'1
        assign \stall_from_top$82 [0] 1'1
      case
    end
  end
  process $23
    assign $8 [0] 1'0
    switch \rst [0]
      case 1'1
        assign $8 [0] 1'0
    end
  end
  cell $dff $24
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $8 [0]
    connect \CLK \clk [0]
    connect \Q \stall_from_top_next
  end
  process $25
    assign $9 [0] 1'0
    switch $7 [0]
      case 1'1
        assign $9 [0] 1'1
    end
    switch \rst [0]
      case 1'1
        assign $9 [0] 1'0
    end
  end
  cell $dff $26
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $9 [0]
    connect \CLK \clk [0]
    connect \Q \stall_from_top_next$96
  end
  process $27
    assign $10 [0] 1'0
    switch $7 [0]
      case 1'1
        assign $10 [0] 1'1
    end
    switch \rst [0]
      case 1'1
        assign $10 [0] 1'0
    end
  end
  cell $dff $28
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $10 [0]
    connect \CLK \clk [0]
    connect \Q \stall_from_top_next$97
  end
  process $29
    assign $11 [31:0] \in_from_prev_pc [31:0]
    switch $5 [0]
      case 1'1
      case
        assign $11 [31:0] $6 [31:0]
    end
    switch $7 [0]
      case 1'1
        assign $11 [31:0] \new_pc [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $11 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $30
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $11 [31:0]
    connect \CLK \clk [0]
    connect \Q \in_from_prev_pc
  end
connect \out_stall \in_from_prev_stall [0]
connect \out_pc \in_from_prev_pc [31:0]
connect \in_from_prev_instruction 32'00000000000000000000000000000000
connect \out_instruction 32'00000000000000000000000000000000
connect \out_stall$9 \in_from_prev_stall$8 [0]
connect \out_pc$11 \in_from_prev_pc$10 [31:0]
connect \out_op \in_from_prev_op [6:0]
connect \out_imm \in_from_prev_imm [31:0]
connect \out_rs1 \in_from_prev_rs1 [31:0]
connect \out_rs2 \in_from_prev_rs2 [31:0]
connect \out_which_rs2 \in_from_prev_which_rs2 [4:0]
connect \out_alu_op \in_from_prev_alu_op [2:0]
connect \out_alu_src1 \in_from_prev_alu_src1 [1:0]
connect \out_alu_src2 \in_from_prev_alu_src2 [1:0]
connect \out_jump_en \in_from_prev_jump_en [0]
connect \out_branch_en \in_from_prev_branch_en [0]
connect \out_branch_op \in_from_prev_branch_op [2:0]
connect \out_shamt_is_rs2 \in_from_prev_shamt_is_rs2 [0]
connect \out_sub \in_from_prev_sub [0]
connect \out_shift_arithmetic \in_from_prev_shift_arithmetic [0]
connect \out_mem_mode \in_from_prev_mem_mode [1:0]
connect \out_mem_width \in_from_prev_mem_width [2:0]
connect \out_which_rd \in_from_prev_which_rd [4:0]
connect \out_wrd_sel \in_from_prev_wrd_sel [1:0]
connect \out_stall$49 \in_from_prev_stall$48 [0]
connect \out_op$51 \in_from_prev_op$50 [6:0]
connect \out_rs2$53 \in_from_prev_rs2$52 [31:0]
connect \out_alu_out \in_from_prev_alu_out [31:0]
connect \out_incr_pc \in_from_prev_incr_pc [31:0]
connect \out_mem_mode$59 \in_from_prev_mem_mode$58 [1:0]
connect \out_mem_width$61 \in_from_prev_mem_width$60 [2:0]
connect \out_which_rd$63 \in_from_prev_which_rd$62 [4:0]
connect \out_wrd_sel$65 \in_from_prev_wrd_sel$64 [1:0]
connect \out_stall$67 \in_from_prev_stall$66 [0]
connect \out_op$69 \in_from_prev_op$68 [6:0]
connect \out_alu_out$71 \in_from_prev_alu_out$70 [31:0]
connect \out_incr_pc$73 \in_from_prev_incr_pc$72 [31:0]
connect \out_mem_read \in_from_prev_mem_read [31:0]
connect \out_which_rd$77 \in_from_prev_which_rd$76 [4:0]
connect \out_wrd_sel$79 \in_from_prev_wrd_sel$78 [1:0]
connect \which_rd \which_rd_from_wb [4:0]
connect \rd \rd_from_wb [31:0]
connect \from_ex \in_from_prev_alu_out [31:0]
connect \rd$89 \from_mem [31:0]
connect \from_wb \rd_from_wb [31:0]
connect \true_stall \in_from_prev_stall$48 [0]
connect \pc \in_from_prev_pc [31:0]
end
module \top.fetch
  wire width 1 \_in_stall
  wire width 1 \out_stall
  wire width 32 \addr
  wire width 32 \out_instruction
  wire width 32 \data
  wire width 32 \out_pc
  wire width 1 \in_from_prev_stall
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 1 output 2  \true_stall
  wire width 1 input 3  \stall_from_top
  wire width 1 input 4  \stall_from_top_next
  wire width 32 input 5  \pc
  wire width 1 $1
  wire width 1 $2
  wire width 1 $3
  cell $or $4
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \_in_stall [0]
    connect \B \stall_from_top_next [0]
    connect \Y $1
  end
  cell $or $5
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $1 [0]
    connect \B \stall_from_top [0]
    connect \Y \true_stall
  end
  cell $not $6
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \stall_from_top [0]
    connect \Y $2
  end
  process $7
    assign $3 [0] \_in_stall [0]
    switch $2 [0]
      case 1'1
        assign $3 [0] 1'0
    end
    switch \rst [0]
      case 1'1
        assign $3 [0] 1'0
    end
  end
  cell $dff $8
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $3 [0]
    connect \CLK \clk [0]
    connect \Q \_in_stall
  end
connect \out_stall \true_stall [0]
connect \addr { 2'00 \pc [31:2] }
connect \out_instruction 32'00000000000000000000000000000000
connect \data 32'00000000000000000000000000000000
connect \out_pc \pc [31:0]
connect \in_from_prev_stall 1'0
end
module \top.id
  wire width 1 \_in_stall
  wire width 1 \out_stall
  wire width 32 \_in_pc
  wire width 7 \op
  wire width 32 \instruction
  wire width 32 \_in_instruction
  wire width 32 \instruction$11
  wire width 32 \imm
  wire width 5 \which_rs1
  wire width 5 \which_rs1$15
  wire width 5 \which_rs2$17
  wire width 5 \which_rd_from_wb
  wire width 5 \which_rd$20
  wire width 1 \do_set
  wire width 1 \build_rd
  wire width 1 \build_rd$23
  wire width 32 \rd_from_wb
  wire width 2 \rs1_src
  wire width 2 \reg0_src
  wire width 2 \reg1_src
  wire width 2 \reg2_src
  wire width 2 \reg3_src
  wire width 2 \reg4_src
  wire width 2 \reg5_src
  wire width 2 \reg6_src
  wire width 2 \reg7_src
  wire width 2 \reg8_src
  wire width 2 \reg9_src
  wire width 2 \reg10_src
  wire width 2 \reg11_src
  wire width 2 \reg12_src
  wire width 2 \reg13_src
  wire width 2 \reg14_src
  wire width 2 \reg15_src
  wire width 2 \reg16_src
  wire width 2 \reg17_src
  wire width 2 \reg18_src
  wire width 2 \reg19_src
  wire width 2 \reg20_src
  wire width 2 \reg21_src
  wire width 2 \reg22_src
  wire width 2 \reg23_src
  wire width 2 \reg24_src
  wire width 2 \reg25_src
  wire width 2 \reg26_src
  wire width 2 \reg27_src
  wire width 2 \reg28_src
  wire width 2 \reg29_src
  wire width 2 \reg30_src
  wire width 2 \reg31_src
  wire width 2 \rs2_src
  wire width 32 \rs1
  wire width 32 \from_wb
  wire width 32 \rs2
  wire width 5 \out_which_rs2
  wire width 1 \write_rd_en
  wire width 1 \need_rs1
  wire width 1 \need_rs2
  wire width 2 \mem_mode
  wire width 3 \mem_width
  wire width 1 \jump_en
  wire width 1 \branch_en
  wire width 3 \branch_op
  wire width 1 \shamt_is_rs2
  wire width 1 \shift_arithmetic
  wire width 1 \sub
  wire width 2 \wrd_sel
  wire width 2 \alu_src1
  wire width 2 \alu_src2
  wire width 3 \alu_op
  wire width 32 \in_from_prev_instruction
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 1 input 2  \ex_is_mem_read
  wire width 1 input 3  \ex_is_jump
  wire width 1 input 4  \in_from_prev_stall
  wire width 1 output 5  \true_stall
  wire width 32 output 6  \out_rs1
  wire width 32 output 7  \out_rs2
  wire width 1 input 8  \stall_from_top
  wire width 1 input 9  \stall_from_top_next
  wire width 32 input 10  \in_from_prev_pc
  wire width 5 output 11  \out_which_rd
  wire width 1 output 12  \stall_because_read
  wire width 32 output 13  \out_pc
  wire width 7 output 14  \out_op
  wire width 3 output 15  \out_mem_width
  wire width 5 output 16  \which_rs2
  wire width 32 output 17  \out_imm
  wire width 2 output 18  \out_alu_src1
  wire width 2 output 19  \out_alu_src2
  wire width 3 output 20  \out_alu_op
  wire width 2 output 21  \out_mem_mode
  wire width 2 output 22  \out_wrd_sel
  wire width 1 output 23  \out_branch_en
  wire width 3 output 24  \out_branch_op
  wire width 1 output 25  \out_jump_en
  wire width 1 output 26  \out_shift_arithmetic
  wire width 1 output 27  \out_shamt_is_rs2
  wire width 1 output 28  \out_sub
  wire width 32 input 29  \from_ex
  wire width 32 input 30  \from_mem
  wire width 5 input 31  \which_rd
  wire width 32 input 32  \rd
  wire width 1 $1
  wire width 1 $2
  wire width 1 $3
  wire width 1 $4
  wire width 2 $5
  wire width 1 $6
  wire width 1 $7
  wire width 1 $8
  wire width 2 $9
  wire width 1 $10
  wire width 1 $11
  wire width 1 $12
  wire width 1 $13
  wire width 1 $14
  wire width 32 $15
  wire width 32 $16
  cell \top.id.regfile \regfile
    connect \clk \clk [0]
    connect \rst \rst [0]
    connect \rs1 \rs1 [31:0]
    connect \rs2 \rs2 [31:0]
    connect \port$254$15 \instruction [24:15]
    connect \which_rd \which_rd [4:0]
    connect \rd \rd [31:0]
  end
  cell \top.id.imm_decoder \imm_decoder
    connect \instruction \instruction [31:0]
    connect \imm \out_imm [31:0]
  end
  cell \top.id.controller \controller
    connect \instruction \instruction [31:0]
    connect \alu_src1 \out_alu_src1 [1:0]
    connect \alu_src2 \out_alu_src2 [1:0]
    connect \alu_op \out_alu_op [2:0]
    connect \mem_mode \out_mem_mode [1:0]
    connect \write_rd_en \write_rd_en [0]
    connect \wrd_sel \out_wrd_sel [1:0]
    connect \need_rs2 \need_rs2 [0]
    connect \build_rd \build_rd$23 [0]
    connect \branch_en \out_branch_en [0]
    connect \branch_op \out_branch_op [2:0]
    connect \jump_en \out_jump_en [0]
    connect \need_rs1 \need_rs1 [0]
    connect \shift_arithmetic \out_shift_arithmetic [0]
    connect \shamt_is_rs2 \out_shamt_is_rs2 [0]
    connect \sub \out_sub [0]
  end
  cell \top.id.forwarder \forwarder
    connect \clk \clk [0]
    connect \rst \rst [0]
    connect \do_set \do_set [0]
    connect \build_rd \build_rd [0]
    connect \which_rd \instruction [11:7]
    connect \reg0_src \reg0_src [1:0]
    connect \reg1_src \reg1_src [1:0]
    connect \reg2_src \reg2_src [1:0]
    connect \reg3_src \reg3_src [1:0]
    connect \reg4_src \reg4_src [1:0]
    connect \reg5_src \reg5_src [1:0]
    connect \reg6_src \reg6_src [1:0]
    connect \reg7_src \reg7_src [1:0]
    connect \reg8_src \reg8_src [1:0]
    connect \reg9_src \reg9_src [1:0]
    connect \reg10_src \reg10_src [1:0]
    connect \reg11_src \reg11_src [1:0]
    connect \reg12_src \reg12_src [1:0]
    connect \reg13_src \reg13_src [1:0]
    connect \reg14_src \reg14_src [1:0]
    connect \reg15_src \reg15_src [1:0]
    connect \reg16_src \reg16_src [1:0]
    connect \reg17_src \reg17_src [1:0]
    connect \reg18_src \reg18_src [1:0]
    connect \reg19_src \reg19_src [1:0]
    connect \reg20_src \reg20_src [1:0]
    connect \reg21_src \reg21_src [1:0]
    connect \reg22_src \reg22_src [1:0]
    connect \reg23_src \reg23_src [1:0]
    connect \reg24_src \reg24_src [1:0]
    connect \reg25_src \reg25_src [1:0]
    connect \reg26_src \reg26_src [1:0]
    connect \reg27_src \reg27_src [1:0]
    connect \reg28_src \reg28_src [1:0]
    connect \reg29_src \reg29_src [1:0]
    connect \reg30_src \reg30_src [1:0]
    connect \reg31_src \reg31_src [1:0]
  end
  cell $or $17
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \_in_stall [0]
    connect \B \stall_from_top_next [0]
    connect \Y $1
  end
  cell $or $18
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $1 [0]
    connect \B \stall_from_top [0]
    connect \Y \true_stall
  end
  cell $not $19
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \_in_stall [0]
    connect \Y \do_set
  end
  cell $not $20
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \ex_is_jump [0]
    connect \Y $2
  end
  cell $and $21
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \build_rd$23 [0]
    connect \B $2 [0]
    connect \Y $3
  end
  cell $not $22
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \true_stall [0]
    connect \Y $4
  end
  cell $and $23
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $3 [0]
    connect \B $4 [0]
    connect \Y \build_rd
  end
  process $24
    assign \rs1_src [1:0] 2'00
    switch \instruction [19:15]
      case 5'00000
        assign \rs1_src [1:0] \reg0_src [1:0]
      case 5'00001
        assign \rs1_src [1:0] \reg1_src [1:0]
      case 5'00010
        assign \rs1_src [1:0] \reg2_src [1:0]
      case 5'00011
        assign \rs1_src [1:0] \reg3_src [1:0]
      case 5'00100
        assign \rs1_src [1:0] \reg4_src [1:0]
      case 5'00101
        assign \rs1_src [1:0] \reg5_src [1:0]
      case 5'00110
        assign \rs1_src [1:0] \reg6_src [1:0]
      case 5'00111
        assign \rs1_src [1:0] \reg7_src [1:0]
      case 5'01000
        assign \rs1_src [1:0] \reg8_src [1:0]
      case 5'01001
        assign \rs1_src [1:0] \reg9_src [1:0]
      case 5'01010
        assign \rs1_src [1:0] \reg10_src [1:0]
      case 5'01011
        assign \rs1_src [1:0] \reg11_src [1:0]
      case 5'01100
        assign \rs1_src [1:0] \reg12_src [1:0]
      case 5'01101
        assign \rs1_src [1:0] \reg13_src [1:0]
      case 5'01110
        assign \rs1_src [1:0] \reg14_src [1:0]
      case 5'01111
        assign \rs1_src [1:0] \reg15_src [1:0]
      case 5'10000
        assign \rs1_src [1:0] \reg16_src [1:0]
      case 5'10001
        assign \rs1_src [1:0] \reg17_src [1:0]
      case 5'10010
        assign \rs1_src [1:0] \reg18_src [1:0]
      case 5'10011
        assign \rs1_src [1:0] \reg19_src [1:0]
      case 5'10100
        assign \rs1_src [1:0] \reg20_src [1:0]
      case 5'10101
        assign \rs1_src [1:0] \reg21_src [1:0]
      case 5'10110
        assign \rs1_src [1:0] \reg22_src [1:0]
      case 5'10111
        assign \rs1_src [1:0] \reg23_src [1:0]
      case 5'11000
        assign \rs1_src [1:0] \reg24_src [1:0]
      case 5'11001
        assign \rs1_src [1:0] \reg25_src [1:0]
      case 5'11010
        assign \rs1_src [1:0] \reg26_src [1:0]
      case 5'11011
        assign \rs1_src [1:0] \reg27_src [1:0]
      case 5'11100
        assign \rs1_src [1:0] \reg28_src [1:0]
      case 5'11101
        assign \rs1_src [1:0] \reg29_src [1:0]
      case 5'11110
        assign \rs1_src [1:0] \reg30_src [1:0]
      case 5'11111
        assign \rs1_src [1:0] \reg31_src [1:0]
    end
  end
  process $25
    assign \rs2_src [1:0] 2'00
    switch \instruction [24:20]
      case 5'00000
        assign \rs2_src [1:0] \reg0_src [1:0]
      case 5'00001
        assign \rs2_src [1:0] \reg1_src [1:0]
      case 5'00010
        assign \rs2_src [1:0] \reg2_src [1:0]
      case 5'00011
        assign \rs2_src [1:0] \reg3_src [1:0]
      case 5'00100
        assign \rs2_src [1:0] \reg4_src [1:0]
      case 5'00101
        assign \rs2_src [1:0] \reg5_src [1:0]
      case 5'00110
        assign \rs2_src [1:0] \reg6_src [1:0]
      case 5'00111
        assign \rs2_src [1:0] \reg7_src [1:0]
      case 5'01000
        assign \rs2_src [1:0] \reg8_src [1:0]
      case 5'01001
        assign \rs2_src [1:0] \reg9_src [1:0]
      case 5'01010
        assign \rs2_src [1:0] \reg10_src [1:0]
      case 5'01011
        assign \rs2_src [1:0] \reg11_src [1:0]
      case 5'01100
        assign \rs2_src [1:0] \reg12_src [1:0]
      case 5'01101
        assign \rs2_src [1:0] \reg13_src [1:0]
      case 5'01110
        assign \rs2_src [1:0] \reg14_src [1:0]
      case 5'01111
        assign \rs2_src [1:0] \reg15_src [1:0]
      case 5'10000
        assign \rs2_src [1:0] \reg16_src [1:0]
      case 5'10001
        assign \rs2_src [1:0] \reg17_src [1:0]
      case 5'10010
        assign \rs2_src [1:0] \reg18_src [1:0]
      case 5'10011
        assign \rs2_src [1:0] \reg19_src [1:0]
      case 5'10100
        assign \rs2_src [1:0] \reg20_src [1:0]
      case 5'10101
        assign \rs2_src [1:0] \reg21_src [1:0]
      case 5'10110
        assign \rs2_src [1:0] \reg22_src [1:0]
      case 5'10111
        assign \rs2_src [1:0] \reg23_src [1:0]
      case 5'11000
        assign \rs2_src [1:0] \reg24_src [1:0]
      case 5'11001
        assign \rs2_src [1:0] \reg25_src [1:0]
      case 5'11010
        assign \rs2_src [1:0] \reg26_src [1:0]
      case 5'11011
        assign \rs2_src [1:0] \reg27_src [1:0]
      case 5'11100
        assign \rs2_src [1:0] \reg28_src [1:0]
      case 5'11101
        assign \rs2_src [1:0] \reg29_src [1:0]
      case 5'11110
        assign \rs2_src [1:0] \reg30_src [1:0]
      case 5'11111
        assign \rs2_src [1:0] \reg31_src [1:0]
    end
  end
  process $26
    assign \out_rs1 [31:0] 32'00000000000000000000000000000000
    switch \rs1_src [1:0]
      case 2'00
        assign \out_rs1 [31:0] \rs1 [31:0]
      case 2'01
        assign \out_rs1 [31:0] \rd [31:0]
      case 2'10
        assign \out_rs1 [31:0] \from_mem [31:0]
      case 2'11
        assign \out_rs1 [31:0] \from_ex [31:0]
    end
  end
  process $27
    assign \out_rs2 [31:0] 32'00000000000000000000000000000000
    switch \rs2_src [1:0]
      case 2'00
        assign \out_rs2 [31:0] \rs2 [31:0]
      case 2'01
        assign \out_rs2 [31:0] \rd [31:0]
      case 2'10
        assign \out_rs2 [31:0] \from_mem [31:0]
      case 2'11
        assign \out_rs2 [31:0] \from_ex [31:0]
    end
  end
  process $28
    assign $5 [1:0] 2'00
    switch \instruction [19:15]
      case 5'00000
        assign $5 [1:0] \reg0_src [1:0]
      case 5'00001
        assign $5 [1:0] \reg1_src [1:0]
      case 5'00010
        assign $5 [1:0] \reg2_src [1:0]
      case 5'00011
        assign $5 [1:0] \reg3_src [1:0]
      case 5'00100
        assign $5 [1:0] \reg4_src [1:0]
      case 5'00101
        assign $5 [1:0] \reg5_src [1:0]
      case 5'00110
        assign $5 [1:0] \reg6_src [1:0]
      case 5'00111
        assign $5 [1:0] \reg7_src [1:0]
      case 5'01000
        assign $5 [1:0] \reg8_src [1:0]
      case 5'01001
        assign $5 [1:0] \reg9_src [1:0]
      case 5'01010
        assign $5 [1:0] \reg10_src [1:0]
      case 5'01011
        assign $5 [1:0] \reg11_src [1:0]
      case 5'01100
        assign $5 [1:0] \reg12_src [1:0]
      case 5'01101
        assign $5 [1:0] \reg13_src [1:0]
      case 5'01110
        assign $5 [1:0] \reg14_src [1:0]
      case 5'01111
        assign $5 [1:0] \reg15_src [1:0]
      case 5'10000
        assign $5 [1:0] \reg16_src [1:0]
      case 5'10001
        assign $5 [1:0] \reg17_src [1:0]
      case 5'10010
        assign $5 [1:0] \reg18_src [1:0]
      case 5'10011
        assign $5 [1:0] \reg19_src [1:0]
      case 5'10100
        assign $5 [1:0] \reg20_src [1:0]
      case 5'10101
        assign $5 [1:0] \reg21_src [1:0]
      case 5'10110
        assign $5 [1:0] \reg22_src [1:0]
      case 5'10111
        assign $5 [1:0] \reg23_src [1:0]
      case 5'11000
        assign $5 [1:0] \reg24_src [1:0]
      case 5'11001
        assign $5 [1:0] \reg25_src [1:0]
      case 5'11010
        assign $5 [1:0] \reg26_src [1:0]
      case 5'11011
        assign $5 [1:0] \reg27_src [1:0]
      case 5'11100
        assign $5 [1:0] \reg28_src [1:0]
      case 5'11101
        assign $5 [1:0] \reg29_src [1:0]
      case 5'11110
        assign $5 [1:0] \reg30_src [1:0]
      case 5'11111
        assign $5 [1:0] \reg31_src [1:0]
    end
  end
  cell $eq $29
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 1
    connect \A $5 [1:0]
    connect \B 2'11
    connect \Y $6
  end
  cell $eq $30
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \need_rs1 [0]
    connect \B 1'1
    connect \Y $7
  end
  cell $and $31
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $6 [0]
    connect \B $7 [0]
    connect \Y $8
  end
  process $32
    assign $9 [1:0] 2'00
    switch \instruction [24:20]
      case 5'00000
        assign $9 [1:0] \reg0_src [1:0]
      case 5'00001
        assign $9 [1:0] \reg1_src [1:0]
      case 5'00010
        assign $9 [1:0] \reg2_src [1:0]
      case 5'00011
        assign $9 [1:0] \reg3_src [1:0]
      case 5'00100
        assign $9 [1:0] \reg4_src [1:0]
      case 5'00101
        assign $9 [1:0] \reg5_src [1:0]
      case 5'00110
        assign $9 [1:0] \reg6_src [1:0]
      case 5'00111
        assign $9 [1:0] \reg7_src [1:0]
      case 5'01000
        assign $9 [1:0] \reg8_src [1:0]
      case 5'01001
        assign $9 [1:0] \reg9_src [1:0]
      case 5'01010
        assign $9 [1:0] \reg10_src [1:0]
      case 5'01011
        assign $9 [1:0] \reg11_src [1:0]
      case 5'01100
        assign $9 [1:0] \reg12_src [1:0]
      case 5'01101
        assign $9 [1:0] \reg13_src [1:0]
      case 5'01110
        assign $9 [1:0] \reg14_src [1:0]
      case 5'01111
        assign $9 [1:0] \reg15_src [1:0]
      case 5'10000
        assign $9 [1:0] \reg16_src [1:0]
      case 5'10001
        assign $9 [1:0] \reg17_src [1:0]
      case 5'10010
        assign $9 [1:0] \reg18_src [1:0]
      case 5'10011
        assign $9 [1:0] \reg19_src [1:0]
      case 5'10100
        assign $9 [1:0] \reg20_src [1:0]
      case 5'10101
        assign $9 [1:0] \reg21_src [1:0]
      case 5'10110
        assign $9 [1:0] \reg22_src [1:0]
      case 5'10111
        assign $9 [1:0] \reg23_src [1:0]
      case 5'11000
        assign $9 [1:0] \reg24_src [1:0]
      case 5'11001
        assign $9 [1:0] \reg25_src [1:0]
      case 5'11010
        assign $9 [1:0] \reg26_src [1:0]
      case 5'11011
        assign $9 [1:0] \reg27_src [1:0]
      case 5'11100
        assign $9 [1:0] \reg28_src [1:0]
      case 5'11101
        assign $9 [1:0] \reg29_src [1:0]
      case 5'11110
        assign $9 [1:0] \reg30_src [1:0]
      case 5'11111
        assign $9 [1:0] \reg31_src [1:0]
    end
  end
  cell $eq $33
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 1
    connect \A $9 [1:0]
    connect \B 2'11
    connect \Y $10
  end
  cell $eq $34
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \need_rs2 [0]
    connect \B 1'1
    connect \Y $11
  end
  cell $and $35
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $10 [0]
    connect \B $11 [0]
    connect \Y $12
  end
  cell $not $36
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \stall_from_top [0]
    connect \Y $13
  end
  process $37
    assign \out_which_rd [4:0] 5'00000
    switch \write_rd_en [0]
      case 1'1
        assign \out_which_rd [4:0] \instruction [11:7]
    end
  end
  process $38
    assign \stall_because_read [0] 1'0
    switch \ex_is_mem_read [0]
      case 1'1
        switch $8 [0]
          case 1'1
            assign \stall_because_read [0] 1'1
        end
        switch $12 [0]
          case 1'1
            assign \stall_because_read [0] 1'1
        end
    end
  end
  process $39
    assign $14 [0] \_in_stall [0]
    switch $13 [0]
      case 1'1
        assign $14 [0] \in_from_prev_stall [0]
    end
    switch \rst [0]
      case 1'1
        assign $14 [0] 1'0
    end
  end
  cell $dff $40
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $14 [0]
    connect \CLK \clk [0]
    connect \Q \_in_stall
  end
  process $41
    assign $15 [31:0] \out_pc [31:0]
    switch $13 [0]
      case 1'1
        assign $15 [31:0] \in_from_prev_pc [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $15 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $42
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $15 [31:0]
    connect \CLK \clk [0]
    connect \Q \out_pc
  end
  process $43
    assign $16 [31:0] \instruction [31:0]
    switch $13 [0]
      case 1'1
        assign $16 [31:0] 32'00000000000000000000000000000000
    end
    switch \rst [0]
      case 1'1
        assign $16 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $44
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $16 [31:0]
    connect \CLK \clk [0]
    connect \Q \instruction
  end
connect \out_stall \true_stall [0]
connect \_in_pc \out_pc [31:0]
connect \op \instruction [6:0]
connect \_in_instruction \instruction [31:0]
connect \instruction$11 \instruction [31:0]
connect \imm \out_imm [31:0]
connect \which_rs1 \instruction [19:15]
connect \which_rs1$15 \instruction [19:15]
connect \which_rs2$17 \instruction [24:20]
connect \which_rd_from_wb \which_rd [4:0]
connect \which_rd$20 \instruction [11:7]
connect \rd_from_wb \rd [31:0]
connect \from_wb \rd [31:0]
connect \out_which_rs2 \instruction [24:20]
connect \mem_mode \out_mem_mode [1:0]
connect \mem_width \instruction [14:12]
connect \jump_en \out_jump_en [0]
connect \branch_en \out_branch_en [0]
connect \branch_op \out_branch_op [2:0]
connect \shamt_is_rs2 \out_shamt_is_rs2 [0]
connect \shift_arithmetic \out_shift_arithmetic [0]
connect \sub \out_sub [0]
connect \wrd_sel \out_wrd_sel [1:0]
connect \alu_src1 \out_alu_src1 [1:0]
connect \alu_src2 \out_alu_src2 [1:0]
connect \alu_op \out_alu_op [2:0]
connect \in_from_prev_instruction 32'00000000000000000000000000000000
connect \out_op \instruction [6:0]
connect \out_mem_width \instruction [14:12]
connect \which_rs2 \instruction [24:20]
end
module \top.id.regfile
  wire width 5 \which_rs1
  wire width 32 \reg0
  wire width 32 \reg1
  wire width 32 \reg2
  wire width 32 \reg3
  wire width 32 \reg4
  wire width 32 \reg5
  wire width 32 \reg6
  wire width 32 \reg7
  wire width 32 \reg8
  wire width 32 \reg9
  wire width 32 \reg10
  wire width 32 \reg11
  wire width 32 \reg12
  wire width 32 \reg13
  wire width 32 \reg14
  wire width 32 \reg15
  wire width 32 \reg16
  wire width 32 \reg17
  wire width 32 \reg18
  wire width 32 \reg19
  wire width 32 \reg20
  wire width 32 \reg21
  wire width 32 \reg22
  wire width 32 \reg23
  wire width 32 \reg24
  wire width 32 \reg25
  wire width 32 \reg26
  wire width 32 \reg27
  wire width 32 \reg28
  wire width 32 \reg29
  wire width 32 \reg30
  wire width 32 \reg31
  wire width 5 \which_rs2
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 32 output 2  \rs1
  wire width 32 output 3  \rs2
  wire width 10 input 4  \port$254$15
  wire width 5 input 5  \which_rd
  wire width 32 input 6  \rd
  wire width 1 $1
  wire width 32 $2
  wire width 32 $3
  wire width 32 $4
  wire width 32 $5
  wire width 32 $6
  wire width 32 $7
  wire width 32 $8
  wire width 32 $9
  wire width 32 $10
  wire width 32 $11
  wire width 32 $12
  wire width 32 $13
  wire width 32 $14
  wire width 32 $15
  wire width 32 $16
  wire width 32 $17
  wire width 32 $18
  wire width 32 $19
  wire width 32 $20
  wire width 32 $21
  wire width 32 $22
  wire width 32 $23
  wire width 32 $24
  wire width 32 $25
  wire width 32 $26
  wire width 32 $27
  wire width 32 $28
  wire width 32 $29
  wire width 32 $30
  wire width 32 $31
  wire width 32 $32
  wire width 32 $33
  process $34
    assign \rs1 [31:0] 32'00000000000000000000000000000000
    switch \port$254$15 [4:0]
      case 5'00000
        assign \rs1 [31:0] \reg0 [31:0]
      case 5'00001
        assign \rs1 [31:0] \reg1 [31:0]
      case 5'00010
        assign \rs1 [31:0] \reg2 [31:0]
      case 5'00011
        assign \rs1 [31:0] \reg3 [31:0]
      case 5'00100
        assign \rs1 [31:0] \reg4 [31:0]
      case 5'00101
        assign \rs1 [31:0] \reg5 [31:0]
      case 5'00110
        assign \rs1 [31:0] \reg6 [31:0]
      case 5'00111
        assign \rs1 [31:0] \reg7 [31:0]
      case 5'01000
        assign \rs1 [31:0] \reg8 [31:0]
      case 5'01001
        assign \rs1 [31:0] \reg9 [31:0]
      case 5'01010
        assign \rs1 [31:0] \reg10 [31:0]
      case 5'01011
        assign \rs1 [31:0] \reg11 [31:0]
      case 5'01100
        assign \rs1 [31:0] \reg12 [31:0]
      case 5'01101
        assign \rs1 [31:0] \reg13 [31:0]
      case 5'01110
        assign \rs1 [31:0] \reg14 [31:0]
      case 5'01111
        assign \rs1 [31:0] \reg15 [31:0]
      case 5'10000
        assign \rs1 [31:0] \reg16 [31:0]
      case 5'10001
        assign \rs1 [31:0] \reg17 [31:0]
      case 5'10010
        assign \rs1 [31:0] \reg18 [31:0]
      case 5'10011
        assign \rs1 [31:0] \reg19 [31:0]
      case 5'10100
        assign \rs1 [31:0] \reg20 [31:0]
      case 5'10101
        assign \rs1 [31:0] \reg21 [31:0]
      case 5'10110
        assign \rs1 [31:0] \reg22 [31:0]
      case 5'10111
        assign \rs1 [31:0] \reg23 [31:0]
      case 5'11000
        assign \rs1 [31:0] \reg24 [31:0]
      case 5'11001
        assign \rs1 [31:0] \reg25 [31:0]
      case 5'11010
        assign \rs1 [31:0] \reg26 [31:0]
      case 5'11011
        assign \rs1 [31:0] \reg27 [31:0]
      case 5'11100
        assign \rs1 [31:0] \reg28 [31:0]
      case 5'11101
        assign \rs1 [31:0] \reg29 [31:0]
      case 5'11110
        assign \rs1 [31:0] \reg30 [31:0]
      case 5'11111
        assign \rs1 [31:0] \reg31 [31:0]
    end
  end
  process $35
    assign \rs2 [31:0] 32'00000000000000000000000000000000
    switch \port$254$15 [9:5]
      case 5'00000
        assign \rs2 [31:0] \reg0 [31:0]
      case 5'00001
        assign \rs2 [31:0] \reg1 [31:0]
      case 5'00010
        assign \rs2 [31:0] \reg2 [31:0]
      case 5'00011
        assign \rs2 [31:0] \reg3 [31:0]
      case 5'00100
        assign \rs2 [31:0] \reg4 [31:0]
      case 5'00101
        assign \rs2 [31:0] \reg5 [31:0]
      case 5'00110
        assign \rs2 [31:0] \reg6 [31:0]
      case 5'00111
        assign \rs2 [31:0] \reg7 [31:0]
      case 5'01000
        assign \rs2 [31:0] \reg8 [31:0]
      case 5'01001
        assign \rs2 [31:0] \reg9 [31:0]
      case 5'01010
        assign \rs2 [31:0] \reg10 [31:0]
      case 5'01011
        assign \rs2 [31:0] \reg11 [31:0]
      case 5'01100
        assign \rs2 [31:0] \reg12 [31:0]
      case 5'01101
        assign \rs2 [31:0] \reg13 [31:0]
      case 5'01110
        assign \rs2 [31:0] \reg14 [31:0]
      case 5'01111
        assign \rs2 [31:0] \reg15 [31:0]
      case 5'10000
        assign \rs2 [31:0] \reg16 [31:0]
      case 5'10001
        assign \rs2 [31:0] \reg17 [31:0]
      case 5'10010
        assign \rs2 [31:0] \reg18 [31:0]
      case 5'10011
        assign \rs2 [31:0] \reg19 [31:0]
      case 5'10100
        assign \rs2 [31:0] \reg20 [31:0]
      case 5'10101
        assign \rs2 [31:0] \reg21 [31:0]
      case 5'10110
        assign \rs2 [31:0] \reg22 [31:0]
      case 5'10111
        assign \rs2 [31:0] \reg23 [31:0]
      case 5'11000
        assign \rs2 [31:0] \reg24 [31:0]
      case 5'11001
        assign \rs2 [31:0] \reg25 [31:0]
      case 5'11010
        assign \rs2 [31:0] \reg26 [31:0]
      case 5'11011
        assign \rs2 [31:0] \reg27 [31:0]
      case 5'11100
        assign \rs2 [31:0] \reg28 [31:0]
      case 5'11101
        assign \rs2 [31:0] \reg29 [31:0]
      case 5'11110
        assign \rs2 [31:0] \reg30 [31:0]
      case 5'11111
        assign \rs2 [31:0] \reg31 [31:0]
    end
  end
  cell $ne $36
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 5
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \which_rd [4:0]
    connect \B {  }
    connect \Y $1
  end
  process $37
    assign $2 [31:0] \reg0 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
            assign $2 [31:0] \rd [31:0]
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $2 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $38
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $2 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg0
  end
  process $39
    assign $3 [31:0] \reg1 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
            assign $3 [31:0] \rd [31:0]
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $3 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $40
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $3 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg1
  end
  process $41
    assign $4 [31:0] \reg2 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
            assign $4 [31:0] \rd [31:0]
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $4 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $42
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $4 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg2
  end
  process $43
    assign $5 [31:0] \reg3 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
            assign $5 [31:0] \rd [31:0]
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $5 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $44
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $5 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg3
  end
  process $45
    assign $6 [31:0] \reg4 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
            assign $6 [31:0] \rd [31:0]
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $6 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $46
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $6 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg4
  end
  process $47
    assign $7 [31:0] \reg5 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
            assign $7 [31:0] \rd [31:0]
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $7 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $48
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $7 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg5
  end
  process $49
    assign $8 [31:0] \reg6 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
            assign $8 [31:0] \rd [31:0]
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $8 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $50
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $8 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg6
  end
  process $51
    assign $9 [31:0] \reg7 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
            assign $9 [31:0] \rd [31:0]
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $9 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $52
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $9 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg7
  end
  process $53
    assign $10 [31:0] \reg8 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
            assign $10 [31:0] \rd [31:0]
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $10 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $54
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $10 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg8
  end
  process $55
    assign $11 [31:0] \reg9 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
            assign $11 [31:0] \rd [31:0]
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $11 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $56
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $11 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg9
  end
  process $57
    assign $12 [31:0] \reg10 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
            assign $12 [31:0] \rd [31:0]
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $12 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $58
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $12 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg10
  end
  process $59
    assign $13 [31:0] \reg11 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
            assign $13 [31:0] \rd [31:0]
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $13 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $60
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $13 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg11
  end
  process $61
    assign $14 [31:0] \reg12 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
            assign $14 [31:0] \rd [31:0]
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $14 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $62
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $14 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg12
  end
  process $63
    assign $15 [31:0] \reg13 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
            assign $15 [31:0] \rd [31:0]
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $15 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $64
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $15 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg13
  end
  process $65
    assign $16 [31:0] \reg14 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
            assign $16 [31:0] \rd [31:0]
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $16 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $66
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $16 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg14
  end
  process $67
    assign $17 [31:0] \reg15 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
            assign $17 [31:0] \rd [31:0]
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $17 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $68
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $17 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg15
  end
  process $69
    assign $18 [31:0] \reg16 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
            assign $18 [31:0] \rd [31:0]
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $18 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $70
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $18 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg16
  end
  process $71
    assign $19 [31:0] \reg17 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
            assign $19 [31:0] \rd [31:0]
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $19 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $72
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $19 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg17
  end
  process $73
    assign $20 [31:0] \reg18 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
            assign $20 [31:0] \rd [31:0]
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $20 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $74
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $20 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg18
  end
  process $75
    assign $21 [31:0] \reg19 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
            assign $21 [31:0] \rd [31:0]
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $21 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $76
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $21 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg19
  end
  process $77
    assign $22 [31:0] \reg20 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
            assign $22 [31:0] \rd [31:0]
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $22 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $78
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $22 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg20
  end
  process $79
    assign $23 [31:0] \reg21 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
            assign $23 [31:0] \rd [31:0]
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $23 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $80
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $23 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg21
  end
  process $81
    assign $24 [31:0] \reg22 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
            assign $24 [31:0] \rd [31:0]
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $24 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $82
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $24 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg22
  end
  process $83
    assign $25 [31:0] \reg23 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
            assign $25 [31:0] \rd [31:0]
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $25 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $84
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $25 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg23
  end
  process $85
    assign $26 [31:0] \reg24 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
            assign $26 [31:0] \rd [31:0]
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $26 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $86
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $26 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg24
  end
  process $87
    assign $27 [31:0] \reg25 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
            assign $27 [31:0] \rd [31:0]
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $27 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $88
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $27 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg25
  end
  process $89
    assign $28 [31:0] \reg26 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
            assign $28 [31:0] \rd [31:0]
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $28 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $90
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $28 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg26
  end
  process $91
    assign $29 [31:0] \reg27 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
            assign $29 [31:0] \rd [31:0]
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $29 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $92
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $29 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg27
  end
  process $93
    assign $30 [31:0] \reg28 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
            assign $30 [31:0] \rd [31:0]
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $30 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $94
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $30 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg28
  end
  process $95
    assign $31 [31:0] \reg29 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
            assign $31 [31:0] \rd [31:0]
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $31 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $96
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $31 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg29
  end
  process $97
    assign $32 [31:0] \reg30 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
            assign $32 [31:0] \rd [31:0]
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $32 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $98
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $32 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg30
  end
  process $99
    assign $33 [31:0] \reg31 [31:0]
    switch $1 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
            assign $33 [31:0] \rd [31:0]
        end
    end
    switch \rst [0]
      case 1'1
        assign $33 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $100
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $33 [31:0]
    connect \CLK \clk [0]
    connect \Q \reg31
  end
connect \which_rs1 \port$254$15 [4:0]
connect \which_rs2 \port$254$15 [9:5]
end
module \top.id.imm_decoder
  wire width 7 \op
  wire width 1 \invalid
  wire width 32 input 0  \instruction
  wire width 32 output 1  \imm
  wire width 1 $1
  wire width 1 $2
  wire width 1 $3
  wire width 1 $4
  wire width 1 $5
  wire width 1 $6
  wire width 1 $7
  wire width 1 $8
  wire width 1 $9
  wire width 1 $10
  cell $eq $11
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 2'11
    connect \Y $1
  end
  cell $eq $12
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 6
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 6'100011
    connect \Y $2
  end
  cell $eq $13
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 7
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 7'1100011
    connect \Y $3
  end
  cell $eq $14
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 7
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 7'1100111
    connect \Y $4
  end
  cell $eq $15
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 7
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 7'1101111
    connect \Y $5
  end
  cell $eq $16
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 5
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 5'10011
    connect \Y $6
  end
  cell $eq $17
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 6
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 6'110011
    connect \Y $7
  end
  cell $eq $18
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 5
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 5'10111
    connect \Y $8
  end
  cell $eq $19
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 6
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 6'110111
    connect \Y $9
  end
  cell $eq $20
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 7
    parameter \B_WIDTH 7
    parameter \Y_WIDTH 1
    connect \A \instruction [6:0]
    connect \B 7'1110011
    connect \Y $10
  end
  process $21
    assign \imm [31:0] 32'00000000000000000000000000000000
    switch $1 [0]
      case 1'1
        assign \imm [0] \instruction [20]
        assign \imm [1] \instruction [21]
        assign \imm [2] \instruction [22]
        assign \imm [3] \instruction [23]
        assign \imm [4] \instruction [24]
        assign \imm [5] \instruction [25]
        assign \imm [6] \instruction [26]
        assign \imm [7] \instruction [27]
        assign \imm [8] \instruction [28]
        assign \imm [9] \instruction [29]
        assign \imm [10] \instruction [30]
        assign \imm [11] \instruction [31]
        assign \imm [12] \instruction [31]
        assign \imm [13] \instruction [31]
        assign \imm [14] \instruction [31]
        assign \imm [15] \instruction [31]
        assign \imm [16] \instruction [31]
        assign \imm [17] \instruction [31]
        assign \imm [18] \instruction [31]
        assign \imm [19] \instruction [31]
        assign \imm [20] \instruction [31]
        assign \imm [21] \instruction [31]
        assign \imm [22] \instruction [31]
        assign \imm [23] \instruction [31]
        assign \imm [24] \instruction [31]
        assign \imm [25] \instruction [31]
        assign \imm [26] \instruction [31]
        assign \imm [27] \instruction [31]
        assign \imm [28] \instruction [31]
        assign \imm [29] \instruction [31]
        assign \imm [30] \instruction [31]
        assign \imm [31] \instruction [31]
    end
    switch $2 [0]
      case 1'1
        assign \imm [0] \instruction [7]
        assign \imm [1] \instruction [8]
        assign \imm [2] \instruction [9]
        assign \imm [3] \instruction [10]
        assign \imm [4] \instruction [11]
        assign \imm [5] \instruction [25]
        assign \imm [6] \instruction [26]
        assign \imm [7] \instruction [27]
        assign \imm [8] \instruction [28]
        assign \imm [9] \instruction [29]
        assign \imm [10] \instruction [30]
        assign \imm [11] \instruction [31]
        assign \imm [12] \instruction [31]
        assign \imm [13] \instruction [31]
        assign \imm [14] \instruction [31]
        assign \imm [15] \instruction [31]
        assign \imm [16] \instruction [31]
        assign \imm [17] \instruction [31]
        assign \imm [18] \instruction [31]
        assign \imm [19] \instruction [31]
        assign \imm [20] \instruction [31]
        assign \imm [21] \instruction [31]
        assign \imm [22] \instruction [31]
        assign \imm [23] \instruction [31]
        assign \imm [24] \instruction [31]
        assign \imm [25] \instruction [31]
        assign \imm [26] \instruction [31]
        assign \imm [27] \instruction [31]
        assign \imm [28] \instruction [31]
        assign \imm [29] \instruction [31]
        assign \imm [30] \instruction [31]
        assign \imm [31] \instruction [31]
    end
    switch $3 [0]
      case 1'1
        assign \imm [1] \instruction [8]
        assign \imm [2] \instruction [9]
        assign \imm [3] \instruction [10]
        assign \imm [4] \instruction [11]
        assign \imm [5] \instruction [25]
        assign \imm [6] \instruction [26]
        assign \imm [7] \instruction [27]
        assign \imm [8] \instruction [28]
        assign \imm [9] \instruction [29]
        assign \imm [10] \instruction [30]
        assign \imm [11] \instruction [7]
        assign \imm [12] \instruction [31]
        assign \imm [13] \instruction [31]
        assign \imm [14] \instruction [31]
        assign \imm [15] \instruction [31]
        assign \imm [16] \instruction [31]
        assign \imm [17] \instruction [31]
        assign \imm [18] \instruction [31]
        assign \imm [19] \instruction [31]
        assign \imm [20] \instruction [31]
        assign \imm [21] \instruction [31]
        assign \imm [22] \instruction [31]
        assign \imm [23] \instruction [31]
        assign \imm [24] \instruction [31]
        assign \imm [25] \instruction [31]
        assign \imm [26] \instruction [31]
        assign \imm [27] \instruction [31]
        assign \imm [28] \instruction [31]
        assign \imm [29] \instruction [31]
        assign \imm [30] \instruction [31]
        assign \imm [31] \instruction [31]
    end
    switch $4 [0]
      case 1'1
        assign \imm [0] \instruction [20]
        assign \imm [1] \instruction [21]
        assign \imm [2] \instruction [22]
        assign \imm [3] \instruction [23]
        assign \imm [4] \instruction [24]
        assign \imm [5] \instruction [25]
        assign \imm [6] \instruction [26]
        assign \imm [7] \instruction [27]
        assign \imm [8] \instruction [28]
        assign \imm [9] \instruction [29]
        assign \imm [10] \instruction [30]
        assign \imm [11] \instruction [31]
        assign \imm [12] \instruction [31]
        assign \imm [13] \instruction [31]
        assign \imm [14] \instruction [31]
        assign \imm [15] \instruction [31]
        assign \imm [16] \instruction [31]
        assign \imm [17] \instruction [31]
        assign \imm [18] \instruction [31]
        assign \imm [19] \instruction [31]
        assign \imm [20] \instruction [31]
        assign \imm [21] \instruction [31]
        assign \imm [22] \instruction [31]
        assign \imm [23] \instruction [31]
        assign \imm [24] \instruction [31]
        assign \imm [25] \instruction [31]
        assign \imm [26] \instruction [31]
        assign \imm [27] \instruction [31]
        assign \imm [28] \instruction [31]
        assign \imm [29] \instruction [31]
        assign \imm [30] \instruction [31]
        assign \imm [31] \instruction [31]
    end
    switch $5 [0]
      case 1'1
        assign \imm [1] \instruction [21]
        assign \imm [2] \instruction [22]
        assign \imm [3] \instruction [23]
        assign \imm [4] \instruction [24]
        assign \imm [5] \instruction [25]
        assign \imm [6] \instruction [26]
        assign \imm [7] \instruction [27]
        assign \imm [8] \instruction [28]
        assign \imm [9] \instruction [29]
        assign \imm [10] \instruction [30]
        assign \imm [11] \instruction [20]
        assign \imm [12] \instruction [12]
        assign \imm [13] \instruction [13]
        assign \imm [14] \instruction [14]
        assign \imm [15] \instruction [15]
        assign \imm [16] \instruction [16]
        assign \imm [17] \instruction [17]
        assign \imm [18] \instruction [18]
        assign \imm [19] \instruction [19]
        assign \imm [20] \instruction [31]
        assign \imm [21] \instruction [31]
        assign \imm [22] \instruction [31]
        assign \imm [23] \instruction [31]
        assign \imm [24] \instruction [31]
        assign \imm [25] \instruction [31]
        assign \imm [26] \instruction [31]
        assign \imm [27] \instruction [31]
        assign \imm [28] \instruction [31]
        assign \imm [29] \instruction [31]
        assign \imm [30] \instruction [31]
        assign \imm [31] \instruction [31]
    end
    switch $6 [0]
      case 1'1
        assign \imm [0] \instruction [20]
        assign \imm [1] \instruction [21]
        assign \imm [2] \instruction [22]
        assign \imm [3] \instruction [23]
        assign \imm [4] \instruction [24]
        assign \imm [5] \instruction [25]
        assign \imm [6] \instruction [26]
        assign \imm [7] \instruction [27]
        assign \imm [8] \instruction [28]
        assign \imm [9] \instruction [29]
        assign \imm [10] \instruction [30]
        assign \imm [11] \instruction [31]
        assign \imm [12] \instruction [31]
        assign \imm [13] \instruction [31]
        assign \imm [14] \instruction [31]
        assign \imm [15] \instruction [31]
        assign \imm [16] \instruction [31]
        assign \imm [17] \instruction [31]
        assign \imm [18] \instruction [31]
        assign \imm [19] \instruction [31]
        assign \imm [20] \instruction [31]
        assign \imm [21] \instruction [31]
        assign \imm [22] \instruction [31]
        assign \imm [23] \instruction [31]
        assign \imm [24] \instruction [31]
        assign \imm [25] \instruction [31]
        assign \imm [26] \instruction [31]
        assign \imm [27] \instruction [31]
        assign \imm [28] \instruction [31]
        assign \imm [29] \instruction [31]
        assign \imm [30] \instruction [31]
        assign \imm [31] \instruction [31]
    end
    switch $8 [0]
      case 1'1
        assign \imm [12] \instruction [12]
        assign \imm [13] \instruction [13]
        assign \imm [14] \instruction [14]
        assign \imm [15] \instruction [15]
        assign \imm [16] \instruction [16]
        assign \imm [17] \instruction [17]
        assign \imm [18] \instruction [18]
        assign \imm [19] \instruction [19]
        assign \imm [20] \instruction [20]
        assign \imm [21] \instruction [21]
        assign \imm [22] \instruction [22]
        assign \imm [23] \instruction [23]
        assign \imm [24] \instruction [24]
        assign \imm [25] \instruction [25]
        assign \imm [26] \instruction [26]
        assign \imm [27] \instruction [27]
        assign \imm [28] \instruction [28]
        assign \imm [29] \instruction [29]
        assign \imm [30] \instruction [30]
        assign \imm [31] \instruction [31]
    end
    switch $9 [0]
      case 1'1
        assign \imm [12] \instruction [12]
        assign \imm [13] \instruction [13]
        assign \imm [14] \instruction [14]
        assign \imm [15] \instruction [15]
        assign \imm [16] \instruction [16]
        assign \imm [17] \instruction [17]
        assign \imm [18] \instruction [18]
        assign \imm [19] \instruction [19]
        assign \imm [20] \instruction [20]
        assign \imm [21] \instruction [21]
        assign \imm [22] \instruction [22]
        assign \imm [23] \instruction [23]
        assign \imm [24] \instruction [24]
        assign \imm [25] \instruction [25]
        assign \imm [26] \instruction [26]
        assign \imm [27] \instruction [27]
        assign \imm [28] \instruction [28]
        assign \imm [29] \instruction [29]
        assign \imm [30] \instruction [30]
        assign \imm [31] \instruction [31]
    end
    switch $10 [0]
      case 1'1
        assign \imm [0] \instruction [20]
        assign \imm [1] \instruction [21]
        assign \imm [2] \instruction [22]
        assign \imm [3] \instruction [23]
        assign \imm [4] \instruction [24]
        assign \imm [5] \instruction [25]
        assign \imm [6] \instruction [26]
        assign \imm [7] \instruction [27]
        assign \imm [8] \instruction [28]
        assign \imm [9] \instruction [29]
        assign \imm [10] \instruction [30]
        assign \imm [11] \instruction [31]
        assign \imm [12] \instruction [31]
        assign \imm [13] \instruction [31]
        assign \imm [14] \instruction [31]
        assign \imm [15] \instruction [31]
        assign \imm [16] \instruction [31]
        assign \imm [17] \instruction [31]
        assign \imm [18] \instruction [31]
        assign \imm [19] \instruction [31]
        assign \imm [20] \instruction [31]
        assign \imm [21] \instruction [31]
        assign \imm [22] \instruction [31]
        assign \imm [23] \instruction [31]
        assign \imm [24] \instruction [31]
        assign \imm [25] \instruction [31]
        assign \imm [26] \instruction [31]
        assign \imm [27] \instruction [31]
        assign \imm [28] \instruction [31]
        assign \imm [29] \instruction [31]
        assign \imm [30] \instruction [31]
        assign \imm [31] \instruction [31]
    end
  end
  process $22
    assign \invalid [0] 1'1
    switch $1 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $2 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $3 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $4 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $5 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $6 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $7 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $8 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $9 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
    switch $10 [0]
      case 1'1
        assign \invalid [0] 1'0
    end
  end
connect \op \instruction [6:0]
end
module \top.id.controller
  wire width 7 \op
  wire width 3 \mem_width
  wire width 1 \ebreak
  wire width 32 input 0  \instruction
  wire width 2 output 1  \alu_src1
  wire width 2 output 2  \alu_src2
  wire width 3 output 3  \alu_op
  wire width 2 output 4  \mem_mode
  wire width 1 output 5  \write_rd_en
  wire width 2 output 6  \wrd_sel
  wire width 1 output 7  \need_rs2
  wire width 1 output 8  \build_rd
  wire width 1 output 9  \branch_en
  wire width 3 output 10  \branch_op
  wire width 1 output 11  \jump_en
  wire width 1 output 12  \need_rs1
  wire width 1 output 13  \shift_arithmetic
  wire width 1 output 14  \shamt_is_rs2
  wire width 1 output 15  \sub
  process $1
    assign \alu_src1 [1:0] 2'11
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
        assign \alu_src1 [1:0] 2'10
      case 7'1100111
      case 7'1101111
        assign \alu_src1 [1:0] 2'10
      case 7'0010011
      case 7'0110011
      case 7'0010111
        assign \alu_src1 [1:0] 2'10
      case 7'0110111
        assign \alu_src1 [1:0] 2'01
      case 7'1110011
    end
  end
  process $2
    assign \alu_src2 [1:0] 2'01
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
        assign \alu_src2 [1:0] 2'10
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $3
    assign \alu_op [2:0] 3'000
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
      case 7'1101111
      case 7'0010011
        assign \alu_op [2:0] \instruction [14:12]
      case 7'0110011
        assign \alu_op [2:0] \instruction [14:12]
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $4
    assign \mem_mode [1:0] 2'01
    switch \instruction [6:0]
      case 7'0000011
        assign \mem_mode [1:0] 2'10
      case 7'0100011
        assign \mem_mode [1:0] 2'11
      case 7'1100011
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $5
    assign \write_rd_en [0] 1'0
    switch \instruction [6:0]
      case 7'0000011
        assign \write_rd_en [0] 1'1
      case 7'0100011
      case 7'1100011
      case 7'1100111
        assign \write_rd_en [0] 1'1
      case 7'1101111
        assign \write_rd_en [0] 1'1
      case 7'0010011
        assign \write_rd_en [0] 1'1
      case 7'0110011
        assign \write_rd_en [0] 1'1
      case 7'0010111
        assign \write_rd_en [0] 1'1
      case 7'0110111
        assign \write_rd_en [0] 1'1
      case 7'1110011
    end
  end
  process $6
    assign \wrd_sel [1:0] 2'00
    switch \instruction [6:0]
      case 7'0000011
        assign \wrd_sel [1:0] 2'10
      case 7'0100011
      case 7'1100011
      case 7'1100111
        assign \wrd_sel [1:0] 2'11
      case 7'1101111
        assign \wrd_sel [1:0] 2'11
      case 7'0010011
        assign \wrd_sel [1:0] 2'01
      case 7'0110011
        assign \wrd_sel [1:0] 2'01
      case 7'0010111
        assign \wrd_sel [1:0] 2'01
      case 7'0110111
        assign \wrd_sel [1:0] 2'01
      case 7'1110011
    end
  end
  process $7
    assign \need_rs2 [0] 1'1
    switch \instruction [6:0]
      case 7'0000011
        assign \need_rs2 [0] 1'0
      case 7'0100011
      case 7'1100011
      case 7'1100111
        assign \need_rs2 [0] 1'0
      case 7'1101111
        assign \need_rs2 [0] 1'0
      case 7'0010011
        assign \need_rs2 [0] 1'0
      case 7'0110011
      case 7'0010111
        assign \need_rs2 [0] 1'0
      case 7'0110111
        assign \need_rs2 [0] 1'0
      case 7'1110011
        assign \need_rs2 [0] 1'0
    end
  end
  process $8
    assign \build_rd [0] 1'1
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
        assign \build_rd [0] 1'0
      case 7'1100011
        assign \build_rd [0] 1'0
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $9
    assign \branch_en [0] 1'0
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
        assign \branch_en [0] 1'1
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $10
    assign \branch_op [2:0] 3'000
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
        assign \branch_op [2:0] \instruction [14:12]
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $11
    assign \jump_en [0] 1'0
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
        assign \jump_en [0] 1'1
      case 7'1101111
        assign \jump_en [0] 1'1
      case 7'0010011
      case 7'0110011
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $12
    assign \need_rs1 [0] 1'1
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
      case 7'1101111
        assign \need_rs1 [0] 1'0
      case 7'0010011
      case 7'0110011
      case 7'0010111
        assign \need_rs1 [0] 1'0
      case 7'0110111
        assign \need_rs1 [0] 1'0
      case 7'1110011
    end
  end
  process $13
    assign \shift_arithmetic [0] 1'0
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
      case 7'1101111
      case 7'0010011
        assign \shift_arithmetic [0] \instruction [30]
      case 7'0110011
        assign \shift_arithmetic [0] \instruction [30]
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $14
    assign \shamt_is_rs2 [0] 1'0
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
        assign \shamt_is_rs2 [0] 1'1
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $15
    assign \sub [0] 1'0
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
        assign \sub [0] \instruction [30]
      case 7'0010111
      case 7'0110111
      case 7'1110011
    end
  end
  process $16
    assign \ebreak [0] 1'0
    switch \instruction [6:0]
      case 7'0000011
      case 7'0100011
      case 7'1100011
      case 7'1100111
      case 7'1101111
      case 7'0010011
      case 7'0110011
      case 7'0010111
      case 7'0110111
      case 7'1110011
        assign \ebreak [0] 1'1
    end
  end
connect \op \instruction [6:0]
connect \mem_width \instruction [14:12]
end
module \top.id.forwarder
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 1 input 2  \do_set
  wire width 1 input 3  \build_rd
  wire width 5 input 4  \which_rd
  wire width 2 output 5  \reg0_src
  wire width 2 output 6  \reg1_src
  wire width 2 output 7  \reg2_src
  wire width 2 output 8  \reg3_src
  wire width 2 output 9  \reg4_src
  wire width 2 output 10  \reg5_src
  wire width 2 output 11  \reg6_src
  wire width 2 output 12  \reg7_src
  wire width 2 output 13  \reg8_src
  wire width 2 output 14  \reg9_src
  wire width 2 output 15  \reg10_src
  wire width 2 output 16  \reg11_src
  wire width 2 output 17  \reg12_src
  wire width 2 output 18  \reg13_src
  wire width 2 output 19  \reg14_src
  wire width 2 output 20  \reg15_src
  wire width 2 output 21  \reg16_src
  wire width 2 output 22  \reg17_src
  wire width 2 output 23  \reg18_src
  wire width 2 output 24  \reg19_src
  wire width 2 output 25  \reg20_src
  wire width 2 output 26  \reg21_src
  wire width 2 output 27  \reg22_src
  wire width 2 output 28  \reg23_src
  wire width 2 output 29  \reg24_src
  wire width 2 output 30  \reg25_src
  wire width 2 output 31  \reg26_src
  wire width 2 output 32  \reg27_src
  wire width 2 output 33  \reg28_src
  wire width 2 output 34  \reg29_src
  wire width 2 output 35  \reg30_src
  wire width 2 output 36  \reg31_src
  wire width 1 $1
  wire width 3 $2
  wire width 1 $3
  wire width 3 $4
  wire width 1 $5
  wire width 3 $6
  wire width 1 $7
  wire width 3 $8
  wire width 1 $9
  wire width 3 $10
  wire width 1 $11
  wire width 3 $12
  wire width 1 $13
  wire width 3 $14
  wire width 1 $15
  wire width 3 $16
  wire width 1 $17
  wire width 3 $18
  wire width 1 $19
  wire width 3 $20
  wire width 1 $21
  wire width 3 $22
  wire width 1 $23
  wire width 3 $24
  wire width 1 $25
  wire width 3 $26
  wire width 1 $27
  wire width 3 $28
  wire width 1 $29
  wire width 3 $30
  wire width 1 $31
  wire width 3 $32
  wire width 1 $33
  wire width 3 $34
  wire width 1 $35
  wire width 3 $36
  wire width 1 $37
  wire width 3 $38
  wire width 1 $39
  wire width 3 $40
  wire width 1 $41
  wire width 3 $42
  wire width 1 $43
  wire width 3 $44
  wire width 1 $45
  wire width 3 $46
  wire width 1 $47
  wire width 3 $48
  wire width 1 $49
  wire width 3 $50
  wire width 1 $51
  wire width 3 $52
  wire width 1 $53
  wire width 3 $54
  wire width 1 $55
  wire width 3 $56
  wire width 1 $57
  wire width 3 $58
  wire width 1 $59
  wire width 3 $60
  wire width 1 $61
  wire width 3 $62
  wire width 1 $63
  wire width 3 $64
  wire width 1 $65
  wire width 1 $66
  wire width 2 $67
  wire width 2 $68
  wire width 2 $69
  wire width 2 $70
  wire width 2 $71
  wire width 2 $72
  wire width 2 $73
  wire width 2 $74
  wire width 2 $75
  wire width 2 $76
  wire width 2 $77
  wire width 2 $78
  wire width 2 $79
  wire width 2 $80
  wire width 2 $81
  wire width 2 $82
  wire width 2 $83
  wire width 2 $84
  wire width 2 $85
  wire width 2 $86
  wire width 2 $87
  wire width 2 $88
  wire width 2 $89
  wire width 2 $90
  wire width 2 $91
  wire width 2 $92
  wire width 2 $93
  wire width 2 $94
  wire width 2 $95
  wire width 2 $96
  wire width 2 $97
  wire width 2 $98
  cell $ne $99
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg0_src [1:0]
    connect \B {  }
    connect \Y $1
  end
  cell $sub $100
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg0_src [1:0]
    connect \B 1'1
    connect \Y $2
  end
  cell $ne $101
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg1_src [1:0]
    connect \B {  }
    connect \Y $3
  end
  cell $sub $102
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg1_src [1:0]
    connect \B 1'1
    connect \Y $4
  end
  cell $ne $103
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg2_src [1:0]
    connect \B {  }
    connect \Y $5
  end
  cell $sub $104
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg2_src [1:0]
    connect \B 1'1
    connect \Y $6
  end
  cell $ne $105
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg3_src [1:0]
    connect \B {  }
    connect \Y $7
  end
  cell $sub $106
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg3_src [1:0]
    connect \B 1'1
    connect \Y $8
  end
  cell $ne $107
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg4_src [1:0]
    connect \B {  }
    connect \Y $9
  end
  cell $sub $108
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg4_src [1:0]
    connect \B 1'1
    connect \Y $10
  end
  cell $ne $109
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg5_src [1:0]
    connect \B {  }
    connect \Y $11
  end
  cell $sub $110
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg5_src [1:0]
    connect \B 1'1
    connect \Y $12
  end
  cell $ne $111
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg6_src [1:0]
    connect \B {  }
    connect \Y $13
  end
  cell $sub $112
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg6_src [1:0]
    connect \B 1'1
    connect \Y $14
  end
  cell $ne $113
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg7_src [1:0]
    connect \B {  }
    connect \Y $15
  end
  cell $sub $114
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg7_src [1:0]
    connect \B 1'1
    connect \Y $16
  end
  cell $ne $115
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg8_src [1:0]
    connect \B {  }
    connect \Y $17
  end
  cell $sub $116
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg8_src [1:0]
    connect \B 1'1
    connect \Y $18
  end
  cell $ne $117
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg9_src [1:0]
    connect \B {  }
    connect \Y $19
  end
  cell $sub $118
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg9_src [1:0]
    connect \B 1'1
    connect \Y $20
  end
  cell $ne $119
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg10_src [1:0]
    connect \B {  }
    connect \Y $21
  end
  cell $sub $120
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg10_src [1:0]
    connect \B 1'1
    connect \Y $22
  end
  cell $ne $121
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg11_src [1:0]
    connect \B {  }
    connect \Y $23
  end
  cell $sub $122
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg11_src [1:0]
    connect \B 1'1
    connect \Y $24
  end
  cell $ne $123
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg12_src [1:0]
    connect \B {  }
    connect \Y $25
  end
  cell $sub $124
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg12_src [1:0]
    connect \B 1'1
    connect \Y $26
  end
  cell $ne $125
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg13_src [1:0]
    connect \B {  }
    connect \Y $27
  end
  cell $sub $126
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg13_src [1:0]
    connect \B 1'1
    connect \Y $28
  end
  cell $ne $127
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg14_src [1:0]
    connect \B {  }
    connect \Y $29
  end
  cell $sub $128
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg14_src [1:0]
    connect \B 1'1
    connect \Y $30
  end
  cell $ne $129
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg15_src [1:0]
    connect \B {  }
    connect \Y $31
  end
  cell $sub $130
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg15_src [1:0]
    connect \B 1'1
    connect \Y $32
  end
  cell $ne $131
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg16_src [1:0]
    connect \B {  }
    connect \Y $33
  end
  cell $sub $132
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg16_src [1:0]
    connect \B 1'1
    connect \Y $34
  end
  cell $ne $133
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg17_src [1:0]
    connect \B {  }
    connect \Y $35
  end
  cell $sub $134
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg17_src [1:0]
    connect \B 1'1
    connect \Y $36
  end
  cell $ne $135
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg18_src [1:0]
    connect \B {  }
    connect \Y $37
  end
  cell $sub $136
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg18_src [1:0]
    connect \B 1'1
    connect \Y $38
  end
  cell $ne $137
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg19_src [1:0]
    connect \B {  }
    connect \Y $39
  end
  cell $sub $138
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg19_src [1:0]
    connect \B 1'1
    connect \Y $40
  end
  cell $ne $139
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg20_src [1:0]
    connect \B {  }
    connect \Y $41
  end
  cell $sub $140
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg20_src [1:0]
    connect \B 1'1
    connect \Y $42
  end
  cell $ne $141
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg21_src [1:0]
    connect \B {  }
    connect \Y $43
  end
  cell $sub $142
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg21_src [1:0]
    connect \B 1'1
    connect \Y $44
  end
  cell $ne $143
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg22_src [1:0]
    connect \B {  }
    connect \Y $45
  end
  cell $sub $144
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg22_src [1:0]
    connect \B 1'1
    connect \Y $46
  end
  cell $ne $145
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg23_src [1:0]
    connect \B {  }
    connect \Y $47
  end
  cell $sub $146
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg23_src [1:0]
    connect \B 1'1
    connect \Y $48
  end
  cell $ne $147
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg24_src [1:0]
    connect \B {  }
    connect \Y $49
  end
  cell $sub $148
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg24_src [1:0]
    connect \B 1'1
    connect \Y $50
  end
  cell $ne $149
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg25_src [1:0]
    connect \B {  }
    connect \Y $51
  end
  cell $sub $150
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg25_src [1:0]
    connect \B 1'1
    connect \Y $52
  end
  cell $ne $151
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg26_src [1:0]
    connect \B {  }
    connect \Y $53
  end
  cell $sub $152
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg26_src [1:0]
    connect \B 1'1
    connect \Y $54
  end
  cell $ne $153
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg27_src [1:0]
    connect \B {  }
    connect \Y $55
  end
  cell $sub $154
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg27_src [1:0]
    connect \B 1'1
    connect \Y $56
  end
  cell $ne $155
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg28_src [1:0]
    connect \B {  }
    connect \Y $57
  end
  cell $sub $156
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg28_src [1:0]
    connect \B 1'1
    connect \Y $58
  end
  cell $ne $157
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg29_src [1:0]
    connect \B {  }
    connect \Y $59
  end
  cell $sub $158
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg29_src [1:0]
    connect \B 1'1
    connect \Y $60
  end
  cell $ne $159
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg30_src [1:0]
    connect \B {  }
    connect \Y $61
  end
  cell $sub $160
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg30_src [1:0]
    connect \B 1'1
    connect \Y $62
  end
  cell $ne $161
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \reg31_src [1:0]
    connect \B {  }
    connect \Y $63
  end
  cell $sub $162
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 3
    connect \A \reg31_src [1:0]
    connect \B 1'1
    connect \Y $64
  end
  cell $eq $163
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \do_set [0]
    connect \B 1'1
    connect \Y $65
  end
  cell $and $164
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $65 [0]
    connect \B \build_rd [0]
    connect \Y $66
  end
  process $165
    assign $67 [1:0] \reg0_src [1:0]
    switch $1 [0]
      case 1'1
        assign $67 [1:0] $2 [1:0]
      case
        assign $67 [1:0] \reg0_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
            assign $67 [1:0] 2'11
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch {}
      case
        assign $67 [1:0] 2'00
    end
    switch \rst [0]
      case 1'1
        assign $67 [1:0] 2'00
    end
  end
  cell $dff $166
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $67 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg0_src
  end
  process $167
    assign $68 [1:0] \reg1_src [1:0]
    switch $3 [0]
      case 1'1
        assign $68 [1:0] $4 [1:0]
      case
        assign $68 [1:0] \reg1_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
            assign $68 [1:0] 2'11
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $68 [1:0] 2'00
    end
  end
  cell $dff $168
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $68 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg1_src
  end
  process $169
    assign $69 [1:0] \reg2_src [1:0]
    switch $5 [0]
      case 1'1
        assign $69 [1:0] $6 [1:0]
      case
        assign $69 [1:0] \reg2_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
            assign $69 [1:0] 2'11
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $69 [1:0] 2'00
    end
  end
  cell $dff $170
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $69 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg2_src
  end
  process $171
    assign $70 [1:0] \reg3_src [1:0]
    switch $7 [0]
      case 1'1
        assign $70 [1:0] $8 [1:0]
      case
        assign $70 [1:0] \reg3_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
            assign $70 [1:0] 2'11
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $70 [1:0] 2'00
    end
  end
  cell $dff $172
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $70 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg3_src
  end
  process $173
    assign $71 [1:0] \reg4_src [1:0]
    switch $9 [0]
      case 1'1
        assign $71 [1:0] $10 [1:0]
      case
        assign $71 [1:0] \reg4_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
            assign $71 [1:0] 2'11
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $71 [1:0] 2'00
    end
  end
  cell $dff $174
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $71 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg4_src
  end
  process $175
    assign $72 [1:0] \reg5_src [1:0]
    switch $11 [0]
      case 1'1
        assign $72 [1:0] $12 [1:0]
      case
        assign $72 [1:0] \reg5_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
            assign $72 [1:0] 2'11
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $72 [1:0] 2'00
    end
  end
  cell $dff $176
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $72 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg5_src
  end
  process $177
    assign $73 [1:0] \reg6_src [1:0]
    switch $13 [0]
      case 1'1
        assign $73 [1:0] $14 [1:0]
      case
        assign $73 [1:0] \reg6_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
            assign $73 [1:0] 2'11
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $73 [1:0] 2'00
    end
  end
  cell $dff $178
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $73 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg6_src
  end
  process $179
    assign $74 [1:0] \reg7_src [1:0]
    switch $15 [0]
      case 1'1
        assign $74 [1:0] $16 [1:0]
      case
        assign $74 [1:0] \reg7_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
            assign $74 [1:0] 2'11
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $74 [1:0] 2'00
    end
  end
  cell $dff $180
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $74 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg7_src
  end
  process $181
    assign $75 [1:0] \reg8_src [1:0]
    switch $17 [0]
      case 1'1
        assign $75 [1:0] $18 [1:0]
      case
        assign $75 [1:0] \reg8_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
            assign $75 [1:0] 2'11
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $75 [1:0] 2'00
    end
  end
  cell $dff $182
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $75 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg8_src
  end
  process $183
    assign $76 [1:0] \reg9_src [1:0]
    switch $19 [0]
      case 1'1
        assign $76 [1:0] $20 [1:0]
      case
        assign $76 [1:0] \reg9_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
            assign $76 [1:0] 2'11
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $76 [1:0] 2'00
    end
  end
  cell $dff $184
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $76 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg9_src
  end
  process $185
    assign $77 [1:0] \reg10_src [1:0]
    switch $21 [0]
      case 1'1
        assign $77 [1:0] $22 [1:0]
      case
        assign $77 [1:0] \reg10_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
            assign $77 [1:0] 2'11
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $77 [1:0] 2'00
    end
  end
  cell $dff $186
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $77 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg10_src
  end
  process $187
    assign $78 [1:0] \reg11_src [1:0]
    switch $23 [0]
      case 1'1
        assign $78 [1:0] $24 [1:0]
      case
        assign $78 [1:0] \reg11_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
            assign $78 [1:0] 2'11
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $78 [1:0] 2'00
    end
  end
  cell $dff $188
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $78 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg11_src
  end
  process $189
    assign $79 [1:0] \reg12_src [1:0]
    switch $25 [0]
      case 1'1
        assign $79 [1:0] $26 [1:0]
      case
        assign $79 [1:0] \reg12_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
            assign $79 [1:0] 2'11
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $79 [1:0] 2'00
    end
  end
  cell $dff $190
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $79 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg12_src
  end
  process $191
    assign $80 [1:0] \reg13_src [1:0]
    switch $27 [0]
      case 1'1
        assign $80 [1:0] $28 [1:0]
      case
        assign $80 [1:0] \reg13_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
            assign $80 [1:0] 2'11
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $80 [1:0] 2'00
    end
  end
  cell $dff $192
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $80 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg13_src
  end
  process $193
    assign $81 [1:0] \reg14_src [1:0]
    switch $29 [0]
      case 1'1
        assign $81 [1:0] $30 [1:0]
      case
        assign $81 [1:0] \reg14_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
            assign $81 [1:0] 2'11
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $81 [1:0] 2'00
    end
  end
  cell $dff $194
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $81 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg14_src
  end
  process $195
    assign $82 [1:0] \reg15_src [1:0]
    switch $31 [0]
      case 1'1
        assign $82 [1:0] $32 [1:0]
      case
        assign $82 [1:0] \reg15_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
            assign $82 [1:0] 2'11
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $82 [1:0] 2'00
    end
  end
  cell $dff $196
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $82 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg15_src
  end
  process $197
    assign $83 [1:0] \reg16_src [1:0]
    switch $33 [0]
      case 1'1
        assign $83 [1:0] $34 [1:0]
      case
        assign $83 [1:0] \reg16_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
            assign $83 [1:0] 2'11
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $83 [1:0] 2'00
    end
  end
  cell $dff $198
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $83 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg16_src
  end
  process $199
    assign $84 [1:0] \reg17_src [1:0]
    switch $35 [0]
      case 1'1
        assign $84 [1:0] $36 [1:0]
      case
        assign $84 [1:0] \reg17_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
            assign $84 [1:0] 2'11
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $84 [1:0] 2'00
    end
  end
  cell $dff $200
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $84 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg17_src
  end
  process $201
    assign $85 [1:0] \reg18_src [1:0]
    switch $37 [0]
      case 1'1
        assign $85 [1:0] $38 [1:0]
      case
        assign $85 [1:0] \reg18_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
            assign $85 [1:0] 2'11
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $85 [1:0] 2'00
    end
  end
  cell $dff $202
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $85 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg18_src
  end
  process $203
    assign $86 [1:0] \reg19_src [1:0]
    switch $39 [0]
      case 1'1
        assign $86 [1:0] $40 [1:0]
      case
        assign $86 [1:0] \reg19_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
            assign $86 [1:0] 2'11
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $86 [1:0] 2'00
    end
  end
  cell $dff $204
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $86 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg19_src
  end
  process $205
    assign $87 [1:0] \reg20_src [1:0]
    switch $41 [0]
      case 1'1
        assign $87 [1:0] $42 [1:0]
      case
        assign $87 [1:0] \reg20_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
            assign $87 [1:0] 2'11
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $87 [1:0] 2'00
    end
  end
  cell $dff $206
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $87 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg20_src
  end
  process $207
    assign $88 [1:0] \reg21_src [1:0]
    switch $43 [0]
      case 1'1
        assign $88 [1:0] $44 [1:0]
      case
        assign $88 [1:0] \reg21_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
            assign $88 [1:0] 2'11
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $88 [1:0] 2'00
    end
  end
  cell $dff $208
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $88 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg21_src
  end
  process $209
    assign $89 [1:0] \reg22_src [1:0]
    switch $45 [0]
      case 1'1
        assign $89 [1:0] $46 [1:0]
      case
        assign $89 [1:0] \reg22_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
            assign $89 [1:0] 2'11
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $89 [1:0] 2'00
    end
  end
  cell $dff $210
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $89 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg22_src
  end
  process $211
    assign $90 [1:0] \reg23_src [1:0]
    switch $47 [0]
      case 1'1
        assign $90 [1:0] $48 [1:0]
      case
        assign $90 [1:0] \reg23_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
            assign $90 [1:0] 2'11
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $90 [1:0] 2'00
    end
  end
  cell $dff $212
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $90 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg23_src
  end
  process $213
    assign $91 [1:0] \reg24_src [1:0]
    switch $49 [0]
      case 1'1
        assign $91 [1:0] $50 [1:0]
      case
        assign $91 [1:0] \reg24_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
            assign $91 [1:0] 2'11
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $91 [1:0] 2'00
    end
  end
  cell $dff $214
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $91 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg24_src
  end
  process $215
    assign $92 [1:0] \reg25_src [1:0]
    switch $51 [0]
      case 1'1
        assign $92 [1:0] $52 [1:0]
      case
        assign $92 [1:0] \reg25_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
            assign $92 [1:0] 2'11
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $92 [1:0] 2'00
    end
  end
  cell $dff $216
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $92 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg25_src
  end
  process $217
    assign $93 [1:0] \reg26_src [1:0]
    switch $53 [0]
      case 1'1
        assign $93 [1:0] $54 [1:0]
      case
        assign $93 [1:0] \reg26_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
            assign $93 [1:0] 2'11
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $93 [1:0] 2'00
    end
  end
  cell $dff $218
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $93 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg26_src
  end
  process $219
    assign $94 [1:0] \reg27_src [1:0]
    switch $55 [0]
      case 1'1
        assign $94 [1:0] $56 [1:0]
      case
        assign $94 [1:0] \reg27_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
            assign $94 [1:0] 2'11
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $94 [1:0] 2'00
    end
  end
  cell $dff $220
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $94 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg27_src
  end
  process $221
    assign $95 [1:0] \reg28_src [1:0]
    switch $57 [0]
      case 1'1
        assign $95 [1:0] $58 [1:0]
      case
        assign $95 [1:0] \reg28_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
            assign $95 [1:0] 2'11
          case 5'11101
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $95 [1:0] 2'00
    end
  end
  cell $dff $222
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $95 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg28_src
  end
  process $223
    assign $96 [1:0] \reg29_src [1:0]
    switch $59 [0]
      case 1'1
        assign $96 [1:0] $60 [1:0]
      case
        assign $96 [1:0] \reg29_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
            assign $96 [1:0] 2'11
          case 5'11110
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $96 [1:0] 2'00
    end
  end
  cell $dff $224
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $96 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg29_src
  end
  process $225
    assign $97 [1:0] \reg30_src [1:0]
    switch $61 [0]
      case 1'1
        assign $97 [1:0] $62 [1:0]
      case
        assign $97 [1:0] \reg30_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
            assign $97 [1:0] 2'11
          case 5'11111
        end
    end
    switch \rst [0]
      case 1'1
        assign $97 [1:0] 2'00
    end
  end
  cell $dff $226
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $97 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg30_src
  end
  process $227
    assign $98 [1:0] \reg31_src [1:0]
    switch $63 [0]
      case 1'1
        assign $98 [1:0] $64 [1:0]
      case
        assign $98 [1:0] \reg31_src [1:0]
    end
    switch $66 [0]
      case 1'1
        switch \which_rd [4:0]
          case 5'00000
          case 5'00001
          case 5'00010
          case 5'00011
          case 5'00100
          case 5'00101
          case 5'00110
          case 5'00111
          case 5'01000
          case 5'01001
          case 5'01010
          case 5'01011
          case 5'01100
          case 5'01101
          case 5'01110
          case 5'01111
          case 5'10000
          case 5'10001
          case 5'10010
          case 5'10011
          case 5'10100
          case 5'10101
          case 5'10110
          case 5'10111
          case 5'11000
          case 5'11001
          case 5'11010
          case 5'11011
          case 5'11100
          case 5'11101
          case 5'11110
          case 5'11111
            assign $98 [1:0] 2'11
        end
    end
    switch \rst [0]
      case 1'1
        assign $98 [1:0] 2'00
    end
  end
  cell $dff $228
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $98 [1:0]
    connect \CLK \clk [0]
    connect \Q \reg31_src
  end
end
module \top.ex
  wire width 1 \_in_stall
  wire width 1 \stall_from_top
  wire width 1 \out_stall
  wire width 7 \_in_op
  wire width 32 \_in_rs2
  wire width 2 \_in_mem_mode
  wire width 3 \_in_mem_width
  wire width 5 \_in_which_rd
  wire width 2 \_in_wrd_sel
  wire width 32 \pc
  wire width 32 \_in_pc
  wire width 32 signed \rs1
  wire width 32 \_in_rs1
  wire width 32 signed \rs2
  wire width 32 signed \imm
  wire width 32 \_in_imm
  wire width 5 \which_rs2
  wire width 5 \_in_which_rs2
  wire width 1 \shamt_is_rs2
  wire width 1 \_in_shamt_is_rs2
  wire width 3 \alu_op
  wire width 3 \_in_alu_op
  wire width 2 \alu_src1
  wire width 2 \_in_alu_src1
  wire width 2 \alu_src2
  wire width 2 \_in_alu_src2
  wire width 1 \shift_arithmetic
  wire width 1 \_in_shift_arithmetic
  wire width 1 \sub
  wire width 1 \_in_sub
  wire width 32 signed \out
  wire width 32 \pc$40
  wire width 32 signed \rs1$41
  wire width 32 signed \rs2$42
  wire width 32 \pc_from_alu
  wire width 1 \branch_en
  wire width 1 \_in_branch_en
  wire width 3 \branch_op
  wire width 3 \_in_branch_op
  wire width 1 \jump_en
  wire width 1 \_in_jump_en
  wire width 32 \incr_pc
  wire width 1 \do_jump$53
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 1 input 2  \in_from_prev_stall
  wire width 32 input 3  \in_from_prev_rs1
  wire width 32 input 4  \in_from_prev_rs2
  wire width 1 output 5  \true_stall
  wire width 32 output 6  \out_incr_pc
  wire width 1 input 7  \stall_from_top_next
  wire width 5 input 8  \in_from_prev_which_rd
  wire width 32 input 9  \in_from_prev_pc
  wire width 7 input 10  \in_from_prev_op
  wire width 3 input 11  \in_from_prev_mem_width
  wire width 5 input 12  \in_from_prev_which_rs2
  wire width 32 input 13  \in_from_prev_imm
  wire width 2 input 14  \in_from_prev_alu_src1
  wire width 2 input 15  \in_from_prev_alu_src2
  wire width 3 input 16  \in_from_prev_alu_op
  wire width 2 input 17  \in_from_prev_mem_mode
  wire width 2 input 18  \in_from_prev_wrd_sel
  wire width 1 input 19  \in_from_prev_branch_en
  wire width 3 input 20  \in_from_prev_branch_op
  wire width 1 input 21  \in_from_prev_jump_en
  wire width 1 input 22  \in_from_prev_shift_arithmetic
  wire width 1 input 23  \in_from_prev_shamt_is_rs2
  wire width 1 input 24  \in_from_prev_sub
  wire width 1 output 25  \do_jump
  wire width 7 output 26  \out_op
  wire width 32 output 27  \out_rs2
  wire width 2 output 28  \out_mem_mode
  wire width 3 output 29  \out_mem_width
  wire width 5 output 30  \out_which_rd
  wire width 2 output 31  \out_wrd_sel
  wire width 32 output 32  \out_alu_out
  wire width 32 output 33  \new_pc
  wire width 1 $1
  wire width 1 $2
  wire width 1 $3
  wire width 1 $4
  wire width 32 $5
  wire width 7 $6
  wire width 32 $7
  wire width 32 $8
  wire width 32 $9
  wire width 5 $10
  wire width 3 $11
  wire width 2 $12
  wire width 2 $13
  wire width 1 $14
  wire width 1 $15
  wire width 3 $16
  wire width 1 $17
  wire width 1 $18
  wire width 1 $19
  wire width 2 $20
  wire width 3 $21
  wire width 5 $22
  wire width 2 $23
  cell \top.ex.alu \alu
    connect \pc \pc [31:0]
    connect \imm \imm [31:0]
    connect \rs1 \rs1 [31:0]
    connect \rs2 \out_rs2 [31:0]
    connect \which_rs2 \which_rs2 [4:0]
    connect \alu_op \alu_op [2:0]
    connect \alu_src1 \alu_src1 [1:0]
    connect \alu_src2 \alu_src2 [1:0]
    connect \shamt_is_rs2 \shamt_is_rs2 [0]
    connect \sub \sub [0]
    connect \shift_arithmetic \shift_arithmetic [0]
    connect \out \out_alu_out [31:0]
  end
  cell \top.ex.brancher \brancher
    connect \incr_pc \out_incr_pc [31:0]
    connect \pc \pc [31:0]
    connect \rs1 \rs1 [31:0]
    connect \rs2 \out_rs2 [31:0]
    connect \jump_en \jump_en [0]
    connect \branch_en \branch_en [0]
    connect \branch_op \branch_op [2:0]
    connect \pc_from_alu \out_alu_out [31:0]
    connect \do_jump \do_jump$53 [0]
    connect \new_pc \new_pc [31:0]
  end
  cell $or $24
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \_in_stall [0]
    connect \B \stall_from_top_next [0]
    connect \Y $1
  end
  cell $or $25
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $1 [0]
    connect \B 1'0
    connect \Y \true_stall
  end
  cell $not $26
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \true_stall [0]
    connect \Y $2
  end
  cell $not $27
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A 1'0
    connect \Y $3
  end
  process $28
    assign \do_jump [0] 1'0
    switch $2 [0]
      case 1'1
        assign \do_jump [0] \do_jump$53 [0]
    end
  end
  process $29
    assign $4 [0] \_in_stall [0]
    switch $3 [0]
      case 1'1
        assign $4 [0] \in_from_prev_stall [0]
    end
    switch \rst [0]
      case 1'1
        assign $4 [0] 1'0
    end
  end
  cell $dff $30
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $4 [0]
    connect \CLK \clk [0]
    connect \Q \_in_stall
  end
  process $31
    assign $5 [31:0] \pc [31:0]
    switch $3 [0]
      case 1'1
        assign $5 [31:0] \in_from_prev_pc [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $5 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $32
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $5 [31:0]
    connect \CLK \clk [0]
    connect \Q \pc
  end
  process $33
    assign $6 [6:0] \out_op [6:0]
    switch $3 [0]
      case 1'1
        assign $6 [6:0] \in_from_prev_op [6:0]
    end
    switch \rst [0]
      case 1'1
        assign $6 [6:0] 7'0000000
    end
  end
  cell $dff $34
    parameter \WIDTH 7
    parameter \CLK_POLARITY 1
    connect \D $6 [6:0]
    connect \CLK \clk [0]
    connect \Q \out_op
  end
  process $35
    assign $7 [31:0] \imm [31:0]
    switch $3 [0]
      case 1'1
        assign $7 [31:0] \in_from_prev_imm [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $7 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $36
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $7 [31:0]
    connect \CLK \clk [0]
    connect \Q \imm
  end
  process $37
    assign $8 [31:0] \rs1 [31:0]
    switch $3 [0]
      case 1'1
        assign $8 [31:0] \in_from_prev_rs1 [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $8 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $38
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $8 [31:0]
    connect \CLK \clk [0]
    connect \Q \rs1
  end
  process $39
    assign $9 [31:0] \out_rs2 [31:0]
    switch $3 [0]
      case 1'1
        assign $9 [31:0] \in_from_prev_rs2 [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $9 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $40
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $9 [31:0]
    connect \CLK \clk [0]
    connect \Q \out_rs2
  end
  process $41
    assign $10 [4:0] \which_rs2 [4:0]
    switch $3 [0]
      case 1'1
        assign $10 [4:0] \in_from_prev_which_rs2 [4:0]
    end
    switch \rst [0]
      case 1'1
        assign $10 [4:0] 5'00000
    end
  end
  cell $dff $42
    parameter \WIDTH 5
    parameter \CLK_POLARITY 1
    connect \D $10 [4:0]
    connect \CLK \clk [0]
    connect \Q \which_rs2
  end
  process $43
    assign $11 [2:0] \alu_op [2:0]
    switch $3 [0]
      case 1'1
        assign $11 [2:0] \in_from_prev_alu_op [2:0]
    end
    switch \rst [0]
      case 1'1
        assign $11 [2:0] 3'000
    end
  end
  cell $dff $44
    parameter \WIDTH 3
    parameter \CLK_POLARITY 1
    connect \D $11 [2:0]
    connect \CLK \clk [0]
    connect \Q \alu_op
  end
  process $45
    assign $12 [1:0] \alu_src1 [1:0]
    switch $3 [0]
      case 1'1
        assign $12 [1:0] \in_from_prev_alu_src1 [1:0]
    end
    switch \rst [0]
      case 1'1
        assign $12 [1:0] 2'00
    end
  end
  cell $dff $46
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $12 [1:0]
    connect \CLK \clk [0]
    connect \Q \alu_src1
  end
  process $47
    assign $13 [1:0] \alu_src2 [1:0]
    switch $3 [0]
      case 1'1
        assign $13 [1:0] \in_from_prev_alu_src2 [1:0]
    end
    switch \rst [0]
      case 1'1
        assign $13 [1:0] 2'00
    end
  end
  cell $dff $48
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $13 [1:0]
    connect \CLK \clk [0]
    connect \Q \alu_src2
  end
  process $49
    assign $14 [0] \jump_en [0]
    switch $3 [0]
      case 1'1
        assign $14 [0] \in_from_prev_jump_en [0]
    end
    switch \rst [0]
      case 1'1
        assign $14 [0] 1'0
    end
  end
  cell $dff $50
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $14 [0]
    connect \CLK \clk [0]
    connect \Q \jump_en
  end
  process $51
    assign $15 [0] \branch_en [0]
    switch $3 [0]
      case 1'1
        assign $15 [0] \in_from_prev_branch_en [0]
    end
    switch \rst [0]
      case 1'1
        assign $15 [0] 1'0
    end
  end
  cell $dff $52
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $15 [0]
    connect \CLK \clk [0]
    connect \Q \branch_en
  end
  process $53
    assign $16 [2:0] \branch_op [2:0]
    switch $3 [0]
      case 1'1
        assign $16 [2:0] \in_from_prev_branch_op [2:0]
    end
    switch \rst [0]
      case 1'1
        assign $16 [2:0] 3'000
    end
  end
  cell $dff $54
    parameter \WIDTH 3
    parameter \CLK_POLARITY 1
    connect \D $16 [2:0]
    connect \CLK \clk [0]
    connect \Q \branch_op
  end
  process $55
    assign $17 [0] \shamt_is_rs2 [0]
    switch $3 [0]
      case 1'1
        assign $17 [0] \in_from_prev_shamt_is_rs2 [0]
    end
    switch \rst [0]
      case 1'1
        assign $17 [0] 1'0
    end
  end
  cell $dff $56
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $17 [0]
    connect \CLK \clk [0]
    connect \Q \shamt_is_rs2
  end
  process $57
    assign $18 [0] \sub [0]
    switch $3 [0]
      case 1'1
        assign $18 [0] \in_from_prev_sub [0]
    end
    switch \rst [0]
      case 1'1
        assign $18 [0] 1'0
    end
  end
  cell $dff $58
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $18 [0]
    connect \CLK \clk [0]
    connect \Q \sub
  end
  process $59
    assign $19 [0] \shift_arithmetic [0]
    switch $3 [0]
      case 1'1
        assign $19 [0] \in_from_prev_shift_arithmetic [0]
    end
    switch \rst [0]
      case 1'1
        assign $19 [0] 1'0
    end
  end
  cell $dff $60
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $19 [0]
    connect \CLK \clk [0]
    connect \Q \shift_arithmetic
  end
  process $61
    assign $20 [1:0] \out_mem_mode [1:0]
    switch $3 [0]
      case 1'1
        assign $20 [1:0] \in_from_prev_mem_mode [1:0]
    end
    switch \rst [0]
      case 1'1
        assign $20 [1:0] 2'00
    end
  end
  cell $dff $62
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $20 [1:0]
    connect \CLK \clk [0]
    connect \Q \out_mem_mode
  end
  process $63
    assign $21 [2:0] \out_mem_width [2:0]
    switch $3 [0]
      case 1'1
        assign $21 [2:0] \in_from_prev_mem_width [2:0]
    end
    switch \rst [0]
      case 1'1
        assign $21 [2:0] 3'000
    end
  end
  cell $dff $64
    parameter \WIDTH 3
    parameter \CLK_POLARITY 1
    connect \D $21 [2:0]
    connect \CLK \clk [0]
    connect \Q \out_mem_width
  end
  process $65
    assign $22 [4:0] \out_which_rd [4:0]
    switch $3 [0]
      case 1'1
        assign $22 [4:0] \in_from_prev_which_rd [4:0]
    end
    switch \rst [0]
      case 1'1
        assign $22 [4:0] 5'00000
    end
  end
  cell $dff $66
    parameter \WIDTH 5
    parameter \CLK_POLARITY 1
    connect \D $22 [4:0]
    connect \CLK \clk [0]
    connect \Q \out_which_rd
  end
  process $67
    assign $23 [1:0] \out_wrd_sel [1:0]
    switch $3 [0]
      case 1'1
        assign $23 [1:0] \in_from_prev_wrd_sel [1:0]
    end
    switch \rst [0]
      case 1'1
        assign $23 [1:0] 2'00
    end
  end
  cell $dff $68
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $23 [1:0]
    connect \CLK \clk [0]
    connect \Q \out_wrd_sel
  end
connect \stall_from_top 1'0
connect \out_stall \true_stall [0]
connect \_in_op \out_op [6:0]
connect \_in_rs2 \out_rs2 [31:0]
connect \_in_mem_mode \out_mem_mode [1:0]
connect \_in_mem_width \out_mem_width [2:0]
connect \_in_which_rd \out_which_rd [4:0]
connect \_in_wrd_sel \out_wrd_sel [1:0]
connect \_in_pc \pc [31:0]
connect \_in_rs1 \rs1 [31:0]
connect \rs2 \out_rs2 [31:0]
connect \_in_imm \imm [31:0]
connect \_in_which_rs2 \which_rs2 [4:0]
connect \_in_shamt_is_rs2 \shamt_is_rs2 [0]
connect \_in_alu_op \alu_op [2:0]
connect \_in_alu_src1 \alu_src1 [1:0]
connect \_in_alu_src2 \alu_src2 [1:0]
connect \_in_shift_arithmetic \shift_arithmetic [0]
connect \_in_sub \sub [0]
connect \out \out_alu_out [31:0]
connect \pc$40 \pc [31:0]
connect \rs1$41 \rs1 [31:0]
connect \rs2$42 \out_rs2 [31:0]
connect \pc_from_alu \out_alu_out [31:0]
connect \_in_branch_en \branch_en [0]
connect \_in_branch_op \branch_op [2:0]
connect \_in_jump_en \jump_en [0]
connect \incr_pc \out_incr_pc [31:0]
end
module \top.ex.alu
  wire width 32 \src1
  wire width 1 \invalid
  wire width 5 \shamt
  wire width 32 \src2
  wire width 32 input 0  \pc
  wire width 32 input 1  signed \imm
  wire width 32 input 2  signed \rs1
  wire width 32 input 3  signed \rs2
  wire width 5 input 4  \which_rs2
  wire width 3 input 5  \alu_op
  wire width 2 input 6  \alu_src1
  wire width 2 input 7  \alu_src2
  wire width 1 input 8  \shamt_is_rs2
  wire width 1 input 9  \sub
  wire width 1 input 10  \shift_arithmetic
  wire width 32 output 11  signed \out
  wire width 33 $1
  wire width 33 $2
  wire width 1 $3
  wire width 1 $4
  wire width 32 $5
  wire width 32 $6
  wire width 32 $7
  wire width 63 $8
  wire width 32 $9
  wire width 32 $10
  cell $sub $11
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 33
    connect \A \src1 [31:0]
    connect \B \src2 [31:0]
    connect \Y $1
  end
  cell $add $12
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 33
    connect \A \src1 [31:0]
    connect \B \src2 [31:0]
    connect \Y $2
  end
  cell $lt $13
    parameter \A_SIGNED 1
    parameter \B_SIGNED 1
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \src1 [31:0]
    connect \B \src2 [31:0]
    connect \Y $3
  end
  cell $lt $14
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \src1 [31:0]
    connect \B \src2 [31:0]
    connect \Y $4
  end
  cell $xor $15
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 32
    connect \A \src1 [31:0]
    connect \B \src2 [31:0]
    connect \Y $5
  end
  cell $or $16
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 32
    connect \A \src1 [31:0]
    connect \B \src2 [31:0]
    connect \Y $6
  end
  cell $and $17
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 32
    connect \A \src1 [31:0]
    connect \B \src2 [31:0]
    connect \Y $7
  end
  cell $shl $18
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 5
    parameter \Y_WIDTH 63
    connect \A \src1 [31:0]
    connect \B \shamt [4:0]
    connect \Y $8
  end
  cell $sshr $19
    parameter \A_SIGNED 1
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 5
    parameter \Y_WIDTH 32
    connect \A \src1 [31:0]
    connect \B \shamt [4:0]
    connect \Y $9
  end
  cell $shr $20
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 5
    parameter \Y_WIDTH 32
    connect \A \src1 [31:0]
    connect \B \shamt [4:0]
    connect \Y $10
  end
  process $21
    assign \src1 [31:0] 32'00000000000000000000000000000000
    switch \alu_src1 [1:0]
      case 2'01
      case 2'10
        assign \src1 [31:0] \pc [31:0]
      case 2'11
        assign \src1 [31:0] \rs1 [31:0]
      case
    end
  end
  process $22
    assign \invalid [0] 1'0
    switch \alu_src1 [1:0]
      case 2'01
      case 2'10
      case 2'11
      case
        assign \invalid [0] 1'1
    end
    switch \alu_src2 [1:0]
      case 2'01
      case 2'10
      case
        assign \invalid [0] 1'1
    end
    switch \alu_op [2:0]
      case 3'000
      case 3'010
      case 3'011
      case 3'100
      case 3'110
      case 3'111
      case 3'001
      case 3'101
      case
        assign \invalid [0] 1'1
    end
  end
  process $23
    assign \shamt [4:0] 5'00000
    switch \shamt_is_rs2 [0]
      case 1'1
        assign \shamt [4:0] \rs2 [4:0]
      case
        assign \shamt [4:0] \which_rs2 [4:0]
    end
  end
  process $24
    assign \src2 [31:0] 32'00000000000000000000000000000000
    switch \alu_src2 [1:0]
      case 2'01
        assign \src2 [31:0] \imm [31:0]
      case 2'10
        assign \src2 [31:0] \rs2 [31:0]
      case
    end
  end
  process $25
    assign \out [31:0] 32'00000000000000000000000000000000
    switch \alu_op [2:0]
      case 3'000
        switch \sub [0]
          case 1'1
            assign \out [31:0] $1 [31:0]
          case
            assign \out [31:0] $2 [31:0]
        end
      case 3'010
        assign \out [31:0] { 31'0000000000000000000000000000000 $3 [0] }
      case 3'011
        assign \out [31:0] { 31'0000000000000000000000000000000 $4 [0] }
      case 3'100
        assign \out [31:0] $5 [31:0]
      case 3'110
        assign \out [31:0] $6 [31:0]
      case 3'111
        assign \out [31:0] $7 [31:0]
      case 3'001
        assign \out [31:0] $8 [31:0]
      case 3'101
        switch \shift_arithmetic [0]
          case 1'1
            assign \out [31:0] $9 [31:0]
          case
            assign \out [31:0] $10 [31:0]
        end
      case
    end
  end
end
module \top.ex.brancher
  wire width 1 \invalid
  wire width 32 output 0  \incr_pc
  wire width 32 input 1  \pc
  wire width 32 input 2  signed \rs1
  wire width 32 input 3  signed \rs2
  wire width 1 input 4  \jump_en
  wire width 1 input 5  \branch_en
  wire width 3 input 6  \branch_op
  wire width 32 input 7  \pc_from_alu
  wire width 1 output 8  \do_jump
  wire width 32 output 9  \new_pc
  wire width 33 $1
  wire width 1 $2
  wire width 1 $3
  wire width 1 $4
  wire width 1 $5
  wire width 1 $6
  wire width 1 $7
  cell $add $8
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 3
    parameter \Y_WIDTH 33
    connect \A \pc [31:0]
    connect \B 3'100
    connect \Y $1
  end
  cell $eq $9
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \rs1 [31:0]
    connect \B \rs2 [31:0]
    connect \Y $2
  end
  cell $ne $10
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \rs1 [31:0]
    connect \B \rs2 [31:0]
    connect \Y $3
  end
  cell $lt $11
    parameter \A_SIGNED 1
    parameter \B_SIGNED 1
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \rs1 [31:0]
    connect \B \rs2 [31:0]
    connect \Y $4
  end
  cell $ge $12
    parameter \A_SIGNED 1
    parameter \B_SIGNED 1
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \rs1 [31:0]
    connect \B \rs2 [31:0]
    connect \Y $5
  end
  cell $lt $13
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \rs1 [31:0]
    connect \B \rs2 [31:0]
    connect \Y $6
  end
  cell $ge $14
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 32
    parameter \B_WIDTH 32
    parameter \Y_WIDTH 1
    connect \A \rs1 [31:0]
    connect \B \rs2 [31:0]
    connect \Y $7
  end
  process $15
    assign \invalid [0] 1'0
    switch { \branch_en [0] \jump_en [0] }
      case 2'-1
      case 2'1-
        switch \branch_op [2:0]
          case 3'000
          case 3'001
          case 3'100
          case 3'101
          case 3'110
          case 3'111
          case
            assign \invalid [0] 1'1
        end
    end
  end
  process $16
    assign \do_jump [0] 1'0
    switch { \branch_en [0] \jump_en [0] }
      case 2'-1
        assign \do_jump [0] 1'1
      case 2'1-
        switch \branch_op [2:0]
          case 3'000
            assign \do_jump [0] $2 [0]
          case 3'001
            assign \do_jump [0] $3 [0]
          case 3'100
            assign \do_jump [0] $4 [0]
          case 3'101
            assign \do_jump [0] $5 [0]
          case 3'110
            assign \do_jump [0] $6 [0]
          case 3'111
            assign \do_jump [0] $7 [0]
          case
        end
    end
  end
  process $17
    assign \new_pc [31:0] 32'00000000000000000000000000000000
    switch \do_jump [0]
      case 1'1
        assign \new_pc [31:0] \pc_from_alu [31:0]
      case
        assign \new_pc [31:0] $1 [31:0]
    end
  end
connect \incr_pc $1 [31:0]
end
module \top.mem
  wire width 1 \_in_stall
  wire width 1 \stall_from_top_next
  wire width 1 \stall_from_top
  wire width 1 \out_stall
  wire width 7 \_in_op
  wire width 32 \_in_alu_out
  wire width 32 \_in_incr_pc
  wire width 5 \_in_which_rd
  wire width 2 \_in_wrd_sel
  wire width 2 \mode
  wire width 2 \_in_mem_mode
  wire width 3 \width
  wire width 3 \_in_mem_width
  wire width 32 \addr
  wire width 32 \write_data
  wire width 32 \_in_rs2
  wire width 32 \read_data
  wire width 32 \alu_out
  wire width 32 \mem_read
  wire width 32 \incr_pc
  wire width 2 \wrd_sel
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 1 input 2  \in_from_prev_stall
  wire width 32 input 3  \in_from_prev_incr_pc
  wire width 1 output 4  \true_stall
  wire width 7 input 5  \in_from_prev_op
  wire width 32 input 6  \in_from_prev_rs2
  wire width 2 input 7  \in_from_prev_mem_mode
  wire width 3 input 8  \in_from_prev_mem_width
  wire width 5 input 9  \in_from_prev_which_rd
  wire width 2 input 10  \in_from_prev_wrd_sel
  wire width 32 input 11  \in_from_prev_alu_out
  wire width 32 output 12  \out_mem_read
  wire width 7 output 13  \out_op
  wire width 32 output 14  \out_alu_out
  wire width 32 output 15  \out_incr_pc
  wire width 5 output 16  \out_which_rd
  wire width 2 output 17  \out_wrd_sel
  wire width 32 output 18  \rd
  wire width 1 $1
  wire width 1 $2
  wire width 1 $3
  wire width 1 $4
  wire width 7 $5
  wire width 32 $6
  wire width 32 $7
  wire width 32 $8
  wire width 2 $9
  wire width 3 $10
  wire width 5 $11
  wire width 2 $12
  cell \top.mem.datamem \datamem
    connect \mode \mode [1:0]
    connect \width \width [2:0]
    connect \addr$1 \addr [31:0]
    connect \write_data \write_data [31:0]
    connect \read_data \read_data [31:0]
  end
  cell \top.mem.reg_mux \reg_mux
    connect \alu_out \out_alu_out [31:0]
    connect \incr_pc \out_incr_pc [31:0]
    connect \wrd_sel \out_wrd_sel [1:0]
    connect \mem_read \read_data [31:0]
    connect \rd \rd [31:0]
  end
  cell $or $13
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \_in_stall [0]
    connect \B 1'0
    connect \Y $1
  end
  cell $or $14
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $1 [0]
    connect \B 1'0
    connect \Y \true_stall
  end
  cell $eq $15
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \true_stall [0]
    connect \B {  }
    connect \Y $2
  end
  cell $not $16
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A 1'0
    connect \Y $3
  end
  process $17
    assign \mode [1:0] 2'00
    switch $2 [0]
      case 1'1
        assign \mode [1:0] \_in_mem_mode [1:0]
    end
  end
  process $18
    assign \width [2:0] 3'000
    switch $2 [0]
      case 1'1
        assign \width [2:0] \_in_mem_width [2:0]
    end
  end
  process $19
    assign \addr [31:0] 32'00000000000000000000000000000000
    switch $2 [0]
      case 1'1
        assign \addr [31:0] \out_alu_out [31:0]
    end
  end
  process $20
    assign \write_data [31:0] 32'00000000000000000000000000000000
    switch $2 [0]
      case 1'1
        assign \write_data [31:0] \_in_rs2 [31:0]
    end
  end
  process $21
    assign \out_mem_read [31:0] 32'00000000000000000000000000000000
    switch $2 [0]
      case 1'1
        assign \out_mem_read [31:0] \read_data [31:0]
    end
  end
  process $22
    assign $4 [0] \_in_stall [0]
    switch $3 [0]
      case 1'1
        assign $4 [0] \in_from_prev_stall [0]
    end
    switch \rst [0]
      case 1'1
        assign $4 [0] 1'0
    end
  end
  cell $dff $23
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $4 [0]
    connect \CLK \clk [0]
    connect \Q \_in_stall
  end
  process $24
    assign $5 [6:0] \out_op [6:0]
    switch $3 [0]
      case 1'1
        assign $5 [6:0] \in_from_prev_op [6:0]
    end
    switch \rst [0]
      case 1'1
        assign $5 [6:0] 7'0000000
    end
  end
  cell $dff $25
    parameter \WIDTH 7
    parameter \CLK_POLARITY 1
    connect \D $5 [6:0]
    connect \CLK \clk [0]
    connect \Q \out_op
  end
  process $26
    assign $6 [31:0] \_in_rs2 [31:0]
    switch $3 [0]
      case 1'1
        assign $6 [31:0] \in_from_prev_rs2 [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $6 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $27
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $6 [31:0]
    connect \CLK \clk [0]
    connect \Q \_in_rs2
  end
  process $28
    assign $7 [31:0] \out_alu_out [31:0]
    switch $3 [0]
      case 1'1
        assign $7 [31:0] \in_from_prev_alu_out [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $7 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $29
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $7 [31:0]
    connect \CLK \clk [0]
    connect \Q \out_alu_out
  end
  process $30
    assign $8 [31:0] \out_incr_pc [31:0]
    switch $3 [0]
      case 1'1
        assign $8 [31:0] \in_from_prev_incr_pc [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $8 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $31
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $8 [31:0]
    connect \CLK \clk [0]
    connect \Q \out_incr_pc
  end
  process $32
    assign $9 [1:0] \_in_mem_mode [1:0]
    switch $3 [0]
      case 1'1
        assign $9 [1:0] \in_from_prev_mem_mode [1:0]
    end
    switch \rst [0]
      case 1'1
        assign $9 [1:0] 2'00
    end
  end
  cell $dff $33
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $9 [1:0]
    connect \CLK \clk [0]
    connect \Q \_in_mem_mode
  end
  process $34
    assign $10 [2:0] \_in_mem_width [2:0]
    switch $3 [0]
      case 1'1
        assign $10 [2:0] \in_from_prev_mem_width [2:0]
    end
    switch \rst [0]
      case 1'1
        assign $10 [2:0] 3'000
    end
  end
  cell $dff $35
    parameter \WIDTH 3
    parameter \CLK_POLARITY 1
    connect \D $10 [2:0]
    connect \CLK \clk [0]
    connect \Q \_in_mem_width
  end
  process $36
    assign $11 [4:0] \out_which_rd [4:0]
    switch $3 [0]
      case 1'1
        assign $11 [4:0] \in_from_prev_which_rd [4:0]
    end
    switch \rst [0]
      case 1'1
        assign $11 [4:0] 5'00000
    end
  end
  cell $dff $37
    parameter \WIDTH 5
    parameter \CLK_POLARITY 1
    connect \D $11 [4:0]
    connect \CLK \clk [0]
    connect \Q \out_which_rd
  end
  process $38
    assign $12 [1:0] \out_wrd_sel [1:0]
    switch $3 [0]
      case 1'1
        assign $12 [1:0] \in_from_prev_wrd_sel [1:0]
    end
    switch \rst [0]
      case 1'1
        assign $12 [1:0] 2'00
    end
  end
  cell $dff $39
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $12 [1:0]
    connect \CLK \clk [0]
    connect \Q \out_wrd_sel
  end
connect \stall_from_top_next 1'0
connect \stall_from_top 1'0
connect \out_stall \true_stall [0]
connect \_in_op \out_op [6:0]
connect \_in_alu_out \out_alu_out [31:0]
connect \_in_incr_pc \out_incr_pc [31:0]
connect \_in_which_rd \out_which_rd [4:0]
connect \_in_wrd_sel \out_wrd_sel [1:0]
connect \alu_out \out_alu_out [31:0]
connect \mem_read \read_data [31:0]
connect \incr_pc \out_incr_pc [31:0]
connect \wrd_sel \out_wrd_sel [1:0]
end
module \top.mem.datamem
  wire width 32 \addr
  wire width 32 \addr$2
  wire width 32 \read_data_shifted
  wire width 32 \data
  wire width 32 \data$7
  wire width 1 \en
  wire width 2 input 0  \mode
  wire width 3 input 1  \width
  wire width 32 input 2  \addr$1
  wire width 32 input 3  \write_data
  wire width 32 output 4  \read_data
  wire width 6 $1
  wire width 1 $2
  wire width 6 $3
  wire width 6 $4
  cell $mul $5
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 4
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 6
    connect \A 4'1000
    connect \B \addr$1 [1:0]
    connect \Y $1
  end
  cell $shr $6
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 0
    parameter \B_WIDTH 6
    parameter \Y_WIDTH 32
    connect \A {  }
    connect \B $1 [5:0]
    connect \Y \read_data_shifted
  end
  cell $eq $7
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 2
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 1
    connect \A \mode [1:0]
    connect \B 2'11
    connect \Y $2
  end
  cell $mul $8
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 4
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 6
    connect \A 4'1000
    connect \B \addr$1 [1:0]
    connect \Y $3
  end
  cell $mul $9
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 4
    parameter \B_WIDTH 2
    parameter \Y_WIDTH 6
    connect \A 4'1000
    connect \B \addr$1 [1:0]
    connect \Y $4
  end
  process $10
    assign \read_data [31:0] 32'00000000000000000000000000000000
    switch \width [2:0]
      case 3'000
        assign \read_data [31:0] { \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7] \read_data_shifted [7:0] }
      case 3'100
        assign \read_data [31:0] { 24'000000000000000000000000 \read_data_shifted [7:0] }
      case 3'001
        assign \read_data [31:0] { \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15] \read_data_shifted [15:0] }
      case 3'101
        assign \read_data [31:0] { 16'0000000000000000 \read_data_shifted [15:0] }
      case 3'010
        assign \read_data [31:0] \read_data_shifted [31:0]
    end
  end
  process $11
    assign \data$7 [31:0] 32'00000000000000000000000000000000
    switch $2 [0]
      case 1'1
        switch \width [2:0]
          case 3'000
            switch $3 [5:0]
              case 6'000000
                assign \data$7 [7:0] \write_data [7:0]
              case 6'000001
                assign \data$7 [8:1] \write_data [7:0]
              case 6'000010
                assign \data$7 [9:2] \write_data [7:0]
              case 6'000011
                assign \data$7 [10:3] \write_data [7:0]
              case 6'000100
                assign \data$7 [11:4] \write_data [7:0]
              case 6'000101
                assign \data$7 [12:5] \write_data [7:0]
              case 6'000110
                assign \data$7 [13:6] \write_data [7:0]
              case 6'000111
                assign \data$7 [14:7] \write_data [7:0]
              case 6'001000
                assign \data$7 [15:8] \write_data [7:0]
              case 6'001001
                assign \data$7 [16:9] \write_data [7:0]
              case 6'001010
                assign \data$7 [17:10] \write_data [7:0]
              case 6'001011
                assign \data$7 [18:11] \write_data [7:0]
              case 6'001100
                assign \data$7 [19:12] \write_data [7:0]
              case 6'001101
                assign \data$7 [20:13] \write_data [7:0]
              case 6'001110
                assign \data$7 [21:14] \write_data [7:0]
              case 6'001111
                assign \data$7 [22:15] \write_data [7:0]
              case 6'010000
                assign \data$7 [23:16] \write_data [7:0]
              case 6'010001
                assign \data$7 [24:17] \write_data [7:0]
              case 6'010010
                assign \data$7 [25:18] \write_data [7:0]
              case 6'010011
                assign \data$7 [26:19] \write_data [7:0]
              case 6'010100
                assign \data$7 [27:20] \write_data [7:0]
              case 6'010101
                assign \data$7 [28:21] \write_data [7:0]
              case 6'010110
                assign \data$7 [29:22] \write_data [7:0]
              case 6'010111
                assign \data$7 [30:23] \write_data [7:0]
              case 6'011000
                assign \data$7 [31:24] \write_data [7:0]
              case 6'011001
                assign \data$7 [31:25] \write_data [6:0]
              case 6'011010
                assign \data$7 [31:26] \write_data [5:0]
              case 6'011011
                assign \data$7 [31:27] \write_data [4:0]
              case 6'011100
                assign \data$7 [31:28] \write_data [3:0]
              case 6'011101
                assign \data$7 [31:29] \write_data [2:0]
              case 6'011110
                assign \data$7 [31:30] \write_data [1:0]
              case 6'011111
                assign \data$7 [31] \write_data [0]
            end
          case 3'001
            switch $4 [5:0]
              case 6'000000
                assign \data$7 [15:0] \write_data [15:0]
              case 6'000001
                assign \data$7 [16:1] \write_data [15:0]
              case 6'000010
                assign \data$7 [17:2] \write_data [15:0]
              case 6'000011
                assign \data$7 [18:3] \write_data [15:0]
              case 6'000100
                assign \data$7 [19:4] \write_data [15:0]
              case 6'000101
                assign \data$7 [20:5] \write_data [15:0]
              case 6'000110
                assign \data$7 [21:6] \write_data [15:0]
              case 6'000111
                assign \data$7 [22:7] \write_data [15:0]
              case 6'001000
                assign \data$7 [23:8] \write_data [15:0]
              case 6'001001
                assign \data$7 [24:9] \write_data [15:0]
              case 6'001010
                assign \data$7 [25:10] \write_data [15:0]
              case 6'001011
                assign \data$7 [26:11] \write_data [15:0]
              case 6'001100
                assign \data$7 [27:12] \write_data [15:0]
              case 6'001101
                assign \data$7 [28:13] \write_data [15:0]
              case 6'001110
                assign \data$7 [29:14] \write_data [15:0]
              case 6'001111
                assign \data$7 [30:15] \write_data [15:0]
              case 6'010000
                assign \data$7 [31:16] \write_data [15:0]
              case 6'010001
                assign \data$7 [31:17] \write_data [14:0]
              case 6'010010
                assign \data$7 [31:18] \write_data [13:0]
              case 6'010011
                assign \data$7 [31:19] \write_data [12:0]
              case 6'010100
                assign \data$7 [31:20] \write_data [11:0]
              case 6'010101
                assign \data$7 [31:21] \write_data [10:0]
              case 6'010110
                assign \data$7 [31:22] \write_data [9:0]
              case 6'010111
                assign \data$7 [31:23] \write_data [8:0]
              case 6'011000
                assign \data$7 [31:24] \write_data [7:0]
              case 6'011001
                assign \data$7 [31:25] \write_data [6:0]
              case 6'011010
                assign \data$7 [31:26] \write_data [5:0]
              case 6'011011
                assign \data$7 [31:27] \write_data [4:0]
              case 6'011100
                assign \data$7 [31:28] \write_data [3:0]
              case 6'011101
                assign \data$7 [31:29] \write_data [2:0]
              case 6'011110
                assign \data$7 [31:30] \write_data [1:0]
              case 6'011111
                assign \data$7 [31] \write_data [0]
            end
          case 3'010
            assign \data$7 [31:0] \write_data [31:0]
        end
    end
  end
  process $12
    assign \en [0] 1'0
    switch $2 [0]
      case 1'1
        assign \en [0] 1'1
    end
  end
connect \addr { 2'00 \addr$1 [31:2] }
connect \addr$2 { 2'00 \addr$1 [31:2] }
connect \data 32'00000000000000000000000000000000
end
module \top.mem.reg_mux
  wire width 1 \invalid
  wire width 32 input 0  \alu_out
  wire width 32 input 1  \incr_pc
  wire width 2 input 2  \wrd_sel
  wire width 32 input 3  \mem_read
  wire width 32 output 4  \rd
  process $1
    assign \rd [31:0] 32'00000000000000000000000000000000
    switch \wrd_sel [1:0]
      case 2'01
        assign \rd [31:0] \alu_out [31:0]
      case 2'10
        assign \rd [31:0] \mem_read [31:0]
      case 2'11
        assign \rd [31:0] \incr_pc [31:0]
      case
    end
  end
  process $2
    assign \invalid [0] 1'0
    switch \wrd_sel [1:0]
      case 2'01
      case 2'10
      case 2'11
      case
        assign \invalid [0] 1'1
    end
  end
end
module \top.wb
  wire width 1 \true_stall
  wire width 1 \_in_stall
  wire width 1 \stall_from_top_next
  wire width 1 \stall_from_top
  wire width 1 \out_stall
  wire width 32 \alu_out
  wire width 32 \_in_alu_out
  wire width 32 \mem_read
  wire width 32 \_in_mem_read
  wire width 32 \incr_pc
  wire width 32 \_in_incr_pc
  wire width 2 \wrd_sel
  wire width 2 \_in_wrd_sel
  wire width 5 \_in_which_rd
  wire width 7 \_in_op
  wire width 1 input 0  \clk
  wire width 1 input 1  \rst
  wire width 1 input 2  \in_from_prev_stall
  wire width 32 input 3  \in_from_prev_mem_read
  wire width 7 input 4  \in_from_prev_op
  wire width 32 input 5  \in_from_prev_alu_out
  wire width 32 input 6  \in_from_prev_incr_pc
  wire width 5 input 7  \in_from_prev_which_rd
  wire width 2 input 8  \in_from_prev_wrd_sel
  wire width 5 output 9  \which_rd
  wire width 32 output 10  \rd
  wire width 1 $1
  wire width 1 $2
  wire width 1 $3
  wire width 1 $4
  wire width 7 $5
  wire width 32 $6
  wire width 32 $7
  wire width 32 $8
  wire width 5 $9
  wire width 2 $10
  cell \top.wb.reg_mux \reg_mux
    connect \alu_out \alu_out [31:0]
    connect \incr_pc \incr_pc [31:0]
    connect \mem_read \mem_read [31:0]
    connect \wrd_sel \wrd_sel [1:0]
    connect \rd \rd [31:0]
  end
  cell $or $11
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A \_in_stall [0]
    connect \B 1'0
    connect \Y $1
  end
  cell $or $12
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A $1 [0]
    connect \B 1'0
    connect \Y \true_stall
  end
  cell $eq $13
    parameter \A_SIGNED 0
    parameter \B_SIGNED 0
    parameter \A_WIDTH 1
    parameter \B_WIDTH 0
    parameter \Y_WIDTH 1
    connect \A \true_stall [0]
    connect \B {  }
    connect \Y $2
  end
  cell $not $14
    parameter \A_SIGNED 0
    parameter \A_WIDTH 1
    parameter \Y_WIDTH 1
    connect \A 1'0
    connect \Y $3
  end
  process $15
    assign \which_rd [4:0] 5'00000
    switch $2 [0]
      case 1'1
        assign \which_rd [4:0] \_in_which_rd [4:0]
    end
  end
  process $16
    assign $4 [0] \_in_stall [0]
    switch $3 [0]
      case 1'1
        assign $4 [0] \in_from_prev_stall [0]
    end
    switch \rst [0]
      case 1'1
        assign $4 [0] 1'0
    end
  end
  cell $dff $17
    parameter \WIDTH 1
    parameter \CLK_POLARITY 1
    connect \D $4 [0]
    connect \CLK \clk [0]
    connect \Q \_in_stall
  end
  process $18
    assign $5 [6:0] \_in_op [6:0]
    switch $3 [0]
      case 1'1
        assign $5 [6:0] \in_from_prev_op [6:0]
    end
    switch \rst [0]
      case 1'1
        assign $5 [6:0] 7'0000000
    end
  end
  cell $dff $19
    parameter \WIDTH 7
    parameter \CLK_POLARITY 1
    connect \D $5 [6:0]
    connect \CLK \clk [0]
    connect \Q \_in_op
  end
  process $20
    assign $6 [31:0] \alu_out [31:0]
    switch $3 [0]
      case 1'1
        assign $6 [31:0] \in_from_prev_alu_out [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $6 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $21
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $6 [31:0]
    connect \CLK \clk [0]
    connect \Q \alu_out
  end
  process $22
    assign $7 [31:0] \incr_pc [31:0]
    switch $3 [0]
      case 1'1
        assign $7 [31:0] \in_from_prev_incr_pc [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $7 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $23
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $7 [31:0]
    connect \CLK \clk [0]
    connect \Q \incr_pc
  end
  process $24
    assign $8 [31:0] \mem_read [31:0]
    switch $3 [0]
      case 1'1
        assign $8 [31:0] \in_from_prev_mem_read [31:0]
    end
    switch \rst [0]
      case 1'1
        assign $8 [31:0] 32'00000000000000000000000000000000
    end
  end
  cell $dff $25
    parameter \WIDTH 32
    parameter \CLK_POLARITY 1
    connect \D $8 [31:0]
    connect \CLK \clk [0]
    connect \Q \mem_read
  end
  process $26
    assign $9 [4:0] \_in_which_rd [4:0]
    switch $3 [0]
      case 1'1
        assign $9 [4:0] \in_from_prev_which_rd [4:0]
    end
    switch \rst [0]
      case 1'1
        assign $9 [4:0] 5'00000
    end
  end
  cell $dff $27
    parameter \WIDTH 5
    parameter \CLK_POLARITY 1
    connect \D $9 [4:0]
    connect \CLK \clk [0]
    connect \Q \_in_which_rd
  end
  process $28
    assign $10 [1:0] \wrd_sel [1:0]
    switch $3 [0]
      case 1'1
        assign $10 [1:0] \in_from_prev_wrd_sel [1:0]
    end
    switch \rst [0]
      case 1'1
        assign $10 [1:0] 2'00
    end
  end
  cell $dff $29
    parameter \WIDTH 2
    parameter \CLK_POLARITY 1
    connect \D $10 [1:0]
    connect \CLK \clk [0]
    connect \Q \wrd_sel
  end
connect \stall_from_top_next 1'0
connect \stall_from_top 1'0
connect \out_stall \true_stall [0]
connect \_in_alu_out \alu_out [31:0]
connect \_in_mem_read \mem_read [31:0]
connect \_in_incr_pc \incr_pc [31:0]
connect \_in_wrd_sel \wrd_sel [1:0]
end
module \top.wb.reg_mux
  wire width 1 \invalid
  wire width 32 input 0  \alu_out
  wire width 32 input 1  \incr_pc
  wire width 32 input 2  \mem_read
  wire width 2 input 3  \wrd_sel
  wire width 32 output 4  \rd
  process $1
    assign \rd [31:0] 32'00000000000000000000000000000000
    switch \wrd_sel [1:0]
      case 2'01
        assign \rd [31:0] \alu_out [31:0]
      case 2'10
        assign \rd [31:0] \mem_read [31:0]
      case 2'11
        assign \rd [31:0] \incr_pc [31:0]
      case
    end
  end
  process $2
    assign \invalid [0] 1'0
    switch \wrd_sel [1:0]
      case 2'01
      case 2'10
      case 2'11
      case
        assign \invalid [0] 1'1
    end
  end
end
