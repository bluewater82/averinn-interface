\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x695807c5717372dc
Maximize
  0 q_1_1 + 0 q_1_2 + 0 q_3_1 + x_3_2 + 0 q_3_2
Subject To
 R0: x_1_1 >= 1
 R1: x_1_1 <= 2
 R2: x_1_2 >= 1
 R3: x_1_2 <= 2
 R4: x_1_1 - x_1_2 - x_2_1 <= 1
 R5: x_1_1 - x_1_2 - x_2_1 + 5 q_2_1 >= 1
 R6: x_2_1 >= 0
 R7: - x_2_1 - 5 q_2_1 >= -5
 R8: 0.5 x_1_1 + 1.5 x_1_2 - x_2_2 <= -1
 R9: 0.5 x_1_1 + 1.5 x_1_2 - x_2_2 + 5 q_2_2 >= -1
 R10: x_2_2 >= 0
 R11: - x_2_2 - 5 q_2_2 >= -5
 R12: x_2_1 + 0.5 x_2_2 - x_3_1 <= -1
 R13: x_2_1 + 0.5 x_2_2 - x_3_1 >= -1
 R14: - x_2_2 - x_3_2 <= -1
 R15: - x_2_2 - x_3_2 >= -1
Bounds
 x_1_1 free
 x_1_2 free
 x_2_1 free
 x_2_2 free
 x_3_1 free
 x_3_2 free
Binaries
 q_1_1 q_1_2 q_2_1 q_2_2 q_3_1 q_3_2
End
