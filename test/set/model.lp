\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xed145b7b9ac732e2
Maximize
  x_1
Subject To
 R0: x_0 - 2 alpha_0 - 4 alpha_1 >= 3
 R1: x_1 - alpha_0 - 3 alpha_1 >= 2
 R2: x_0 - 3 alpha_0 - 5 alpha_1 <= 4
 R3: x_1 - 2 alpha_0 - 4 alpha_1 <= 3
 R4: alpha_0 + alpha_1 <= 1
 R5: - alpha_0 <= 0
 R6: - alpha_1 <= 0
 R7: alpha_0 <= 1
 R8: alpha_1 <= 1
Bounds
 x_0 free
 x_1 free
 alpha_0 free
 alpha_1 free
End
