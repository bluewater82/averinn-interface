\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x9013bc69ed466a85
Maximize
  + Constant
Subject To
 R0: X_1 - 2 alpha_1 - 4 alpha_2 >= 3
 R1: X_2 - alpha_1 - 3 alpha_2 >= 2
 R2: X_1 - 3 alpha_1 - 5 alpha_2 <= 4
 R3: X_2 - 2 alpha_1 - 4 alpha_2 <= 3
 R4: alpha_1 + alpha_2 <= 1
 R5: - alpha_1 <= 0
 R6: - alpha_2 <= 0
 R7: alpha_1 <= 1
 R8: alpha_2 <= 1
 R9: - 3 alpha_1 - 5 alpha_2 <= 4
 R10: alpha_1 + 3 alpha_2 <= -2
Bounds
 X_1 free
 X_2 free
 alpha_1 free
 alpha_2 free
 Constant = 1
End
