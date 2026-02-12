\ Model _copy
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x7eb614cb2bae8625
Maximize
  + Constant
Subject To
 R0: X_1 - 46.523798685 alpha_1 - 5.65591e-05 alpha_3
   - 18.45405071243239 alpha_4 + 1.72879046894266 alpha_5
   + 1.557136378192311 alpha_6 >= 129.0628291490991
 R1: X_2 - 0.0996513961473047 alpha_2 - 8.61207e-05 alpha_3
   >= 20.90888409748963
 R2: X_3 + 1.7241822179750175e-04 alpha_2 - 2.68574e-05 alpha_3
   >= -2.023831000592401
 R3: X_4 - 8.61207e-05 alpha_3 - 18.44107652006983 alpha_4
   + 2.632374699498443 alpha_5 + 4.088672792964069 alpha_6
   >= 15.90148204948961
 R4: X_5 + 1.7241822179750175e-04 alpha_2 - 0.0319070052658575 alpha_4
   - 0.8209246713125223 alpha_5 - 6.225693054236311 alpha_6
   >= -5.845001178859809
 R5: X_1 - 46.523798685 alpha_1 - 5.65591e-05 alpha_3
   - 18.45405071243239 alpha_4 + 1.72879046894266 alpha_5
   + 1.557136378192311 alpha_6 <= 129.0628291490991
 R6: X_2 - 0.0996513961473047 alpha_2 - 8.61207e-05 alpha_3
   <= 20.90888409748963
 R7: X_3 + 1.7241822179750175e-04 alpha_2 - 2.68574e-05 alpha_3
   <= -2.023831000592401
 R8: X_4 - 8.61207e-05 alpha_3 - 18.44107652006983 alpha_4
   + 2.632374699498443 alpha_5 + 4.088672792964069 alpha_6
   <= 15.90148204948961
 R9: X_5 + 1.7241822179750175e-04 alpha_2 - 0.0319070052658575 alpha_4
   - 0.8209246713125223 alpha_5 - 6.225693054236311 alpha_6
   <= -5.845001178859809
 R10: alpha_1 <= 1
 R11: - alpha_1 <= 1
 R12: alpha_2 <= 1
 R13: - alpha_2 <= 1
 R14: alpha_3 <= 1
 R15: - alpha_3 <= 1
 R16: alpha_4 <= 1
 R17: - alpha_4 <= 1
 R18: alpha_5 <= 1
 R19: - alpha_5 <= 1
 R20: alpha_6 <= 1
 R21: - alpha_6 <= 1
 R22: - X_1 + X_4 + 1.4 X_5 <= -10
Bounds
 X_1 free
 X_2 free
 X_3 free
 X_4 free
 X_5 free
 alpha_1 free
 alpha_2 free
 alpha_3 free
 alpha_4 free
 alpha_5 free
 alpha_6 free
 Constant = 1
End
