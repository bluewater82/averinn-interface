"""
Author: Ratan Lal
Date : 20 September, 2020
"""

class DtDyn:
	"""
	Class representing discrete time linear dynamics
		X(k+1) = AX(k) + Bu(k) + C
	Attributes:
		intDim (int): dimension of the system
		intIpDim (int): dimension of inputs
		A (matrix): system matrix of size intDim x intDim
		B (matrix): control input matrix of size intDim x intIpDim	
		C (column vector): matrix of size nx1
	"""
	
	def __init__(self, intDim=None, intIpDim=None, A=None, B=None, C=None):
		self.intDim = intDim
		self.intIpDim = intIpDim
		self.A = A
		self.B = B
		self.C = C

		
