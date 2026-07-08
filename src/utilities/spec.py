from typing import Tuple

from src.set.box import Box
from src.set.set import Set
import numpy.typing as npt
import numpy as np


class Spec:
    """
    Extract input and output from spec
    """

    @staticmethod
    def getInput(spec) -> Set:
        """
        Return an input set
        :param spec: A list of arrays
        :type spec: List[npt.ArrayLike]
        :return: (objSet -> Set)
        """
        # input size
        inputSize: int = len(spec[0][0])
        cvecLow: npt.ArrayLike = np.array([0.0 for i in range(inputSize)])
        cvecHigh: npt.ArrayLike = np.array([0.0 for i in range(inputSize)])
        for i in range(inputSize):
            cvecLow[i] = spec[0][0][i][0]
            cvecHigh[i] = spec[0][0][i][1]

        objSet: Set = Box(cvecLow, cvecHigh)

        return objSet

    @staticmethod
    def getOutput(spec) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Return an output in the form of (A, b) Ax <= b
        :param spec: A list of arrays
        :type spec: List[npt.ArrayLike]
        :return: ((A,b) -> Tuple[npt.ArrayLike, npt.ArrayLike])
        """
        listA = []
        listB = []
        for i in range(len(spec[0][1])):
            listA.append(spec[0][1][i][0])
            listB.append(spec[0][1][i][1])

        listA = np.array(listA)
        listB = np.array(listB)

        return listA, listB

    def display(outputConstr: npt.ArrayLike)->str:
        """
        Return formatted output string
        """
        numOfProperties: int = len(outputConstr[0])
        propertyMsg = ""
        for i in range(numOfProperties):
            propertyMsg += "        Property "+str(i+1)+"\n"
            A = outputConstr[0][i]
            b = outputConstr[1][i]
            for j in range(A.shape[0]):
                propertyMsg += "        "+str(A[j]) + "      "+ str(b[j]) + "\n"

        return propertyMsg