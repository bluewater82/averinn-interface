import numpy as np
import numpy.typing as npt
from pandas.core.arrays import IntervalArray
from src.types.datatype import DataType


class IntervalArray:
    """
    The class IntervalArray captures a list of real intervals, expressed
    as a pair of two arrays
    """

    def __init__(self, arrayLow: npt.ArrayLike, arrayHigh: npt.ArrayLike):
        """
        Interval Array as a pair of low and high arrays
        :param arrayLow: a one dimensional array of real values
        :type arrayLow: npt.ArrayLike
        :param arrayHigh: a one dimensional array of real values
        :type arrayHigh: npt.ArrayLike
        """
        self.__arrayLow__: npt.ArrayLike = arrayLow
        self.__arrayHigh__: npt.ArrayLike = arrayHigh

    ############################################
    ########## Methods for attributes  #########
    ############################################
    def getArrayLow(self) -> npt.ArrayLike:
        """
        Returns the lower array
        :return: (arrayLow -> npt.ArrayLike)
        """
        return self.__arrayLow__

    def getArrayHigh(self) -> npt.ArrayLike:
        """
        Returns the upper array
        :return: (arrayHigh -> npt.ArrayLike)
        """
        return self.__arrayHigh__

    ############################################
    ##########      Other Methods      #########
    ############################################
    def addition(self, objIA: 'IntervalArray') -> 'IntervalArray':
        """
        Addition of two interval arrays
        :param objIA: an instance of IntervalArray
         :type objIA: 'IntervalArray'
        :return: (objSumAI -> 'IntervalArray')
        """
        arrayLowAdd: npt.ArrayLike = self.__arrayLow__ + objIA.getArrayLow()
        arrayHighAdd: npt.ArrayLike = self.__arrayHigh__ + objIA.getArrayHigh()

        # Create IntervalArray for addition
        objAIAdd: IntervalArray = IntervalArray(arrayLowAdd, arrayHighAdd)

        return objAIAdd

    def elementWiseProd(self, objIA: 'IntervalArray') -> 'IntervalArray':
        """
        Product of two interval arrays
        :param objIA: an instance of IntervalArray
         :type objIA: 'IntervalArray'
        :return: (objSumAI -> 'IntervalArray')
        """
        arrayLowLow: npt.ArrayLike = self.__arrayLow__ * objIA.getArrayLow()
        arrayLowHigh: npt.ArrayLike = self.__arrayLow__ * objIA.getArrayHigh()
        arrayHighLow: npt.ArrayLike = self.__arrayHigh__ * objIA.getArrayLow()
        arrayHighHigh: npt.ArrayLike = self.__arrayHigh__ * objIA.getArrayHigh()

        # Create IntervalArray for Product
        arrayLow: npt.ArrayLike = np.array([min(arrayLowLow[i], arrayLowHigh[i], arrayHighLow[i],
                                                arrayHighHigh[i]) for i in range(len(arrayLowLow))],
                                           dtype=DataType.RealType)
        arrayHigh: npt.ArrayLike = np.array([max(arrayLowLow[i], arrayLowHigh[i], arrayHighLow[i],
                                                 arrayHighHigh[i]) for i in range(len(arrayLowLow))],
                                            dtype=DataType.RealType)

        objAIProd: IntervalArray = IntervalArray(arrayLow, arrayHigh)

        return objAIProd
