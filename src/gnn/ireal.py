"""
Author: Ratan Lal
Date : November 4, 2024
"""
from abc import ABC
from src.gnn.number import Number
from src.types.datatype import DataType


class IReal(Number, ABC):
    """
    This class captures real number
    """

    def __init__(self, floatLow: DataType.RealType, floatHigh: DataType.RealType):
        """

        :param floatNum: a real number
        :type floatNum: float
        """
        self.__floatLow__ = floatLow
        self.__floatHigh__ = floatHigh

    def getLower(self) -> DataType.RealType:
        """
        Returns lower bound of an interval
        :return: (floatLow:float)
        a lower bound of an interval
        """
        return self.__floatLow__

    def setLower(self, value: DataType.RealType):
        """
        Set lower bound of an interval
        :param value: a real value
        :type value: DataType.RealType
        """
        self.__floatLow__ = value

    def getUpper(self) -> DataType.RealType:
        """
        Returns upper bound of an interval
        :return: (floatHigh: float)
        an upper bound of a real number
        """
        return self.__floatHigh__

    def setUpper(self, value: DataType.RealType):
        """
        Set upper bound of an interval
        :param value: a real value
        :type value: DataType.RealType
        """
        self.__floatHigh__ = value
