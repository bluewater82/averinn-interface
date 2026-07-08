"""
Author: Ratan Lal
Date : November 4, 2024
"""
from abc import ABC
from src.gnn.number import Number
from src.types.datatype import DataType


class Real(Number, ABC):
    """
    This class captures real number
    """

    def __init__(self, real: DataType.RealType):
        """

        :param floatNum: a real number
        :type floatNum: float
        """
        self.__real__ = real

    def getLower(self) -> DataType.RealType:
        """
        Returns lower bound of a real number
        :return: (real:float)
        a lower bound of a real number
        """
        return self.__real__

    def setLower(self, value: DataType.RealType):
        """
        Set lower bound of a real number
        :param value: a real value
        :type value: DataType.RealType
        """
        self.__real__ = value

    def getUpper(self) -> DataType.RealType:
        """
        Returns upper bound of a real number
        :return: (real: float)
        an upper bound of a real number
        """
        return self.__real__

    def setUpper(self, value: DataType.RealType):
        """
        Set upper bound of a real number
        :param value: a real value
        :type value: DataType.RealType
        """
        self.__real__ = value
