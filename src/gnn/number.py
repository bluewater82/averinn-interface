"""
Author: Ratan Lal
Date : November 4, 2024
"""
from abc import ABCMeta, abstractmethod
from src.types.datatype import DataType


class Number(metaclass=ABCMeta):
    """
    Abstract class for real and interval number
    """

    @abstractmethod
    def getLower(self) -> DataType.RealType:
        """
        Return lower bound of the interval or just real number
        :return: (floatLow: float)
        Lower bound of the interval or just real number
        """
        pass

    @abstractmethod
    def getUpper(self) -> DataType.RealType:
        """
        Return upper bound of the interval or just real number
        :return: (floatHigh: float)
        Upper bound of the interval or just real number
        """
        pass

    @abstractmethod
    def setLower(self, value: DataType.RealType):
        """
        Set value with the lower bound of the number
        :param value: a real number
        :type value: DataType.RealType
        """
        pass

    @abstractmethod
    def setUpper(self, value: DataType.RealType) -> DataType.RealType:
        """
        Set value with the upper bound of the number
        :param value: a real number
        :type value: DataType.RealType
        """
        pass
