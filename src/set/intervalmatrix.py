import numpy.typing as npt
import numpy as np

from src.set.intervalarray import IntervalArray
from src.types.datatype import DataType


class IntervalMatrix:
    """
    This class captures interval matrix
    """

    def __init__(self, matLow: npt.ArrayLike, matHigh: npt.ArrayLike):
        """
        Interval matrix is represented as a pair of two matrices
        :param matLow: a two-dimensional numpy array
        :type matLow: npt.ArrayLike
        :param matHigh: a two-dimensional numpy array
        :type matHigh: npt.ArrayLike
        """
        self.__matLow__: npt.ArrayLike = matLow
        self.__matHigh__: npt.ArrayLike = matHigh

    ############################################
    ########## Methods for attributes  #########
    ############################################
    def getMatLow(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the interval matrix
        :return: (matLow -> npt.ArrayLike)
        """
        return self.__matLow__

    def getMatHigh(self) -> npt.ArrayLike:
        """
        Returns the upper bound of the interval matrix
        :return: (matHigh -> npt.ArrayLike)
        """
        return self.__matHigh__

    ############################################
    ##########      Other Methods      #########
    ############################################
    def getNumberOfRows(self) -> int:
        """
        Returns the number of rows of the interval matrix
        :return: (numOfRows -> int)
        """
        return self.__matLow__.shape[0]

    def getNumberOfColumns(self) -> int:
        """
        Returns the number of rows of the interval matrix
        :return: (numOfRows -> int)
        """
        return self.__matLow__.shape[1]

    def getRowIAByIndex(self, rowIndex: int) -> IntervalArray:
        """
        Returns interval array in column index colIndex
        :param rowIndex: index of a row
        :type rowIndex: int
        :return: (objIA -> IntervalArray)
        """
        arrayLow: npt.ArrayLike = self.__matLow__[rowIndex, :]
        arrayHigh: npt.ArrayLike = self.__matHigh__[rowIndex, :]

        # Create an instance of IntervalArray
        objIA: IntervalArray = IntervalArray(arrayLow, arrayHigh)

        return objIA

    def getColumnIAByIndex(self, colIndex: int) -> IntervalArray:
        """
        Returns interval array in column index colIndex
        :param colIndex: index of a column
        :type colIndex: int
        :return: (objIA -> IntervalArray)
        """
        arrayLow: npt.ArrayLike = self.__matLow__[:, colIndex]
        arrayHigh: npt.ArrayLike = self.__matHigh__[:, colIndex]

        # Create an instance of IntervalArray
        objIA: IntervalArray = IntervalArray(arrayLow, arrayHigh)

        return objIA

    def setColumnIAByIndex(self, objIA: IntervalArray, colIndex: int):
        """
        Returns interval array in column index colIndex
        :param objIA: an instance of IntervalArray
        :type objIA: 'IntervalArray'
        :param colIndex: index of a column
        :type colIndex: int
        :return: None
        """
        self.__matLow__[:, colIndex] = objIA.getArrayLow()
        self.__matHigh__[:, colIndex] = objIA.getArrayHigh()

    def addition(self, objIM: 'IntervalMatrix') -> 'IntervalMatrix':
        """
        Returns the interval matrix with addition of two interval matrices
        :param objIM: an instance of IntervalMatrix
        :type objIM: 'IntervalMatrix'
        :return: (objIMAdd -> IntervalMatrix)
        """
        # Compute lower and upper bound on interval matrix
        row: int = self.getNumberOfRows()
        col: int = self.getNumberOfColumns()
        matLow: npt.ArrayLike = np.array([[0.0 for j in range(col)] for i in range(row)], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[0.0 for j in range(col)] for i in range(row)], dtype=np.float64)
        for i in range(row):
            for j in range(col):
                matLow[i][j] = min(self.__matLow__[i][j], objIM.__matLow__[i][j])
                matHigh[i][j] = max(self.__matHigh__[i][j], objIM.__matHigh__[i][j])

        # Compute lower and upper bound for sum of two interval matrices
        objIMAdd: IntervalMatrix = IntervalMatrix(matLow, matHigh)

        return objIMAdd

    def product(self, objIM: 'IntervalMatrix') -> 'IntervalMatrix':
        """
        Returns the interval matrix with addition of two interval matrices
        :param objIM: an instance of IntervalMatrix
        :type objIM: 'IntervalMatrix'
        :return: (objIMAdd -> IntervalMatrix)
        """
        matLowLow: npt.ArrayLike = np.matmul(self.__matLow__, objIM.getMatLow())
        matLowHigh: npt.ArrayLike = np.matmul(self.__matLow__, objIM.getMatHigh())
        matHighLow: npt.ArrayLike = np.matmul(self.__matHigh__, objIM.getMatLow())
        matHighHigh: npt.ArrayLike = np.matmul(self.__matHigh__, objIM.getMatHigh())

        # Compute lower and upper bound on interval matrix
        row: int = matLowLow.shape[0]
        col: int = matLowLow.shape[1]
        matLow: npt.ArrayLike = np.array([[0.0 for j in range(col)] for i in range(row)], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[0.0 for j in range(col)] for i in range(row)], dtype=np.float64)
        for i in range(row):
            for j in range(col):
                matLow[i][j] = min(matLowLow[i][j], matLowHigh[i][j], matHighLow[i][j], matHighHigh[i][j])
                matHigh[i][j] = max(matLowLow[i][j], matLowHigh[i][j], matHighLow[i][j], matHighHigh[i][j])

        # Construct an Interval Matrix
        objIMProd: IntervalMatrix = IntervalMatrix(matLow, matHigh)

        return objIMProd
