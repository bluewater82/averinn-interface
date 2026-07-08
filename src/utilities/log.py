"""
Author: Ratan Lal
Date : November 4, 2023
"""


class Log:

    @staticmethod
    def message(msg: str):
        """
        :param msg:  log related messages
        :type msg: str
        """
        f = open("log.txt", 'a')
        f.write(msg)
        f.close()



































































         




















