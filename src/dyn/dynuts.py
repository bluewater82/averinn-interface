from src.dyn.dtdyn import DtDyn
from src.set.set import Set


class DynUTS:
    """
    Utilities for DtDyn and other Dyn classes
    """

    @staticmethod
    def superposition(objDtDyn: DtDyn, objSetState: Set, objSetInput: Set) -> Set:
        """
        Compute next reach state in the form of a set
        :param objDyn:
        :param iniStarSet:
        :param ipStarSet:
        :return: (reachStarSet -> StarSet) next reach
        """
        nextSetState: Set = None

        if objDtDyn.B is None and objDtDyn.C is None:
            nextSetState = objSetState.linearMap(objDtDyn.A)
        elif objDtDyn.B is None and objDtDyn.C is not None:
            nextSetState = objSetState.affineMap(objDtDyn.A, objDtDyn.C)
        elif objDtDyn.B is not None and objDtDyn.C is None:
            # AX
            setState: Set = objSetState.linearMap(objDtDyn.A)
            # Bu
            ipSetState: Set = objSetInput.linearMap(objDtDyn.B)
            # AX + Bu
            nextSetState = setState.minkowskiSum(ipSetState)
        else:
            # AX
            setState: Set = objSetState.linearMap(objDtDyn.A)
            # Bu + C
            ipSetState: Set = objSetInput.affineMap(objDtDyn.B, objDtDyn.C)
            # AX + Bu + C
            nextSetState = setState.minkowskiSum(ipSetState)

        return nextSetState
