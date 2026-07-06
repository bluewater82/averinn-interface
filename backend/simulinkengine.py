import matlab.engine as engine
import matlab
import os
import pandas
import numpy as np

class sl:
    def __init__(self, link:str):
        self.__link__ = link
        self.eng = engine.start_matlab()
        self.eng.load_system(link)
        self.model_name = os.path.splitext(os.path.basename(link.strip("/")))[0]
        self.inport_paths = self.eng.find_system(self.model_name, "searchdepth", "1", "blocktype", "inport")
        self.outport_paths = self.eng.find_system(self.model_name, "searchdepth", "1", "blocktype", "outport")
        self.hasNNChanged = False

    def getModel(self):
        return self.eng

    def getNumOut(self):
        return len(self.outport_paths)

    def getNumIn(self):
        return len(self.inport_paths)

    def hasNNChanged(self):
        return self.hasNNChanged

    @staticmethod
    def getfstr(num: int):
        r = ""
        for i in range(num):
            r += f"var{i}{',' if i != num - 1 else ''} "
        return r

    @staticmethod
    def getlsstr(num: int):
        r = []
        for i in range(num):
            r.append(f"var{i}")
        return r
    def params(self, stepSize, timePeriod):
        self.eng.set_param(self.model_name, "MaxStep", f"{stepSize}", nargout=0)
        self.eng.set_param(self.model_name, "StartTime", f"0.0", nargout=0)
        self.eng.set_param(self.model_name, "StopTime", f"{timePeriod}", nargout=0)

    def params(self, stepSize, timePeriod, initState):
        self.eng.set_param(self.model_name, "MaxStep", f"{stepSize}", nargout=0)
        self.eng.set_param(self.model_name, "StartTime", f"0.0", nargout=0)
        self.eng.set_param(self.model_name, "StopTime", f"{timePeriod}", nargout=0)
        delayXBlock = self.eng.find_system(self.model_name, "blocktype", "UnitDelay")
        print(delayXBlock)
        for block in delayXBlock:
            self.eng.set_param(block, "InitialCondition", initState, nargout=0)


    def changeModelNN(self, path: str):
        modelNN = self.eng.find_system(self.model_name, "masktype", "ONNXModelPredict")
        assert len(modelNN)
        self.hasNNChanged = True
        for n in modelNN:
            self.eng.set_param(n, "ModelFile", path, nargout=0)

    def close(self):
        self.eng.close_system(self.model_name, 0, nargout=0)
        del self

    def sim(self):
        self.eng.set_param(self.model_name, "SaveFormat", "dataset", nargout=0)

        ls = self.eng.sim(self.model_name)
        self.eng.workspace['yout'] = self.eng.get(ls, "yout")
        self.eng.workspace['b'] = self.eng.get(ls, "tout")
        self.eng.eval('a = extractTimetable(yout).Data_1;', nargout=0)
        #self.eng.eval('b = extractTimetable(tout)', nargout=0)

        yout = np.array(self.eng.workspace['a'])
        print(yout)
        tout = np.array(self.eng.workspace['b'])

        return [tout, yout]

a = sl("./ri_simulink")