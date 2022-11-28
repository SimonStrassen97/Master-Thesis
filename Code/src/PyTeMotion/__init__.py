import os
import sys
sys.path.append(os.path.realpath('..\src\PyTeMotion\Support.AxisWrappers\\bin\Debug'))


import clr
clr.AddReference("Support.AxisWrappers")

from typing import SupportsFloat as Numeric

from Support.AxisWrappers import AxisWrapper
from Support.AxisWrappers import ConfigurationHelper



class Axis:

    def __init__(self, modul: str, axis: str, configfile: str) -> None:
        self.axisAbstraction = AxisWrapper(ConfigurationHelper.GetAxis(modul, axis, configfile), modul+"."+axis+"_log")
        pass

    def StartTeControl(self) -> None:
        self.axisAbstraction.StartTeControl()
        pass

    def Initialize(self) -> None:
        self.axisAbstraction.Initialize()
        pass

    def MoveTo(self, target: Numeric) -> None:
        self.axisAbstraction.MoveTo(float(target))
        pass

    def MoveFor(self, distance: Numeric) -> None:
        self.axisAbstraction.MoveFor(float(distance))
        pass

    def GetCurrentPosition(self) -> float:
        return self.axisAbstraction.GetCurrentPosition()

    def WriteLog(self) -> None:
        self.axisAbstraction.WriteLog()
        pass
