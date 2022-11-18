import os
import sys
sys.path.append(os.path.realpath('..\Support.AxisWrappers\\bin\Debug'))

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


if __name__ == "__main__":

    CGA_y = Axis("CGA", "y",  '..\..\..\..\ConfigurationData\Fluent_CGA_Config.xml')

    CGA_y.StartTeControl()
    CGA_y.Initialize()

    for x in [111.0, 222.0, 333.0, 222.0, 111.0, 123.0]:
        CGA_y.MoveTo(x)
        pass
