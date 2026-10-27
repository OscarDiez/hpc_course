"""Use ParaView's Python, not the notebook's Python, for this file."""
from array import array
from pathlib import Path
import sys
from paraview.simple import (OpenDataFile, GetActiveViewOrCreate, Show, ColorBy,
                            GetColorTransferFunction, SaveScreenshot)

n = int(sys.argv[1])
values = array('d')
values.frombytes(Path(sys.argv[2]).read_bytes())
assert len(values) == n*n
# VTK adds grid geometry and a scalar name to the solver's raw numbers.
with open('temperature.vtk', 'w') as file:
    file.write(f'# vtk DataFile Version 3.0\nHeat plate, Celsius\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS {n} {n} 1\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA {n*n}\nSCALARS Temperature double 1\nLOOKUP_TABLE default\n')
    file.write('\n'.join(map(str, values)))
field = OpenDataFile(str(Path('temperature.vtk').resolve()))
field.UpdatePipeline()
assert field.GetDataInformation().GetNumberOfPoints() == n*n
assert field.PointData['Temperature'].GetRange(0) == (0.0, 100.0)
view = GetActiveViewOrCreate('RenderView')
image = Show(field, view)
ColorBy(image, ('POINTS', 'Temperature'))
GetColorTransferFunction('Temperature').RescaleTransferFunction(0, 100)
image.SetScalarBarVisibility(view, True)
view.CameraParallelProjection = 1
view.CameraPosition = [(n-1)/2, (n-1)/2, 2*n]
view.CameraFocalPoint = [(n-1)/2, (n-1)/2, 0]
view.CameraViewUp = [0, -1, 0]
view.CameraParallelScale = n*0.6
# Unlike simple.Render, StillRender does not create an X-window interactor.
view.StillRender()
SaveScreenshot('paraview.png', view, ImageResolution=[900, 700])
print('PARAVIEW_PASS')
