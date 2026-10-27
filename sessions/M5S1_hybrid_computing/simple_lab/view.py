# Run with pvbatch. The solver has already written the grid in VTK format.
from pathlib import Path
import sys
from paraview.simple import (OpenDataFile, GetActiveViewOrCreate, Show, ColorBy,
                            GetColorTransferFunction, SaveScreenshot)

n = int(sys.argv[1])
field = OpenDataFile(str(Path(sys.argv[2]).resolve()))
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
view.StillRender()
SaveScreenshot(sys.argv[3], view, ImageResolution=[900, 700])
print('PARAVIEW_PASS')
