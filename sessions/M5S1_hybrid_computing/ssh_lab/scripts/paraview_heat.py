"""Run with pvpython, not the notebook's Python interpreter."""
from pathlib import Path
import json,sys
from paraview.simple import (OpenDataFile,GetActiveViewOrCreate,Show,ColorBy,
 GetColorTransferFunction,Render,SaveScreenshot,# A state file saves the pipeline/view settings, not the input field itself.
SaveState,SaveData,PlotOverLine)
source=Path(sys.argv[1]).resolve();n=int(sys.argv[2]);out=Path(sys.argv[3]).resolve();out.mkdir(parents=True,exist_ok=True)
# Read the grid into ParaView and confirm the dimensions and scalar range.
field=OpenDataFile(str(source));field.UpdatePipeline()
info=field.GetDataInformation();points=info.GetNumberOfPoints()
array_info=field.GetPointDataInformation().GetArrayInformation('Temperature')
if array_info is None:raise RuntimeError('Temperature array missing')
low,high=array_info.GetComponentRange(0)
assert points==n*n,(points,n*n)
assert abs(low)<1e-10 and abs(high-100)<1e-10,(low,high)
# VTI stores the structured field; the CSV samples the centre column.
SaveData(str(out/'temperature.vti'),proxy=field)
line=PlotOverLine(Input=field)
geometry=line if hasattr(line,'Point1') else line.Source
geometry.Point1=[n//2,0,0];geometry.Point2=[n//2,n-1,0];geometry.Resolution=n-1
line.UpdatePipeline();SaveData(str(out/'centre-line.csv'),proxy=line)
# Build a view with a fixed 0-100 Celsius colour scale for honest comparisons.
view=GetActiveViewOrCreate('RenderView');display=Show(field,view)
ColorBy(display,('POINTS','Temperature'));GetColorTransferFunction('Temperature').RescaleTransferFunction(0,100)
display.SetScalarBarVisibility(view,True)
view.ViewSize=[1000,800];view.CameraParallelProjection=1
view.CameraPosition=[(n-1)/2,(n-1)/2,2*n];view.CameraFocalPoint=[(n-1)/2,(n-1)/2,0]
view.CameraViewUp=[0,-1,0];view.CameraParallelScale=n*.6
# The batch launcher requests offscreen rendering; save the actual rendered image.
Render(view);SaveScreenshot(str(out/'paraview-temperature.png'),view,ImageResolution=[1000,800])
SaveState(str(out/'heat.pvsm'))
assert (out/'paraview-temperature.png').is_file()
(out/'validation.json').write_text(json.dumps({'points':points,'n':n,'minimum':low,'maximum':high,'status':'PASS'},indent=2))
print('PARAVIEW_VALIDATION_PASS',points,'points, range',low,high)
