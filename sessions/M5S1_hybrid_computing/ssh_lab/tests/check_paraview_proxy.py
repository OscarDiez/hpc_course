"""Check the ParaView metadata API without pretending to render an image.

ParaView's Python PointData wrapper returns ArrayInformation with GetRange.
It is not the raw VTK object exposing GetArrayInformation/GetComponentRange.
"""
from pathlib import Path
from types import SimpleNamespace
script=(Path(__file__).resolve().parents[1]/'scripts/paraview_heat.py').read_text()
validation=script[script.index('field=OpenDataFile'):script.index('# VTI stores')]
class ArrayInfo:
 def GetRange(self,component):
  assert component==0
  return (0.,100.)
class Field:
 PointData={'Temperature':ArrayInfo()}
 def UpdatePipeline(self):pass
 def GetDataInformation(self):return SimpleNamespace(GetNumberOfPoints=lambda:128*128)
namespace={'OpenDataFile':lambda path:Field(),'source':'field.vtk','n':128}
exec(validation,namespace)
assert namespace['points']==16384 and namespace['low']==0 and namespace['high']==100
print('PASS ParaView Python-proxy metadata access; actual rendering requires pvpython')
