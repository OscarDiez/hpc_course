"""Generate temperature, profile and timing SVGs without extra Python packages."""
from visualize import write_plots
import sys
for path in write_plots(sys.argv[1]):print(path)
