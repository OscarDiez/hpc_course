import csv, hashlib, json
from pathlib import Path
path=Path('temperatures.csv')
with path.open(newline='', encoding='utf-8') as f:
    count=sum(1 for row in csv.DictReader(f))
record={'rows':count, 'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'message':'The container read the saved host CSV and wrote this file.'}
Path('written_by_container.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
