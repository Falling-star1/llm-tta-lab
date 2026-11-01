"""Synthetic CPU plumbing check. Does not load data, models or GPU libraries."""
import json
from pathlib import Path
import sys

output=Path(sys.argv[1])
output.mkdir(parents=True,exist_ok=False)
(output/'metrics.json').write_text(json.dumps({
    'smoke_check':1,'quality_status':'NOT_APPLICABLE',
    'note':'Synthetic CPU workflow check; not a scientific evaluation.'
},indent=2)+'\n',encoding='utf-8')
print('CPU workflow smoke completed')
