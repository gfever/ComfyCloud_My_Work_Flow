import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    raw = f.read()
first = raw.find('{')
if first==-1:
    print('no json')
    sys.exit(1)
js = raw[first:]
obj = json.loads(js)
print('Top type:', type(obj))
print('Has keys:', list(obj.keys())[:10])
cells = obj.get('cells')
print('cells type:', type(cells))
if isinstance(cells, list) and len(cells)>0:
    c0 = cells[0]
    print('cell0 type:', c0.get('cell_type'))
    src = c0.get('source')
    print('source type:', type(src))
    if isinstance(src, list) and len(src)>0:
        print('first source element type:', type(src[0]))
        s0 = src[0]
        print('first 200 chars of source[0]:')
        print(repr(s0[:200]))
    else:
        print('source empty or not list')
else:
    print('cells missing or not list')

