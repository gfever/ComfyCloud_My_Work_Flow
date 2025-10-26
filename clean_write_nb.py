import sys
import json
import nbformat

if len(sys.argv) < 2:
    print("Usage: python clean_write_nb.py <input.ipynb> [output.ipynb]")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2] if len(sys.argv) > 2 else infile

with open(infile, 'r', encoding='utf-8') as f:
    raw = f.read()

# find JSON start
first = raw.find('{')
if first == -1:
    print('No JSON object found in file')
    sys.exit(2)

js = raw[first:]
try:
    nb = json.loads(js)
except Exception as e:
    print('Failed to parse JSON:', e)
    sys.exit(3)

# If first cell is raw and contains JSON, extract
cells = nb.get('cells')
if isinstance(cells, list) and len(cells) and cells[0].get('cell_type') == 'raw':
    first_src = cells[0].get('source')
    if isinstance(first_src, list) and first_src and isinstance(first_src[0], str) and first_src[0].lstrip().startswith('{'):
        try:
            inner = json.loads(''.join(first_src))
            if isinstance(inner, dict) and 'cells' in inner:
                nb = inner
                print('Extracted inner notebook JSON')
        except Exception as e:
            print('Failed to parse embedded JSON:', e)

# Ensure nbformat fields
nb.setdefault('nbformat', 4)
nb.setdefault('nbformat_minor', 0)

# Normalize cell source to list of strings
for cell in nb.get('cells', []):
    src = cell.get('source')
    if isinstance(src, str):
        cell['source'] = [src]

# Convert to NotebookNode and write using nbformat
try:
    nbnode = nbformat.from_dict(nb)
    with open(outfile, 'w', encoding='utf-8') as f:
        nbformat.write(nbnode, f)
    print('Wrote cleaned notebook to', outfile)
except Exception as e:
    print('Failed to write notebook via nbformat:', e)
    sys.exit(4)

# Quick validation
try:
    nb2 = nbformat.read(outfile, as_version=4)
    print('Validation ok: cells=', len(nb2.get('cells', [])), 'nbformat=', nb2.nbformat)
except Exception as e:
    print('Validation failed:', e)
    sys.exit(5)

