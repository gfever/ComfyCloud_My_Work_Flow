import sys
import nbformat
import json

if len(sys.argv) < 2:
    print("Usage: python convert_nb_to_v4.py <input.ipynb> [output.ipynb]")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2] if len(sys.argv) > 2 else infile

print(f"Reading {infile} (will convert to nbformat v4)...")
# Read raw text and strip any non-JSON wrapper (like ```jupyter and filepath comments)
with open(infile, 'r', encoding='utf-8') as f:
    raw = f.read()

# Find the first '{' that starts the JSON document
first_brace = raw.find('{')
if first_brace == -1:
    print("Error: no JSON object found in the input file")
    sys.exit(2)

json_text = raw[first_brace:]

# Load via json to allow editing nbformat field even if nbformat library is older
try:
    nb_dict = json.loads(json_text)
except Exception as e:
    print("Failed to parse JSON directly:", e)
    sys.exit(3)

# If file was double-wrapped: first cell is raw and contains the full notebook JSON as text
cells = nb_dict.get('cells')
if isinstance(cells, list) and len(cells) > 0 and cells[0].get('cell_type') == 'raw':
    first_src = cells[0].get('source')
    if isinstance(first_src, list) and len(first_src) > 0 and isinstance(first_src[0], str) and first_src[0].lstrip().startswith('{'):
        print('Detected notebook JSON embedded in first raw cell — extracting...')
        embedded_json = ''.join(first_src)
        try:
            inner = json.loads(embedded_json)
            # if inner looks like a notebook, replace nb_dict with inner
            if isinstance(inner, dict) and 'cells' in inner:
                nb_dict = inner
        except Exception as e:
            print('Failed to parse embedded JSON:', e)

# Ensure nbformat fields exist and are compatible
if not isinstance(nb_dict, dict):
    print('Parsed notebook is not a dict')
    sys.exit(4)

nb_dict.setdefault('nbformat', 4)
nb_dict.setdefault('nbformat_minor', 0)

# Normalize cells: ensure each cell source is list of strings
for cell in nb_dict.get('cells', []):
    src = cell.get('source')
    if isinstance(src, str):
        cell['source'] = [src]

# Write the cleaned notebook directly as JSON
with open(outfile, 'w', encoding='utf-8') as f:
    json.dump(nb_dict, f, ensure_ascii=False, indent=2)

print(f"Wrote converted notebook to {outfile}")
