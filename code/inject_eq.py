"""Replace equation placeholders in the manuscript with native Word (OMML) equations.
LaTeX is converted by pandoc; each <m:oMath> is placed inline between the centre and
right tab stops, so the equation is centred and its number right-aligned."""
import json, re, subprocess, zipfile, shutil, sys, os, tempfile

DOC = sys.argv[1] if len(sys.argv) > 1 else 'CertiRisk_Manuscript.docx'
eqs = json.load(open('equations.json'))
tmp = tempfile.mkdtemp()
md = '\n\nEQBREAK\n\n'.join('$$' + e + '$$' for e in eqs)
open(os.path.join(tmp, 'eq.md'), 'w').write(md)
subprocess.run(['pandoc', os.path.join(tmp, 'eq.md'), '-f', 'markdown', '-o', os.path.join(tmp, 'eq.docx')], check=True)
x = zipfile.ZipFile(os.path.join(tmp, 'eq.docx')).read('word/document.xml').decode('utf8')
maths = re.findall(r'<m:oMathPara>.*?</m:oMathPara>', x, re.S)
assert len(maths) == len(eqs), (len(maths), len(eqs))
inner = []
for m in maths:
    om = re.search(r'<m:oMath>.*</m:oMath>', m, re.S).group(0)
    assert '<m:r>' in om or '<m:f>' in om
    inner.append(om)


def _fix_schema(om):
    """Reorder OMML property children to the order required by the ECMA schema."""
    def rpr(m):
        body = m.group(1)
        kids = re.findall(r'<m:(lit|nor|scr|sty|brk|aln)\b[^>]*/>', body)
        tags = {k: re.search(r'<m:%s\b[^>]*/>' % k, body).group(0) for k in kids}
        order = ['lit'] + (['nor'] if 'nor' in tags else ['scr', 'sty']) + ['brk', 'aln']
        return '<m:rPr>' + ''.join(tags[k] for k in order if k in tags) + '</m:rPr>'
    om = re.sub(r'<m:rPr>(.*?)</m:rPr>', rpr, om, flags=re.S)
    def dpr(m):
        body = m.group(1)
        parts = re.findall(r'<m:(begChr|sepChr|endChr|grow|shp|ctrlPr)\b(?:[^>]*/>|.*?</m:\1>)', body, re.S)
        tags = {}
        for k in ['begChr', 'sepChr', 'endChr', 'grow', 'shp', 'ctrlPr']:
            mm = re.search(r'<m:%s\b(?:[^>]*/>|.*?</m:%s>)' % (k, k), body, re.S)
            if mm: tags[k] = mm.group(0)
        return '<m:dPr>' + ''.join(tags[k] for k in ['begChr', 'sepChr', 'endChr', 'grow', 'shp', 'ctrlPr'] if k in tags) + '</m:dPr>'
    om = re.sub(r'<m:dPr>(.*?)</m:dPr>', dpr, om, flags=re.S)
    return om

inner = [_fix_schema(o) for o in inner]
src = zipfile.ZipFile(DOC)
doc = src.read('word/document.xml').decode('utf8')
for k, om in enumerate(inner, 1):
    pat = re.compile(r'<w:r>(?:(?!<w:r>).)*?EQPH%dEQPH</w:t></w:r>' % k, re.S)
    doc, n = pat.subn(lambda _: om, doc, count=1)
    assert n == 1, ('placeholder not found', k)
assert 'EQPH' not in doc
out = DOC + '.tmp'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for it in src.infolist():
        data = doc.encode('utf8') if it.filename == 'word/document.xml' else src.read(it.filename)
        z.writestr(it, data)
src.close(); shutil.move(out, DOC)
print('injected', len(inner), 'equations')
