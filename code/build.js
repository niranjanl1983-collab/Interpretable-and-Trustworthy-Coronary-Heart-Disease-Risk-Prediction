const fs = require('fs');
const D = require('docx');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  HeadingLevel, ShadingType, BorderStyle, LevelFormat, TabStopType, Footer, PageNumber, HighlightColor, ImageRun } = D;
function pngSize(f){const b=fs.readFileSync(f);return [b.readUInt32BE(16),b.readUInt32BE(20)];}
const C = require('./content.js');
const REFS = require('./refs.js');

const FONT = 'Times New Roman';
const W = 9638; // text width (A4, 2 cm margins)

// ---- citation numbering by first appearance ----
const order = [];
function cite(text) {
  return text.replace(/\{([a-z0-9_,]+)\}/g, (m, keys) => {
    if (!keys.split(',').every(k => REFS[k])) return m;
    const nums = keys.split(',').map(k => {
      if (!REFS[k]) throw new Error('missing ref ' + k);
      if (!order.includes(k)) order.push(k);
      return order.indexOf(k) + 1;
    }).sort((a, b) => a - b);
    // compress ranges
    const out = []; let i = 0;
    while (i < nums.length) {
      let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
      out.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : nums.slice(i, j + 1).join(', '));
      i = j + 1;
    }
    return '[' + out.join(', ') + ']';
  });
}

// ---- inline markup ----
function runs(text, base = {}) {
  text = cite(text);
  const parts = text.split(/(\[\[[^\]]*\]\]|\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*)/);
  const out = [];
  for (const s of parts) {
    if (!s) continue;
    if (s.startsWith('[[')) out.push(new TextRun({ ...base, text: s, font: FONT, highlight: 'yellow' }));
    else if (s.startsWith('***')) out.push(new TextRun({ ...base, text: s.slice(3, -3), bold: true, italics: true, font: FONT }));
    else if (s.startsWith('**')) out.push(new TextRun({ ...base, text: s.slice(2, -2), bold: true, font: FONT }));
    else if (s.startsWith('*') && s.length > 2) out.push(new TextRun({ ...base, text: s.slice(1, -1), italics: true, font: FONT }));
    else out.push(new TextRun({ ...base, text: s, font: FONT }));
  }
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
const borders = { top: border, bottom: border, left: border, right: border };

function table(o) {
  const out = [];
  out.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 200, after: 80 }, keepNext: true, children: runs('**' + o.title + '.** ' + o.caption, { size: 19 }) }));
  const total = o.widths.reduce((a, b) => a + b, 0);
  const widths = o.widths.map(w => Math.round(w * W / total));
  widths[widths.length - 1] += W - widths.reduce((a, b) => a + b, 0);
  const mk = (cells, header) => new TableRow({ tableHeader: header, children: cells.map((c, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: header ? { fill: 'E7E6E6', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { after: 0 }, children: runs(header ? '**' + c + '**' : c, { size: 16 }) })] })) });
  out.push(new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(o.headers, true), ...o.rows.map(r => mk(r, false))] }));
  if (o.note) out.push(new Paragraph({ spacing: { before: 40, after: 200 }, children: runs(o.note, { size: 16, italics: true }) }));
  return out;
}

// ---- automatic numbering of labelled tables and figures, in order of appearance ----
const TLAB = {}, FLAB = {}, ELAB = {}; let _e = 0;
for (const b of C) {
  if (b[0] === 'eq') { _e++; if (b[2]) ELAB[b[2]] = _e; }
  if (b[0] === 'tbl' && b[1].label) TLAB[b[1].label] = Object.keys(TLAB).length + 1;
  if (b[0] === 'img' && b[3]) FLAB[b[3]] = Object.keys(FLAB).length + 1;
}
const lab = t => typeof t !== 'string' ? t : t.replace(/\{T:([a-z0-9_]+)\}/g, (m, k) => { if (!TLAB[k]) throw new Error('table label ' + k); return TLAB[k]; })
  .replace(/\{F:([a-z0-9_]+)\}/g, (m, k) => { if (!FLAB[k]) throw new Error('figure label ' + k); return FLAB[k]; })
  .replace(/\{E:([a-z0-9_]+)\}/g, (m, k) => { if (!ELAB[k]) throw new Error('equation label ' + k); return ELAB[k]; });
for (const b of C) {
  if (b[0] === 'tbl') { const o = b[1]; if (o.label) o.title = 'Table ' + TLAB[o.label]; o.caption = lab(o.caption); o.note = lab(o.note); o.rows = o.rows.map(r => r.map(lab)); }
  else if (b[0] !== 'eq' && b[0] !== 'img') { b[1] = lab(b[1]); if (Array.isArray(b[2])) b[2] = b[2].map(lab); }
}
const body = []; const EQS = [];
let eqn = 0;
for (const b of C) {
  const [k, t] = b;
  if (k === 'notetitle') body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs('**' + t + '**', { size: 24, color: 'C00000' }) }));
  else if (k === 'note') body.push(new Paragraph({ spacing: { after: 140 }, shading: { fill: 'FFF2CC', type: ShadingType.CLEAR, color: 'auto' },
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'BF9000', space: 6 } }, children: runs(t, { size: 20 }) }));
  else if (k === 'pb') body.push(new Paragraph({ pageBreakBefore: true, children: [] }));
  else if (k === 'title') body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: runs('**' + t + '**', { size: 32 }) }));
  else if (k === 'author') body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs(t, { size: 20 }) }));
  else if (k === 'abstract') { body.push(new Paragraph({ spacing: { before: 240, after: 80 }, children: runs('**Abstract**', { size: 22 }) }));
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 }, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: '808080', space: 6 } }, children: runs(t, { size: 20 }) })); }
  else if (k === 'keywords') body.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 60, after: 240 }, children: runs('**Keywords:** ' + t, { size: 20 }) }));
  else if (k === 'affil') body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: runs(t, { size: 18, italics: true }) }));
  else if (k === 'h1') body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.LEFT, spacing: { before: 320, after: 120 }, keepNext: true, children: runs('**' + t + '**', { size: 24 }) }));
  else if (k === 'h2') body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, alignment: AlignmentType.LEFT, spacing: { before: 200, after: 80 }, keepNext: true, children: runs('**' + t + '**', { size: 22 }) }));
  else if (k === 'h3') body.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 140, after: 60 }, keepNext: true, children: runs('***' + t + '***', { size: 21 }) }));
  else if (k === 'p') body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 }, indent: { firstLine: 284 }, children: runs(t, { size: 21 }) }));
  else if (k === 'bul') body.push(new Paragraph({ numbering: { reference: 'bul', level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 264 }, children: runs(t, { size: 21 }) }));
  else if (k === 'eq') { eqn++; EQS.push(t); body.push(new Paragraph({ tabStops: [{ type: TabStopType.CENTER, position: W / 2 }, { type: TabStopType.RIGHT, position: W }], spacing: { before: 100, after: 140 }, keepLines: true,
      children: [new TextRun({ text: '\t', font: FONT, size: 21 }), new TextRun({ text: `EQPH${eqn}EQPH`, font: FONT, size: 21 }), new TextRun({ text: `\t(${eqn})`, font: FONT, size: 21 })] })); }
  else if (k === 'fig') body.push(new Paragraph({ spacing: { before: 120, after: 160 }, border: { top: { style: BorderStyle.DASHED, size: 4, color: '999999', space: 4 }, bottom: { style: BorderStyle.DASHED, size: 4, color: '999999', space: 4 } }, children: runs(t, { size: 18 }) }));
  else if (k === 'img') {
    const [pw, ph] = pngSize(t); const wpx = Math.round((b[2] || 1) * 642); const hpx = Math.round(wpx * ph / pw);
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: fs.readFileSync(t), transformation: { width: wpx, height: hpx } })] }));
  }
  else if (k === 'cap') body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: runs(t, { size: 18 }) }));
  else if (k === 'box') t.split('\n').forEach((ln, i, arr) => body.push(new Paragraph({ spacing: { before: i === 0 ? 120 : 0, after: i === arr.length - 1 ? 80 : 0, line: 240 },
      shading: { fill: 'F4F3EF', type: ShadingType.CLEAR, color: 'auto' }, keepNext: true, keepLines: true,
      border: { left: { style: BorderStyle.SINGLE, size: 12, color: '2A78D6', space: 6 } }, indent: { left: 200, right: 200 },
      children: [new TextRun({ text: ln.replace(/^ +/, m => '\u00A0'.repeat(m.length)), font: 'Consolas', size: 15 })] })));
  else if (k === 'tbl') body.push(...table(t));
  else if (k === 'alg') {
    const lines = b[2];
    body.push(new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [
      new TableRow({ children: [new TableCell({ borders, width: { size: W, type: WidthType.DXA }, shading: { fill: 'E7E6E6', type: ShadingType.CLEAR, color: 'auto' }, margins: { top: 60, bottom: 60, left: 100, right: 100 }, children: [new Paragraph({ children: runs('**' + t + '**', { size: 18 }) })] })] }),
      new TableRow({ children: [new TableCell({ borders, width: { size: W, type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 100, right: 100 }, children: lines.map(l => new Paragraph({ spacing: { after: 20 }, children: runs(l, { size: 18 }) })) })] }),
    ] }));
    body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  }
}

// references
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.LEFT, spacing: { before: 320, after: 120 }, children: runs('**References**', { size: 24 }) }));
order.forEach((k, i) => body.push(new Paragraph({ tabStops: [{ type: TabStopType.LEFT, position: 567 }], indent: { left: 567, hanging: 567 }, spacing: { after: 60 },
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 18 }), ...runs(REFS[k], { size: 18 })] })));
const unused = Object.keys(REFS).filter(k => !order.includes(k));
if (unused.length) console.log('UNUSED REFS:', unused.join(', '));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 22, bold: true, color: '000000' }, paragraph: { outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 21, italics: true, color: '000000' }, paragraph: { outlineLevel: 1 } },
    ] },
  numbering: { config: [{ reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 284 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(process.env.OUT || 'Manuscript_v4.docx', b); fs.writeFileSync('equations.json', JSON.stringify(EQS)); console.log('ok, refs:', order.length, 'eqs:', eqn); });
