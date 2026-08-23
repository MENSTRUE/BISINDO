#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv

VIDEO_EXTS = {'.mp4','.avi','.mov','.mkv','.webm','.mpeg','.mpg','.m4v','.wmv'}
IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
SPLITS = {
    'train':'Train','training':'Train',
    'test':'Test','testing':'Test',
    'val':'Validation','valid':'Validation','validation':'Validation','dev':'Validation'
}
STRUCTURAL = {
    'video','videos','image','images','train','training','test','testing',
    'val','valid','validation','dev','dataset','datasets','data'
}

def get_split(parts):
    for p in parts:
        k = p.lower().strip()
        if k in SPLITS:
            return SPLITS[k]
    return 'Unspecified'

def media_type(path):
    return 'Video' if path.suffix.lower() in VIDEO_EXTS else 'Image'

def infer_label(parts):
    if len(parts) < 2:
        return 'UNRESOLVED', 'File directly inside dataset root'
    parent = parts[-2]
    if parent.lower().strip() in STRUCTURAL:
        return 'UNRESOLVED', f"Parent folder '{parent}' looks structural"
    return parent, ''

def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def print_table(headers, rows, max_rows=100):
    if not rows:
        print('(empty)'); return
    rows = rows[:max_rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i,v in enumerate(row): widths[i] = min(max(widths[i], len(str(v))), 36)
    def cut(v,w):
        v=str(v); return v if len(v)<=w else v[:w-3]+'...'
    print(' | '.join(cut(h,widths[i]).ljust(widths[i]) for i,h in enumerate(headers)))
    print('-+-'.join('-'*w for w in widths))
    for row in rows:
        print(' | '.join(cut(v,widths[i]).ljust(widths[i]) for i,v in enumerate(row)))

def main():
    ap = argparse.ArgumentParser(description='Audit BISINDO dataset folders without modifying them.')
    ap.add_argument('--root', default='dataset')
    ap.add_argument('--output', default='audit_output')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.output).resolve()
    if not root.is_dir():
        raise SystemExit(f'[ERROR] Dataset folder not found: {root}')
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in (VIDEO_EXTS|IMAGE_EXTS))
    if not files:
        raise SystemExit('[WARN] No supported image/video files found.')

    details=[]; unresolved=[]
    class_counts=Counter(); class_split=defaultdict(Counter); class_type=defaultdict(Counter)
    dataset_counts=Counter()

    for p in files:
        rel=p.relative_to(root); parts=rel.parts
        dataset = parts[0] if len(parts)>=2 else 'ROOT'
        split = get_split(parts)
        label,note = infer_label(parts)
        mtype = media_type(p)
        row={
            'dataset':dataset,'split':split,'class':label,'media_type':mtype,
            'extension':p.suffix.lower(),'relative_path':str(rel),'note':note
        }
        details.append(row)
        if label=='UNRESOLVED': unresolved.append(row)
        key=(dataset,label)
        class_counts[key]+=1; class_split[key][split]+=1; class_type[key][mtype]+=1; dataset_counts[dataset]+=1

    class_rows=[]
    for (dataset,label),total in sorted(class_counts.items(), key=lambda x:(x[0][0].lower(),x[0][1].lower())):
        class_rows.append({
            'dataset':dataset,'class':label,
            'train':class_split[(dataset,label)]['Train'],
            'validation':class_split[(dataset,label)]['Validation'],
            'test':class_split[(dataset,label)]['Test'],
            'unspecified':class_split[(dataset,label)]['Unspecified'],
            'video':class_type[(dataset,label)]['Video'],
            'image':class_type[(dataset,label)]['Image'],
            'total':total
        })

    dataset_rows=[]
    for dataset,total in sorted(dataset_counts.items()):
        sub=[r for r in details if r['dataset']==dataset]
        labels={r['class'] for r in sub if r['class']!='UNRESOLVED'}
        dataset_rows.append({
            'dataset':dataset,'classes_detected':len(labels),
            'train':sum(r['split']=='Train' for r in sub),
            'validation':sum(r['split']=='Validation' for r in sub),
            'test':sum(r['split']=='Test' for r in sub),
            'unspecified':sum(r['split']=='Unspecified' for r in sub),
            'video':sum(r['media_type']=='Video' for r in sub),
            'image':sum(r['media_type']=='Image' for r in sub),
            'unresolved':sum(r['class']=='UNRESOLVED' for r in sub),
            'total':total
        })

    write_csv(out/'dataset_audit.csv', ['dataset','split','class','media_type','extension','relative_path','note'], details)
    write_csv(out/'class_summary.csv', ['dataset','class','train','validation','test','unspecified','video','image','total'], class_rows)
    write_csv(out/'dataset_summary.csv', ['dataset','classes_detected','train','validation','test','unspecified','video','image','unresolved','total'], dataset_rows)
    write_csv(out/'unresolved_files.csv', ['dataset','split','class','media_type','extension','relative_path','note'], unresolved)

    print('='*72)
    print('BISINDO DATASET AUDIT')
    print('='*72)
    print('Root  :', root)
    print('Media :', len(files), '| Videos:', sum(r['media_type']=='Video' for r in details), '| Images:', sum(r['media_type']=='Image' for r in details))
    print('Unresolved:', len(unresolved))
    print('\nDATASET SUMMARY')
    print_table(['Dataset','Classes','Train','Val','Test','Video','Image','Unresolved','Total'], [
        [r['dataset'],r['classes_detected'],r['train'],r['validation'],r['test'],r['video'],r['image'],r['unresolved'],r['total']] for r in dataset_rows
    ])
    print('\nCLASS SUMMARY')
    print_table(['Dataset','Class','Train','Val','Test','Video','Image','Total'], [
        [r['dataset'],r['class'],r['train'],r['validation'],r['test'],r['video'],r['image'],r['total']] for r in class_rows
    ], max_rows=150)
    print('\nOUTPUT:')
    for name in ['dataset_audit.csv','class_summary.csv','dataset_summary.csv','unresolved_files.csv']:
        print(' -', out/name)
    print('\nNEXT: kirim class_summary.csv + dataset_summary.csv ke ChatGPT.')

if __name__ == '__main__':
    main()
