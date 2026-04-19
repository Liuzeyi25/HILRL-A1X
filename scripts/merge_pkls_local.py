#!/usr/bin/env python3
"""
Merge all .pkl files in a directory into one pickle file.
Writes output and prints a short verification.

Usage:
    python3 scripts/merge_pkls_local.py /path/to/dir [output_filename]
"""
import sys
import os
from pathlib import Path
import glob
import pickle


def load_pickle(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, p):
    with open(p, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def merge(loaded):
    # loaded: list of (filename, obj)
    if all(isinstance(o, list) for _, o in loaded):
        out = []
        for _, o in loaded:
            out.extend(o)
        return out
    if all(isinstance(o, dict) for _, o in loaded):
        out = {}
        for _, d in loaded:
            for k, v in d.items():
                out.setdefault(k, [])
                if isinstance(v, list):
                    out[k].extend(v)
                else:
                    out[k].append(v)
        return out
    # fallback
    return [{'filename': os.path.basename(fn), 'data': obj} for fn, obj in loaded]


def summarize(obj):
    if isinstance(obj, list):
        return f'list, len={len(obj)}'
    if isinstance(obj, dict):
        return f'dict, keys={len(obj)}'
    return str(type(obj))


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/merge_pkls_local.py /path/to/dir [output_filename]')
        sys.exit(2)

    target = Path(sys.argv[1]).expanduser().resolve()
    if not target.is_dir():
        print('Target is not a directory:', target)
        sys.exit(1)

    outname = sys.argv[2] if len(sys.argv) > 2 else 'all_demos_merged.pkl'
    outpath = target / outname

    pkl_files = sorted(glob.glob(str(target / '*.pkl')))
    # exclude possible existing merged files
    pkl_files = [p for p in pkl_files if Path(p).resolve() != outpath.resolve()]
    pkl_files = [p for p in pkl_files if not os.path.basename(p).startswith('all_demos_')]

    if not pkl_files:
        print('No .pkl files to merge in', target)
        sys.exit(0)

    print(f'Found {len(pkl_files)} .pkl files. Loading...')
    loaded = []
    for p in pkl_files:
        try:
            obj = load_pickle(p)
            loaded.append((p, obj))
        except Exception as e:
            print('Warning: failed to load', p, '->', e)

    if not loaded:
        print('No files could be loaded. Exiting.')
        sys.exit(1)

    merged = merge(loaded)
    save_pickle(merged, outpath)
    print('Saved merged pickle to', outpath)
    try:
        check = load_pickle(outpath)
        print('Verification:', summarize(check))
    except Exception as e:
        print('Warning: failed to load merged pickle for verification:', e)

if __name__ == '__main__':
    main()
