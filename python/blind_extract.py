#!/usr/bin/env python3
"""Extract a downloaded test-set archive blind: images under neutral content-hash names, everything that can carry labels sealed.

Some releases encode the label in the archive layout (Paraguay: one folder per DR grade) or ship annotations only for diseased eyes
(SUSTech-SYSU exudate files). Before a lock nothing may expose that, so this tool

* writes every image whose archive path matches ``--images`` to ``<out>/images/<prefix><sha256[:16]>.<ext>`` (name from the image
  bytes only; sorted listings are therefore in random label order);
* extracts every other member (label sheets, annotations, read-me files) under ``<sealed>/`` with its original path;
* writes ``<sealed>/mapping.csv`` (neutral name, original archive path, sha256) - the only link between images and labels;
* prints counts only (images written, byte-identical duplicates, sealed members), never a path or a name.

The sealed folder is registered as a reserved label path in the exposure registry; only the locked evaluation may read it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import zipfile
from pathlib import Path


def blind_extract(archive: Path, out: Path, sealed: Path, image_pattern: str, prefix: str) -> dict:
    images_dir = out / 'images'
    if images_dir.exists() and any(images_dir.iterdir()):
        raise FileExistsError(f'{images_dir} is not empty; extraction is done once')
    images_dir.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(image_pattern)
    rows, seen, duplicates, sealed_members = [], {}, 0, 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if pattern.search(name):
                data = zf.read(info)
                digest = hashlib.sha256(data).hexdigest()
                ext = Path(name).suffix.lower()
                neutral = f'{prefix}{digest[:16]}{ext}'
                if digest in seen:
                    duplicates += 1
                else:
                    (images_dir / neutral).write_bytes(data)
                    seen[digest] = neutral
                rows.append({'image_id': seen[digest], 'archive_path': name, 'sha256': digest})
            else:
                target = sealed / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
                sealed_members += 1
    with open(sealed / 'mapping.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_id', 'archive_path', 'sha256'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['image_id']))
    return {'images_written': len(seen), 'archive_images': len(rows), 'byte_identical_duplicates': duplicates, 'sealed_members': sealed_members}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='dataset folder; images go to <out>/images')
    parser.add_argument('--sealed', type=Path, required=True)
    parser.add_argument('--images', default=r'\.(jpe?g|png|tif)$', help='regex (case-insensitive by default suffixes) of archive members that are images to blind')
    parser.add_argument('--prefix', default='img_')
    args = parser.parse_args(argv)
    print(blind_extract(args.archive, args.out, args.sealed, '(?i)' + args.images, args.prefix))


if __name__ == '__main__':
    main()
