#!/usr/bin/env python3
"""Extract a downloaded test-set release blind: images under neutral content-hash names, everything that can carry labels sealed.

Some releases encode the label in the archive layout (Paraguay: one folder per DR grade) or ship annotations only for diseased eyes
(SUSTech-SYSU exudate files). Before a lock nothing may expose that, so this tool

* reads a ``.zip`` or ``.tar``/``.tar.gz``/``.tgz`` archive or a downloaded directory, including archives nested inside it (members are
  addressed as ``outer.zip!/inner/path``);
* writes every image whose path matches ``--images`` to ``<out>/images/<prefix><sha256[:16]>.<ext>`` (name from the image bytes only;
  sorted listings are therefore in random label order);
* extracts every other member (label sheets, annotations, read-me files) under ``<sealed>/`` with its original path (``!/`` -> ``/``);
* writes ``<sealed>/mapping.csv`` (neutral name, original path, sha256) - the only link between images and labels;
* prints counts only (images written, byte-identical duplicates, sealed members), never a path or a name.

Member paths that are absolute or contain ``..`` are refused (a release is untrusted data), symbolic links are skipped. The sealed folder
is registered as a reserved label path in the exposure registry; only the locked evaluation may read it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE = re.compile(r'(?i)\.(zip|tar|tgz|tar\.gz)$')


def safe_parts(name: str) -> tuple[str, ...]:
    """Path components of an archive member; absolute paths and '..' are refused."""
    path = PurePosixPath(name.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or re.match(r'^[A-Za-z]:', name):
        raise ValueError('Unsafe member path in the release (absolute or containing ..); refusing to extract')
    return tuple(p for p in path.parts if p not in ('', '.'))


def _archive_members(name: str, data: bytes | None = None, path: Path | None = None):
    """(member path, bytes) of every regular file of a zip/tar archive, recursing into nested archives."""
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(path if path is not None else io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    yield from _member(f'{name}!/{info.filename}', info.filename, zf.read(info))
    else:
        with tarfile.open(name=str(path) if path is not None else None, fileobj=None if path is not None else io.BytesIO(data)) as tf:
            for info in tf.getmembers():
                if info.isfile():
                    yield from _member(f'{name}!/{info.name}', info.name, tf.extractfile(info).read())


def _member(label: str, inner: str, data: bytes):
    safe_parts(inner)
    if ARCHIVE.search(inner):
        yield from _archive_members(label, data=data)
    else:
        yield label, data


def release_members(source: Path):
    """Yield (path, bytes) for every file of the release; paths are relative to the source and keep nesting as ``a.zip!/b``."""
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob('*')):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            if ARCHIVE.search(relative):
                yield from _archive_members(relative, path=path)
            else:
                yield relative, path.read_bytes()
    elif ARCHIVE.search(source.name):
        for label, data in _archive_members(source.name, path=source):
            yield label.split('!/', 1)[1], data          # the top-level archive name is not part of the member path
    else:
        raise ValueError('The release must be a directory or a .zip/.tar/.tar.gz/.tgz archive')


def blind_extract(archive: Path, out: Path, sealed: Path, image_pattern: str, prefix: str) -> dict:
    images_dir = out / 'images'
    if images_dir.exists() and any(images_dir.iterdir()):
        raise FileExistsError(f'{images_dir} is not empty; extraction is done once')
    images_dir.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(image_pattern)
    rows, seen, duplicates, sealed_members = [], {}, 0, 0
    for name, data in release_members(archive):
        if pattern.search(name):
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
            target = sealed.joinpath(*safe_parts(name.replace('!/', '/')))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            sealed_members += 1
    with open(sealed / 'mapping.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_id', 'archive_path', 'sha256'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['image_id']))
    return {'images_written': len(seen), 'archive_images': len(rows), 'byte_identical_duplicates': duplicates, 'sealed_members': sealed_members}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', type=Path, required=True, help='the downloaded release: a .zip/.tar/.tar.gz/.tgz archive or a directory')
    parser.add_argument('--out', type=Path, required=True, help='dataset folder; images go to <out>/images')
    parser.add_argument('--sealed', type=Path, required=True)
    parser.add_argument('--images', default=r'\.(jpe?g|png|tif)$', help='regex (case-insensitive by default suffixes) of archive members that are images to blind')
    parser.add_argument('--prefix', default='img_')
    args = parser.parse_args(argv)
    print(blind_extract(args.archive, args.out, args.sealed, '(?i)' + args.images, args.prefix))


if __name__ == '__main__':
    main()
