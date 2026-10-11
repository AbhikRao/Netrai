"""Research CLI with no dataset or model download side effects."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(prog='netrai')
    parser.add_argument('command', choices=('doctor', 'smoke', 'predict', 'evaluate', 'train', 'manifest'))
    args, remaining = parser.parse_known_args()
    if args.command == 'doctor':
        import torch
        import weights
        from utils.bundle import verify_bundle
        manifest = verify_bundle(Path(weights.__file__).parent / 'bundle.json')
        print(json.dumps({'python': sys.version.split()[0], 'cuda_available': torch.cuda.is_available(),
                          'bundle_status': manifest['status'],
                          'packages': {name: importlib.metadata.version(name) for name in
                                       ('torch', 'torchvision', 'numpy', 'opencv-python-headless', 'timm')},
                          'dataset_required': 'Your own legally obtained RGB fundus image for clinical-domain testing',
                          'clinical_validation': False}, indent=2))
    elif args.command == 'smoke':
        import numpy as np
        from modules.quality import assess_image_quality
        from utils.preprocessing import preprocess_for_model
        constant = np.full((64, 64, 3), 127, np.uint8)
        assert assess_image_quality(constant)[0] == 0
        assert np.isfinite(preprocess_for_model(constant)).all()
        print('PASS: synthetic preprocessing/invalid-image smoke; no medical accuracy claim')
    else:
        modules = {'predict': 'inference', 'evaluate': 'batch_validate',
                   'train': 'train_grading', 'manifest': 'prepare_training_manifest'}
        module = importlib.import_module(modules[args.command])
        sys.argv = [f'netrai {args.command}', *remaining]
        result = module.main()
        if isinstance(result, int):
            raise SystemExit(result)


if __name__ == '__main__':
    main()
