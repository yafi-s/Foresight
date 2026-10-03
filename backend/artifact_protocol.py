"""Causal artifact identity; refuses legacy/mixed bundles without loading models."""
import hashlib
import json
from pathlib import Path
import re


PROTOCOL='causal-v2'


def paths(directory,symbol):
    if not re.fullmatch(r'[A-Za-z0-9.^-]{1,16}',symbol):
        raise ValueError('Invalid symbol')
    root=Path(directory)
    return root,root/f'{symbol}_protocol.json',[root/f'{symbol}{suffix}' for suffix in (
        '_best.keras','_best.h5','_feature_scaler.joblib','_target_scaler.joblib')]


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(65536),b''):
            result.update(block)
    return result.hexdigest()


def write_bundle_metadata(directory,symbol,feature_cols):
    root,marker,files=paths(directory,symbol)
    metadata={'protocol':PROTOCOL,'symbol':symbol,'feature_cols':list(feature_cols),
              'close_channel':'per_observation','target':'cumulative_return_from_origin',
              'files':{p.name:sha(p)for p in files}}
    temporary=marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata,indent=2),encoding='utf8')
    temporary.replace(marker)
    return metadata


def validate_bundle(directory,symbol):
    root,marker,files=paths(directory,symbol)
    if not marker.is_file():
        raise ValueError('Legacy model bundle: retrain with causal-v2 before serving')
    metadata=json.loads(marker.read_text(encoding='utf8'))
    if metadata.get('protocol')!=PROTOCOL or metadata.get('symbol')!=symbol:
        raise ValueError('Incompatible model protocol')
    expected=metadata.get('files',{})
    if set(expected)!={p.name for p in files} or any(not p.is_file() or sha(p)!=expected[p.name] for p in files):
        raise ValueError('Model bundle changed or incomplete: retrain before serving')
    return metadata
