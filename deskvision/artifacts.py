"""Validation helpers for artifact paths declared by route reports."""

from pathlib import Path


def artifact_names(report: dict) -> set[str]:
    names = set()
    if isinstance(report.get('artifacts'), dict):
        names.update(value for value in report['artifacts'].values() if isinstance(value, str))
    for instance in report.get('segmentation', {}).get('instances', []):
        names.update(instance.get(key) for key in ('mask_file', 'contours_file')
                     if isinstance(instance.get(key), str))
    return {name for name in names if name != 'result.json' and '\\' not in name
            and not Path(name).is_absolute() and ':' not in name
            and all(part not in ('', '.', '..') for part in name.split('/'))}
