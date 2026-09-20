"""Assemble the closed native-scale validation comparison.

This is a read-only post-processing utility.  It does not retrain or select
parameters; the vanilla values come from the already completed standard-CPS
E0/E1 diagnostics and the S/SC values come from their fixed formal runs.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path('outputs/native_scale_training')
S = ROOT / 'validation_shared_errors_native_s.csv'
SC = ROOT / 'validation_shared_errors_native_sc.csv'
SN_S = ROOT / 'validation_native_states_native_s.csv'
SN_SC = ROOT / 'validation_native_states_native_sc.csv'
EV_S = ROOT / 'validation_events_native_s.csv'
EV_SC = ROOT / 'validation_events_native_sc.csv'


def _last(path: Path, epoch: int = 200):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return next(row for row in reversed(rows) if int(row['epoch']) == epoch)


def _log_metrics(path: Path, epoch: int = 200):
    pat = re.compile(
        r'epoch=(\d+)/\d+.*?val_dice_student1=([0-9.]+).*?'
        r'val_dice_student2=([0-9.]+).*?val_jaccard_student1=([0-9.]+).*?'
        r'val_jaccard_student2=([0-9.]+).*?student12_both_correct=([0-9.]+).*?'
        r'student12_both_wrong=([0-9.]+)'
    )
    found = None
    for line in path.read_text(encoding='utf-8').splitlines():
        m = pat.search(line)
        if m and int(m.group(1)) == epoch:
            found = m
    if found is None:
        return {}
    return {
        'Dice': (float(found.group(2)) + float(found.group(3))) / 2.0,
        'IoU': (float(found.group(4)) + float(found.group(5))) / 2.0,
        'CC_rate': float(found.group(6)),
        'WW_rate': float(found.group(7)),
    }


def _native_row(tag: str, run_dir: Path, shared: Path, state: Path, events: Path):
    row = _last(shared)
    state_row = _last(state)
    event_row = _last(events)
    log_row = _log_metrics(run_dir / 'train.log')
    return {
        'experiment': tag,
        'epoch': 200,
        **log_row,
        'CC_rate': row['CC_rate'],
        'WW_rate': row['WW_rate'], 'HC90WW_rate': row['HC90WW_rate'],
        'Shared_FP_rate': row['Shared_FP_rate'], 'Shared_FN_rate': row['Shared_FN_rate'],
        'WW_events': event_row['WW_events'],
        'HC90WW_events': event_row['HC90WW_events'],
        'HC_token_MAE': state_row['HC_token_MAE'],
    }


def main():
    standard_log = Path(
        'checkpoints/semi_10_seed_24_rgbstain/GLaS/'
        'CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image-input_mode=rgb_stain/train.log'
    )
    # These are the locked E0/E1 standard-CPS reference values; no new test
    # evaluation is performed here.
    rows = [{
        'experiment': 'vanilla', 'epoch': 200,
        **_log_metrics(standard_log),
        'CC_rate': _log_metrics(standard_log).get('CC_rate', ''),
        'WW_rate': _log_metrics(standard_log).get('WW_rate', ''), 'HC90WW_rate': '',
        'Shared_FP_rate': '', 'Shared_FN_rate': '',
        'WW_events': '0.5096630974144686',
        'HC90WW_events': '0.4606012969149145',
        'HC_token_MAE': '',
    }]
    rows.append(_native_row(
        'native_s',
        Path('checkpoints/native_s_seed24/GLaS/'
             'CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=standard-reference_mode=native_suppress'),
        S, SN_S, EV_S,
    ))
    rows.append(_native_row(
        'native_sc',
        Path('checkpoints/native_sc_seed24/GLaS/'
             'CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=standard-reference_mode=native_suppress_hc'),
        SC, SN_SC, EV_SC,
    ))
    output = ROOT / 'comparison_summary.csv'
    fields = ['experiment', 'epoch', 'Dice', 'IoU', 'CC_rate', 'WW_rate',
              'HC90WW_rate', 'Shared_FP_rate', 'Shared_FN_rate',
              'WW_events', 'HC90WW_events', 'HC_token_MAE']
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == '__main__':
    main()
