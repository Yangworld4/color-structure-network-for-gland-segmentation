"""Detached q(D) diagnostics for native disroute CPS training."""

import csv
import math
from pathlib import Path

import numpy as np


GROUPS = ('corr_FG2BG', 'corr_BG2FG', 'corr_ALL', 'dis_FG2BG', 'dis_BG2FG', 'dis_ALL')
MILESTONES = {40, 80, 120, 160, 200}
Q_EDGES = np.asarray([0.0, .2, .4, .6, .8, 1.0 + 1e-7], dtype=np.float64)
Q_BINS = ('q_bin_0_02_frac', 'q_bin_02_04_frac', 'q_bin_04_06_frac',
          'q_bin_06_08_frac', 'q_bin_08_10_frac')

CSV_FIELDS = [
    'epoch', 'region', 'direction', 'N_pixels', 'mean_D', 'std_D', 'mean_q', 'std_q',
    'q_min', 'q25', 'q50', 'q75', 'q90', 'q95', 'q_max', 'sum_q', *Q_BINS,
    'q_lt_0_2_frac', 'q_gt_0_5_frac', 'q_gt_0_8_frac',
    'mean_raw_cps_loss', 'mean_raw_opp_loss', 'mean_weighted_cps_loss',
    'mean_weighted_opp_loss', 'sum_weighted_cps_loss', 'sum_weighted_opp_loss',
    'mean_harmful_weight', 'N_dis_routed_pixels', 'N_dis_A_aligned', 'N_dis_B_aligned',
]

RECORD_FIELDS = [
    'epoch', 'batch_idx', 'image_id', 'region', 'direction', 'token_id', 'pixel_y',
    'pixel_x', 'D', 'q', 'S_cur', 'S_opp', 'Ct', 'Ut', 'A', 'B', 'conf_A', 'conf_B', 'rho',
]


def _safe_mean(total, count):
    return float(total / count) if count else float('nan')


class _Bucket:
    def __init__(self):
        self.n = 0
        self.sum_d = self.sum_d2 = self.sum_q = self.sum_q2 = 0.0
        self.min_q = float('inf')
        self.max_q = float('-inf')
        self.q_values = []
        self.hist = np.zeros(5, dtype=np.int64)
        self.lt02 = self.gt05 = self.gt08 = 0
        self.raw_cps_sum = self.raw_opp_sum = 0.0
        self.weighted_cps_sum = self.weighted_opp_sum = 0.0
        self.dis_aligned_a = self.dis_aligned_b = 0

    def add(self, d, q, raw_cps=None, raw_opp=None, weighted_cps=None, weighted_opp=None,
            aligned_a=None, aligned_b=None):
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if not q.size:
            return
        if d.size != q.size or not np.isfinite(d).all() or not np.isfinite(q).all():
            raise ValueError('q-dynamics received invalid D/q arrays')
        if np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError('q-dynamics q escaped [0,1]')
        self.n += int(q.size)
        self.sum_d += float(d.sum())
        self.sum_d2 += float(np.square(d).sum())
        self.sum_q += float(q.sum())
        self.sum_q2 += float(np.square(q).sum())
        self.min_q = min(self.min_q, float(q.min()))
        self.max_q = max(self.max_q, float(q.max()))
        self.q_values.append(q.copy())
        self.hist += np.histogram(q, bins=Q_EDGES)[0]
        self.lt02 += int((q < .2).sum())
        self.gt05 += int((q > .5).sum())
        self.gt08 += int((q > .8).sum())
        if raw_cps is not None:
            self.raw_cps_sum += float(np.asarray(raw_cps, dtype=np.float64).sum())
            self.raw_opp_sum += float(np.asarray(raw_opp, dtype=np.float64).sum())
            self.weighted_cps_sum += float(np.asarray(weighted_cps, dtype=np.float64).sum())
            self.weighted_opp_sum += float(np.asarray(weighted_opp, dtype=np.float64).sum())
        if aligned_a is not None:
            self.dis_aligned_a += int(np.asarray(aligned_a, dtype=bool).sum())
            self.dis_aligned_b += int(np.asarray(aligned_b, dtype=bool).sum())

    def quantiles(self):
        if not self.n:
            return [float('nan')] * 7
        return np.quantile(np.concatenate(self.q_values), [.0, .25, .5, .75, .9, .95, 1.0]).tolist()

    def stat_values(self):
        d_mean = _safe_mean(self.sum_d, self.n)
        q_mean = _safe_mean(self.sum_q, self.n)
        d_var = max(0.0, _safe_mean(self.sum_d2, self.n) - d_mean * d_mean) if self.n else float('nan')
        q_var = max(0.0, _safe_mean(self.sum_q2, self.n) - q_mean * q_mean) if self.n else float('nan')
        quantiles = self.quantiles()
        hist = self.hist / self.n if self.n else np.full(5, np.nan)
        return {
            'N_pixels': self.n, 'mean_D': d_mean, 'std_D': math.sqrt(d_var) if self.n else float('nan'),
            'mean_q': q_mean, 'std_q': math.sqrt(q_var) if self.n else float('nan'),
            'q_min': quantiles[0], 'q25': quantiles[1], 'q50': quantiles[2], 'q75': quantiles[3],
            'q90': quantiles[4], 'q95': quantiles[5], 'q_max': quantiles[6], 'sum_q': self.sum_q,
            **dict(zip(Q_BINS, hist.tolist())),
            'q_lt_0_2_frac': _safe_mean(self.lt02, self.n),
            'q_gt_0_5_frac': _safe_mean(self.gt05, self.n),
            'q_gt_0_8_frac': _safe_mean(self.gt08, self.n),
            'mean_raw_cps_loss': _safe_mean(self.raw_cps_sum, self.n),
            'mean_raw_opp_loss': _safe_mean(self.raw_opp_sum, self.n),
            'mean_weighted_cps_loss': _safe_mean(self.weighted_cps_sum, self.n),
            'mean_weighted_opp_loss': _safe_mean(self.weighted_opp_sum, self.n),
            'sum_weighted_cps_loss': self.weighted_cps_sum,
            'sum_weighted_opp_loss': self.weighted_opp_sum,
            'mean_harmful_weight': 1.0 - q_mean if self.n else float('nan'),
            'N_dis_routed_pixels': self.n if 'dis_' in getattr(self, '_name', '') else 0,
            'N_dis_A_aligned': self.dis_aligned_a, 'N_dis_B_aligned': self.dis_aligned_b,
        }


class QDynamicsTracker:
    """Collect per-pixel D/q statistics without retaining tensors or graphs."""

    def __init__(self, output_dir, token_grid, resume=False, has_disroute=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token_grid = int(token_grid)
        self.has_disroute = bool(has_disroute)
        self.csv_path = self.output_dir / 'q_dynamics_by_epoch.csv'
        existing_rows = []
        if resume and self.csv_path.is_file():
            with self.csv_path.open('r', newline='', encoding='utf-8') as previous:
                existing_rows = list(csv.DictReader(previous))
        self._file = self.csv_path.open(
            'a' if resume and self.csv_path.is_file() else 'w',
            newline='', encoding='utf-8'
        )
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if not (resume and existing_rows):
            self._writer.writeheader()
        self._file.flush()
        self.epoch = None
        self.buckets = None
        self.milestone_records = []
        cached = {}
        for row in existing_rows:
            if row['direction'] != 'ALL':
                continue
            epoch = int(row['epoch'])
            region = row['region']
            stats = {}
            for key, value in row.items():
                if key in ('epoch', 'region', 'direction'):
                    continue
                try:
                    stats[key] = int(value) if key in ('N_pixels', 'N_dis_routed_pixels', 'N_dis_A_aligned', 'N_dis_B_aligned') else float(value)
                except (TypeError, ValueError):
                    stats[key] = float('nan')
            cached.setdefault(epoch, {})[region] = stats
        self.epoch_cache = [
            (epoch, pair['corr'], pair['dis'])
            for epoch, pair in sorted(cached.items())
            if 'corr' in pair and 'dis' in pair
        ]

    def start_epoch(self, epoch):
        self.epoch = int(epoch)
        self.buckets = {name: _Bucket() for name in GROUPS}
        self.milestone_records = []

    @staticmethod
    def _as_numpy(value):
        if hasattr(value, 'detach'):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    def update(self, batch_idx, image_ids, fine, log_values):
        """Update from detached fine maps and loss logging maps."""
        masks = {
            'corr': self._as_numpy(fine['M_corr_pixel'])[:, 0].astype(bool),
            'dis': self._as_numpy(fine['M_dis_pixel'])[:, 0].astype(bool),
        }
        directions = {
            'FG2BG': self._as_numpy(fine['direction_fg2bg'])[:, 0].astype(bool),
            'BG2FG': self._as_numpy(fine['direction_bg2fg'])[:, 0].astype(bool),
        }
        k = self._as_numpy(fine['K_pixel'])[:, 0].astype(bool)
        if fine.get('q_mode_has_disroute', True):
            if not np.array_equal(k, masks['corr'] | masks['dis'] | self._as_numpy(fine['M_keep_pixel'])[:, 0].astype(bool)):
                raise AssertionError('q logging masks changed the existing K partition')
        elif np.any(masks['corr'] & ~k) or np.any(masks['dis']):
            raise AssertionError('Non-disroute q logging must contain M_corr only')
        if np.any(masks['corr'] & masks['dis']) or np.any(directions['FG2BG'] & directions['BG2FG']):
            raise AssertionError('q logging subgroup masks overlap')
        maps = {'d_pixel': self._as_numpy(fine['d_pixel'])[:, 0]}
        maps['q_log_pixel'] = self._as_numpy(log_values['q_pixel'])
        active = masks['corr'] | masks['dis']
        if np.any(maps['d_pixel'][active] <= 0.0):
            raise AssertionError('Only D>0 intervention pixels may enter q dynamics')
        loss_maps = {key: self._as_numpy(log_values[key]) for key in (
            'raw_cps', 'raw_opp', 'weighted_cps', 'weighted_opp')}
        aligned_a = self._as_numpy(fine['aligned_A_pixel'])[:, 0].astype(bool)
        aligned_b = self._as_numpy(fine['aligned_B_pixel'])[:, 0].astype(bool)
        detail_maps = None
        if self.epoch in MILESTONES:
            detail_maps = {key: self._as_numpy(fine[key])[:, 0] for key in (
                'S_cur_pixel', 'S_opp_pixel', 'Ct_pixel', 'Ut_pixel', 'A_pixel',
                'B_pixel', 'conf_A_pixel', 'conf_B_pixel', 'token_id_pixel')}
            detail_maps.update(maps)
        for bi in range(masks['corr'].shape[0]):
            image_id = str(image_ids[bi]) if image_ids is not None else str(bi)
            for region, region_mask in masks.items():
                for direction in ('FG2BG', 'BG2FG', 'ALL'):
                    selection = region_mask[bi]
                    if direction != 'ALL':
                        selection = selection & directions[direction][bi]
                    if not selection.any():
                        continue
                    name = '{}_{}'.format(region, direction)
                    vals = {key: value[bi][selection] for key, value in maps.items()}
                    kw = {}
                    if region == 'corr':
                        kw = {key: loss_maps[key][bi][selection] for key in loss_maps}
                    else:
                        kw = {'aligned_a': aligned_a[bi][selection], 'aligned_b': aligned_b[bi][selection]}
                    self.buckets[name].add(vals['d_pixel'], vals['q_log_pixel'], **kw)
                    if self.epoch in MILESTONES and direction != 'ALL':
                        ys, xs = np.nonzero(selection)
                        for y, x in zip(ys.tolist(), xs.tolist()):
                            self.milestone_records.append({
                                'epoch': self.epoch, 'batch_idx': int(batch_idx), 'image_id': image_id,
                                'region': region, 'direction': direction,
                                'token_id': int(detail_maps['token_id_pixel'][bi, y, x]), 'pixel_y': y, 'pixel_x': x,
                                'D': float(detail_maps['d_pixel'][bi, y, x]),
                                'q': float(detail_maps['q_log_pixel'][bi, y, x]),
                                'S_cur': float(detail_maps['S_cur_pixel'][bi, y, x]),
                                'S_opp': float(detail_maps['S_opp_pixel'][bi, y, x]),
                                'Ct': int(detail_maps['Ct_pixel'][bi, y, x]),
                                'Ut': int(detail_maps['Ut_pixel'][bi, y, x]),
                                'A': int(detail_maps['A_pixel'][bi, y, x]),
                                'B': int(detail_maps['B_pixel'][bi, y, x]),
                                'conf_A': float(detail_maps['conf_A_pixel'][bi, y, x]),
                                'conf_B': float(detail_maps['conf_B_pixel'][bi, y, x]), 'rho': '',
                            })

    def finalize_epoch(self, epoch):
        rows = []
        for name in GROUPS:
            bucket = self.buckets[name]
            region, direction = name.split('_', 1)
            row = {'epoch': int(epoch), 'region': region, 'direction': direction, **bucket.stat_values()}
            if region != 'dis':
                row['N_dis_routed_pixels'] = 0
                row['N_dis_A_aligned'] = 0
                row['N_dis_B_aligned'] = 0
                row['mean_harmful_weight'] = float('nan')
            else:
                row['N_dis_routed_pixels'] = bucket.n
            rows.append(row)
            self._writer.writerow(row)
        self._file.flush()
        corr = self.buckets['corr_ALL'].stat_values()
        dis = self.buckets['dis_ALL'].stat_values()
        for region in ('corr', 'dis'):
            all_count = self.buckets['{}_ALL'.format(region)].n
            directional_count = (
                self.buckets['{}_FG2BG'.format(region)].n
                + self.buckets['{}_BG2FG'.format(region)].n
            )
            if all_count != directional_count:
                raise AssertionError('{} direction counts do not sum to ALL'.format(region))
            bucket = self.buckets['{}_ALL'.format(region)]
            if bucket.n and abs(float((bucket.hist / bucket.n).sum()) - 1.0) > 1e-12:
                raise AssertionError('{} q histogram fractions do not sum to one'.format(region))
            branch_stats = corr if region == 'corr' else dis
            if bucket.n and abs(bucket.sum_q / bucket.n - branch_stats['mean_q']) > 1e-12:
                raise AssertionError('{} q mean does not equal sum_q/N'.format(region))
        self.epoch_cache.append((int(epoch), corr, dis))
        if int(epoch) in MILESTONES:
            path = self.output_dir / 'q_records_epoch_{:03d}.csv'.format(int(epoch))
            with path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=RECORD_FIELDS)
                writer.writeheader()
                writer.writerows(self.milestone_records)
        return corr, dis

    def print_epoch(self, epoch, print_epoch):
        if not print_epoch:
            return
        corr = self.buckets['corr_ALL'].stat_values()
        dis = self.buckets['dis_ALL'].stat_values()
        print('[Q-Dynamics][Epoch {:03d}]'.format(epoch))
        print('M_corr: N={} meanD={:.6g} meanQ={:.6g} q50={:.6g} q90={:.6g} sumQ={:.6g}'.format(
            corr['N_pixels'], corr['mean_D'], corr['mean_q'], corr['q50'], corr['q90'], corr['sum_q']))
        print('M_dis: N={} meanD={:.6g} meanQ={:.6g} q50={:.6g} q90={:.6g} sumQ={:.6g} harmful_w={:.6g}'.format(
            dis['N_pixels'], dis['mean_D'], dis['mean_q'], dis['q50'], dis['q90'], dis['sum_q'], dis['mean_harmful_weight']))

    def finalize_run(self):
        self._file.close()
        if not self.epoch_cache:
            return
        lines = ['q(D) diagnostics are descriptive statistics only.',
                 'Unavailable milestone field: rho (not produced by the current training path).']
        if not self.has_disroute:
            lines.append('This CPS mode has no M_dis routing branch; M_dis rows are empty by definition.')
        full = {}
        for branch in ('corr', 'dis'):
            sums = {'n': 0, 'q': 0.0, 'lt02': 0.0, 'gt05': 0.0, 'gt08': 0.0,
                    'weighted_cps': 0.0, 'weighted_opp': 0.0}
            for _, c, d in self.epoch_cache:
                x = c if branch == 'corr' else d
                n = int(x['N_pixels'])
                sums['n'] += n
                sums['q'] += float(x['sum_q'])
                sums['lt02'] += float(x['q_lt_0_2_frac']) * n if n else 0
                sums['gt05'] += float(x['q_gt_0_5_frac']) * n if n else 0
                sums['gt08'] += float(x['q_gt_0_8_frac']) * n if n else 0
                if branch == 'corr':
                    sums['weighted_cps'] += float(x['sum_weighted_cps_loss'])
                    sums['weighted_opp'] += float(x['sum_weighted_opp_loss'])
            mean_q = _safe_mean(sums['q'], sums['n'])
            full[branch] = mean_q
            if not sums['n']:
                lines += ['', '[{}]'.format(branch), 'No active intervention pixels in this CPS mode; statistics are undefined.']
                continue
            lines += ['', '[{}]'.format(branch), 'training pixel-weighted mean q = {:.8g}'.format(mean_q),
                      'q < 0.2 fraction = {:.8g}'.format(_safe_mean(sums['lt02'], sums['n'])),
                      'q > 0.5 fraction = {:.8g}'.format(_safe_mean(sums['gt05'], sums['n'])),
                      'q > 0.8 fraction = {:.8g}'.format(_safe_mean(sums['gt08'], sums['n']))]
            values = [x[1 if branch == 'corr' else 2]['mean_q'] for x in self.epoch_cache]
            if values:
                change = values[-1] - values[0]
                lines.append('epoch mean q trend (last-first) = {:.8g} ({})'.format(
                    change, 'increased' if change > 1e-6 else 'decreased' if change < -1e-6 else 'stable'))
            if branch == 'corr':
                total = sums['weighted_cps'] + sums['weighted_opp']
                lines.append('weighted opposite / total M_corr intervention loss = {:.8g}'.format(
                    _safe_mean(sums['weighted_opp'], total)))
            else:
                lines.append('average harmful-edge remaining weight mean(1-q) = {:.8g}'.format(1.0-mean_q))
        for branch in (('corr', 'dis') if self.has_disroute else ('corr',)):
            fg = self._all_epoch_group(branch, 'FG2BG')
            bg = self._all_epoch_group(branch, 'BG2FG')
            lines.append('{} FG2BG mean q = {:.8g}; BG2FG mean q = {:.8g}; delta = {:.8g}'.format(
                branch, fg, bg, fg-bg if np.isfinite(fg) and np.isfinite(bg) else float('nan')))
        (self.output_dir / 'q_dynamics_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        self._make_plots()

    def _all_epoch_group(self, branch, direction):
        # Direction-specific q means are retained in epoch rows; reload exact pixel-weighted counts.
        total_n = total_q = 0.0
        with self.csv_path.open('r', newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                if row['region'] == branch and row['direction'] == direction:
                    n = int(row['N_pixels'])
                    total_n += n
                    total_q += float(row['sum_q'])
        return _safe_mean(total_q, total_n)

    def _make_plots(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if not self.epoch_cache:
            return
        epochs = [x[0] for x in self.epoch_cache]
        branches = [('M_corr', 1)]
        if self.has_disroute:
            branches.append(('M_dis', 2))
        for key, filename, ylabel in (
            ('mean_q', 'q_mean_vs_epoch.png', 'Mean q'),
            ('sum_q', 'intervention_budget_vs_epoch.png', 'sum q'),
        ):
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for branch, index in branches:
                ax.plot(epochs, [x[index][key] for x in self.epoch_cache], marker='o', ms=3, label=branch)
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.legend(frameon=False); fig.tight_layout()
            fig.savefig(self.output_dir / filename, dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        quantile_lines = [
            ('M_corr', 1, 'q50', 'M_corr median', '-'),
            ('M_corr', 1, 'q90', 'M_corr p90', '--'),
        ]
        if self.has_disroute:
            quantile_lines += [
                ('M_dis', 2, 'q50', 'M_dis median', '-'),
                ('M_dis', 2, 'q90', 'M_dis p90', '--'),
            ]
        for branch, index, key, label, style in quantile_lines:
            ax.plot(epochs, [x[index][key] for x in self.epoch_cache], marker='o', ms=3, ls=style, label=label)
        ax.set_xlabel('Epoch'); ax.set_ylabel('q'); ax.legend(frameon=False, ncol=2); fig.tight_layout()
        fig.savefig(self.output_dir / 'q_median_p90_vs_epoch.png', dpi=180); plt.close(fig)
        milestones = [e for e in sorted(MILESTONES) if e in epochs]
        if not milestones:
            return
        for branch in (('corr', 'dis') if self.has_disroute else ('corr',)):
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for epoch in milestones:
                with (self.output_dir / 'q_records_epoch_{:03d}.csv'.format(epoch)).open('r', newline='', encoding='utf-8') as file:
                    vals = [float(r['q']) for r in csv.DictReader(file) if r['region'] == branch]
                if vals:
                    ax.hist(vals, bins=np.linspace(0, 1, 21), density=True, histtype='step', lw=1.5, label=str(epoch))
            ax.set_xlabel('q'); ax.set_ylabel('Density'); ax.legend(title='Epoch', frameon=False); fig.tight_layout()
            fig.savefig(self.output_dir / 'q_distribution_{}_milestones.png'.format(branch), dpi=180); plt.close(fig)
