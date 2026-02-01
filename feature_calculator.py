#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature calculation module
Responsible for computing grid features and composite scores
"""

import numpy as np
import pandas as pd
import concurrent.futures
from scipy import stats
from scipy.signal import find_peaks
from typing import Dict, Tuple, List
from data_structures import PointCloudData
from logging_utils import get_logger


class FeatureCalculator:
    """Feature calculator"""
    
    def __init__(self, z_bins: int, intensity_bins: int, alpha: float,
                 entropy_weight_a: float, entropy_weight_b: float, entropy_weight_c: float,
                 min_composite_score_threshold: float, min_valid_regions: int,
                 min_regions: int, max_regions: int, n_threads: int = None):
        self.z_bins = z_bins
        self.intensity_bins = intensity_bins
        self.alpha = alpha
        self.entropy_weight_a = entropy_weight_a
        self.entropy_weight_b = entropy_weight_b
        self.entropy_weight_c = entropy_weight_c
        self.min_composite_score_threshold = min_composite_score_threshold
        self.min_valid_regions = min_valid_regions
        self.min_regions = min_regions
        self.max_regions = max_regions
        self.n_threads = n_threads
        self.global_ranges = None
        self.logger = get_logger()
    
    def compute_unified_global_ranges(self, strip1_data: PointCloudData,
                                      strip2_data: PointCloudData,
                                      height_percentiles: Tuple[float, float],
                                      intensity_percentiles: Tuple[float, float]) -> Dict[str, Tuple[float, float]]:
        """Compute unified global data ranges (independent clipping intervals)"""
        self.logger.info("Computing unified global data ranges...")

        # Concatenate all data
        all_z = np.concatenate([strip1_data.points[:, 2], strip2_data.points[:, 2]])
        all_intensity = np.concatenate([strip1_data.intensity, strip2_data.intensity])

        self.logger.debug(f"Total points: {len(all_z):,}")
        self.logger.debug(f"Raw elevation range: [{np.min(all_z):.2f}, {np.max(all_z):.2f}] m")
        self.logger.debug(f"Raw intensity range: [{np.min(all_intensity):.0f}, {np.max(all_intensity):.0f}]")

        # Compute percentiles independently for clipping
        z_p_low, z_p_high = np.percentile(all_z, height_percentiles)
        i_p_low, i_p_high = np.percentile(all_intensity, intensity_percentiles)

        global_ranges = {
            'z_range': (float(z_p_low), float(z_p_high)),
            'intensity_range': (float(i_p_low), float(i_p_high))
        }

        self.logger.info(f"Height clipping range ({height_percentiles[0]}%-{height_percentiles[1]}%): [{z_p_low:.2f}, {z_p_high:.2f}] m")
        self.logger.info(f"Intensity clipping range ({intensity_percentiles[0]}%-{intensity_percentiles[1]}%): [{i_p_low:.0f}, {i_p_high:.0f}]")

        # Store global ranges
        self.global_ranges = global_ranges

        return global_ranges

    def compute_grid_features_unified(self, point_data: PointCloudData,
                                      grid_ids: np.ndarray, unique_ids: np.ndarray,
                                      strip_name: str) -> pd.DataFrame:
        """Compute grid features using unified global ranges"""
        self.logger.info(f"Computing {len(unique_ids)} grid features for {strip_name}...")

        if self.global_ranges is None:
            self.logger.error("Global ranges must be set first!")
            raise ValueError("Global ranges must be set first!")

        z_range = self.global_ranges['z_range']
        i_range = self.global_ranges['intensity_range']

        self.logger.debug(f"Using height range: [{z_range[0]:.2f}, {z_range[1]:.2f}]")
        self.logger.debug(f"Using intensity range: [{i_range[0]:.0f}, {i_range[1]:.0f}]")

        features_list = []

        # Sort by grid_id first to avoid repeated full masks per grid
        sort_idx = np.argsort(grid_ids)
        sorted_grid_ids = grid_ids[sort_idx]
        sorted_points = point_data.points[sort_idx]
        sorted_intensity = point_data.intensity[sort_idx]

        unique_sorted_ids, start_indices = np.unique(sorted_grid_ids, return_index=True)
        end_indices = np.r_[start_indices[1:], len(sorted_grid_ids)]

        def _compute_features_for_slice(grid_id: int, start_idx: int, end_idx: int) -> Dict:
            grid_points = sorted_points[start_idx:end_idx]
            grid_intensity = sorted_intensity[start_idx:end_idx]
            n_points = len(grid_points)

            if n_points == 0:
                return None

            z_values = grid_points[:, 2]

            # Basic statistics
            z_stats = {
                'z_mean': np.mean(z_values), 'z_std': np.std(z_values),
                'z_min': np.min(z_values), 'z_max': np.max(z_values),
                'z_range': np.max(z_values) - np.min(z_values)
            }

            i_stats = {
                'intensity_mean': np.mean(grid_intensity),
                'intensity_std': np.std(grid_intensity),
                'intensity_min': np.min(grid_intensity),
                'intensity_max': np.max(grid_intensity)
            }

            # Compute histograms using global ranges
            z_hist, _ = np.histogram(z_values, bins=self.z_bins, range=z_range)
            i_hist, _ = np.histogram(grid_intensity, bins=self.intensity_bins, range=i_range)

            # Convert to probability distributions
            z_hist = z_hist.astype(float)
            i_hist = i_hist.astype(float)

            # Smooth and normalize
            z_hist_smooth = z_hist + self.alpha
            i_hist_smooth = i_hist + self.alpha
            z_prob = z_hist_smooth / np.sum(z_hist_smooth)
            i_prob = i_hist_smooth / np.sum(i_hist_smooth)

            # Compute entropy
            z_entropy = stats.entropy(z_prob)
            i_entropy = stats.entropy(i_prob)

            # 2D histogram and mutual information
            hist_2d, _, _ = np.histogram2d(z_values, grid_intensity,
                                           bins=[self.z_bins, self.intensity_bins],
                                           range=[z_range, i_range])
            hist_2d = hist_2d.astype(float)
            hist_2d_smooth = hist_2d + self.alpha
            joint_prob = hist_2d_smooth / np.sum(hist_2d_smooth)

            # Mutual information calculation
            mutual_info = 0
            for i in range(self.z_bins):
                for j in range(self.intensity_bins):
                    if joint_prob[i, j] > 0 and z_prob[i] > 0 and i_prob[j] > 0:
                        mutual_info += joint_prob[i, j] * np.log(
                            joint_prob[i, j] / (z_prob[i] * i_prob[j]))

            # Normalized mutual information (NMI)
            h_z_i = z_entropy + i_entropy
            if h_z_i > 0:
                nmi = 2 * mutual_info / h_z_i
            else:
                nmi = 0

            return {
                'grid_id': grid_id, 'n_points': n_points,
                'z_entropy': z_entropy, 'intensity_entropy': i_entropy,
                'mutual_information': mutual_info,
                'normalized_mutual_information': nmi,
                **z_stats, **i_stats
            }

        use_parallel = self.n_threads is not None and self.n_threads > 1 and len(unique_sorted_ids) > 1
        if use_parallel:
            max_workers = self.n_threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_compute_features_for_slice, grid_id, start_idx, end_idx)
                    for grid_id, start_idx, end_idx in zip(unique_sorted_ids, start_indices, end_indices)
                ]
                for future in concurrent.futures.as_completed(futures):
                    features = future.result()
                    if features is not None:
                        features_list.append(features)
        else:
            for grid_id, start_idx, end_idx in zip(unique_sorted_ids, start_indices, end_indices):
                features = _compute_features_for_slice(grid_id, start_idx, end_idx)
                if features is not None:
                    features_list.append(features)

        features_df = pd.DataFrame(features_list)

        if len(features_df) > 0:
            # Normalized entropy
            max_z_entropy = np.log(self.z_bins)
            max_i_entropy = np.log(self.intensity_bins)
            features_df['z_entropy_norm'] = features_df['z_entropy'] / max_z_entropy
            features_df['intensity_entropy_norm'] = features_df['intensity_entropy'] / max_i_entropy

        self.logger.info(f"Feature computation complete: {len(features_df)} grids, {len(features_df.columns)} feature dimensions")

        return features_df

    def compute_composite_scores_unified(self, features1_df: pd.DataFrame, features2_df: pd.DataFrame) -> pd.DataFrame:
        """Compute joint composite feature scores using three-entropy linear combination"""
        self.logger.info("Computing joint grid composite scores...")
        self.logger.info(f"Three entropy weights: a={self.entropy_weight_a:.2f} (height), b={self.entropy_weight_b:.2f} (intensity), c={self.entropy_weight_c:.2f} (mutual information)")

        # Find common grids
        common_grids = set(features1_df['grid_id']) & set(features2_df['grid_id'])
        self.logger.info(f"Number of common grids: {len(common_grids)}")

        if len(common_grids) == 0:
            self.logger.warning("No common valid grids!")
            return pd.DataFrame()

        # Compute composite scores
        merged_features = []

        for grid_id in common_grids:
            feat1 = features1_df[features1_df['grid_id'] == grid_id].iloc[0]
            feat2 = features2_df[features2_df['grid_id'] == grid_id].iloc[0]

            # Extract three entropy values for each strip
            h1_z_norm = feat1['z_entropy_norm']
            h1_i_norm = feat1['intensity_entropy_norm']
            nmi1 = feat1['normalized_mutual_information']

            h2_z_norm = feat2['z_entropy_norm']
            h2_i_norm = feat2['intensity_entropy_norm']
            nmi2 = feat2['normalized_mutual_information']

            # Compute raw score
            raw_score = (
                                self.entropy_weight_a * (h1_z_norm + h2_z_norm) +
                                self.entropy_weight_b * (h1_i_norm + h2_i_norm) +
                                self.entropy_weight_c * (nmi1 + nmi2)
                        ) / 2.0

            # Compute density balance factor
            n1 = feat1['n_points']
            n2 = feat2['n_points']

            if n1 > 0 and n2 > 0:
                density_balance_factor = 2.0 * min(n1, n2) / (max(n1, n2) + min(n1, n2))
            else:
                density_balance_factor = 0.0

            # Final score
            final_score = density_balance_factor * raw_score

            merged_feature = {
                'grid_id': grid_id,
                'total_points': n1 + n2,
                'strip1_points': n1,
                'strip2_points': n2,
                'strip1_z_entropy_norm': h1_z_norm,
                'strip1_intensity_entropy_norm': h1_i_norm,
                'strip1_nmi': nmi1,
                'strip2_z_entropy_norm': h2_z_norm,
                'strip2_intensity_entropy_norm': h2_i_norm,
                'strip2_nmi': nmi2,
                'raw_score': raw_score,
                'density_balance_factor': density_balance_factor,
                'composite_score': final_score,
                'height_entropy_contribution': self.entropy_weight_a * (h1_z_norm + h2_z_norm) / 2.0,
                'intensity_entropy_contribution': self.entropy_weight_b * (h1_i_norm + h2_i_norm) / 2.0,
                'mutual_info_contribution': self.entropy_weight_c * (nmi1 + nmi2) / 2.0,
            }

            merged_features.append(merged_feature)

        merged_df = pd.DataFrame(merged_features)

        if len(merged_df) > 0:
            self.logger.info("Composite score computation complete:")
            self.logger.debug(f"Raw score range: [{merged_df['raw_score'].min():.4f}, {merged_df['raw_score'].max():.4f}]")
            self.logger.debug(f"Density balance factor range: [{merged_df['density_balance_factor'].min():.4f}, {merged_df['density_balance_factor'].max():.4f}]")
            self.logger.info(f"Final score range: [{merged_df['composite_score'].min():.4f}, {merged_df['composite_score'].max():.4f}]")
            self.logger.debug(f"Final score mean: {merged_df['composite_score'].mean():.4f}")

        return merged_df

    def select_high_quality_regions_unified(self, unified_features_df: pd.DataFrame, 
                                           enable_fallback_to_identity: bool) -> pd.DataFrame:
        """Select high-quality regions based on joint features"""
        self.logger.info("Selecting high-quality regions based on joint features...")

        if len(unified_features_df) == 0:
            self.logger.error("No unified feature data available")
            return pd.DataFrame()

        sorted_df = unified_features_df.sort_values('composite_score', ascending=False).copy()
        scores = sorted_df['composite_score'].values

        self.logger.info(f"Number of candidate regions: {len(scores)}")
        self.logger.debug(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Score threshold pre-check
        valid_score_mask = scores >= self.min_composite_score_threshold
        valid_score_count = np.sum(valid_score_mask)

        self.logger.info(f"Threshold check: {valid_score_count}/{len(scores)} regions exceed minimum score threshold")

        if valid_score_count == 0:
            self.logger.warning(f"No regions have composite score above threshold {self.min_composite_score_threshold:.4f}")
            if enable_fallback_to_identity:
                self.logger.info("Fallback: enable identity matrix fallback")
                return pd.DataFrame()
            else:
                self.logger.info("Continue processing: fallback disabled, selecting highest scored regions")

        elif valid_score_count < self.min_valid_regions:
            self.logger.warning(f"Valid regions count ({valid_score_count}) is less than required ({self.min_valid_regions})")
            if enable_fallback_to_identity:
                self.logger.info("Fallback: enable identity matrix fallback")
                return pd.DataFrame()
            else:
                self.logger.info("Continue processing: fallback disabled, using all valid regions")

        # If enough valid regions, proceed with standard region selection
        if valid_score_count >= self.min_valid_regions or not enable_fallback_to_identity:
            if valid_score_count > 0:
                filtered_df = sorted_df[valid_score_mask]
                filtered_scores = filtered_df['composite_score'].values
                self.logger.debug(f"After filtering: {len(filtered_scores)} regions for further selection")
            else:
                filtered_df = sorted_df
                filtered_scores = scores
                self.logger.debug(f"No filtering: using all {len(filtered_scores)} regions")

            # Multiple selection methods
            elbow_indices = self._find_elbow_point(filtered_scores)
            threshold_indices = self._find_score_threshold_regions(filtered_scores)
            gap_indices = self._find_score_gap_regions(filtered_scores)

            # Combine selection results
            all_selected = set(elbow_indices) | set(threshold_indices) | set(gap_indices)
            final_selected = list(all_selected)

            # Limit to a reasonable range
            if len(final_selected) < self.min_regions:
                additional = self.min_regions - len(final_selected)
                remaining = [i for i in range(len(filtered_scores)) if i not in final_selected]
                final_selected.extend(remaining[:additional])
            elif len(final_selected) > self.max_regions:
                final_selected = sorted(final_selected, key=lambda i: filtered_scores[i], reverse=True)[
                                 :self.max_regions]

            selected_df = filtered_df.iloc[final_selected].copy()
            selected_df['selection_rank'] = range(len(selected_df))

            self.logger.info(f"Joint region selection complete: {len(selected_df)} regions")

        else:
            selected_df = pd.DataFrame()

        return selected_df

    def _find_elbow_point(self, scores: np.ndarray) -> List[int]:
        """Elbow point detection"""
        if len(scores) < 3:
            return list(range(len(scores)))

        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        curvature = np.zeros(len(scores))
        for i in range(1, len(scores) - 1):
            curvature[i] = abs(normalized_scores[i - 1] - 2 * normalized_scores[i] + normalized_scores[i + 1])

        peaks, _ = find_peaks(curvature, height=np.std(curvature) * 0.5)

        if len(peaks) == 0:
            n_select = max(self.min_regions, int(len(scores) * 0.3))
            return list(range(min(n_select, len(scores))))

        first_elbow = peaks[0] if len(peaks) > 0 else len(scores) // 3
        return list(range(min(max(first_elbow, self.min_regions), len(scores))))

    def _find_score_threshold_regions(self, scores: np.ndarray) -> List[int]:
        """Score threshold method"""
        if len(scores) == 0:
            return []

        threshold = scores[0] * 0.8
        selected_indices = []

        for i, score in enumerate(scores):
            if score >= threshold:
                selected_indices.append(i)
            else:
                break

        return selected_indices

    def _find_score_gap_regions(self, scores: np.ndarray) -> List[int]:
        """Score gap detection"""
        if len(scores) < 2:
            return list(range(len(scores)))

        score_diffs = np.diff(scores)
        mean_diff = np.mean(score_diffs)
        std_diff = np.std(score_diffs)

        significant_drops = []
        for i, diff in enumerate(score_diffs):
            if abs(diff) > abs(mean_diff) + std_diff:
                significant_drops.append(i + 1)

        if len(significant_drops) == 0:
            n_select = max(self.min_regions, len(scores) // 4)
            return list(range(min(n_select, len(scores))))

        first_drop = significant_drops[0]
        return list(range(min(max(first_drop, self.min_regions), len(scores))))
