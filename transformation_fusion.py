#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformation Fusion Module
Responsible for similarity analysis and fusion of transformation matrices
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict
from logging_utils import get_logger


class TransformationFusion:
    """Transformation Fuser"""
    
    def __init__(self, offset_center: np.ndarray = None):
        self.logger = get_logger()
        self.offset_center = offset_center
        self.valid_grid_ids = set()
    
    def set_offset_center(self, offset_center: np.ndarray):
        """Set decentralization offset"""
        self.offset_center = offset_center
    
    def analyze_transformation_similarity_MAD_S2S(self, region_results: List[Dict]) -> List[Dict]:
        """Analyze transformation matrix similarity, filter abnormal transformations"""
        self.logger.info("Analyzing transformation matrix similarity...")

        if len(region_results) <= 2:
            self.logger.info(f"Few regions ({len(region_results)}), skipping similarity analysis")
            self.valid_grid_ids = set(result['grid_id'] for result in region_results)
            return region_results

        # Extract transformation information
        transformations = []
        for result in region_results:
            T = result['transformation']
            translation = T[:3, 3].copy()
            rotation_matrix = np.array(T[:3, :3], dtype=np.float64)
            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=True)

            translation_magnitude = np.linalg.norm(translation)
            rotation_magnitude = np.linalg.norm(euler_angles)

            transformations.append({
                'result': result,
                'translation': translation,
                'rotation_angles': euler_angles,
                'translation_magnitude': translation_magnitude,
                'rotation_magnitude': rotation_magnitude,
                'fitness': result['fitness']
            })

        # Extract XYZ components
        trans_pitch = np.array([t['rotation_angles'][0] for t in transformations])
        trans_roll = np.array([t['rotation_angles'][1] for t in transformations])
        trans_yaw = np.array([t['rotation_angles'][2] for t in transformations])
        trans_x = np.array([t['translation'][0] for t in transformations])
        trans_y = np.array([t['translation'][1] for t in transformations])
        trans_z = np.array([t['translation'][2] for t in transformations])
        
        transformations_to_use = transformations
        
        # Update data for subsequent MAD analysis
        
        trans_pitch = np.array([t['rotation_angles'][0] for t in transformations_to_use])
        trans_roll = np.array([t['rotation_angles'][1] for t in transformations_to_use])
        trans_yaw = np.array([t['rotation_angles'][2] for t in transformations_to_use])
        trans_x = np.array([t['translation'][0] for t in transformations_to_use])
        trans_y = np.array([t['translation'][1] for t in transformations_to_use])
        trans_z = np.array([t['translation'][2] for t in transformations_to_use])
        self.logger.info("Starting MAD filtering")
        
        def detect_outliers_median(data, threshold=1.5, min_mad=None, component_name=""):
            if len(data) < 2:
                return np.zeros(len(data), dtype=bool)

            median = np.median(data)
            mad = np.median(np.abs(data - median))

            if min_mad is not None and mad < min_mad:
                mad = min_mad

            if mad < 1e-10:
                return np.zeros(len(data), dtype=bool)

            scores = np.abs(data - median) / mad
            outliers = scores > threshold

            self.logger.debug(f"{component_name} Median={median:.3f}, MAD={mad:.3f}")
            for i, val in enumerate(data):
                score = scores[i]
                status = "Abnormal" if outliers[i] else "Normal"
                grid_id = transformations_to_use[i]['result']['grid_id']
                self.logger.debug(f"Grid {grid_id}: {val:.3f} (Score={score:.1f}, {status})")
            return outliers

        # Detect outliers for XYZ components separately
        self.logger.debug("X component outlier detection:")
        x_outliers = detect_outliers_median(trans_x, threshold=1, min_mad=0.5, component_name="X")

        self.logger.debug("Y component outlier detection:")
        y_outliers = detect_outliers_median(trans_y, threshold=1, component_name="Y")

        self.logger.debug("Z component outlier detection:")
        z_outliers = detect_outliers_median(trans_z, threshold=2, min_mad=0.2, component_name="Z")
        
        self.logger.info("Pitch component outlier detection:")
        pitch_outliers = detect_outliers_median(trans_pitch, threshold=2, component_name="Pitch")
        
        self.logger.info("Roll component outlier detection:")
        roll_outliers = detect_outliers_median(trans_roll, threshold=2, component_name="Roll")
        
        self.logger.info("Yaw component outlier detection:")
        yaw_outliers = detect_outliers_median(trans_yaw, threshold=2, component_name="Yaw")

        # Filter valid results and update valid grid_id set
        valid_results = []
        outlier_count = 0
        self.valid_grid_ids = set()

        self.logger.info("Filter results:")
        for i in range(len(transformations_to_use)):
            is_outlier = x_outliers[i] or y_outliers[i] or z_outliers[i] or pitch_outliers[i] or roll_outliers[i] or yaw_outliers[i]

            grid_id = transformations_to_use[i]['result']['grid_id']

            if is_outlier:
                outlier_count += 1
                reasons = []
                if x_outliers[i]:
                    reasons.append(f"X abnormal({transformations_to_use[i]['translation'][0]:.3f})")
                if y_outliers[i]:
                    reasons.append(f"Y abnormal({transformations_to_use[i]['translation'][1]:.3f})")
                if z_outliers[i]:
                    reasons.append(f"Z abnormal({transformations_to_use[i]['translation'][2]:.3f})")
                if pitch_outliers[i]:
                    reasons.append(f"Pitch abnormal({transformations_to_use[i]['rotation_angles'][0]:.3f})")
                if roll_outliers[i]:
                    reasons.append(f"Roll abnormal({transformations_to_use[i]['rotation_angles'][1]:.3f})")
                if yaw_outliers[i]:
                    reasons.append(f"Yaw abnormal({transformations_to_use[i]['rotation_angles'][2]:.3f})")
                self.logger.info(f"Grid {grid_id} rejected: {', '.join(reasons)}")
            else:
                valid_results.append(transformations_to_use[i]['result'])
                self.valid_grid_ids.add(grid_id)
                self.logger.debug(f"Grid {grid_id} kept: Fitness={transformations_to_use[i]['fitness']:.3f}")

        # Fallback strategy
        if len(valid_results) < 2:
            self.logger.warning(f"Too few remaining ({len(valid_results)}), relaxing threshold and retrying")
            x_outliers = detect_outliers_median(trans_x, threshold=1.5, min_mad=0.5)
            y_outliers = detect_outliers_median(trans_y, threshold=1.5)
            z_outliers = detect_outliers_median(trans_z, threshold=2, min_mad=0.2)

            valid_results = []
            self.valid_grid_ids = set()
            for i in range(len(transformations_to_use)):
                is_outlier = x_outliers[i] or y_outliers[i] or z_outliers[i]
                grid_id = transformations_to_use[i]['result']['grid_id']
                if not is_outlier:
                    valid_results.append(transformations_to_use[i]['result'])
                    self.valid_grid_ids.add(grid_id)

            self.logger.info(f"Kept {len(valid_results)} regions after relaxation")

        if len(valid_results) < 1:
            self.logger.warning(f"Still too few, keeping top {max(3, len(region_results) // 2)} sorted by fitness")
            sorted_results = sorted(region_results, key=lambda x: x['fitness'], reverse=True)
            n_keep = max(3, len(region_results) // 2)
            valid_results = sorted_results[:n_keep]
            self.valid_grid_ids = set(result['grid_id'] for result in valid_results)

        self.logger.info(f"Final results: Total {len(region_results)}, Rejected {len(region_results) - len(valid_results)}, Kept {len(valid_results)}")
        self.logger.info(f"Valid grid_id set: {sorted(self.valid_grid_ids)}")

        return valid_results
    def analyze_transformation_similarity(self, region_results: List[Dict]) -> List[Dict]:
        """Analyze transformation matrix similarity and filter abnormal transformations
        (based on distance in transformation parameter space / main cluster filtering).
        Preserve original interface and return format: input `region_results`, output filtered `region_results`.
        """
        self.logger.info("Analyzing transformation matrix similarity...")

        try:
            if len(region_results) <= 3:
                self.logger.info(f"Few regions ({len(region_results)}), skipping similarity analysis")
                self.valid_grid_ids = set(result['grid_id'] for result in region_results)
                return region_results

            # -----------------------------
            # 1) Extract transformation: Translation + Euler angles (deg)
            #    Critical fix: Force copy to avoid read-only buffer
            # -----------------------------
            transformations = []
            for result in region_results:
                T_in = result['transformation']

                # Force transformation to be writable, contiguous float64 ndarray
                T = np.array(T_in, dtype=np.float64, copy=True)
                T = np.ascontiguousarray(T)

                translation = np.array(T[:3, 3], dtype=np.float64, copy=True)
                rotation_matrix = np.array(T[:3, :3], dtype=np.float64, copy=True)
                rotation_matrix = np.ascontiguousarray(rotation_matrix)

                rotation = Rotation.from_matrix(rotation_matrix)
                euler_angles = rotation.as_euler('xyz', degrees=True)
                euler_angles = np.array(euler_angles, dtype=np.float64, copy=True)

                transformations.append({
                    'result': result,
                    'translation': translation,
                    'rotation_angles': euler_angles,
                    'fitness': float(result.get('fitness', 0.0))
                })

            self.logger.debug("All translation vectors:")
            for t in transformations:
                gid = t['result']['grid_id']
                tr = t['translation']
                self.logger.debug(f"Region {gid}: [{tr[0]:6.3f}, {tr[1]:6.3f}, {tr[2]:6.3f}]")

            # -----------------------------
            # 1.5) Initial Filtering: Based on translation range
            # -----------------------------
            self.logger.info("Starting initial filtering: Based on translation range")
            x_range = [(-1000, 1000), (-1000, 1000)]   # x range: (-6,-5) or (5,6)
            y_range = (-1000, 1000)   # y range: -1 to 1
            z_range = (-1000, 1000)   # z range: -1 to 1
            
            initial_filtered = []
            initial_filtered_count = 0
            
            for t in transformations:
                tx, ty, tz = t['translation']
                grid_id = t['result']['grid_id']
                
                # Check if within range
                x_valid = any(lo <= tx < hi for lo, hi in x_range)
                y_valid = y_range[0] <= ty <= y_range[1]
                z_valid = z_range[0] <= tz <= z_range[1]
                
                if x_valid and y_valid and z_valid:
                    initial_filtered.append(t)
                    self.logger.debug(f"Grid {grid_id} passed initial filter: X={tx:.3f}, Y={ty:.3f}, Z={tz:.3f}")
                else:
                    initial_filtered_count += 1
                    reasons = []
                    if not x_valid:
                        reasons.append(f"X={tx:.3f} not in {x_range}")
                    if not y_valid:
                        reasons.append(f"Y={ty:.3f} not in {y_range}")
                    if not z_valid:
                        reasons.append(f"Z={tz:.3f} not in {z_range}")
                    self.logger.info(f"Grid {grid_id} rejected by initial filter: {', '.join(reasons)}")
            
            self.logger.info(f"Initial filter results: Total {len(transformations)}, Rejected {initial_filtered_count}, Kept {len(initial_filtered)}")
            
            # If too few remain after initial filter, skip initial filter and use all data
            if len(initial_filtered) < 1:
                self.logger.warning(f"Too few remaining after initial filter ({len(initial_filtered)}), skipping initial filter and using all data")
                transformations = transformations
            else:
                transformations = initial_filtered
                self.logger.info(f"Using {len(transformations)} regions after initial filter for subsequent analysis")

            # -----------------------------
            # 2) Construct 6D parameter vector and perform robust normalization
            # -----------------------------
            X_raw = np.zeros((len(transformations), 6), dtype=np.float64)
            for i, t in enumerate(transformations):
                tx, ty, tz = t['translation']
                rx, ry, rz = t['rotation_angles']  # degrees
                X_raw[i, :] = [tx, ty, tz, rx, ry, rz]

            def robust_scale_mad(X, min_mad_per_dim=None, eps=1e-12):
                med = np.median(X, axis=0)
                mad = np.median(np.abs(X - med), axis=0)

                if min_mad_per_dim is not None:
                    mad = np.maximum(mad, np.asarray(min_mad_per_dim, dtype=np.float64))

                mad = np.maximum(mad, eps)
                Xs = (X - med) / mad
                return Xs, med, mad

            min_mad = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02], dtype=np.float64)
            X, med, mad = robust_scale_mad(X_raw, min_mad_per_dim=min_mad)

            # -----------------------------
            # 3) Distance matrix (normalized space)
            # -----------------------------
            N = X.shape[0]
            D = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                diff = X[i] - X
                D[i] = np.linalg.norm(diff, axis=1)

            # -----------------------------
            # 4) medoid (Main cluster center)
            # -----------------------------
            sum_dist = np.sum(D, axis=1)
            medoid_idx = int(np.argmin(sum_dist))
            medoid_gid = transformations[medoid_idx]['result']['grid_id']

            self.logger.info(f"Similarity center (Main cluster representative) grid: {medoid_gid}")
            d_to_medoid = D[medoid_idx].copy()

            # -----------------------------
            # 5) Robust threshold inlier filtering: median + k*MAD
            # -----------------------------
            def robust_inlier_mask(dist, k=2.5, min_mad_dist=0.05):
                d_med = float(np.median(dist))
                d_mad = float(np.median(np.abs(dist - d_med)))
                d_mad = max(d_mad, min_mad_dist)
                thr = d_med + k * d_mad
                mask = dist <= thr
                return mask, thr, d_med, d_mad

            target_min_keep = 3
            ks = [2.5, 3.5, 4.5]
            inlier_mask = None
            last_info = None

            for k in ks:
                mask, thr, d_med, d_mad = robust_inlier_mask(d_to_medoid, k=k, min_mad_dist=0.05)
                keep_cnt = int(np.sum(mask))
                last_info = (k, thr, d_med, d_mad, keep_cnt)
                if keep_cnt >= max(2, min(target_min_keep, N)):
                    inlier_mask = mask
                    break

            if inlier_mask is None:
                inlier_mask = np.zeros(N, dtype=bool)
                idx_sorted = np.argsort(d_to_medoid)
                k_keep = max(2, N // 2)
                inlier_mask[idx_sorted[:k_keep]] = True
                self.logger.warning(f"Main cluster filtering too strict, changed to keep top {k_keep} regions by distance")
            else:
                k_used, thr, d_med, d_mad, keep_cnt = last_info
                self.logger.info(
                    f"Main cluster filtering: k={k_used:.1f}, dist_median={d_med:.3f}, dist_MAD={d_mad:.3f}, thr={thr:.3f}, Kept {keep_cnt}/{N}"
                )

            # -----------------------------
            # 5.5) Angle outlier detection (Added)
            # -----------------------------
            self.logger.info("Starting angle outlier detection:")
            
            # Extract angle components
            angles_pitch = np.array([t['rotation_angles'][0] for t in transformations])
            angles_roll = np.array([t['rotation_angles'][1] for t in transformations])
            angles_yaw = np.array([t['rotation_angles'][2] for t in transformations])
            
            def detect_angle_outliers(angles, threshold=2.0, component_name=""):
                """Angle outlier detection based on MAD"""
                if len(angles) < 4:
                    return np.zeros(len(angles), dtype=bool)
                
                median = np.median(angles)
                mad = np.median(np.abs(angles - median))
                
                if mad < 1e-10:
                    return np.zeros(len(angles), dtype=bool)
                
                scores = np.abs(angles - median) / mad
                outliers = scores > threshold
                
                self.logger.info(f"{component_name}: Median={median:.3f}°, MAD={mad:.3f}°, Threshold={threshold}")
                for i, val in enumerate(angles):
                    score = scores[i]
                    status = "Abnormal" if outliers[i] else "Normal"
                    grid_id = transformations[i]['result']['grid_id']
                    self.logger.debug(f"  Grid {grid_id}: {val:.3f}° (Score={score:.1f}, {status})")
                return outliers
            
            pitch_outliers = detect_angle_outliers(angles_pitch, threshold=2.0, component_name="Pitch Angle")
            roll_outliers = detect_angle_outliers(angles_roll, threshold=2.0, component_name="Roll Angle")
            yaw_outliers = detect_angle_outliers(angles_yaw, threshold=2.0, component_name="Yaw Angle")
            
            # Merge angle outlier flags to inlier_mask
            angle_outliers = pitch_outliers | roll_outliers | yaw_outliers
            angle_filtered_count = np.sum(angle_outliers)
            
            if angle_filtered_count > 0:
                self.logger.info(f"Angle outlier detection: Found {angle_filtered_count} angle outlier regions")
                for i in range(len(transformations)):
                    if angle_outliers[i] and inlier_mask[i]:
                        gid = transformations[i]['result']['grid_id']
                        reasons = []
                        if pitch_outliers[i]:
                            reasons.append(f"Pitch={angles_pitch[i]:.3f}°")
                        if roll_outliers[i]:
                            reasons.append(f"Roll={angles_roll[i]:.3f}°")
                        if yaw_outliers[i]:
                            reasons.append(f"Yaw={angles_yaw[i]:.3f}°")
                        self.logger.info(f"  Grid {gid} rejected due to angle outlier: {', '.join(reasons)}")
                        inlier_mask[i] = False
            else:
                self.logger.info("Angle outlier detection: No angle outliers found")
            
            # -----------------------------
            # 6) Output valid results + Update valid_grid_ids
            # -----------------------------
            valid_results = []
            self.valid_grid_ids = set()

            self.logger.info("Filter results (Distance to main cluster center + Angle detection):")
            for i, t in enumerate(transformations):
                gid = t['result']['grid_id']
                if inlier_mask[i]:
                    valid_results.append(t['result'])
                    self.valid_grid_ids.add(gid)
                    self.logger.debug(f"Grid {gid} kept: dist={d_to_medoid[i]:.3f}, fitness={t['fitness']:.3f}")
                else:
                    self.logger.debug(f"Grid {gid} rejected: dist={d_to_medoid[i]:.3f}, fitness={t['fitness']:.3f}")

            # -----------------------------
            # 7) Fallback
            # -----------------------------
            if len(valid_results) < 1:
                self.logger.warning(f"Too few remaining ({len(valid_results)}), keeping top {max(3, len(region_results) // 2)} sorted by fitness")
                sorted_results = sorted(region_results, key=lambda x: x.get('fitness', 0.0), reverse=True)
                n_keep = max(3, len(region_results) // 2)
                valid_results = sorted_results[:n_keep]
                self.valid_grid_ids = set(result['grid_id'] for result in valid_results)

            self.logger.info(f"Final results: Total {len(region_results)}, Rejected {len(region_results) - len(valid_results)}, Kept {len(valid_results)}")
            self.logger.info(f"Valid grid_id set: {sorted(self.valid_grid_ids)}")
            return valid_results

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            self.logger.error("Analysis failed!")
            # Do not block flow on error: Return original results (Fallback thought consistent with original logic)
            self.valid_grid_ids = set(result['grid_id'] for result in region_results)
            return region_results


    def fuse_multiple_transformations(self, region_results: List[Dict], fitness_threshold: float,
                                     enable_fallback_to_identity: bool) -> Dict:
        """Fuse multiple transformation matrices"""
        self.logger.info("Fusing transformation matrices...")

        if len(region_results) == 0:
            self.logger.error("No valid registration results")
            if enable_fallback_to_identity:
                self.logger.info("Fallback mechanism: Return identity matrix")
                return self._create_identity_transformation_result("no_valid_registration")
            else:
                return None

        # When grid count is too low (<=3), directly select region with min RMSE
        if len(region_results) <= 2:
            self.logger.info(f"Low grid count ({len(region_results)}), directly selecting region with min RMSE")
            best_result = min(region_results, key=lambda x: x['rmse'])
            self.valid_grid_ids = {best_result['grid_id']}
            
            centered_transform = best_result['transformation']
            final_transform = self._convert_centered_to_original_transform(centered_transform)
            
            self.logger.info(f"Select region {best_result['grid_id']}: RMSE={best_result['rmse']:.4f}m, "
                           f"MAE={best_result['mae']:.4f}m, Fitness={best_result['fitness']:.4f}")
            
            return {
                'fused_transformation': final_transform,
                'centered_transformation': centered_transform,
                'fusion_method': 'best_rmse_single_region',
                'n_regions': 1,
                'confidence': best_result['fitness'],
                'center_offset': self.offset_center.copy(),
                'quality_check_passed': True,
                'composite_confidence': best_result.get('composite_score', 0),
                'valid_grid_ids': list(self.valid_grid_ids)
            }

        # Fitness filtering and similarity analysis
        valid_results = [r for r in region_results if r['fitness'] >= fitness_threshold]

        if len(valid_results) == 0:
            self.logger.warning("No regions meeting fitness criteria")
            if enable_fallback_to_identity:
                self.logger.info("Fallback mechanism: No qualifying areas, return identity matrix")
                return self._create_identity_transformation_result("low_fitness")
            else:
                return None

        similarity_filtered_results = self.analyze_transformation_similarity_MAD_S2S(valid_results)

        if len(similarity_filtered_results) == 1:
            centered_transform = similarity_filtered_results[0]['transformation']
            final_transform = self._convert_centered_to_original_transform(centered_transform)

            return {
                'fused_transformation': final_transform,
                'centered_transformation': centered_transform,
                'fusion_method': 'single_valid_region',
                'n_regions': 1,
                'confidence': similarity_filtered_results[0]['fitness'],
                'center_offset': self.offset_center.copy(),
                'quality_check_passed': True,
                'composite_confidence': similarity_filtered_results[0].get('composite_score', 0),
                'valid_grid_ids': list(self.valid_grid_ids)
            }

        if len(similarity_filtered_results) < 2:
            self.logger.warning(f"Only {len(similarity_filtered_results)} valid regions after similarity filtering, selecting region with highest composite_score")
            # Select region with highest composite_score from original valid_results
            best_result = max(valid_results, key=lambda x: x.get('composite_score', 0))
            self.valid_grid_ids = {best_result['grid_id']}
            
            centered_transform = best_result['transformation']
            final_transform = self._convert_centered_to_original_transform(centered_transform)
            
            self.logger.info(f"Select region {best_result['grid_id']}: composite_score={best_result.get('composite_score', 0):.4f}, "
                           f"RMSE={best_result['rmse']:.4f}m, MAE={best_result['mae']:.4f}m, "
                           f"Fitness={best_result['fitness']:.4f}")
            
            return {
                'fused_transformation': final_transform,
                'centered_transformation': centered_transform,
                'fusion_method': 'best_score_after_similarity_filter',
                'n_regions': 1,
                'confidence': best_result['fitness'],
                'center_offset': self.offset_center.copy(),
                'quality_check_passed': True,
                'composite_confidence': best_result.get('composite_score', 0),
                'valid_grid_ids': list(self.valid_grid_ids)
            }

        # Calculate fusion weights
        weights = []
        transformations = []

        for result in similarity_filtered_results:
            feature_weight = result.get('composite_score', 0)
            rmse_weight = result['rmse']**2
            fitness_weight = result['fitness']
            n_corr = result['n_correspondences']
            combined_weight = 1 /(n_corr* rmse_weight)
            weights.append(combined_weight)
            transformations.append(result['transformation'])

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Fuse transformation matrices
        fused_centered_transform = self._weighted_average_transformations(transformations, weights)
        fused_original_transform = self._convert_centered_to_original_transform(fused_centered_transform)

        # Calculate confidence
        weighted_fitness = np.average([r['fitness'] for r in similarity_filtered_results], weights=weights)
        weighted_composite_score = np.average([r.get('composite_score', 0) for r in similarity_filtered_results],
                                              weights=weights)

        fusion_result = {
            'fused_transformation': fused_original_transform,
            'centered_transformation': fused_centered_transform,
            'fusion_method': 'composite_score_weighted',
            'n_regions': len(similarity_filtered_results),
            'confidence': weighted_fitness,
            'composite_confidence': weighted_composite_score,
            'center_offset': self.offset_center.copy(),
            'quality_check_passed': True,
            'valid_grid_ids': list(self.valid_grid_ids)
        }

        self.logger.info(f"Fusion complete: {len(similarity_filtered_results)} final regions")
        self.logger.info(f"Registration confidence: {weighted_fitness:.4f}")
        self.logger.info(f"Composite score confidence: {weighted_composite_score:.4f}")

        return fusion_result

    def _create_identity_transformation_result(self, reason: str, score: float = 0.0) -> Dict:
        """Create identity matrix fallback result"""
        self.logger.info(f"Creating identity matrix fallback result, reason: {reason}")

        identity_matrix = np.eye(4, dtype=np.float64)

        return {
            'fused_transformation': identity_matrix,
            'centered_transformation': identity_matrix,
            'fusion_method': 'identity_fallback',
            'n_regions': 0,
            'confidence': 0.0,
            'composite_confidence': score,
            'center_offset': self.offset_center.copy() if self.offset_center is not None else np.zeros(3),
            'quality_check_passed': False,
            'fallback_reason': reason,
            'is_identity': True,
            'valid_grid_ids': []
        }

    def _convert_centered_to_original_transform(self, centered_transform: np.ndarray) -> np.ndarray:
        """Convert transformation matrix from decentralized space to original coordinate system"""
        if self.offset_center is None:
            self.logger.warning("No offset information, returning original transformation matrix directly")
            return centered_transform

        offset = self.offset_center

        T_offset_neg = np.eye(4, dtype=np.float64)
        T_offset_neg[:3, 3] = -offset

        T_offset_pos = np.eye(4, dtype=np.float64)
        T_offset_pos[:3, 3] = offset

        original_transform = T_offset_pos @ centered_transform @ T_offset_neg

        self.logger.debug("Transformation matrix coordinate system conversion complete")

        return original_transform

    def _weighted_average_transformations(self, transformations: List[np.ndarray],
                                          weights: List[float]) -> np.ndarray:
        """Weighted fusion of transformation matrices"""
        try:
            from scipy.linalg import logm, expm

            weights = np.array(weights, dtype=np.float64)
            weights = weights / np.sum(weights)

            # Process rotation and translation separately
            rotation_matrices, translations = [], []

            for T in transformations:
                T_copy = np.array(T, dtype=np.float64)
                R, t = T_copy[:3, :3], T_copy[:3, 3]

                # Ensure rotation matrix is strictly orthogonal
                U, s, Vt = np.linalg.svd(R)
                R_clean = U @ Vt

                if np.linalg.det(R_clean) < 0:
                    U[:, -1] *= -1
                    R_clean = U @ Vt

                rotation_matrices.append(R_clean)
                translations.append(t.copy())

            rotation_matrices = np.array(rotation_matrices)
            translations = np.array(translations)

            # Lie algebra fusion
            lie_algebras = []
            for R in rotation_matrices:
                log_R = logm(R)
                omega = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
                lie_algebras.append(omega)

            lie_algebras = np.array(lie_algebras)
            avg_omega = np.average(lie_algebras, axis=0, weights=weights)

            # Construct skew-symmetric matrix
            avg_log_R = np.array([
                [0, -avg_omega[2], avg_omega[1]],
                [avg_omega[2], 0, -avg_omega[0]],
                [-avg_omega[1], avg_omega[0], 0]
            ])

            avg_rotation_matrix = expm(avg_log_R)
            avg_translation = np.average(translations, axis=0, weights=weights)

            # Construct final transformation matrix
            fused_transformation = np.eye(4, dtype=np.float64)
            fused_transformation[:3, :3] = avg_rotation_matrix
            fused_transformation[:3, 3] = avg_translation

            return fused_transformation
            
        except Exception as e:
            self.logger.error(f"Weighted fusion of transformation matrices failed: {e}")
            # Fallback to simple translation average
            avg_translation = np.average([T[:3, 3] for T in transformations], axis=0, weights=weights)
            fallback_transform = np.eye(4, dtype=np.float64)
            fallback_transform[:3, 3] = avg_translation
            return fallback_transform
