#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Airborne strip matching main class module
Main class that integrates all submodule functionality
"""

import numpy as np
import laspy
import os
import time
import psutil
import gc
import concurrent.futures
from typing import Dict, Any, List, Tuple
from data_structures import PointCloudData
from pointcloud_loader import PointCloudLoader
from grid_manager import GridManager
from feature_calculator import FeatureCalculator
from registration import RegistrationManager
from transformation_fusion import TransformationFusion
from logging_utils import get_logger


class AirborneStripMatching:
    """Airborne strip matching algorithm - enhanced: MAE + global analysis + outlier filtering + memory logging + structured logging"""

    def __init__(self, grid_rows=10, grid_cols=10, buffer_delta=10.0,
                 z_bins=30, intensity_bins=30, alpha=0.1, n_threads=2,
                 registration_method='robust_icp_cauchy',
                 height_percentiles=(1, 99), intensity_percentiles=(2, 98),
                 entropy_weight_a=1.0, entropy_weight_b=1.0, entropy_weight_c=1.0,
                 min_composite_score_threshold=0.01,
                 min_valid_regions=1,
                 enable_fallback_to_identity=False, min_regions=3, max_region=8,
                 feature_type='fpfh',
                 max_points_per_region: int = None,
                 store_correspondences: bool = False,
                 max_correspondences_per_grid: int = None,
                 sample_seed: int = None,
                 voxel_size: float = 0.5):
        """Initialize matching algorithm - supports robust_icp_cauchy only"""
        self.logger = get_logger()
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.buffer_delta = buffer_delta
        self.z_bins = z_bins
        self.intensity_bins = intensity_bins
        self.alpha = alpha
        self.n_threads = n_threads
        self.registration_method = registration_method
        self.feature_type = feature_type
        self.max_points_per_region = max_points_per_region
        self.store_correspondences = store_correspondences
        self.max_correspondences_per_grid = max_correspondences_per_grid
        self.sample_seed = sample_seed

        self.height_percentiles = height_percentiles
        self.intensity_percentiles = intensity_percentiles

        self.entropy_weight_a = entropy_weight_a
        self.entropy_weight_b = entropy_weight_b
        self.entropy_weight_c = entropy_weight_c

        self.min_composite_score_threshold = min_composite_score_threshold
        self.min_valid_regions = min_valid_regions
        self.enable_fallback_to_identity = enable_fallback_to_identity

        self.min_regions = min_regions
        self.max_regions = max_region

        self.global_ranges = None
        self.offset_center = None
        self.voxel_size = voxel_size
        # Initialize submodules
        self.loader = PointCloudLoader()
        self.grid_manager = GridManager(grid_rows, grid_cols, n_threads)
        self.feature_calculator = FeatureCalculator(
            z_bins, intensity_bins, alpha,
            entropy_weight_a, entropy_weight_b, entropy_weight_c,
            min_composite_score_threshold, min_valid_regions,
            min_regions, max_region, n_threads
        )
        self.registration_manager = RegistrationManager(
            method=registration_method,
            feature_type=feature_type,
            store_correspondences=store_correspondences,
            max_correspondences_per_grid=max_correspondences_per_grid,
            voxel_size=self.voxel_size  # Default voxel size; adjust as needed
        )
        self.transformation_fusion = TransformationFusion()

        # Memory monitoring
        self.process = psutil.Process()

        self.logger.info(f"Airborne strip matching initialized (MAE + global analysis + outlier filtering + memory logging)")
        self.logger.info(f"Grid settings: {grid_rows}×{grid_cols} = {grid_rows * grid_cols} grids")
        self.logger.info(f"Registration method: {registration_method}")
        self.logger.info(f"New features: MAE calculation + global correspondence analysis + outlier filtering + memory monitoring + logging")

    def _get_memory_usage_mb(self) -> float:
        """Get system used memory (MB), consistent with Task Manager"""
        try:
            vm = psutil.virtual_memory()
            return vm.used / (1024 * 1024)
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def _format_memory_size(self, size_mb: float) -> str:
        """Format memory size string"""
        if size_mb < 1024:
            return f"{size_mb:.1f} MB"
        else:
            return f"{size_mb / 1024:.1f} GB"

    def set_correspondence_save_info(self, save_dir: str, file_pair_name: str):
        """Set correspondence save information"""
        self.registration_manager.set_correspondence_save_info(save_dir, file_pair_name)

    def register_selected_regions(self, strip1_data: Dict, strip2_data: Dict,
                                  selected_regions, baseline_memory_mb: float = None) -> List[Dict]:
        """Register selected corresponding regions - enhanced: adds MAE, translation output and memory monitoring"""
        self.logger.info("Registering selected regions (robust_icp_cauchy + MAE + translation analysis + memory monitoring)...")
        self.logger.info(f"Number of selected regions: {len(selected_regions)}")

        initial_memory = baseline_memory_mb if baseline_memory_mb is not None else self._get_memory_usage_mb()
        peak_memory = initial_memory
        memory_records = []

        self.logger.debug(f"Initial memory before registration: {self._format_memory_size(initial_memory)}")

        region_results = []
        self.registration_manager.all_correspondences = {}

        rng = np.random.default_rng(self.sample_seed) if self.sample_seed is not None else None

        def _register_one_region(idx: int, region_row) -> Dict:
            grid_id = region_row['grid_id']
            pre_region_memory = self._get_memory_usage_mb()

            # Extract corresponding grid point clouds from both strips
            mask1 = strip1_data['grid_ids'] == grid_id
            mask2 = strip2_data['grid_ids'] == grid_id

            if not np.any(mask1) or not np.any(mask2):
                self.logger.warning(f"Skipping grid {grid_id}: missing corresponding point cloud")
                return {'skipped': True, 'grid_id': grid_id}

            idx1 = np.flatnonzero(mask1)
            idx2 = np.flatnonzero(mask2)

            if self.max_points_per_region is not None:
                if len(idx1) > self.max_points_per_region:
                    if rng is None:
                        idx1 = np.random.choice(idx1, self.max_points_per_region, replace=False)
                    else:
                        idx1 = rng.choice(idx1, self.max_points_per_region, replace=False)
                if len(idx2) > self.max_points_per_region:
                    if rng is None:
                        idx2 = np.random.choice(idx2, self.max_points_per_region, replace=False)
                    else:
                        idx2 = rng.choice(idx2, self.max_points_per_region, replace=False)

            # Extract point cloud data
            region1_data = PointCloudData(
                points=strip1_data['point_data'].points[idx1],
                intensity=strip1_data['point_data'].intensity[idx1],
                colors=strip1_data['point_data'].colors[idx1] if strip1_data['point_data'].has_colors else None,
                has_colors=strip1_data['point_data'].has_colors
            )

            region2_data = PointCloudData(
                points=strip2_data['point_data'].points[idx2],
                intensity=strip2_data['point_data'].intensity[idx2],
                colors=strip2_data['point_data'].colors[idx2] if strip2_data['point_data'].has_colors else None,
                has_colors=strip2_data['point_data'].has_colors
            )

            self.logger.info(f"Registering grid {grid_id} ({idx + 1}/{len(selected_regions)}): {len(region1_data.points)} pts <-> {len(region2_data.points)} pts")

            def _cleanup_region_memory():
                """Clean up temporary memory for the current grid"""
                try:
                    del region1_data
                except Exception:
                    pass
                try:
                    del region2_data
                except Exception:
                    pass
                try:
                    del mask1, mask2, idx1, idx2
                except Exception:
                    pass
                gc.collect()

            try:
                reg_result = self.registration_manager.method_robust_icp_cauchy_enhanced(
                    region1_data.points, region2_data.points, grid_id)

                post_region_memory = self._get_memory_usage_mb()
                region_memory_usage = post_region_memory - pre_region_memory

                memory_record = {
                    'grid_id': grid_id,
                    'region_index': idx + 1,
                    'pre_memory_mb': pre_region_memory,
                    'post_memory_mb': post_region_memory,
                    'memory_increase_mb': region_memory_usage,
                    'points_strip1': len(region1_data.points),
                    'points_strip2': len(region2_data.points),
                    'total_points': len(region1_data.points) + len(region2_data.points)
                }

                if reg_result['success']:
                    translation_vector = reg_result['transformation'][:3, 3]

                    result = {
                        'grid_id': grid_id,
                        'composite_score': region_row['composite_score'],
                        'raw_score': region_row['raw_score'],
                        'density_balance_factor': region_row['density_balance_factor'],
                        'transformation': reg_result['transformation'],
                        'translation_vector': translation_vector,
                        'fitness': reg_result['fitness'],
                        'rmse': reg_result['rmse'],
                        'mae': reg_result['mae'],
                        'runtime': reg_result['runtime'],
                        'success': True,
                        'strip1_points': len(region1_data.points),
                        'strip2_points': len(region2_data.points),
                        'n_correspondences': reg_result['n_corr'],
                        'registration_method': 'robust_icp_cauchy',
                        'memory_usage_mb': region_memory_usage,
                        'memory_efficiency': region_memory_usage / memory_record['total_points'] * 1000
                    }

                    self.logger.info(f"Success: fitness={reg_result['fitness']:.4f}, RMSE={reg_result['rmse']:.4f} m, MAE={reg_result['mae']:.4f} m")
                    self.logger.debug(f"Translation vector: [{translation_vector[0]:.4f}, {translation_vector[1]:.4f}, {translation_vector[2]:.4f}] m")
                    self.logger.debug(f"Memory usage: {self._format_memory_size(region_memory_usage)} (efficiency: {result['memory_efficiency']:.2f} MB/1000 pts)")
                else:
                    result = None
                    self.logger.warning(f"Failure: {reg_result.get('error', 'unknown error')}")
                    self.logger.debug(f"Memory usage: {self._format_memory_size(region_memory_usage)}")

                payload = {
                    'result': result,
                    'memory_record': memory_record,
                    'post_memory_mb': post_region_memory
                }
                _cleanup_region_memory()
                return payload

            except Exception as e:
                post_region_memory = self._get_memory_usage_mb()
                region_memory_usage = post_region_memory - pre_region_memory

                memory_record = {
                    'grid_id': grid_id,
                    'region_index': idx + 1,
                    'pre_memory_mb': pre_region_memory,
                    'post_memory_mb': post_region_memory,
                    'memory_increase_mb': region_memory_usage,
                    'points_strip1': len(region1_data.points),
                    'points_strip2': len(region2_data.points),
                    'total_points': len(region1_data.points) + len(region2_data.points),
                    'error': str(e)
                }

                self.logger.error(f"Exception: {e}")
                self.logger.debug(f"Memory usage: {self._format_memory_size(region_memory_usage)}")

                payload = {
                    'result': None,
                    'memory_record': memory_record,
                    'post_memory_mb': post_region_memory
                }
                _cleanup_region_memory()
                return payload

        use_parallel = self.n_threads is not None and self.n_threads > 1 and len(selected_regions) > 1
        if use_parallel:
            max_workers = self.n_threads
            region_items = list(selected_regions.iterrows())
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_register_one_region, idx, row): (idx, row)
                    for idx, (_, row) in enumerate(region_items)
                }
                for future in concurrent.futures.as_completed(future_map):
                    payload = future.result()
                    if payload.get('skipped'):
                        continue
                    memory_record = payload.get('memory_record')
                    if memory_record:
                        memory_records.append(memory_record)
                        post_memory_mb = payload.get('post_memory_mb', initial_memory)
                        if post_memory_mb > peak_memory:
                            peak_memory = post_memory_mb
                    result = payload.get('result')
                    if result:
                        region_results.append(result)
        else:
            for idx, (_, region_row) in enumerate(selected_regions.iterrows()):
                payload = _register_one_region(idx, region_row)
                if payload.get('skipped'):
                    continue
                memory_record = payload.get('memory_record')
                if memory_record:
                    memory_records.append(memory_record)
                    post_memory_mb = payload.get('post_memory_mb', initial_memory)
                    if post_memory_mb > peak_memory:
                        peak_memory = post_memory_mb
                result = payload.get('result')
                if result:
                    region_results.append(result)

                if (idx + 1) % 5 == 0:
                    gc.collect()
                    gc_memory = self._get_memory_usage_mb()
                    self.logger.debug(f"Memory after GC: {self._format_memory_size(gc_memory)}")

        # Memory statistics and report
        final_memory = self._get_memory_usage_mb()
        avg_memory = final_memory
        if memory_records:
            avg_memory = float(np.mean([r['post_memory_mb'] for r in memory_records]))
        avg_memory_increase = avg_memory - initial_memory
        max_region_memory_increase = None
        if memory_records:
            memory_increases = [r['memory_increase_mb'] for r in memory_records if 'error' not in r]
            if memory_increases:
                max_region_memory_increase = float(np.max(memory_increases))

        self.logger.info("Registration memory usage report:")
        self.logger.info(f"Initial memory: {self._format_memory_size(initial_memory)}")
        self.logger.info(f"Peak memory: {self._format_memory_size(peak_memory)}")
        self.logger.info(f"Final memory: {self._format_memory_size(final_memory)}")
        self.logger.info(f"Average memory: {self._format_memory_size(avg_memory)}")
        self.logger.info(f"Average memory increase: {self._format_memory_size(avg_memory_increase)}")
        if max_region_memory_increase is not None:
            self.logger.info(f"Max region memory increase: {self._format_memory_size(max_region_memory_increase)}")

        # Save to instance for analysis report output
        self.last_memory_report = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'avg_memory_mb': avg_memory,
            'avg_memory_increase_mb': avg_memory_increase,
            'max_region_memory_increase_mb': max_region_memory_increase
        }

        if memory_records:
            memory_increases = [r['memory_increase_mb'] for r in memory_records if 'error' not in r]
            if memory_increases:
                avg_memory_per_region = np.mean(memory_increases)
                max_memory_per_region = np.max(memory_increases)
                min_memory_per_region = np.min(memory_increases)

                self.logger.debug(f"Average memory increase per region: {self._format_memory_size(avg_memory_per_region)}")
                self.logger.debug(f"Max memory increase per region: {self._format_memory_size(max_memory_per_region)}")
                self.logger.debug(f"Min memory increase per region: {self._format_memory_size(min_memory_per_region)}")
                self.last_memory_report['max_region_memory_increase_mb'] = float(max_memory_per_region)

                total_points_processed = sum(r['total_points'] for r in memory_records if 'error' not in r)
                if total_points_processed > 0:
                    overall_efficiency = avg_memory_increase / total_points_processed * 1000
                    self.logger.debug(f"Overall memory efficiency: {overall_efficiency:.2f} MB/1000 pts")

        # Save detailed memory usage records
        if self.registration_manager.correspondence_save_dir and memory_records:
            memory_log_file = os.path.join(self.registration_manager.correspondence_save_dir, 
                                          f"memory_usage_{self.registration_manager.current_file_pair}.txt")
            try:
                with open(memory_log_file, 'w') as f:
                    f.write("# Detailed memory usage records for registration process\n")
                    f.write(f"# File pair: {self.registration_manager.current_file_pair}\n")
                    f.write(f"# Initial memory: {initial_memory:.2f} MB\n")
                    f.write(f"# Peak memory: {peak_memory:.2f} MB\n")
                    f.write(f"# Final memory: {final_memory:.2f} MB\n")
                    f.write(f"# Average memory: {avg_memory:.2f} MB\n")
                    f.write(f"# Average memory increase: {avg_memory_increase:.2f} MB\n")
                    f.write("# Format: grid_id region_index pre_memory(MB) post_memory(MB) memory_increase(MB) strip1_points strip2_points total_points memory_efficiency(MB/1000pts) [ERROR]\n")

                    for record in memory_records:
                        efficiency = record['memory_increase_mb'] / record['total_points'] * 1000 if record['total_points'] > 0 else 0
                        error_info = f" ERROR: {record['error']}" if 'error' in record else ""
                        f.write(f"{record['grid_id']} {record['region_index']} {record['pre_memory_mb']:.2f} "
                                f"{record['post_memory_mb']:.2f} {record['memory_increase_mb']:.2f} "
                                f"{record['points_strip1']} {record['points_strip2']} {record['total_points']} "
                                f"{efficiency:.2f}{error_info}\n")

                self.logger.debug(f"Memory usage log saved: {os.path.basename(memory_log_file)}")
                
            except Exception as e:
                self.logger.error(f"Failed to save memory usage record: {e}")

        self.logger.info(f"Region registration complete: {len(region_results)}/{len(selected_regions)} succeeded")
        self.logger.debug(f"Global correspondence collection complete: {len(self.registration_manager.all_correspondences)} regions")

        return region_results

    def apply_transformation_to_pointcloud(self, las_filepath: str, transformation_matrix: np.ndarray,
                                           output_filepath: str = None) -> str:
        """Apply a transformation matrix to the whole point cloud and save"""
        self.logger.info(f"Applying transformation to point cloud: {os.path.basename(las_filepath)}")
        start_time = time.time()

        is_identity = np.allclose(transformation_matrix, np.eye(4), atol=1e-6)
        if is_identity:
            self.logger.info("Identity matrix detected: copying file without transformation")

        if output_filepath is None:
            base_name = os.path.splitext(os.path.basename(las_filepath))[0]
            output_dir = os.path.dirname(las_filepath)
            suffix = "_identity" if is_identity else "_transformed"
            output_filepath = os.path.join(output_dir, f"{base_name}{suffix}{self.grid_cols}_{self.grid_rows}.las")

        self.logger.debug(f"Output file: {os.path.basename(output_filepath)}")

        try:
            self.logger.debug("Reading point cloud data...")
            las_file = laspy.read(las_filepath)
            total_points = len(las_file.points)

            self.logger.info(f"Total points: {total_points:,}")

            if is_identity:
                self.logger.debug("Identity handling: saving original file copy")
                las_file.write(output_filepath)
            else:
                self.logger.debug("Applying transformation...")
                x = np.array(las_file.x, dtype=np.float64)
                y = np.array(las_file.y, dtype=np.float64)
                z = np.array(las_file.z, dtype=np.float64)

                points_homogeneous = np.column_stack([x, y, z, np.ones(len(x))])
                transformed_points = (transformation_matrix @ points_homogeneous.T).T

                las_file.x = transformed_points[:, 0]
                las_file.y = transformed_points[:, 1]
                las_file.z = transformed_points[:, 2]

                las_file.header.min = [
                    np.min(las_file.x), np.min(las_file.y), np.min(las_file.z)
                ]
                las_file.header.max = [
                    np.max(las_file.x), np.max(las_file.y), np.max(las_file.z)
                ]

                self.logger.debug("Writing transformed point cloud...")
                las_file.write(output_filepath)

        except Exception as e:
            self.logger.error(f"Point cloud transform failed - {e}")
            if os.path.exists(output_filepath):
                os.remove(output_filepath)
            raise e

        transform_time = time.time() - start_time
        processing_type = "copy" if is_identity else "transform"
        self.logger.info(f"Point cloud {processing_type} complete: {total_points:,} points")
        self.logger.debug(f"Processing time: {transform_time:.2f}s")

        return output_filepath

    def save_transformation_matrix(self, transformation_matrix: np.ndarray, filepath: str):
        """Save transformation matrix to file"""
        self.logger.info(f"Saving transformation matrix to: {os.path.basename(filepath)}")

        is_identity = np.allclose(transformation_matrix, np.eye(4), atol=1e-6)

        try:
            np.save(filepath.replace('.txt', '.npy'), transformation_matrix)

            with open(filepath, 'w') as f:
                f.write("# Airborne strip matching - transformation matrix (outlier filtering + memory logging + structured logging)\n")
                f.write(f"# Generated time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Format: 4x4 homogeneous transformation matrix\n")
                f.write("# Application: transformed_point = T @ [x, y, z, 1]^T\n")
                f.write(f"# Matrix type: {'Identity matrix (no transform)' if is_identity else 'Transformation matrix'}\n")
                f.write(f"# Based on filtered valid regions: {sorted(self.transformation_fusion.valid_grid_ids) if hasattr(self.transformation_fusion, 'valid_grid_ids') else 'unknown'}\n")
                f.write("\n")

                for i in range(4):
                    row_str = " ".join([f"{transformation_matrix[i, j]:12.8f}" for j in range(4)])
                    f.write(f"{row_str}\n")

            matrix_type = "identity matrix" if is_identity else "transformation matrix"
            
            self.logger.info(f"{matrix_type} saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save transformation matrix: {e}")
