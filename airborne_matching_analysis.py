#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Airborne strip matching main class extension - complete analysis pipeline
"""

import os
import time
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Try importing target_cal
try:
    from target_cal import (
        find_control_point_with_reference,
        analyze_target_points
    )
    HAS_TARGET_CAL = True
except ImportError:
    HAS_TARGET_CAL = False



def run_complete_analysis(self, las_file1: str, las_file2: str,
                          apply_transform: bool = True,
                          output_dir: str = None,
                          reference_file: str = None) -> Dict[str, Any]:
    """Run complete airborne strip matching analysis - outlier region correspondence filtering + enhanced memory logging + logging + target accuracy assessment"""
    self.logger.info("=" * 80)
    self.logger.info("Complete airborne strip matching analysis - outlier filtering + enhanced memory logging + reporting")
    self.logger.info("=" * 80)

    total_start = time.time()
    registration_time = None
    registration_start = None
    analysis_results = {}
    
    # Load reference points
    reference_points = None
    if reference_file is not None and os.path.exists(reference_file):
        try:
            reference_points = np.loadtxt(reference_file)
            self.logger.info(f"Loaded reference points file: {reference_file}, total {len(reference_points)} points")
        except Exception as e:
            self.logger.warning(f"Failed to load reference points file: {e}")
    elif reference_file is not None:
         self.logger.warning(f"Reference points file does not exist: {reference_file}")

    try:
        # Set correspondence save information
        if output_dir is None:
            output_dir = os.path.dirname(las_file1)

        correspondence_dir = os.path.join(output_dir, "correspondences")
        file1_name = os.path.splitext(os.path.basename(las_file1))[0]
        file2_name = os.path.splitext(os.path.basename(las_file2))[0]
        file_pair_name = f"{file1_name}_to_{file2_name}"

        self.set_correspondence_save_info(correspondence_dir, file_pair_name)
        self.logger.info(f"Correspondence save directory: {correspondence_dir}")

        # 1. Read file info
        self.logger.info("[Step 1/10] Read strip information")
        info1 = self.loader.load_las_header_info(las_file1)
        info2 = self.loader.load_las_header_info(las_file2)

        if not info1 or not info2:
            self.logger.error("Failed to read strip information")
            return None

        analysis_results['strip_info'] = {'strip1': info1, 'strip2': info2}

        # 2. Compute overlap region
        self.logger.info("[Step 2/10] Compute overlap region")
        overlap_info = self.grid_manager.compute_overlap_region(info1, info2, self.buffer_delta)

        if not overlap_info:
            self.logger.error("Failed to compute overlap region")
            return None

        analysis_results['overlap_info'] = overlap_info
        self.offset_center = overlap_info['center_offset']
        self.transformation_fusion.set_offset_center(self.offset_center)

        # 3. Load point cloud data
        self.logger.info("[Step 3/10] Load strip point cloud data")
        strip1_data = self.loader.load_overlap_points_chunked(las_file1, overlap_info)
        strip2_data = self.loader.load_overlap_points_chunked(las_file2, overlap_info)

        if len(strip1_data.points) == 0 or len(strip2_data.points) == 0:
            self.logger.error("Missing valid overlap point cloud data")
            return None

        # Registration timing start: after pointclouds are loaded and overlap computed (before grid partitioning)
        registration_start = time.time()

        # 4. Compute unified global ranges
        self.logger.info("[Step 4/10] Compute unified global data ranges")
        global_ranges = self.feature_calculator.compute_unified_global_ranges(
            strip1_data, strip2_data, self.height_percentiles, self.intensity_percentiles)
        analysis_results['global_ranges'] = global_ranges
        self.global_ranges = global_ranges

        # 5. Create unified grid
        self.logger.info("[Step 5/10] Create unified grid system")
        unified_grid_info = self.grid_manager.create_unified_grid(overlap_info)
        analysis_results['unified_grid_info'] = unified_grid_info

        # 6. Assign point clouds to grid
        self.logger.info("[Step 6/10] Assign point clouds to grid")

        grid_ids1, unique_ids1, counts1 = self.grid_manager.assign_points_to_unified_grid(
            strip1_data, unified_grid_info, "Strip1")
        features1_df = self.feature_calculator.compute_grid_features_unified(
            strip1_data, grid_ids1, unique_ids1, "Strip1")

        grid_ids2, unique_ids2, counts2 = self.grid_manager.assign_points_to_unified_grid(
            strip2_data, unified_grid_info, "Strip2")
        features2_df = self.feature_calculator.compute_grid_features_unified(
            strip2_data, grid_ids2, unique_ids2, "Strip2")

        strip1_processed = {
            'point_data': strip1_data,
            'grid_ids': grid_ids1,
            'features_df': features1_df
        }
        strip2_processed = {
            'point_data': strip2_data,
            'grid_ids': grid_ids2,
            'features_df': features2_df
        }

        analysis_results['strip1_data'] = strip1_processed
        analysis_results['strip2_data'] = strip2_processed
        baseline_memory_mb = self._get_memory_usage_mb()
        analysis_results['baseline_memory_mb'] = baseline_memory_mb
        # 7. Joint feature evaluation and region selection
        self.logger.info("[Step 7/10] Joint feature evaluation and region selection")
        unified_features_df = self.feature_calculator.compute_composite_scores_unified(
            features1_df, features2_df)

        if len(unified_features_df) == 0:
            self.logger.error("No unified feature data available")
            return None

        selected_regions = self.feature_calculator.select_high_quality_regions_unified(
            unified_features_df, self.enable_fallback_to_identity)

        if len(selected_regions) == 0:
            self.logger.warning("Score threshold check: no high-quality regions selected")
            if self.enable_fallback_to_identity:
                self.logger.info("Fallback: identity matrix triggered")
                fusion_result = self.transformation_fusion._create_identity_transformation_result(
                    "no_quality_regions_selected")
                analysis_results['unified_features'] = unified_features_df
                analysis_results['selected_regions'] = None
                analysis_results['region_registration_results'] = []
                analysis_results['fusion_result'] = fusion_result
                analysis_results['quality_check_failed'] = True
            else:
                self.logger.error("No high-quality regions selected and fallback disabled")
                return None

        analysis_results['unified_features'] = unified_features_df
        analysis_results['selected_regions'] = selected_regions

        # 8. Region registration and fusion
        self.logger.info("[Step 8/10] Region registration and transformation fusion (outlier filtering + enhanced memory logging)")

        if len(selected_regions) > 0:
            region_results = self.register_selected_regions(
                strip1_processed,
                strip2_processed,
                selected_regions,
                baseline_memory_mb=analysis_results.get('baseline_memory_mb')
            )

            if len(region_results) == 0:
                self.logger.error("All region registrations failed")
                if self.enable_fallback_to_identity:
                    self.logger.info("Fallback: registration failed, returning identity matrix")
                    fusion_result = self.transformation_fusion._create_identity_transformation_result(
                        "all_registration_failed")
                    analysis_results['region_registration_results'] = []
                    analysis_results['fusion_result'] = fusion_result
                    analysis_results['quality_check_failed'] = True
                else:
                    self.logger.error("All registrations failed and fallback disabled")
                    return None
            else:
                fusion_result = self.transformation_fusion.fuse_multiple_transformations(
                    region_results, 0.3, self.enable_fallback_to_identity)
                
                if not fusion_result:
                    self.logger.error("Transformation fusion failed")
                    return None
                
                # Update valid_grid_ids
                self.registration_manager.valid_grid_ids = self.transformation_fusion.valid_grid_ids
                
                # Filter global correspondences
                self.registration_manager.filter_global_correspondences_by_valid_grids()
                
                analysis_results['region_registration_results'] = region_results
                analysis_results['fusion_result'] = fusion_result

        fusion_result_outpath = os.path.join(output_dir,
                                             f"{self.registration_manager.current_file_pair}_{self.grid_rows}_{self.grid_cols}.npy")
        
        try:
            np.save(fusion_result_outpath, fusion_result['centered_transformation'])
            self.logger.debug(f"Fusion result matrix saved: {fusion_result_outpath}")
        except Exception as e:
            self.logger.error(f"Failed to save fusion result matrix: {e}")

        # Registration timing ends when global transformation matrix computation is completed
        if registration_start is not None:
            registration_time = time.time() - registration_start
        
        # 9. Compute global registration metrics
        self.logger.info("[Step 9/10] Compute global registration metrics (based on filtered correspondences)")
        fusion_result = analysis_results['fusion_result']
        is_identity = fusion_result.get('is_identity', False)

        if not self.registration_manager.store_correspondences:
            self.logger.info("Correspondence saving disabled, skipping global metrics computation (can compute later with open3d)")
            analysis_results['global_metrics'] = {
                'global_rmse': 0.0, 'global_mae': 0.0, 'global_fitness': 0.0,
                'total_correspondences': 0, 'valid_correspondences': 0
            }
        elif not is_identity and len(self.registration_manager.all_correspondences) > 0:
            global_metrics = self.registration_manager.compute_global_registration_metrics(
                fusion_result['centered_transformation'])
            analysis_results['global_metrics'] = global_metrics

            fusion_result['global_rmse'] = global_metrics['global_rmse']
            fusion_result['global_mae'] = global_metrics['global_mae']
            fusion_result['global_fitness'] = global_metrics['global_fitness']
            fusion_result['total_correspondences'] = global_metrics['total_correspondences']
            fusion_result['valid_correspondences'] = global_metrics['valid_correspondences']
        else:
            self.logger.info("Skipping global metrics: identity fallback or no correspondence data")
            analysis_results['global_metrics'] = {
                'global_rmse': 0.0, 'global_mae': 0.0, 'global_fitness': 0.0,
                'total_correspondences': 0, 'valid_correspondences': 0
            }

        # 10. Apply transformation and save results
        if apply_transform:
            self.logger.info("[Step 10/10] Apply transformation and save results")

            os.makedirs(output_dir, exist_ok=True)

            transformation_matrix = fusion_result['fused_transformation']

            is_identity = fusion_result.get('is_identity', False)
            matrix_suffix = "identity_fallback" if is_identity else "filtered_enhanced_memory_logged"

            # Save transformation matrix
            matrix_file = os.path.join(output_dir, f"transformation_matrix_{matrix_suffix}.txt")
            self.save_transformation_matrix(transformation_matrix, matrix_file)
            analysis_results['transformation_matrix_file'] = matrix_file

            # Apply transformation to strip1
            output_las1 = os.path.join(output_dir,
                                       f"strip1_transformed_{self.registration_manager.current_file_pair}_{self.grid_cols}_{self.grid_rows}.las")
            transformed_file1 = self.apply_transformation_to_pointcloud(las_file1, transformation_matrix,
                                                                        output_las1)
            analysis_results['transformed_strip1'] = transformed_file1

            # Save reference strip2
            output_las2 = os.path.join(output_dir,
                                       f"strip2_reference_{self.registration_manager.current_file_pair}_{self.grid_cols}_{self.grid_rows}.las")
            identity_matrix = np.eye(4)
            reference_file2 = self.apply_transformation_to_pointcloud(las_file2, identity_matrix, output_las2)
            analysis_results['reference_strip2'] = reference_file2

            transform_type = "identity_fallback" if is_identity else "outlier_region_cauchy_icp"

            self.logger.info(f"{transform_type} applied:")
            self.logger.info(f"Transformation matrix: {os.path.basename(matrix_file)}")
            self.logger.info(f"Transformed strip1: {os.path.basename(transformed_file1)}")
            self.logger.info(f"Reference strip2: {os.path.basename(reference_file2)}")
            if self.registration_manager.store_correspondences:
                self.logger.info(f"Correspondence files saved in: {correspondence_dir}")

        # 11. Target accuracy assessment
        if HAS_TARGET_CAL and reference_points is not None and apply_transform:
            self.logger.info("[Step 11/11] Target accuracy assessment")
            try:
                self.logger.info("Extracting and analyzing target accuracy...")
                src_points_df = find_control_point_with_reference([Path(transformed_file1)], reference_points)
                # Use original las_file2 or saved reference_file2 as destination
                dst_points_df = find_control_point_with_reference([Path(las_file2)], reference_points)

                if src_points_df is None or dst_points_df is None or len(src_points_df) == 0 or len(dst_points_df) == 0:
                    self.logger.warning("Unable to extract target points, skipping accuracy assessment")
                else:
                    src_results, _ = analyze_target_points(src_points_df, reference_points)
                    dst_results, _ = analyze_target_points(dst_points_df, reference_points)
                    
                    if src_results is None or dst_results is None or len(src_results) == 0 or len(dst_results) == 0:
                        self.logger.warning("Target analysis failed")
                    else:
                        src_ref_ids = set(src_results['reference_id'].unique())
                        dst_ref_ids = set(dst_results['reference_id'].unique())
                        common_ref_ids = src_ref_ids & dst_ref_ids

                        if len(common_ref_ids) == 0:
                            self.logger.warning("No common target points found")
                        else:
                            self.logger.info(f"Found {len(common_ref_ids)} common targets")
                            
                            # Compute accuracy metrics
                            position_errors_3d = []
                            distance_to_ref_errors = []
                            normal_vector_angles = []
                            normal_z_angle_diffs = []

                            for ref_id in common_ref_ids:
                                src_target = src_results[src_results['reference_id'] == ref_id].iloc[0]
                                dst_target = dst_results[dst_results['reference_id'] == ref_id].iloc[0]
                                
                                # 1. Relative position error
                                pos_src = np.array([src_target['center_x'], src_target['center_y'], src_target['center_z']])
                                pos_dst = np.array([dst_target['center_x'], dst_target['center_y'], dst_target['center_z']])
                                position_errors_3d.append(np.linalg.norm(pos_src - pos_dst))
                                
                                # 2. Distance-to-reference error
                                distance_to_ref_errors.append(abs(src_target['distance_to_ref'] - dst_target['distance_to_ref']))
                                
                                # 3. Normal vector angle
                                normal_src = np.array([src_target['normal_x'], src_target['normal_y'], src_target['normal_z']])
                                normal_dst = np.array([dst_target['normal_x'], dst_target['normal_y'], dst_target['normal_z']])
                                
                                dot_product = np.clip(np.dot(normal_src, normal_dst), -1.0, 1.0)
                                angle = np.degrees(np.arccos(dot_product))
                                normal_vector_angles.append(min(angle, 180 - angle))
                                
                                # 4. Plane Z-axis angle difference
                                normal_z_angle_diffs.append(abs(src_target['normal_angle'] - dst_target['normal_angle']))
                            
                            accuracy_metrics = {
                                'target_count': len(common_ref_ids),
                                'position_3d_mean': float(np.mean(position_errors_3d)),
                                'position_ref_mean': float(np.mean(distance_to_ref_errors)),
                                'plane_vector_angle_mean': float(np.mean(normal_vector_angles)),
                                'plane_z_angle_mean': float(np.mean(normal_z_angle_diffs))
                            }
                            
                            analysis_results['target_accuracy'] = accuracy_metrics
                            
                            self.logger.info("Target accuracy assessment results:")
                            self.logger.info(f"  Position accuracy 1 (point-to-point): {accuracy_metrics['position_3d_mean']:.4f} m")
                            self.logger.info(f"  Position accuracy 2 (to reference): {accuracy_metrics['position_ref_mean']:.4f} m")
                            self.logger.info(f"  Plane accuracy 1 (normal vector angle): {accuracy_metrics['plane_vector_angle_mean']:.4f}°")
                            self.logger.info(f"  Plane accuracy 2 (Z-axis angle diff): {accuracy_metrics['plane_z_angle_mean']:.4f}°")

            except Exception as e:
                self.logger.error(f"Target accuracy computation error: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        total_time = time.time() - total_start

        # Generate report
        _generate_summary_report(self, analysis_results, registration_time, total_time)
        analysis_results['registration_time'] = registration_time
        analysis_results['total_time'] = total_time

        return analysis_results

    except Exception as e:
        self.logger.error(f"Error during analysis: {e}")
        return None


def _generate_summary_report(self, results: Dict, registration_time: float, total_time: float):
    """Generate summary report"""
    self.logger.info("=" * 80)
    self.logger.info("Airborne strip matching analysis report - outlier filtering + enhanced memory logging")
    self.logger.info("=" * 80)

    strip1_info = results['strip_info']['strip1']
    strip2_info = results['strip_info']['strip2']
    overlap_info = results['overlap_info']

    self.logger.info("Data overview:")
    self.logger.info(f"Strip1: {strip1_info['n_points']:,} points")
    self.logger.info(f"Strip2: {strip2_info['n_points']:,} points")
    self.logger.info(f"Overlap region: {overlap_info['width']:.1f} × {overlap_info['height']:.1f} m")

    self.logger.info("Registration method: robust_icp_cauchy (Cauchy robust ICP)")
    self.logger.info("Enhanced features: MAE calculation + translation vector analysis + global metrics + outlier filtering + memory monitoring + logging")
    if self.registration_manager.store_correspondences:
        self.logger.info(f"Correspondence saving: {self.registration_manager.current_file_pair}")
    else:
        self.logger.info("Correspondence saving: disabled")

    unified_features = results['unified_features']
    selected_regions = results['selected_regions']

    self.logger.info("Processing results:")
    self.logger.info(f"Grid settings: {self.grid_rows}×{self.grid_cols} = {self.grid_rows * self.grid_cols} grids")
    self.logger.info(f"Valid common grids: {len(unified_features)}")
    self.logger.info(f"Selected high-quality regions: {len(selected_regions) if selected_regions is not None else 0}")

    quality_failed = results.get('quality_check_failed', False)
    if quality_failed:
        self.logger.warning("Quality check: FAILED - identity fallback triggered")
        fusion_result = results['fusion_result']
        fallback_reason = fusion_result.get('fallback_reason', 'unknown')
        self.logger.warning(f"Fallback reason: {fallback_reason}")
    else:
        self.logger.info("Quality check: PASSED")

    region_results = results.get('region_registration_results', [])
    fusion_result = results['fusion_result']

    self.logger.info("Registration & fusion results:")
    self.logger.info(f"Number of successfully registered regions: {len(region_results)}")
    self.logger.info("Registration method: robust_icp_cauchy (outlier filtering + memory logging + structured logging)")
    self.logger.info(f"Fusion method: {fusion_result['fusion_method']}")
    self.logger.info(f"Registration confidence: {fusion_result['confidence']:.4f}")

    valid_grids = fusion_result.get('valid_grid_ids', [])
    if valid_grids:
        self.logger.info(f"Valid regions after similarity analysis: {sorted(valid_grids)}")
        self.logger.info(f"Number of valid regions: {len(valid_grids)}")

    is_identity = fusion_result.get('is_identity', False)
    if is_identity:
        self.logger.warning("Identity fallback details:")
        self.logger.warning(f"Fallback reason: {fusion_result.get('fallback_reason', 'unknown')}")
        self.logger.warning("Applied transform: Identity (no change)")
    else:
        global_metrics = results.get('global_metrics', {})
        if self.registration_manager.store_correspondences and global_metrics.get('total_correspondences', 0) > 0:
            self.logger.info("Global registration metrics (based on filtered valid regions):")
            self.logger.info(f"Global RMSE: {global_metrics['global_rmse']:.4f} m")
            self.logger.info(f"Global MAE: {global_metrics['global_mae']:.4f} m")
            self.logger.info(f"Global fitness: {global_metrics['global_fitness']:.4f}")
            self.logger.info(f"Total correspondences: {global_metrics['total_correspondences']:,}")
            self.logger.info(f"Valid correspondences: {global_metrics.get('valid_correspondences', 0):,} (after outlier removal)")
        elif not self.registration_manager.store_correspondences:
            self.logger.info("Global registration metrics: correspondence saving disabled, skipped")

    if 'target_accuracy' in results:
        target_acc = results['target_accuracy']
        self.logger.info("Target accuracy evaluation:")
        self.logger.info(f"Common target count: {target_acc['target_count']}")
        self.logger.info(f"Position error (point-to-point): {target_acc['position_3d_mean']:.4f} m")
        self.logger.info(f"Position error (to reference): {target_acc['position_ref_mean']:.4f} m")
        self.logger.info(f"Plane error (normal vector): {target_acc['plane_vector_angle_mean']:.4f} °")
        self.logger.info(f"Plane error (Z-axis angle): {target_acc['plane_z_angle_mean']:.4f} °")

    # if region_results:
    #     best = max(region_results, key=lambda x: x.get('composite_score', 0))
    #     self.logger.info("Best region registration:")
    #     self.logger.info(f"Grid ID: {best['grid_id']}")
    #     self.logger.info(f"Composite score: {best.get('composite_score', 0):.4f}")
    #     self.logger.info(f"Registration fitness: {best['fitness']:.4f}")
    #     self.logger.info(f"Registration accuracy (RMSE): {best['rmse']:.4f} m")
    #     self.logger.info(f"Mean absolute error (MAE): {best.get('mae', 0):.4f} m")

    #     translation = best.get('translation_vector', np.zeros(3))
    #     self.logger.info(f"Translation vector: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}] m")

    #     if 'memory_usage_mb' in best:
    #         memory_usage = best['memory_usage_mb']
    #         memory_efficiency = best.get('memory_efficiency', 0)
    #         self.logger.info(f"Memory usage: {self._format_memory_size(memory_usage)} (efficiency: {memory_efficiency:.2f} MB/1000 pts)")

    #     if len(valid_grids) > 1:
    #         self.logger.debug("Translation vectors for valid regions after similarity analysis:")
    #         valid_results = [r for r in region_results if r['grid_id'] in valid_grids]
    #         for result in valid_results:
    #             trans = result.get('translation_vector', np.zeros(3))
    #             mae_val = result.get('mae', 0)
    #             mem_usage = result.get('memory_usage_mb', 0)
    #             mem_eff = result.get('memory_efficiency', 0)
    #             fitness = result['fitness']
    #             rmse = result['rmse']
    #             self.logger.debug(f"Grid {result['grid_id']}: [{trans[0]:6.3f}, {trans[1]:6.3f}, {trans[2]:6.3f}] m "
    #                               f"(MAE: {mae_val:.4f}m, Memory: {self._format_memory_size(mem_usage)}, Efficiency: {mem_eff:.2f} MB/1000 pts), "
    #                               f"Fitness:{fitness:.4f}, RMSE: {rmse:.4f}")

    execution_result = "Identity fallback" if is_identity else "Outlier-region filtered Cauchy robust ICP registration completed"
    if registration_time is not None:
        self.logger.info(f"Registration time: {registration_time:.2f} s")
    self.logger.info(f"Execution time: {total_time:.2f} s")
    self.logger.info(f"Execution result: {execution_result}")
    # Memory usage summary
    memory_report = getattr(self, 'last_memory_report', None)
    if memory_report:
        self.logger.info("Memory usage summary:")
        self.logger.info(f"Baseline memory: {self._format_memory_size(memory_report['initial_memory_mb'])}")
        self.logger.info(f"Peak memory: {self._format_memory_size(memory_report['peak_memory_mb'])}")
        self.logger.info(f"Final memory: {self._format_memory_size(memory_report['final_memory_mb'])}")
        self.logger.info(f"Average memory: {self._format_memory_size(memory_report['avg_memory_mb'])}")
        self.logger.info(f"Average memory increase: {self._format_memory_size(memory_report['avg_memory_increase_mb'])}")
        if 'max_region_memory_increase_mb' in memory_report:
            self.logger.info(f"Max region memory increase: {self._format_memory_size(memory_report['max_region_memory_increase_mb'])}")

    if self.registration_manager.store_correspondences:
        self.logger.info(f"Correspondence save directory: {self.registration_manager.correspondence_save_dir}")
        if not is_identity:
            self.logger.info(f"Global error analysis files: {self.registration_manager.correspondence_save_dir}/global_*_filtered.txt")
        self.logger.info(f"Memory usage log file: {self.registration_manager.correspondence_save_dir}/memory_usage_{self.registration_manager.current_file_pair}.txt")
    self.logger.info("=" * 80)


# Bind methods to AirborneStripMatching class
from airborne_matching import AirborneStripMatching
AirborneStripMatching.run_complete_analysis = run_complete_analysis
AirborneStripMatching._generate_summary_report = _generate_summary_report
