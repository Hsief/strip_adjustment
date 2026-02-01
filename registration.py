#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registration module
Responsible for point cloud registration, correspondence calculation and saving
"""

import numpy as np
import open3d as o3d
import os
import time
import threading
import pandas as pd
import psutil
try:
    import torch
except ImportError:
    torch = None
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List
from data_structures import PointCloudData
from logging_utils import get_logger


class RegistrationManager:
    """Registration manager"""
    
    def __init__(self, method='p2plane_icp', feature_type='fpfh',
                 store_correspondences: bool = True,
                 max_correspondences_per_grid: int = None,
                 voxel_size: float = 0.5):
        """
        Initialize RegistrationManager

        Parameters:
            method: registration method, options include:
                - 'p2plane_icp': Point-to-Plane ICP (default)
                - 'p2point_icp': Point-to-Point ICP
                - 'teaser++': TEASER++ robust registration
                - 'ndt': Normal Distributions Transform
                - 'ransac': RANSAC + ICP
                - 'fgr': Fast Global Registration
                - 'mac': MAC (Maximal Cliques) robust registration
                - 'turboreg': TurboReg GPU accelerated registration
            feature_type: feature type used (mainly for teaser++): 'fpfh' or 'normal'
        """
        self.logger = get_logger()
        self._correspondence_lock = threading.Lock()
        self.store_correspondences = store_correspondences
        self.max_correspondences_per_grid = max_correspondences_per_grid
        self.correspondence_save_dir = None
        self.current_file_pair = None
        self.all_correspondences = {}
        self.valid_grid_ids = set()
        self.method = method
        self.feature_type = feature_type
        self.logger.info(f"Registration method: {method}")
        self.voxel_size = voxel_size
        if method == 'teaser++':
            self.logger.info(f"Feature type: {feature_type}")
    
    def set_correspondence_save_info(self, save_dir: str, file_pair_name: str):
        """Set correspondence save information"""
        self.correspondence_save_dir = save_dir
        self.current_file_pair = file_pair_name
        os.makedirs(save_dir, exist_ok=True)
        self.logger.debug(f"Correspondence save directory set: {save_dir}")

    def save_correspondences(self, source_points: np.ndarray, target_points: np.ndarray,
                             correspondences: np.ndarray, transformation: np.ndarray,
                             grid_id: int, fitness: float):
        """Save correspondences to txt file - includes source points before and after transformation"""
        # if self.correspondence_save_dir is None or self.current_file_pair is None:
        #     self.logger.warning("Correspondence save info not set, skipping save")
        #     return

        # # Extract correspondences
        # source_corr = source_points[correspondences[:, 0]]
        # target_corr = target_points[correspondences[:, 1]]

        # # Compute transformed source correspondences
        # source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
        # transformed_source_corr = (transformation @ source_homogeneous.T).T[:, :3]

        # # Assign colors for different point types
        # source_colors = np.tile([1.0, 0.0, 0.0], (len(source_corr), 1))
        # transformed_source_colors = np.tile([0.0, 1.0, 0.0], (len(transformed_source_corr), 1))
        # target_colors = np.tile([0.0, 0.0, 1.0], (len(target_corr), 1))

        # # Merge data: coordinates + colors
        # source_data = np.column_stack([source_corr, source_colors])
        # transformed_source_data = np.column_stack([transformed_source_corr, transformed_source_colors])
        # target_data = np.column_stack([target_corr, target_colors])

        # # Compute distance improvement before/after registration
        # distances_before = np.linalg.norm(source_corr - target_corr, axis=1)
        # distances_after = np.linalg.norm(transformed_source_corr - target_corr, axis=1)
        # avg_distance_before = np.mean(distances_before)
        # avg_distance_after = np.mean(distances_after)
        # distance_improvement = avg_distance_before - avg_distance_after

        # # Save filename
        # filename = f"correspondences_{self.current_file_pair}_grid{grid_id}_fitness{fitness:.4f}.txt"
        # filepath = os.path.join(self.correspondence_save_dir, filename)

        # try:
        #     with open(filepath, 'w') as f:
        #         # Write original source correspondences (red)
        #         f.write("# === Original source correspondences (red) ===\n")
        #         for point in source_data:
        #             f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.3f} {point[4]:.3f} {point[5]:.3f}\n")

        #         # Write transformed source correspondences (green)
        #         f.write("# === Transformed source correspondences (green) ===\n")
        #         for point in transformed_source_data:
        #             f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.3f} {point[4]:.3f} {point[5]:.3f}\n")

        #         # Write target correspondences (blue)
        #         f.write("# === Target correspondences (blue) ===\n")
        #         for point in target_data:
        #             f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.3f} {point[4]:.3f} {point[5]:.3f}\n")

        #     self.logger.debug(f"Correspondences saved: {filename} ({len(correspondences)} pairs)")
        #     self.logger.debug(f"Distance improvement: {avg_distance_before:.4f}m → {avg_distance_after:.4f}m (improved {distance_improvement:.4f}m)")
            
        # except Exception as e:
        #     self.logger.error(f"Failed to save correspondences file: {e}")

    def store_correspondence_for_global_analysis(self, source_points: np.ndarray, target_points: np.ndarray,
                                                 correspondences: np.ndarray, grid_id: int):
        """Store correspondences for global analysis"""
        if not self.store_correspondences:
            return

        if self.max_correspondences_per_grid is not None and len(correspondences) > self.max_correspondences_per_grid:
            idx = np.random.choice(len(correspondences), self.max_correspondences_per_grid, replace=False)
            correspondences = correspondences[idx]

        source_corr = source_points[correspondences[:, 0]]
        target_corr = target_points[correspondences[:, 1]]

        correspondence_data = {
            'grid_id': grid_id,
            'source_points': source_corr.astype(np.float32, copy=True),
            'target_points': target_corr.astype(np.float32, copy=True),
            'n_correspondences': len(correspondences)
        }

        with self._correspondence_lock:
            self.all_correspondences[grid_id] = correspondence_data
            self.logger.debug(f"Correspondences stored for global analysis: grid {grid_id}, {len(correspondences)} pairs")

    def filter_global_correspondences_by_valid_grids(self):
        """Filter global correspondences by valid grid ids"""
        self.logger.info("Filtering global correspondences...")

        total_grids_before = len(self.all_correspondences)
        valid_grids_count = len(self.valid_grid_ids)

        self.logger.info(f"Total grids before filtering: {total_grids_before}")
        self.logger.info(f"Grids passed similarity analysis: {valid_grids_count}")

        if valid_grids_count == 0:
            self.logger.warning("No valid grids passed similarity analysis")
            self.all_correspondences = {}
            return

        # Filter correspondences
        filtered_correspondences = {}
        total_points_before = 0
        total_points_after = 0

        for grid_id, corr_data in self.all_correspondences.items():
            total_points_before += corr_data['n_correspondences']

            if grid_id in self.valid_grid_ids:
                filtered_correspondences[grid_id] = corr_data
                total_points_after += corr_data['n_correspondences']
                self.logger.debug(f"Keeping grid {grid_id}: {corr_data['n_correspondences']} correspondences")
            else:
                self.logger.debug(f"Removing grid {grid_id}: {corr_data['n_correspondences']} correspondences (outlier transform)")

        self.all_correspondences = filtered_correspondences

        self.logger.info("Filtering results:")
        self.logger.info(f"Grids: {total_grids_before} → {len(filtered_correspondences)}")
        self.logger.info(f"Correspondences: {total_points_before} → {total_points_after}")
        self.logger.info(f"Correspondences removed: {total_points_before - total_points_after}")

    def method_robust_icp_cauchy_enhanced(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """Enhanced robust ICP - route to selected method implementation"""
        # Route to the selected registration method
        # if self.method == 'p2plane_icp':
        return self._method_p2plane_icp(source, target, grid_id)

    def _select_voxel_size_for_icp(self, source_count: int, target_count: int) -> float:
        """Adaptively select voxel size based on point counts (more points -> stronger downsampling)"""
        # max_points = max(source_count, target_count)
        # if max_points >= 10_000_000:
        #     return self.voxel_size  if self.voxel_size*0.8>0.1 else 0.1
        # if max_points >= 5_000_000:
        #     return self.voxel_size*0.6 if self.voxel_size*0.6>0.1 else 0.1
        # if max_points >= 2_000_000:
        #     return self.voxel_size*0.4 if self.voxel_size*0.4>0.1 else 0.1
        # if max_points >= 1_000_000:
        #     return self.voxel_size*0.2 if self.voxel_size*0.2>0.1 else 0.1
        return self.voxel_size
    
    def _method_p2plane_icp(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """Point-to-Plane ICP with Cauchy loss function"""
        start_time = time.time()

        try:
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)

            voxel_size = self._select_voxel_size_for_icp(len(source), len(target))
            # voxel_size = 0.4
            self.logger.debug(
                f"Grid {grid_id} voxel downsample size: {voxel_size:.2f} m "
                f"(source={len(source):,}, target={len(target):,})"
            )
            source_pcd = source_pcd.voxel_down_sample(voxel_size)
            target_pcd = target_pcd.voxel_down_sample(voxel_size)

            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))

            loss = o3d.pipelines.registration.CauchyLoss(k=2)
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 0.8, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )

                # Get downsampled point coordinates
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)

            # Compute correspondences
            correspondences = self._find_correspondences(source_points, target_points, reg_result.transformation,max_distance=0.8)

            # Compute MAE
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]

                # Transform source points
                source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
                transformed_source = (reg_result.transformation @ source_homogeneous.T).T[:, :3]

                # Compute MAE
                distances = np.linalg.norm(transformed_source - target_corr, axis=1)
                mae = np.mean(distances)

                # Store correspondences for global analysis
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)

            # Save correspondences
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                          reg_result.transformation, grid_id, reg_result.fitness)

            return {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'rmse': reg_result.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Robust ICP registration failed: {e}")
            return {
                'transformation': np.eye(4),
                'fitness': 0.0,
                'rmse': float('inf'),
                'mae': float('inf'),
                'runtime': time.time() - start_time,
                'n_corr': 0,
                'success': False,
                'error': str(e)
            }

    def _find_correspondences(self, source_points: np.ndarray, target_points: np.ndarray,
                              transformation: np.ndarray, max_distance: float = 1) -> np.ndarray:
        """Find correspondences after applying transformation to source points"""
        try:
            # Transform source points to target space
            source_homogeneous = np.column_stack([source_points, np.ones(len(source_points))])
            transformed_source = (transformation @ source_homogeneous.T).T[:, :3]

            # Build KDTree to search nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
            nn.fit(target_points)

            distances, indices = nn.kneighbors(transformed_source)

            # Filter correspondences with large distance
            valid_mask = distances.flatten() < max_distance
            valid_source_indices = np.where(valid_mask)[0]
            valid_target_indices = indices.flatten()[valid_mask]

            correspondences = np.column_stack([valid_source_indices, valid_target_indices])

            return correspondences
            
        except Exception as e:
            self.logger.error(f"Failed to compute correspondences: {e}")
            return np.empty((0, 2), dtype=int)
    
    def _method_p2point_icp(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """Point-to-Point ICP registration"""
        start_time = time.time()
        
        try:
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)
            
            source_pcd = source_pcd.voxel_down_sample(0.1)
            target_pcd = target_pcd.voxel_down_sample(0.1)
            
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 0.8, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)
            correspondences = self._find_correspondences(source_points, target_points, reg_result.transformation)
            
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]
                source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
                transformed_source = (reg_result.transformation @ source_homogeneous.T).T[:, :3]
                distances = np.linalg.norm(transformed_source - target_corr, axis=1)
                mae = np.mean(distances)
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)
            
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                        reg_result.transformation, grid_id, reg_result.fitness)
            
            return {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'rmse': reg_result.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Point-to-Point ICP registration failed: {e}")
            return self._get_failed_result(start_time, str(e))
    
    def _method_teaser(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """TEASER++ robust registration"""
        start_time = time.time()
        
        try:
            # Try importing TEASER++
            try:
                import sys
                from pathlib import Path
                import shutil
                
                # Get project root (main/registration.py -> main -> root)
                project_root = Path(__file__).resolve().parent.parent
                teaser_root = project_root / "comparasion/teaser/TEASER-plusplus"
                teaser_build_path = teaser_root / "build/python"
                teaser_src_path = teaser_root / "python/teaserpp_python"
                
                # Ensure build directory has __init__.py
                build_init = teaser_build_path / "teaserpp_python/__init__.py"
                src_init = teaser_src_path / "__init__.py"
                
                if teaser_build_path.exists():
                    if not build_init.exists() and src_init.exists():
                        try:
                            shutil.copy2(src_init, build_init)
                        except Exception:
                            pass
                            
                    if str(teaser_build_path) not in sys.path:
                        sys.path.insert(0, str(teaser_build_path))
                        
                import teaserpp_python
            except ImportError:
                self.logger.error("TEASER++ not installed, please install TEASER++")
                return self._get_failed_result(start_time, "TEASER++ not installed")
            
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)
            
            # Downsample
            source_pcd = source_pcd.voxel_down_sample(0.1)
            target_pcd = target_pcd.voxel_down_sample(0.1)
            
            # Estimate normals
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
            
            src_features = None
            dst_features = None
            
            if self.feature_type == 'fpfh':
                radius_feature = 2.5
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                src_features = np.array(source_fpfh.data).T
                dst_features = np.array(target_fpfh.data).T
            elif self.feature_type == 'normal':
                src_features = np.asarray(source_pcd.normals)
                dst_features = np.asarray(target_pcd.normals)
            else:
                self.logger.warning(f"Unknown feature type: {self.feature_type}, defaulting to FPFH")
                radius_feature = 2.5
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                src_features = np.array(source_fpfh.data).T
                dst_features = np.array(target_fpfh.data).T
            
            # Feature matching
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(dst_features)
            distances, indices = nbrs.kneighbors(src_features)
            
            # Mutual matching filtering
            nbrs_src = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(src_features)
            distances_d2s, indices_d2s = nbrs_src.kneighbors(dst_features)
            
            mutual_matches = []
            for i in range(len(src_features)):
                j = indices[i, 0]
                if indices_d2s[j, 0] == i:
                    mutual_matches.append([i, j])
            
            corrs = np.array(mutual_matches)
            
            # Limit number of correspondences
            if len(corrs) > 20000:
                dist_vals = distances[corrs[:, 0]].flatten()
                best_indices = np.argsort(dist_vals, kind='stable')[:20000]
                corrs = corrs[best_indices]
            
            # Prepare TEASER++ inputs
            src_points = np.asarray(source_pcd.points)
            dst_points = np.asarray(target_pcd.points)
            src_corr_points = src_points[corrs[:, 0]].T
            dst_corr_points = dst_points[corrs[:, 1]].T
            
            # Solve with TEASER++
            params = teaserpp_python.RobustRegistrationSolver.Params()
            params.cbar2 = 1.0
            params.noise_bound = 0.1
            params.estimate_scaling = False
            params.rotation_estimation_algorithm = \
                teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            params.rotation_max_iterations = 100
            
            solver = teaserpp_python.RobustRegistrationSolver(params)
            solver.solve(src_corr_points, dst_corr_points)
            solution = solver.getSolution()
            
            # Build transformation matrix
            transformation = np.eye(4)
            transformation[:3, :3] = solution.rotation
            transformation[:3, 3] = solution.translation
            
            # Evaluate fitness and RMSE
            source_pcd.transform(transformation)
            evaluation = o3d.pipelines.registration.evaluate_registration(
                source_pcd, target_pcd, 0.8, np.eye(4))
            
            # Compute correspondences and MAE
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)
            correspondences = self._find_correspondences(source_points, target_points, np.eye(4))
            
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]
                distances = np.linalg.norm(source_corr - target_corr, axis=1)
                mae = np.mean(distances)
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)
            
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                        transformation, grid_id, evaluation.fitness)
            
            return {
                'transformation': transformation,
                'fitness': evaluation.fitness,
                'rmse': evaluation.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"TEASER++ registration failed: {e}")
            return self._get_failed_result(start_time, str(e))
    
    def _method_ndt(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """NDT (Normal Distributions Transform) registration"""
        start_time = time.time()
        
        try:
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)
            
            source_pcd = source_pcd.voxel_down_sample(0.5)
            target_pcd = target_pcd.voxel_down_sample(0.5)
            
            # NDT registration
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 2.0, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)
            correspondences = self._find_correspondences(source_points, target_points, reg_result.transformation)
            
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]
                source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
                transformed_source = (reg_result.transformation @ source_homogeneous.T).T[:, :3]
                distances = np.linalg.norm(transformed_source - target_corr, axis=1)
                mae = np.mean(distances)
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)
            
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                        reg_result.transformation, grid_id, reg_result.fitness)
            
            return {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'rmse': reg_result.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"NDT registration failed: {e}")
            return self._get_failed_result(start_time, str(e))
    
    def _method_ransac(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """RANSAC + ICP registration"""
        start_time = time.time()
        
        try:
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)
            
            source_pcd = source_pcd.voxel_down_sample(0.5)
            target_pcd = target_pcd.voxel_down_sample(0.5)
            
            # Feature computation
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            
            source_feature = o3d.pipelines.registration.Feature()
            target_feature = o3d.pipelines.registration.Feature()

            if self.feature_type == 'fpfh':
                radius_feature = 2.5
                source_feature = o3d.pipelines.registration.compute_fpfh_feature(
                    source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_feature = o3d.pipelines.registration.compute_fpfh_feature(
                    target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            elif self.feature_type == 'normal':
                source_feature.data = np.asarray(source_pcd.normals).T
                target_feature.data = np.asarray(target_pcd.normals).T
            else:
                self.logger.warning(f"Unknown feature type: {self.feature_type}, defaulting to FPFH")
                radius_feature = 2.5
                source_feature = o3d.pipelines.registration.compute_fpfh_feature(
                    source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_feature = o3d.pipelines.registration.compute_fpfh_feature(
                    target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

            # RANSAC registration
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_pcd, target_pcd, source_feature, target_feature, True, 0.5,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.5)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )
            
            # ICP fine-tuning
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 0.5, result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)
            correspondences = self._find_correspondences(source_points, target_points, reg_result.transformation)
            
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]
                source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
                transformed_source = (reg_result.transformation @ source_homogeneous.T).T[:, :3]
                distances = np.linalg.norm(transformed_source - target_corr, axis=1)
                mae = np.mean(distances)
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)
            
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                        reg_result.transformation, grid_id, reg_result.fitness)
            
            return {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'rmse': reg_result.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"RANSAC registration failed: {e}")
            return self._get_failed_result(start_time, str(e))
    
    def _method_fgr(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """Fast Global Registration (FGR) registration"""
        start_time = time.time()
        
        try:
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)
            
            source_pcd = source_pcd.voxel_down_sample(0.5)
            target_pcd = target_pcd.voxel_down_sample(0.5)
            
            # Compute FPFH features
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=100))
            
            # FGR registration
            result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_pcd, target_pcd, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=0.5)
            )
            
            # ICP fine-tuning
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 0.5, result_fgr.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)
            correspondences = self._find_correspondences(source_points, target_points, reg_result.transformation)
            
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]
                source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
                transformed_source = (reg_result.transformation @ source_homogeneous.T).T[:, :3]
                distances = np.linalg.norm(transformed_source - target_corr, axis=1)
                mae = np.mean(distances)
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)
            
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                        reg_result.transformation, grid_id, reg_result.fitness)
            
            return {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'rmse': reg_result.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"FGR registration failed: {e}")
            return self._get_failed_result(start_time, str(e))
    
    def _method_mac(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """MAC (Maximal Cliques) robust registration"""
        start_time = time.time()
        
        if torch is None:
            self.logger.error("MAC registration requires Torch")
            return self._get_failed_result(start_time, "torch not installed")
        
        try:
            # Try importing igraph
            try:
                from igraph import Graph
            except ImportError:
                self.logger.error("MAC registration requires python-igraph: pip install python-igraph")
                return self._get_failed_result(start_time, "python-igraph not installed")

            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, MAC registration will be very slow or may not run")
                device = "cpu"
            else:
                device = "cuda:0"

            # Preprocessing: downsample and compute FPFH
            voxel_size = 0.5
            
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)
            
            source_down = source_pcd.voxel_down_sample(voxel_size)
            target_down = target_pcd.voxel_down_sample(voxel_size)
            
            radius_normal = voxel_size * 2
            source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            
            src_features = None
            dst_features = None

            if self.feature_type == 'fpfh':
                radius_feature = voxel_size * 5 
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                src_features = np.array(source_fpfh.data).T
                dst_features = np.array(target_fpfh.data).T
            elif self.feature_type == 'normal':
                src_features = np.asarray(source_down.normals)
                dst_features = np.asarray(target_down.normals)
            else:
                self.logger.warning(f"Unknown feature type: {self.feature_type}, defaulting to FPFH")
                radius_feature = voxel_size * 5 
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                src_features = np.array(source_fpfh.data).T
                dst_features = np.array(target_fpfh.data).T

            # Normalize features
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            dst_features = dst_features / (np.linalg.norm(dst_features, axis=1, keepdims=True) + 1e-6)

            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine').fit(dst_features)
            distances, indices = nbrs.kneighbors(src_features)
            
            source_idx = indices.flatten()
            corrs = np.column_stack([np.arange(len(src_features)), source_idx])
            
            # max_corr limit
            max_corr = 10000
            if len(corrs) > max_corr:
                best_indices = np.argsort(distances.flatten(), kind='stable')[:max_corr]
                corrs = corrs[best_indices]
            
            # Prepare MAC inputs
            src_corr_points = np.array(source_down.points)[corrs[:, 0]]
            dst_corr_points = np.array(target_down.points)[corrs[:, 1]]
            
            # Solve MAC optimization
            transformation = self._solve_mac_optimization(src_corr_points, dst_corr_points, device=device)
            
            # ICP refinement
            reg_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, 0.5, transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )

            # Result processing
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)
            correspondences = self._find_correspondences(source_points, target_points, reg_result.transformation)
            
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]
                source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
                transformed_source = (reg_result.transformation @ source_homogeneous.T).T[:, :3]
                distances = np.linalg.norm(transformed_source - target_corr, axis=1)
                mae = np.mean(distances)
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)
            
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                        reg_result.transformation, grid_id, reg_result.fitness)

            return {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'rmse': reg_result.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"MAC registration failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_failed_result(start_time, str(e))

    def _solve_mac_optimization(self, src_corr_points, dst_corr_points, device="cuda:0", inlier_threshold=0.1, min_clique_size=3):
        """MAC graph matching optimization solver"""
        from igraph import Graph
        
        # Convert to torch tensors
        src_pts = torch.from_numpy(src_corr_points).to(device).float()
        dst_pts = torch.from_numpy(dst_corr_points).to(device).float()
        
        # Build graph
        src_dist = ((src_pts[:, None, :] - src_pts[None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((dst_pts[:, None, :] - dst_pts[None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dis = torch.abs(src_dist - tgt_dist)
        
        # First-order compatibility graph (FCG)
        FCG = torch.clamp(1 - cross_dis ** 2 / inlier_threshold ** 2, min=0)
        FCG = FCG - torch.diag_embed(torch.diag(FCG))
        FCG[FCG < 0.99] = 0
        
        # Second-order compatibility graph (SCG)
        SCG = torch.matmul(FCG, FCG) * FCG
        
        # Search for maximal cliques
        SCG_np = SCG.cpu().numpy()
        graph = Graph.Adjacency((SCG_np > 0).tolist())
        graph.es['weight'] = SCG_np[SCG_np.nonzero()]
        graph.vs['label'] = range(0, len(src_pts))
        graph.to_undirected()
        
        macs = graph.maximal_cliques(min=min_clique_size)
        
        clique_weight = np.zeros(len(macs), dtype=float)
        for ind in range(len(macs)):
            mac = list(macs[ind])
            if len(mac) >= min_clique_size:
                sub_graph = SCG_np[np.ix_(mac, mac)]
                clique_weight[ind] = np.sum(sub_graph)
                
        if len(macs) == 0:
            return np.eye(4)

        best_ind = np.argmax(clique_weight)
        best_mac = list(macs[best_ind])
        
        src_final = src_pts[best_mac]
        dst_final = dst_pts[best_mac]
        
        # Compute centroids
        centroid_A = torch.mean(src_final, dim=0)
        centroid_B = torch.mean(dst_final, dim=0)
        
        Am = src_final - centroid_A
        Bm = dst_final - centroid_B
        
        H = Am.T @ Bm
        U, S, Vt = torch.svd(H)
        R = Vt.T @ U.T
        
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        t = centroid_B - R @ centroid_A
        
        trans = np.eye(4)
        trans[:3, :3] = R.cpu().numpy()
        trans[:3, 3] = t.cpu().numpy()
        
        return trans

    def _method_turboreg(self, source: np.ndarray, target: np.ndarray, grid_id: int = -1) -> Dict:
        """TurboReg GPU robust registration"""
        start_time = time.time()
        
        if torch is None:
            self.logger.error("TurboReg requires Torch")
            return self._get_failed_result(start_time, "torch not installed")
            
        try:
            try:
                import turboreg_gpu
            except ImportError:
                self.logger.error("TurboReg GPU module not found")
                return self._get_failed_result(start_time, "turboreg_gpu not installed")

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                self.logger.warning("TurboReg GPU requires CUDA")

            # Preprocessing
            voxel_size = 0.5
            source_pcd = self._create_point_cloud(source)
            target_pcd = self._create_point_cloud(target)
            
            source_down = source_pcd.voxel_down_sample(voxel_size)
            target_down = target_pcd.voxel_down_sample(voxel_size)
            
            radius_normal = voxel_size * 2
            source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            
            src_features = None
            dst_features = None

            if self.feature_type == 'fpfh':
                radius_feature = voxel_size * 2
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                src_features = np.array(source_fpfh.data).T
                dst_features = np.array(target_fpfh.data).T
            elif self.feature_type == 'normal':
                src_features = np.asarray(source_down.normals)
                dst_features = np.asarray(target_down.normals)
            else:
                self.logger.warning(f"Unknown feature type: {self.feature_type}, defaulting to FPFH")
                radius_feature = voxel_size * 2
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                src_features = np.array(source_fpfh.data).T
                dst_features = np.array(target_fpfh.data).T

             # Establish correspondences (mutual matching)

            nbrs_dst = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst_features)
            distances_s2d, indices_s2d = nbrs_dst.kneighbors(src_features)
            
            nbrs_src = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(src_features)
            distances_d2s, indices_d2s = nbrs_src.kneighbors(dst_features)
            
            mutual_matches = []
            for i in range(len(src_features)):
                j = indices_s2d[i, 0]
                if indices_d2s[j, 0] == i:
                     mutual_matches.append([i, j])
            
            corrs = np.array(mutual_matches)
            
            max_corr = 6000
            if len(corrs) > max_corr:
                distances = distances_s2d[corrs[:, 0]].flatten()
                best_indices = np.argsort(distances, kind='stable')[:max_corr]
                corrs = corrs[best_indices]
            
            src_corr_points = np.array(source_down.points)[corrs[:, 0]]
            dst_corr_points = np.array(target_down.points)[corrs[:, 1]]
            
            # TurboReg solver
            reger = turboreg_gpu.TurboRegGPU(
                max_corr, 0.1, 2500, 0.15, 0.4, "IN"
            )
            
            src_points_torch = torch.tensor(src_corr_points, device=device, dtype=torch.float32)
            dst_points_torch = torch.tensor(dst_corr_points, device=device, dtype=torch.float32)
            
            trans_torch = reger.run_reg(src_points_torch, dst_points_torch)
            transformation = trans_torch.cpu().numpy()

            # ICP refinement
            reg_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, 0.5, transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            # Result processing
            source_points = np.asarray(source_pcd.points)
            target_points = np.asarray(target_pcd.points)
            correspondences = self._find_correspondences(source_points, target_points, reg_result.transformation)
            
            mae = 0.0
            if len(correspondences) > 0:
                source_corr = source_points[correspondences[:, 0]]
                target_corr = target_points[correspondences[:, 1]]
                source_homogeneous = np.column_stack([source_corr, np.ones(len(source_corr))])
                transformed_source = (reg_result.transformation @ source_homogeneous.T).T[:, :3]
                distances = np.linalg.norm(transformed_source - target_corr, axis=1)
                mae = np.mean(distances)
                self.store_correspondence_for_global_analysis(source_points, target_points, correspondences, grid_id)
            
            if len(correspondences) > 0:
                self.save_correspondences(source_points, target_points, correspondences,
                                        reg_result.transformation, grid_id, reg_result.fitness)

            return {
                'transformation': reg_result.transformation,
                'fitness': reg_result.fitness,
                'rmse': reg_result.inlier_rmse,
                'mae': mae,
                'runtime': time.time() - start_time,
                'n_corr': len(correspondences),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"TurboReg registration failed: {e}")
            return self._get_failed_result(start_time, str(e))

    def _get_failed_result(self, start_time: float, error_msg: str) -> Dict:
        """Return a failed registration result structure"""
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'rmse': float('inf'),
            'mae': float('inf'),
            'runtime': time.time() - start_time,
            'n_corr': 0,
            'success': False,
            'error': error_msg
        }

    def _create_point_cloud(self, points: np.ndarray):
        """Create an Open3D point cloud object"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def compute_global_registration_metrics(self, fused_transformation: np.ndarray) -> Dict[str, float]:
        """Compute global registration metrics using fused transformation on filtered correspondences"""
        self.logger.info("Computing global registration metrics (based on filtered correspondences)...")

        if len(self.all_correspondences) == 0:
            self.logger.warning("No filtered correspondences available")
            return {'global_rmse': 0.0, 'global_mae': 0.0, 'global_fitness': 0.0, 'total_correspondences': 0}

        # Merge correspondences from all filtered regions
        all_source_points = []
        all_target_points = []

        for grid_id, corr_data in self.all_correspondences.items():
            all_source_points.append(corr_data['source_points'])
            all_target_points.append(corr_data['target_points'])

        global_source_points = np.vstack(all_source_points)
        global_target_points = np.vstack(all_target_points)

        total_correspondences = len(global_source_points)
        self.logger.info(f"Filtered correspondences: {total_correspondences} pairs from {len(self.all_correspondences)} valid regions")

        # Transform all source points using fused transformation
        source_homogeneous = np.column_stack([global_source_points, np.ones(len(global_source_points))])
        transformed_global_source = (fused_transformation @ source_homogeneous.T).T[:, :3]

        # Compute post-registration distance errors
        distances = np.linalg.norm(transformed_global_source - global_target_points, axis=1)

        # Percentile-based outlier removal
        percentile = 100
        threshold = np.percentile(distances, percentile)
        valid_mask = distances <= threshold
        distances_filtered = distances[valid_mask]

        self.logger.info(f"Percentile removal: {np.sum(~valid_mask)}/{len(distances)} worst points removed")

        # Compute global metrics
        global_rmse = np.sqrt(np.mean(distances_filtered ** 2))
        global_mae = np.mean(distances_filtered)

        # Compute global fitness
        fitness_threshold = 0.8
        inlier_count = np.sum(distances < fitness_threshold)
        global_fitness = inlier_count / len(distances) if len(distances) > 0 else 0.0

        self.logger.info(f"Global RMSE: {global_rmse:.4f} m (based on {len(distances_filtered)} valid correspondences)")
        self.logger.info(f"Global MAE: {global_mae:.4f} m")
        self.logger.info(f"Global fitness: {global_fitness:.4f} ({inlier_count}/{len(distances)} points within {fitness_threshold} m)")

        # Save global correspondence error analysis files
        self.save_global_correspondence_errors(global_source_points, global_target_points,
                               transformed_global_source, distances, fused_transformation)

        return {
            'global_rmse': global_rmse,
            'global_mae': global_mae,
            'global_fitness': global_fitness,
            'total_correspondences': total_correspondences,
            'valid_correspondences': len(distances_filtered)
        }

    def save_global_correspondence_errors(self, source_points: np.ndarray, target_points: np.ndarray,
                                          transformed_source_points: np.ndarray, distances: np.ndarray,
                                          transformation: np.ndarray):
        """Save global correspondence error analysis files"""
        if self.correspondence_save_dir is None:
            self.logger.warning("Correspondence save directory not set, skipping global error analysis saving")
            return

        # Error statistics
        rmse = np.sqrt(np.mean(distances ** 2))
        mae = np.mean(distances)
        max_error = np.max(distances)
        min_error = np.min(distances)
        std_error = np.std(distances)

        # Save detailed error file
        error_file = os.path.join(self.correspondence_save_dir,
                      f"global_correspondence_errors_{self.current_file_pair}_filtered.txt")

        try:
            with open(error_file, 'w') as f:
                f.write("# Global correspondence error analysis (based on filtered valid regions)\n")
                f.write(f"# File pair: {self.current_file_pair}\n")
                f.write(f"# Number of valid regions after filtering: {len(self.all_correspondences)}\n")
                f.write(f"# Total correspondences: {len(distances)}\n")
                f.write(f"# Global RMSE: {rmse:.6f} m\n")
                f.write(f"# Global MAE: {mae:.6f} m\n")
                f.write(f"# Max error: {max_error:.6f} m\n")
                f.write(f"# Min error: {min_error:.6f} m\n")
                f.write(f"# Std error: {std_error:.6f} m\n")
                f.write("# Format: srcX srcY srcZ transformedSrcX transformedSrcY transformedSrcZ tgtX tgtY tgtZ errorDistance\n")

                for i in range(len(distances)):
                    f.write(f"{source_points[i, 0]:.6f} {source_points[i, 1]:.6f} {source_points[i, 2]:.6f} ")
                    f.write(
                        f"{transformed_source_points[i, 0]:.6f} {transformed_source_points[i, 1]:.6f} {transformed_source_points[i, 2]:.6f} ")
                    f.write(f"{target_points[i, 0]:.6f} {target_points[i, 1]:.6f} {target_points[i, 2]:.6f} ")
                    f.write(f"{distances[i]:.6f}\n")

            # Save summary statistics file
            summary_file = os.path.join(self.correspondence_save_dir,
                                        f"global_error_summary_{self.current_file_pair}_filtered.txt")

            with open(summary_file, 'w') as f:
                f.write("# Global registration error summary (based on filtered valid regions)\n")
                f.write(f"File pair: {self.current_file_pair}\n")
                f.write(f"Number of valid regions after filtering: {len(self.all_correspondences)}\n")
                f.write(f"Total correspondences: {len(distances)}\n")
                f.write(f"Global RMSE: {rmse:.6f} m\n")
                f.write(f"Global MAE: {mae:.6f} m\n")
                f.write(f"Max error: {max_error:.6f} m\n")
                f.write(f"Min error: {min_error:.6f} m\n")
                f.write(f"Std error: {std_error:.6f} m\n")

                # Save valid regions list
                f.write("\nValid regions list:\n")
                for grid_id, corr_data in self.all_correspondences.items():
                    f.write(f"Grid {grid_id}: {corr_data['n_correspondences']} correspondences\n")

                # Error distribution statistics
                percentiles = [50, 75, 90, 95, 99]
                f.write("\nError percentiles:\n")
                for p in percentiles:
                    percentile_value = np.percentile(distances, p)
                    f.write(f"{p}%: {percentile_value:.6f} m\n")

                # Distance threshold statistics
                thresholds = [0.5, 1.0, 2.0, 5.0]
                f.write("\nCounts within distance thresholds:\n")
                for thresh in thresholds:
                    count = np.sum(distances < thresh)
                    percentage = count / len(distances) * 100
                    f.write(f"< {thresh} m: {count} ({percentage:.1f}%)\n")

            self.logger.debug(f"Global error analysis saved (based on filtered valid regions):")
            self.logger.debug(f"Detail file: {os.path.basename(error_file)}")
            self.logger.debug(f"Summary: {os.path.basename(summary_file)}")
            
        except Exception as e:
            self.logger.error(f"Failed to save global error analysis: {e}")
