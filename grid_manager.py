#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid management module
Responsible for grid creation and point cloud assignment
"""

import numpy as np
import concurrent.futures
from typing import Dict, Tuple, Any
from data_structures import PointCloudData
from logging_utils import get_logger


class GridManager:
    """Grid manager"""
    
    def __init__(self, grid_rows: int, grid_cols: int, n_threads: int = None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_threads = n_threads
        self.logger = get_logger()
    
    def compute_overlap_region(self, info1: Dict, info2: Dict, buffer_delta: float) -> Dict:
        """Compute overlap region and determine decentering offset"""
        self.logger.info("Computing overlap region...")

        b1, b2 = info1['bounds'], info2['bounds']

        self.logger.debug(f"Strip1 bounds: X=[{b1['x_min']:.2f}, {b1['x_max']:.2f}], Y=[{b1['y_min']:.2f}, {b1['y_max']:.2f}]")
        self.logger.debug(f"Strip2 bounds: X=[{b2['x_min']:.2f}, {b2['x_max']:.2f}], Y=[{b2['y_min']:.2f}, {b2['y_max']:.2f}]")

        # Compute original overlap region
        orig_x_min = max(b1['x_min'], b2['x_min'])
        orig_x_max = min(b1['x_max'], b2['x_max'])
        orig_y_min = max(b1['y_min'], b2['y_min'])
        orig_y_max = min(b1['y_max'], b2['y_max'])

        self.logger.debug(f"Original overlap: X=[{orig_x_min:.2f}, {orig_x_max:.2f}], Y=[{orig_y_min:.2f}, {orig_y_max:.2f}]")

        # Check overlap
        if orig_x_min >= orig_x_max or orig_y_min >= orig_y_max:
            self.logger.error("No overlap between strips!")
            return None

        orig_width = orig_x_max - orig_x_min
        orig_height = orig_y_max - orig_y_min
        orig_area = orig_width * orig_height

        self.logger.info(f"Original overlap region: {orig_width:.2f} × {orig_height:.2f} m, area: {orig_area:.2f} m²")

        # Compute overlap region center as decentering offset
        overlap_center_x = (orig_x_min + orig_x_max) / 2.0
        overlap_center_y = (orig_y_min + orig_y_max) / 2.0
        overlap_center_z = (min(b1['z_min'], b2['z_min']) + max(b1['z_max'], b2['z_max'])) / 2.0

        # Decentering offset
        offset_center = np.array([overlap_center_x, overlap_center_y, overlap_center_z], dtype=np.float64)

        self.logger.debug(f"Decentering center: [{overlap_center_x:.3f}, {overlap_center_y:.3f}, {overlap_center_z:.3f}]")

        # Add buffer
        x_min = orig_x_min - buffer_delta
        x_max = orig_x_max + buffer_delta
        y_min = orig_y_min - buffer_delta
        y_max = orig_y_max + buffer_delta

        overlap_info = {
            'bounds': {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max},
            'width': x_max - x_min,
            'height': y_max - y_min,
            'area': (x_max - x_min) * (y_max - y_min),
            'buffer': buffer_delta,
            'original_area': orig_area,
            'center_offset': offset_center.copy(),
            'original_bounds': {
                'x_min': orig_x_min, 'x_max': orig_x_max,
                'y_min': orig_y_min, 'y_max': orig_y_max
            }
        }

        self.logger.debug(f"Buffered overlap region: {overlap_info['width']:.2f} × {overlap_info['height']:.2f} m")

        return overlap_info

    def create_unified_grid(self, overlap_info: Dict) -> Dict[str, Any]:
        """Create unified grid system (based on overlap region bounds)"""
        self.logger.info(f"Creating unified {self.grid_rows}×{self.grid_cols} grid...")

        # Get overlap region bounds (with buffer)
        bounds = overlap_info['bounds']
        center_offset = overlap_info['center_offset']

        # Compute decentered overlap bounds
        x_min_centered = bounds['x_min'] - center_offset[0]
        x_max_centered = bounds['x_max'] - center_offset[0]
        y_min_centered = bounds['y_min'] - center_offset[1]
        y_max_centered = bounds['y_max'] - center_offset[1]

        width = x_max_centered - x_min_centered
        height = y_max_centered - y_min_centered

        cell_width = width / self.grid_cols
        cell_height = height / self.grid_rows

        x_edges = np.linspace(x_min_centered, x_max_centered, self.grid_cols + 1)
        y_edges = np.linspace(y_min_centered, y_max_centered, self.grid_rows + 1)

        grid_info = {
            'rows': self.grid_rows, 'cols': self.grid_cols,
            'cell_width': cell_width, 'cell_height': cell_height,
            'cell_area': cell_width * cell_height,
            'x_edges': x_edges, 'y_edges': y_edges,
            'total_cells': self.grid_rows * self.grid_cols,
            'bounds': {
                'x_min': x_min_centered, 'x_max': x_max_centered,
                'y_min': y_min_centered, 'y_max': y_max_centered
            },
            'original_bounds': bounds,
            'center_offset': center_offset.copy()
        }

        self.logger.info(f"Grid creation complete: {grid_info['total_cells']} cells")
        self.logger.debug(f"Original overlap bounds: X=[{bounds['x_min']:.2f}, {bounds['x_max']:.2f}], Y=[{bounds['y_min']:.2f}, {bounds['y_max']:.2f}]")
        self.logger.debug(f"Decentered overlap bounds: X=[{x_min_centered:.3f}, {x_max_centered:.3f}], Y=[{y_min_centered:.3f}, {y_max_centered:.3f}]")
        self.logger.debug(f"Cell size: {cell_width:.3f} × {cell_height:.3f} m")
        self.logger.debug(f"Cell area: {grid_info['cell_area']:.2f} m²")

        return grid_info

    def assign_points_to_unified_grid(self, point_data: PointCloudData, grid_info: Dict, strip_name: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Assign point cloud to unified grid"""
        self.logger.info(f"Assigning {len(point_data.points):,} points from {strip_name} to unified grid...")

        points = point_data.points

        bounds = grid_info['bounds']
        cell_width = grid_info['cell_width']
        cell_height = grid_info['cell_height']

        # Use direct index calculation for uniform grid, faster than digitize
        use_parallel = self.n_threads is not None and self.n_threads > 1 and len(points) > 0
        if use_parallel:
            x_indices = np.empty(len(points), dtype=np.int64)
            y_indices = np.empty(len(points), dtype=np.int64)

            def _compute_chunk(start_idx: int, end_idx: int):
                x_chunk = np.floor((points[start_idx:end_idx, 0] - bounds['x_min']) / cell_width).astype(np.int64)
                y_chunk = np.floor((points[start_idx:end_idx, 1] - bounds['y_min']) / cell_height).astype(np.int64)
                return start_idx, x_chunk, y_chunk

            chunk_count = min(self.n_threads, max(1, len(points) // 50000))
            if chunk_count <= 1:
                x_indices = np.floor((points[:, 0] - bounds['x_min']) / cell_width).astype(np.int64)
                y_indices = np.floor((points[:, 1] - bounds['y_min']) / cell_height).astype(np.int64)
            else:
                chunk_size = int(np.ceil(len(points) / chunk_count))
                ranges = [(i, min(i + chunk_size, len(points))) for i in range(0, len(points), chunk_size)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                    futures = [executor.submit(_compute_chunk, s, e) for s, e in ranges]
                    for future in concurrent.futures.as_completed(futures):
                        start_idx, x_chunk, y_chunk = future.result()
                        end_idx = start_idx + len(x_chunk)
                        x_indices[start_idx:end_idx] = x_chunk
                        y_indices[start_idx:end_idx] = y_chunk
        else:
            x_indices = np.floor((points[:, 0] - bounds['x_min']) / cell_width).astype(np.int64)
            y_indices = np.floor((points[:, 1] - bounds['y_min']) / cell_height).astype(np.int64)

        # Ensure all indices are within valid range
        x_indices = np.clip(x_indices, 0, self.grid_cols - 1)
        y_indices = np.clip(y_indices, 0, self.grid_rows - 1)

        grid_ids = y_indices * self.grid_cols + x_indices
        unique_ids, counts = np.unique(grid_ids, return_counts=True)

        self.logger.info(f"Assignment complete: {len(unique_ids)} valid grids")
        self.logger.debug(f"Average points/grid: {np.mean(counts):.1f}")
        self.logger.debug(f"Point count range: [{np.min(counts)}, {np.max(counts)}]")
        self.logger.debug(f"Empty grid count: {grid_info['total_cells'] - len(unique_ids)}")

        return grid_ids, unique_ids, counts
