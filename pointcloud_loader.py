#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point cloud loading module
Responsible for reading LAS files and loading overlap region point clouds
"""

import numpy as np
import laspy
import os
import time
from typing import Dict, Any
from data_structures import PointCloudData
from logging_utils import get_logger


class PointCloudLoader:
    """Point cloud loader"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def load_las_header_info(self, filepath: str) -> Dict[str, Any]:
        """Quickly read LAS file header information"""
        self.logger.info(f"Reading LAS file: {os.path.basename(filepath)}")

        if not os.path.exists(filepath):
            self.logger.error(f"File does not exist - {filepath}")
            return None

        file_size = os.path.getsize(filepath) / (1024 ** 2)  # MB
        self.logger.debug(f"File size: {file_size:.2f} MB")

        start_time = time.time()

        try:
            with laspy.open(filepath) as las_file:
                header = las_file.header
                las_data = las_file.read()

                self.logger.debug(f"LAS version: {header.version}")
                self.logger.debug(f"Point format: {header.point_format}")
                self.logger.info(f"Point count: {header.point_count:,}")
                self.logger.debug(f"Geographic bounds: X=[{header.min[0]:.2f}, {header.max[0]:.2f}]")
                self.logger.debug(f"                 Y=[{header.min[1]:.2f}, {header.max[1]:.2f}]")
                self.logger.debug(f"                 Z=[{header.min[2]:.2f}, {header.max[2]:.2f}]")

                # Check attributes
                attributes = []
                has_intensity = hasattr(las_data, 'intensity') and las_data.intensity is not None
                has_colors = (hasattr(las_data, 'red') and hasattr(las_data, 'green') and
                              hasattr(las_data, 'blue') and las_data.red is not None)

                if has_intensity:
                    attributes.append("intensity")
                    self.logger.debug(f"Intensity range: [{np.min(las_data.intensity):.0f}, {np.max(las_data.intensity):.0f}]")

                if has_colors:
                    attributes.append("RGB colors")

                if hasattr(las_data, 'gps_time') and las_data.gps_time is not None:
                    attributes.append("GPS time")
                if hasattr(las_data, 'classification') and las_data.classification is not None:
                    attributes.append("classification")

                self.logger.debug(f"Available attributes: {', '.join(attributes) if attributes else 'coordinates only'}")

                info = {
                    'filepath': filepath,
                    'bounds': {
                        'x_min': header.min[0], 'x_max': header.max[0],
                        'y_min': header.min[1], 'y_max': header.max[1],
                        'z_min': header.min[2], 'z_max': header.max[2]
                    },
                    'n_points': header.point_count,
                    'point_format': header.point_format,
                    'has_intensity': has_intensity,
                    'has_colors': has_colors,
                    'attributes': attributes
                }

                x_range = header.max[0] - header.min[0]
                y_range = header.max[1] - header.min[1]
                info['ranges'] = {'x': x_range, 'y': y_range}
                info['area'] = x_range * y_range

                self.logger.debug(f"Coverage: {x_range:.2f} × {y_range:.2f} m")

        except Exception as e:
            self.logger.error(f"Failed to read LAS file - {e}")
            return None

        load_time = time.time() - start_time
        self.logger.debug(f"Read time: {load_time:.3f}s")

        return info

    def load_overlap_points_chunked(self, filepath: str, overlap_info: Dict) -> PointCloudData:
        """Chunked load of overlap region point cloud (includes full color and decentering)"""
        self.logger.info(f"Loading overlap region point cloud: {os.path.basename(filepath)}")
        start_time = time.time()

        bounds = overlap_info['bounds']
        center_offset = overlap_info['center_offset']

        self.logger.debug(f"Filter bounds: X=[{bounds['x_min']:.2f}, {bounds['x_max']:.2f}]")
        self.logger.debug(f"             Y=[{bounds['y_min']:.2f}, {bounds['y_max']:.2f}]")

        try:
            with laspy.open(filepath) as las_file:
                header = las_file.header
                total_file_points = header.point_count
                self.logger.debug(f"Total points: {total_file_points:,}")

                # Read point cloud data
                las_data = las_file.read()

                # Process coordinates
                x = np.array(las_data.x, dtype=np.float64)
                y = np.array(las_data.y, dtype=np.float64)
                z = np.array(las_data.z, dtype=np.float64)

                # Get intensity information
                if hasattr(las_data, 'intensity') and las_data.intensity is not None:
                    intensity = np.array(las_data.intensity, dtype=np.float32)
                else:
                    intensity = np.ones(len(x), dtype=np.float32) * 100
                    self.logger.warning("No intensity information, using default values")

                # Get color information
                colors = None
                has_colors = False
                if (hasattr(las_data, 'red') and hasattr(las_data, 'green') and
                        hasattr(las_data, 'blue') and las_data.red is not None):

                    red = np.array(las_data.red, dtype=np.float32)
                    green = np.array(las_data.green, dtype=np.float32)
                    blue = np.array(las_data.blue, dtype=np.float32)

                    # Check color range and normalize
                    max_rgb = max(np.max(red), np.max(green), np.max(blue))
                    if max_rgb > 1.0:  # 16-bit color values
                        red = red / 65535.0
                        green = green / 65535.0
                        blue = blue / 65535.0
                        self.logger.debug("Detected 16-bit RGB, normalized to [0,1]")

                    colors = np.column_stack([red, green, blue])
                    has_colors = True

                # Spatial filtering
                self.logger.debug("Starting spatial filtering...")
                mask = ((x >= bounds['x_min']) & (x <= bounds['x_max']) &
                        (y >= bounds['y_min']) & (y <= bounds['y_max']))

                overlap_count = np.sum(mask)
                self.logger.info(f"Filter results: {overlap_count:,} / {len(x):,} points in overlap region")

                if overlap_count > 0:
                    # Extract overlap points
                    overlap_x = x[mask]
                    overlap_y = y[mask]
                    overlap_z = z[mask]

                    # Decentering - subtract offset
                    centered_x = overlap_x - center_offset[0]
                    centered_y = overlap_y - center_offset[1]
                    centered_z = overlap_z - center_offset[2]

                    final_points = np.column_stack([centered_x, centered_y, centered_z])
                    final_intensity = intensity[mask]
                    final_colors = colors[mask] if has_colors else None

                    # Log decentered bounds
                    self.logger.debug(f"Decentered bounds: X=[{np.min(centered_x):.3f}, {np.max(centered_x):.3f}]")
                    self.logger.debug(f"                  Y=[{np.min(centered_y):.3f}, {np.max(centered_y):.3f}]")
                    self.logger.debug(f"                  Z=[{np.min(centered_z):.3f}, {np.max(centered_z):.3f}]")

                else:
                    final_points = np.empty((0, 3))
                    final_intensity = np.empty(0)
                    final_colors = None
                    self.logger.warning("No points in overlap region!")

        except Exception as e:
            self.logger.error(f"Point cloud loading failed - {e}")
            return PointCloudData(np.empty((0, 3)), np.empty(0))

        load_time = time.time() - start_time
        retention_rate = len(final_points) / total_file_points * 100 if total_file_points > 0 else 0

        self.logger.info(f"Load complete: {len(final_points):,} points")
        self.logger.debug(f"Retention rate: {retention_rate:.2f}%")
        self.logger.debug(f"Contains colors: {'Yes' if has_colors else 'No'}")
        self.logger.debug(f"Load time: {load_time:.2f}s")

        return PointCloudData(
            points=final_points,
            intensity=final_intensity,
            colors=final_colors,
            has_colors=has_colors
        )
