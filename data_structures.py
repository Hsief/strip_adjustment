#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data structures module
Defines point cloud data structures
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PointCloudData:
    """Point cloud data structure"""
    points: np.ndarray
    intensity: np.ndarray
    colors: Optional[np.ndarray] = None  # RGB colors [N, 3]
    has_colors: bool = False

    def __post_init__(self):
        if self.colors is not None and len(self.colors) > 0:
            self.has_colors = True
