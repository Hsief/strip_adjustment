#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import warnings
import logging
import time
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Import modular components
from logging_utils import setup_logging, get_logger
from airborne_matching import AirborneStripMatching
import airborne_matching_analysis  

def append_log(log_path, values_7):
    """
    values_7: [time_s, memory_mb, rmse, fitness, mae, plane_angle_deg, position_spacing_m]
    """
    need_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write("time(s)\tmemory(MB)\tRMSE(m)\tfitness\tMAE(m)\tPlane Accuracy (Normal Angle)\tPosition Accuracy (Point Spacing)\n")
        f.write(
            f"{values_7[0]:.3f}\t"
            f"{values_7[1]:.1f}\t"
            f"{values_7[2]:.4f}\t"
            f"{values_7[3]:.4f}\t"
            f"{values_7[4]:.4f}\t"
            f"{values_7[5]:.4f}\t"
            f"{values_7[6]:.4f}\n"
        )

def clear_memory():
    # 1) First perform garbage collection (CPU side Python objects)
    gc.collect()


def main():
    data_dir = r"./data/hills"
    # data_dir=""
    source_file = os.path.join(data_dir, f"wuding ({2}).las")
    target_file = os.path.join(data_dir, f"wuding ({3}).las")
    reference_file = os.path.join(data_dir, "target_positions.txt")
    
    # Set result output directory
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "result" / f"Ours"
    log_dir = output_dir / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"airborne_matching_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(
        log_level=logging.INFO,
        log_file=log_file,
        enable_console=True
    )
    
    logger.info("Starting strip matching analysis - Log system started")
    # output_dir = r"E:\Paper\opendata"
    os.makedirs(output_dir, exist_ok=True)
    #THE HILLS DATA'S HYPRPARAMETERS
    matcher = AirborneStripMatching(
        grid_rows=16, grid_cols=8,
        buffer_delta=10.0,
        z_bins=20,
        intensity_bins=30,
        alpha=1,
        height_percentiles=(0.01, 99.99),
        intensity_percentiles=(0, 99.5),
        entropy_weight_a=0.4,
        entropy_weight_b=0.4,
        entropy_weight_c=0.2,
        min_composite_score_threshold=0.05,
        min_valid_regions=2,
        enable_fallback_to_identity=True,
        max_region=8,
        store_correspondences=True,
        n_threads=2,
        voxel_size=0.2
    )
    #THE court DATA'S HYPRPARAMETERS
    # matcher = AirborneStripMatching(
    #     grid_rows=4, grid_cols=1,
    #     buffer_delta=10.0,
    #     z_bins=20,
    #     intensity_bins=30,
    #     alpha=1,
    #     registration_method=method_name,
    #     height_percentiles=(0.01, 99.99),
    #     intensity_percentiles=(0, 99.5),
    #     entropy_weight_a=0.4,
    #     entropy_weight_b=0.4,
    #     entropy_weight_c=0.2,
    #     min_composite_score_threshold=0.05,
    #     min_valid_regions=1,
    #     enable_fallback_to_identity=True,
    #     max_region=8,
    #     store_correspondences=True,
    #     n_threads=1,
    #     voxel_size=0.3
    # )

    logger.info("Starting strip matching analysis...")
    results = matcher.run_complete_analysis(
        source_file, target_file,
        apply_transform=True,
        output_dir=output_dir,
        reference_file=reference_file
    )

    if results:
        fusion_result = results['fusion_result']
        logger.info("=" * 60)
        logger.info("Final Result Summary")
        logger.info("=" * 60)
        logger.info(f"Number of fusion regions: {fusion_result['n_regions']}")
        logger.info(f"Registration confidence: {fusion_result['confidence']:.4f}")
        logger.info(f"Fusion method: {fusion_result['fusion_method']}")

        valid_grids = fusion_result.get('valid_grid_ids', [])
        if valid_grids:
            logger.info(f"List of valid regions: {sorted(valid_grids)}")

        is_identity = fusion_result.get('is_identity', False)
        logger.info(f"Transformation type: {'Identity Matrix (No Transformation)' if is_identity else 'Cauchy Robust ICP Transformation based on RMSE weights'}")

        if not is_identity:
            if 'global_rmse' in fusion_result:
                logger.info("Global registration accuracy (based on filtered valid regions):")
                logger.info(f"Global RMSE: {fusion_result['global_rmse']:.4f} m")
                logger.info(f"Global MAE: {fusion_result['global_mae']:.4f} m")
                logger.info(f"Global fitness: {fusion_result['global_fitness']:.4f}")
                logger.info(f"Total correspondences: {fusion_result['total_correspondences']:,}")
                logger.info(f"Valid correspondences: {fusion_result.get('valid_correspondences', 0):,}")

        logger.info("Output files:")
        logger.info(f"Correspondence files: {output_dir}/correspondences/")
        logger.info(f"Global error analysis: {output_dir}/correspondences/global_*_filtered.txt")
        logger.info(f"Memory usage record: {output_dir}/correspondences/memory_usage_*.txt")
        logger.info(f"Log file: {log_file}")
    else:
        logger.error("Analysis failed!")

def _print_results(results, output_dir, log_file):
    """Helper function to print results"""
    logger = get_logger()
    fusion_result = results['fusion_result']
    logger.info("=" * 60)
    logger.info("Final Result Summary")
    logger.info("=" * 60)
    logger.info(f"Number of fusion regions: {fusion_result['n_regions']}")
    logger.info(f"Registration confidence: {fusion_result['confidence']:.4f}")
    logger.info(f"Fusion method: {fusion_result['fusion_method']}")

    valid_grids = fusion_result.get('valid_grid_ids', [])
    if valid_grids:
        logger.info(f"List of valid regions: {sorted(valid_grids)}")

    is_identity = fusion_result.get('is_identity', False)
    logger.info(f"Transformation type: {'Identity Matrix (No Transformation)' if is_identity else 'Cauchy Robust ICP Transformation based on RMSE weights'}")

    if not is_identity and 'global_rmse' in fusion_result:
        logger.info("Global registration accuracy (based on filtered valid regions):")
        logger.info(f"Global RMSE: {fusion_result['global_rmse']:.4f} m")
        logger.info(f"Global MAE: {fusion_result['global_mae']:.4f} m")
        logger.info(f"Global fitness: {fusion_result['global_fitness']:.4f}")
        logger.info(f"Total correspondences: {fusion_result['total_correspondences']:,}")
        logger.info(f"Valid correspondences: {fusion_result.get('valid_correspondences', 0):,}")

    if 'target_accuracy' in results:
        target_acc = results['target_accuracy']
        logger.info("Target accuracy evaluation:")
        logger.info(f"Number of common targets: {target_acc['target_count']}")
        logger.info(f"Position error (point spacing): {target_acc['position_3d_mean']:.4f} m")
        logger.info(f"Position error (reference point): {target_acc['position_ref_mean']:.4f} m")
        logger.info(f"Plane error (normal vector): {target_acc['plane_vector_angle_mean']:.4f} °")
        logger.info(f"Plane error (Z-axis angle): {target_acc['plane_z_angle_mean']:.4f} °")

    logger.info("Output files:")
    logger.info(f"Correspondence files: {output_dir}/correspondences/")
    logger.info(f"Global error analysis: {output_dir}/correspondences/global_*_filtered.txt")
    logger.info(f"Memory usage record: {output_dir}/correspondences/memory_usage_*.txt")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main() 
