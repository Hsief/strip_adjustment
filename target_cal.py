"""
Target point calibration - enhanced full version
Extract target points, perform clustering and planarity analysis
Includes comprehensive plotting and data saving utilities
"""

import numpy as np
import pandas as pd
import laspy
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, curve_fit
from scipy import stats
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import csv
import re
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def log_info(message):
    """Info log"""
    print(f"[INFO] {message}")

def log_debug(message):
    """Debug log"""
    print(f"[DEBUG] {message}")

def log_warning(message):
    """Warning log"""
    print(f"[WARNING] {message}")

def log_success(message):
    """Success log"""
    print(f"[SUCCESS] {message}")

def create_output_directory(base_path, prefix="analysis"):
    """Create output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_path / f"{prefix}_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    return output_dir

def save_plot_and_data(fig, data, output_dir, plot_name, data_filename=None):
    """Save plot and corresponding data"""
    # Save plot
    if fig is not None:
        plot_path = output_dir / f"{plot_name}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        log_success(f"Plot saved: {plot_path}")
    
    # Save data
    if data is not None:
        if data_filename is None:
            data_filename = f"{plot_name}_data.csv"
        data_path = output_dir / data_filename
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(data_path, index=False, encoding='utf-8-sig')
        elif isinstance(data, dict):
            pd.DataFrame([data]).to_csv(data_path, index=False, encoding='utf-8-sig')
        elif isinstance(data, (list, tuple)):
            pd.DataFrame(data).to_csv(data_path, index=False, encoding='utf-8-sig')
        else:
            # Try converting to DataFrame
            try:
                pd.DataFrame(data).to_csv(data_path, index=False, encoding='utf-8-sig')
            except:
                # If unable to convert, save as plain text
                with open(data_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(str(data))
        
        log_success(f"Data saved: {data_path}")

def filter_by_color_similarity_with_mad(df, intensity_threshold=1300,
                                        spatial_radius=0.5,
                                        mad_multiplier=1,
                                        min_neighbors=5):
    """Filter color-similar points using MAD and spatial distance"""
    log_info("Starting MAD-based color similarity filtering...")
    log_debug(f"Filter params - intensity_threshold: {intensity_threshold}, spatial_radius: {spatial_radius}m, mad_multiplier: {mad_multiplier}")

    def calculate_mad(values, axis=None):
        """Calculate median absolute deviation (MAD)"""
        median = np.median(values, axis=axis, keepdims=True)
        mad = np.median(np.abs(values - median), axis=axis)
        return mad

    # Select high-intensity points as seeds first
    high_intensity_mask = df['intensity'] > intensity_threshold
    if high_intensity_mask.sum() == 0:
        log_warning("No seed points met the intensity threshold")
        return df.iloc[0:0]

    log_info(f"Found {high_intensity_mask.sum():,} high-intensity seed points")

    if mad_multiplier == 0:
        log_info(f"MAD multiplier is 0, returning {high_intensity_mask.sum():,} seed points")
        return df[high_intensity_mask]

    # Get coordinates and color data for all points
    xyz_columns = ['x', 'y', 'z']
    rgb_columns = ['r', 'g', 'b']
    all_coords = df[xyz_columns].values
    all_colors = df[rgb_columns].values

    high_intensity_coords = df[high_intensity_mask][xyz_columns].values
    high_intensity_colors = df[high_intensity_mask][rgb_columns].values

    # Use KDTree for fast spatial neighborhood queries
    log_info("Building KDTree index for spatial neighborhood search...")
    nbrs = NearestNeighbors(radius=spatial_radius, algorithm='kd_tree').fit(all_coords)

    selected_indices = set()
    seed_original_indices = df[high_intensity_mask].index.tolist()
    selected_indices.update(seed_original_indices)

    log_info(f"Processing {len(high_intensity_coords):,} high-intensity seed points...")

    with tqdm(total=len(high_intensity_coords), desc="MAD color filtering progress") as pbar:
        for i, (seed_coord, seed_color) in enumerate(zip(high_intensity_coords, high_intensity_colors)):
            try:
                neighbor_indices = nbrs.radius_neighbors([seed_coord], return_distance=False)[0]

                if len(neighbor_indices) < min_neighbors:
                    pbar.update(1)
                    continue

                valid_neighbor_indices = neighbor_indices[neighbor_indices < len(all_colors)]
                
                if len(valid_neighbor_indices) < min_neighbors:
                    pbar.update(1)
                    continue

                neighbor_colors = all_colors[valid_neighbor_indices]

                rgb_medians = np.median(neighbor_colors, axis=0)
                rgb_mads = calculate_mad(neighbor_colors, axis=0)
                rgb_mads = np.maximum(rgb_mads, 0.1)

                deviations = np.abs(neighbor_colors - rgb_medians)
                within_threshold = deviations <= (rgb_mads * mad_multiplier)
                valid_neighbors_mask = np.all(within_threshold, axis=1)

                mad_valid_indices = valid_neighbor_indices[valid_neighbors_mask]
                original_indices = df.iloc[mad_valid_indices].index.tolist()
                selected_indices.update(original_indices)

            except Exception as e:
                log_debug(f"Error processing seed {i}: {str(e)}")
                
            pbar.update(1)

    log_success(f"MAD filtering complete, selected {len(selected_indices):,} points")
    return df.loc[list(selected_indices)]

def las_to_df(lasdata, method='estimate'):
    """Convert LAS data to a pandas DataFrame"""
    xyz = np.vstack([lasdata.x, lasdata.y, lasdata.z]).transpose()
    intensity = lasdata.intensity
    
    if hasattr(lasdata, 'red') and hasattr(lasdata, 'green') and hasattr(lasdata, 'blue'):
        red = lasdata.red
        green = lasdata.green
        blue = lasdata.blue
    else:
        red = np.zeros_like(intensity)
        green = np.zeros_like(intensity)
        blue = np.zeros_like(intensity)
    
    df = pd.DataFrame({
        'x': xyz[:, 0],
        'y': xyz[:, 1],
        'z': xyz[:, 2],
        'intensity': intensity,
        'r': red,
        'g': green,
        'b': blue
    })
    
    return df

def find_control_point_with_reference(las_files, reference_centers, buffer_distance=10.0,
                                     intensity_threshold=1300, spatial_radius=0.5, 
                                     mad_multiplier=0, z_max=None):
    """Find target points in LAS files based on reference centers and buffer"""
    currentpoint = []

    x_min, x_max = np.min(reference_centers[:, 0]) - buffer_distance, np.max(reference_centers[:, 0]) + buffer_distance
    y_min, y_max = np.min(reference_centers[:, 1]) - buffer_distance, np.max(reference_centers[:, 1]) + buffer_distance

    log_info(f"Reference area bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
    log_info('Searching target points in strips...')

    for las_file in tqdm(las_files, desc="Processing LAS files"):
        try:
            log_info(f"Reading file: {las_file.name}")
            lasdata = laspy.read(las_file)
            df = las_to_df(lasdata)
            
            log_info(f"Original point cloud count: {len(df)}")
            
            ori_df = df[(df['x'] >= x_min) & (df['x'] <= x_max) & 
                       (df['y'] >= y_min) & (df['y'] <= y_max)].copy()
            
            log_info(f"Point count after region clipping: {len(ori_df)}")
            
            if len(ori_df) == 0:
                log_warning(f"File {las_file.name} has no points within reference area")
                continue
            
            ori_df = ori_df.reset_index(drop=True)
            
            point = filter_by_color_similarity_with_mad(
                ori_df,
                intensity_threshold=intensity_threshold,
                spatial_radius=spatial_radius,
                mad_multiplier=mad_multiplier
            )
            
            log_info(f'File {las_file.name} filter result: {len(point)} points')
            
            if len(point) == 0:
                continue

            try:
                numbers = re.findall(r'\d+', las_file.stem)
                if numbers:
                    strip_id = int(numbers[0])
                else:
                    strip_id = 0
            except:
                strip_id = 0
                
            point = point.copy()
            point['strip_id'] = strip_id
            currentpoint.append(point)
            
        except Exception as e:
            log_warning(f"Error processing file {las_file.name}: {str(e)}")
            log_debug(f"Detailed error info: {traceback.format_exc()}")
            continue

    if len(currentpoint) == 0:
        log_warning("No target points found")
        return pd.DataFrame()

    combined_df = pd.concat(currentpoint, ignore_index=True)
    if z_max is None:
        df = combined_df
    else:
        df = combined_df[combined_df['z'] < z_max]
        if len(df) == 0:
            log_warning(
                f"No points after Z filter (z_max={z_max}), using unfiltered set for analysis"
            )
            df = combined_df
    
    log_success(f"Total found {len(df)} target points")
    return df

def cluster_data(df, n_clusters=5, features=['x', 'y', 'z']):
    """Perform KMeans clustering on the data"""
    X = df[features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    df['cluster'] = kmeans.fit_predict(X)
    cluster_means = df.groupby('cluster')[features].mean().reset_index()
    return df, cluster_means, kmeans

def merge_close_clusters(centers, threshold=10.0):
    """Merge nearby cluster centers"""
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='single')
    labels = clustering.fit_predict(centers)
    merged_centers = []

    for label in np.unique(labels):
        group = centers[labels == label]
        merged_center = group.mean(axis=0)
        merged_centers.append(merged_center)

    return np.array(merged_centers), labels

def calculate_plane_properties(points):
    """Calculate planarity, normal vector, normal angle and center of a point set"""
    if len(points) < 3:
        return None, None, None, None
    
    center = np.mean(points, axis=0)
    centered_points = points - center
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    if eigenvalues[1] > 1e-8:
        planarity = eigenvalues[2] / eigenvalues[1]
    else:
        planarity = 0.0
    
    normal_vector = eigenvectors[:, 2]
    
    if normal_vector[2] < 0:
        normal_vector = -normal_vector
    
    z_axis = np.array([0, 0, 1])
    cos_angle = np.dot(normal_vector, z_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    normal_angle = np.degrees(np.arccos(cos_angle))
    
    return planarity, normal_vector, normal_angle, center

def analyze_target_points(df, reference_centers, min_center_distance=5.0, expected_clusters=None):
    """Analyze target points: clustering, matching to reference, and plane properties"""
    if len(df) < 10:
        log_warning('Too few target points, skipping analysis')
        return None
    
    if expected_clusters is None:
        expected_clusters = len(reference_centers)
    
    log_info(f"Starting analysis of {len(df)} target points, expected clusters: {expected_clusters}")
    
    df, cluster_means, _ = cluster_data(df, n_clusters=expected_clusters)
    centers = cluster_means[['x', 'y', 'z']].to_numpy()
    
    centers_merged, merge_labels = merge_close_clusters(centers, threshold=min_center_distance)
    actual_n = centers_merged.shape[0]
    
    if actual_n == 0:
        log_warning('No valid clusters after merging')
        return None
    
    log_info(f"After clustering obtained {actual_n} valid clusters")
    
    cost_matrix = cdist(centers_merged, reference_centers)
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matched_actual = centers_merged[row_idx]
    matched_reference = reference_centers[col_idx]
    
    cluster_means['merged_label'] = merge_labels
    df = df.merge(cluster_means[['cluster', 'merged_label']], on='cluster', how='left')
    
    used_merge_labels = np.unique(merge_labels)[row_idx]
    label2ref = {label: (matched_reference[i], col_idx[i]) for i, label in enumerate(used_merge_labels)}
    
    results = []
    
    for merge_label in used_merge_labels:
        cluster_points = df[df['merged_label'] == merge_label][['x', 'y', 'z']].values
        
        if len(cluster_points) < 3:
            continue
        
        planarity, normal_vector, normal_angle, center = calculate_plane_properties(cluster_points)
        ref_point, ref_idx = label2ref[merge_label]
        
        result = {
            'cluster_id': merge_label,
            'reference_id': ref_idx,
            'point_count': len(cluster_points),
            'center_x': center[0],
            'center_y': center[1],
            'center_z': center[2],
            'ref_x': ref_point[0],
            'ref_y': ref_point[1],
            'ref_z': ref_point[2],
            'planarity': planarity,
            'normal_x': normal_vector[0],
            'normal_y': normal_vector[1],
            'normal_z': normal_vector[2],
            'normal_angle': normal_angle,
            'distance_to_ref': np.linalg.norm(center - ref_point)
            
        }
        
        results.append(result)
    
    return pd.DataFrame(results), df

def plot_intensity_distribution(df, output_dir):
    """Plot intensity distribution histogram"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Intensity Distribution Analysis', fontsize=16)
    
    # 1. Basic histogram
    axes[0, 0].hist(df['intensity'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Intensity Histogram')
    axes[0, 0].set_xlabel('Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Log-scale histogram
    axes[0, 1].hist(df['intensity'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Intensity Histogram (Log Scale)')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('Log Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cumulative distribution function
    sorted_intensity = np.sort(df['intensity'])
    cumulative = np.arange(1, len(sorted_intensity) + 1) / len(sorted_intensity)
    axes[1, 0].plot(sorted_intensity, cumulative, color='red', linewidth=2)
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    axes[1, 1].boxplot(df['intensity'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 1].set_title('Intensity Box Plot')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Prepare statistics
    intensity_stats = {
        'mean': df['intensity'].mean(),
        'median': df['intensity'].median(),
        'std': df['intensity'].std(),
        'min': df['intensity'].min(),
        'max': df['intensity'].max(),
        'q25': df['intensity'].quantile(0.25),
        'q75': df['intensity'].quantile(0.75),
        'skewness': stats.skew(df['intensity']),
        'kurtosis': stats.kurtosis(df['intensity'])
    }
    
    save_plot_and_data(fig, intensity_stats, output_dir, "01_intensity_distribution")
    plt.close(fig)

def plot_rgb_distribution(df, output_dir):
    """Plot RGB color distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RGB Color Distribution Analysis', fontsize=16)
    
    colors = ['red', 'green', 'blue']
    rgb_cols = ['r', 'g', 'b']
    
    # RGB histograms
    for i, (col, color) in enumerate(zip(rgb_cols, colors)):
        axes[0, i].hist(df[col], bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].set_title(f'{col.upper()} Channel Histogram')
        axes[0, i].set_xlabel(f'{col.upper()} Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
    
    # RGB scatter plots
    axes[1, 0].scatter(df['r'], df['g'], alpha=0.5, s=1)
    axes[1, 0].set_title('R vs G Scatter Plot')
    axes[1, 0].set_xlabel('Red')
    axes[1, 0].set_ylabel('Green')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(df['g'], df['b'], alpha=0.5, s=1)
    axes[1, 1].set_title('G vs B Scatter Plot')
    axes[1, 1].set_xlabel('Green')
    axes[1, 1].set_ylabel('Blue')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(df['r'], df['b'], alpha=0.5, s=1)
    axes[1, 2].set_title('R vs B Scatter Plot')
    axes[1, 2].set_xlabel('Red')
    axes[1, 2].set_ylabel('Blue')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Prepare RGB statistics
    rgb_stats = {}
    for col in rgb_cols:
        rgb_stats[f'{col}_mean'] = df[col].mean()
        rgb_stats[f'{col}_std'] = df[col].std()
        rgb_stats[f'{col}_min'] = df[col].min()
        rgb_stats[f'{col}_max'] = df[col].max()
    
    save_plot_and_data(fig, rgb_stats, output_dir, "02_rgb_distribution")
    plt.close(fig)

def plot_spatial_distribution(df, output_dir):
    """Plot spatial distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Spatial Distribution Analysis', fontsize=16)
    
    # XY scatter plot
    scatter = axes[0, 0].scatter(df['x'], df['y'], c=df['z'], cmap='viridis', s=1, alpha=0.6)
    axes[0, 0].set_title('XY Distribution (colored by Z)')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].axis('equal')
    plt.colorbar(scatter, ax=axes[0, 0], label='Z (m)')
    
    # XZ profile plot
    axes[0, 1].scatter(df['x'], df['z'], c=df['intensity'], cmap='plasma', s=1, alpha=0.6)
    axes[0, 1].set_title('XZ Profile (colored by Intensity)')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Z (m)')
    
    # YZ profile plot
    axes[1, 0].scatter(df['y'], df['z'], c=df['intensity'], cmap='plasma', s=1, alpha=0.6)
    axes[1, 0].set_title('YZ Profile (colored by Intensity)')
    axes[1, 0].set_xlabel('Y (m)')
    axes[1, 0].set_ylabel('Z (m)')
    
    # Elevation distribution histogram
    axes[1, 1].hist(df['z'], bins=50, alpha=0.7, color='brown', edgecolor='black')
    axes[1, 1].set_title('Elevation Distribution')
    axes[1, 1].set_xlabel('Z (m)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Spatial statistics
    spatial_stats = {
        'x_range': df['x'].max() - df['x'].min(),
        'y_range': df['y'].max() - df['y'].min(),
        'z_range': df['z'].max() - df['z'].min(),
        'x_center': df['x'].mean(),
        'y_center': df['y'].mean(),
        'z_center': df['z'].mean(),
        'point_density': len(df) / ((df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min()))
    }
    
    save_plot_and_data(fig, spatial_stats, output_dir, "03_spatial_distribution")
    plt.close(fig)

def plot_cluster_analysis(df, results_df, reference_centers, output_dir):
    """Plot cluster analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cluster Analysis', fontsize=16)
    
    # 3D cluster visualization
    ax_3d = fig.add_subplot(2, 3, 1, projection='3d')
    if 'merged_label' in df.columns:
        unique_labels = df['merged_label'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if pd.isna(label):
                continue
            cluster_data = df[df['merged_label'] == label]
            ax_3d.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'], 
                         c=[colors[i]], label=f'Cluster {int(label)}', s=10, alpha=0.6)
    
    # Show cluster centers and reference points
    if results_df is not None:
        ax_3d.scatter(results_df['center_x'], results_df['center_y'], results_df['center_z'], 
                     c='red', s=100, marker='*', label='Cluster Centers')
    
    if reference_centers is not None:
        ax_3d.scatter(reference_centers[:, 0], reference_centers[:, 1], reference_centers[:, 2], 
                     c='green', s=100, marker='^', label='Reference Points')
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Cluster Visualization')
    ax_3d.legend()
    
    if results_df is not None and len(results_df) > 0:
        # Cluster size distribution
        axes[0, 1].bar(results_df['cluster_id'], results_df['point_count'])
        axes[0, 1].set_title('Points per Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Point Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Planarity distribution
        axes[0, 2].hist(results_df['planarity'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Planarity Distribution')
        axes[0, 2].set_xlabel('Planarity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Normal angle distribution
        axes[1, 0].hist(results_df['normal_angle'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Normal Angle Distribution')
        axes[1, 0].set_xlabel('Angle (degrees)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distance error distribution
        axes[1, 1].hist(results_df['distance_to_ref'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Distance to Reference Distribution')
        axes[1, 1].set_xlabel('Distance (m)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Cluster quality metrics
        quality_metrics = {
            'avg_planarity': results_df['planarity'].mean(),
            'avg_distance_error': results_df['distance_to_ref'].mean(),
            'max_distance_error': results_df['distance_to_ref'].max(),
            'std_distance_error': results_df['distance_to_ref'].std()
        }
        
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in quality_metrics.items()])
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 2].set_title('Cluster Quality Metrics')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Cluster analysis data
    cluster_data = results_df if results_df is not None else pd.DataFrame()
    
    save_plot_and_data(fig, cluster_data, output_dir, "04_cluster_analysis")
    plt.close(fig)

def plot_accuracy_assessment(results_df, output_dir):
    """Plot accuracy assessment figures"""
    if results_df is None or len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Accuracy Assessment', fontsize=16)
    
    # Distance error vs Reference ID
    axes[0, 0].plot(results_df['reference_id'], results_df['distance_to_ref'], 'o-')
    axes[0, 0].set_title('Distance Error by Reference Point')
    axes[0, 0].set_xlabel('Reference Point ID')
    axes[0, 0].set_ylabel('Distance Error (m)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error vector plot
    axes[0, 1].quiver(results_df['ref_x'], results_df['ref_y'],
                     results_df['center_x'] - results_df['ref_x'],
                     results_df['center_y'] - results_df['ref_y'],
                     scale_units='xy', scale=1, color='red', alpha=0.7)
    axes[0, 1].scatter(results_df['ref_x'], results_df['ref_y'], c='blue', s=50, label='Reference')
    axes[0, 1].scatter(results_df['center_x'], results_df['center_y'], c='red', s=50, label='Detected')
    axes[0, 1].set_title('XY Error Vectors')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].legend()
    axes[0, 1].axis('equal')
    
    # Elevation (Z) error
    z_error = results_df['center_z'] - results_df['ref_z']
    axes[0, 2].bar(results_df['reference_id'], z_error)
    axes[0, 2].set_title('Z Height Error by Reference Point')
    axes[0, 2].set_xlabel('Reference Point ID')
    axes[0, 2].set_ylabel('Z Error (m)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Error distribution histogram
    axes[1, 0].hist(results_df['distance_to_ref'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_title('Distance Error Distribution')
    axes[1, 0].set_xlabel('Distance Error (m)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residual Q-Q plot
    from scipy import stats
    stats.probplot(results_df['distance_to_ref'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Distance Error Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy statistics
    accuracy_stats = {
        'mean_distance_error': results_df['distance_to_ref'].mean(),
        'std_distance_error': results_df['distance_to_ref'].std(),
        'max_distance_error': results_df['distance_to_ref'].max(),
        'rmse': np.sqrt(np.mean(results_df['distance_to_ref']**2)),
        'mean_z_error': z_error.mean(),
        'std_z_error': z_error.std(),
        'rmse_z': np.sqrt(np.mean(z_error**2))
    }
    
    stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in accuracy_stats.items()])
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('Accuracy Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Accuracy assessment data
    accuracy_data = results_df.copy()
    accuracy_data['z_error'] = z_error
    for key, value in accuracy_stats.items():
        accuracy_data[f'stat_{key}'] = value
    
    save_plot_and_data(fig, accuracy_data, output_dir, "05_accuracy_assessment")
    plt.close(fig)

def plot_correlation_matrix(results_df, output_dir):
    """Plot correlation matrix"""
    if results_df is None or len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Correlation Analysis', fontsize=16)
    
    # Select numeric columns
    numeric_cols = ['point_count', 'planarity', 'normal_angle', 'distance_to_ref']
    corr_data = results_df[numeric_cols]
    
    # Correlation matrix heatmap
    corr_matrix = corr_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[0])
    axes[0].set_title('Correlation Matrix')
    
    # Scatter plot matrix (selected variables)
    if len(results_df) > 1:
        axes[1].scatter(results_df['planarity'], results_df['distance_to_ref'], alpha=0.7)
        axes[1].set_xlabel('Planarity')
        axes[1].set_ylabel('Distance to Reference (m)')
        axes[1].set_title('Planarity vs Distance Error')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_plot_and_data(fig, corr_matrix, output_dir, "06_correlation_matrix")
    plt.close(fig)

def plot_quality_metrics(results_df, output_dir):
    """Plot quality metrics"""
    if results_df is None or len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quality Metrics Analysis', fontsize=16)
    
    # Planarity vs Point Count
    axes[0, 0].scatter(results_df['point_count'], results_df['planarity'], alpha=0.7)
    axes[0, 0].set_xlabel('Point Count')
    axes[0, 0].set_ylabel('Planarity')
    axes[0, 0].set_title('Planarity vs Point Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Normal angle vs Planarity
    axes[0, 1].scatter(results_df['normal_angle'], results_df['planarity'], alpha=0.7)
    axes[0, 1].set_xlabel('Normal Angle (degrees)')
    axes[0, 1].set_ylabel('Planarity')
    axes[0, 1].set_title('Planarity vs Normal Angle')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quality score calculation (composite metric)
    # Quality score = (1 - normalized distance error) * normalized planarity * (1 - normalized angle deviation/90)
    norm_distance = results_df['distance_to_ref'] / results_df['distance_to_ref'].max()
    norm_planarity = results_df['planarity'] / results_df['planarity'].max() if results_df['planarity'].max() > 0 else 0
    norm_angle = results_df['normal_angle'] / 90.0
    
    quality_score = (1 - norm_distance) * norm_planarity * (1 - norm_angle)
    results_df_copy = results_df.copy()
    results_df_copy['quality_score'] = quality_score
    
    axes[1, 0].bar(results_df['reference_id'], quality_score)
    axes[1, 0].set_title('Quality Score by Reference Point')
    axes[1, 0].set_xlabel('Reference Point ID')
    axes[1, 0].set_ylabel('Quality Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall quality assessment radar chart
    from math import pi
    
    # Compute average normalized metrics
    avg_metrics = {
        'Distance Accuracy': 1 - norm_distance.mean(),
        'Planarity': norm_planarity if isinstance(norm_planarity, (int, float)) else norm_planarity.mean(),
        'Angle Accuracy': 1 - norm_angle.mean(),
        'Point Density': results_df['point_count'].mean() / results_df['point_count'].max()
    }
    
    categories = list(avg_metrics.keys())
    values = list(avg_metrics.values())
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    values += values[:1]
    
    axes[1, 1] = fig.add_subplot(2, 2, 4, projection='polar')
    axes[1, 1].plot(angles, values, 'o-', linewidth=2)
    axes[1, 1].fill(angles, values, alpha=0.25)
    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Overall Quality Assessment', y=1.1)
    
    plt.tight_layout()
    
    save_plot_and_data(fig, results_df_copy, output_dir, "07_quality_metrics")
    plt.close(fig)

def plot_spectral_analysis(df, output_dir):
    """Plot spectral analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Spectral Analysis', fontsize=16)
    
    # Intensity vs RGB relationships
    for i, (col, color) in enumerate([('r', 'red'), ('g', 'green'), ('b', 'blue')]):
        axes[0, i].scatter(df[col], df['intensity'], alpha=0.5, s=1, c=color)
        axes[0, i].set_title(f'Intensity vs {col.upper()}')
        axes[0, i].set_xlabel(f'{col.upper()} Value')
        axes[0, i].set_ylabel('Intensity')
        axes[0, i].grid(True, alpha=0.3)
    
    # Color ratio analysis
    df_copy = df.copy()
    total_rgb = df['r'] + df['g'] + df['b']
    mask = total_rgb > 0
    
    df_copy.loc[mask, 'r_ratio'] = df.loc[mask, 'r'] / total_rgb[mask]
    df_copy.loc[mask, 'g_ratio'] = df.loc[mask, 'g'] / total_rgb[mask]
    df_copy.loc[mask, 'b_ratio'] = df.loc[mask, 'b'] / total_rgb[mask]
    
    # Color ratio histograms
    for i, (col, color) in enumerate([('r_ratio', 'red'), ('g_ratio', 'green'), ('b_ratio', 'blue')]):
        if col in df_copy.columns:
            axes[1, i].hist(df_copy[col].dropna(), bins=30, alpha=0.7, color=color, edgecolor='black')
            axes[1, i].set_title(f'{col.split("_")[0].upper()} Ratio Distribution')
            axes[1, i].set_xlabel('Color Ratio')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Spectral statistics
    spectral_stats = {
        'intensity_r_corr': df['intensity'].corr(df['r']),
        'intensity_g_corr': df['intensity'].corr(df['g']),
        'intensity_b_corr': df['intensity'].corr(df['b']),
        'r_g_corr': df['r'].corr(df['g']),
        'g_b_corr': df['g'].corr(df['b']),
        'r_b_corr': df['r'].corr(df['b'])
    }
    
    if mask.any():
        spectral_stats['mean_r_ratio'] = df_copy.loc[mask, 'r_ratio'].mean()
        spectral_stats['mean_g_ratio'] = df_copy.loc[mask, 'g_ratio'].mean()
        spectral_stats['mean_b_ratio'] = df_copy.loc[mask, 'b_ratio'].mean()
    
    save_plot_and_data(fig, spectral_stats, output_dir, "08_spectral_analysis")
    plt.close(fig)

def plot_density_maps(df, output_dir):
    """Plot point density maps"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Point Density Maps', fontsize=16)
    
    # 2D density heatmap - XY plane
    h1 = axes[0, 0].hist2d(df['x'], df['y'], bins=50, cmap='hot')
    axes[0, 0].set_title('XY Density Heatmap')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].axis('equal')
    plt.colorbar(h1[3], ax=axes[0, 0], label='Point Count')
    
    # 2D density heatmap - XZ plane
    h2 = axes[0, 1].hist2d(df['x'], df['z'], bins=50, cmap='hot')
    axes[0, 1].set_title('XZ Density Heatmap')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Z (m)')
    plt.colorbar(h2[3], ax=axes[0, 1], label='Point Count')
    
    # 2D density heatmap - YZ plane
    h3 = axes[1, 0].hist2d(df['y'], df['z'], bins=50, cmap='hot')
    axes[1, 0].set_title('YZ Density Heatmap')
    axes[1, 0].set_xlabel('Y (m)')
    axes[1, 0].set_ylabel('Z (m)')
    plt.colorbar(h3[3], ax=axes[1, 0], label='Point Count')
    
    # Kernel density estimation
    from scipy.stats import gaussian_kde
    
    if len(df) < 10000:  # Limit points to avoid long computation time
        xy = np.vstack([df['x'], df['y']])
        kernel = gaussian_kde(xy)
        
        xi, yi = np.mgrid[df['x'].min():df['x'].max():100j, 
                         df['y'].min():df['y'].max():100j]
        zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        
        contour = axes[1, 1].contourf(xi, yi, zi, levels=20, cmap='viridis')
        axes[1, 1].set_title('Kernel Density Estimation (XY)')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        axes[1, 1].axis('equal')
        plt.colorbar(contour, ax=axes[1, 1], label='Density')
    else:
        axes[1, 1].text(0.5, 0.5, 'Too many points for KDE\n(> 10000 points)', 
                       transform=axes[1, 1].transAxes,
                       horizontalalignment='center', verticalalignment='center')
        axes[1, 1].set_title('Kernel Density Estimation (Skipped)')
    
    plt.tight_layout()
    
    # Density statistics
    density_stats = {
        'total_points': len(df),
        'xy_area': (df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min()),
        'xy_density': len(df) / ((df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min())),
        'volume': (df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min()) * (df['z'].max() - df['z'].min()),
        'volume_density': len(df) / ((df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min()) * (df['z'].max() - df['z'].min()))
    }
    
    save_plot_and_data(fig, density_stats, output_dir, "09_density_maps")
    plt.close(fig)

def plot_outlier_analysis(df, output_dir):
    """Plot outlier analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Outlier Analysis', fontsize=16)
    
    # Detect outliers using the IQR method
    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    # Intensity outliers
    intensity_outliers = detect_outliers_iqr(df, 'intensity')
    axes[0, 0].scatter(df.index, df['intensity'], c=['red' if x else 'blue' for x in intensity_outliers], 
                      s=1, alpha=0.5)
    axes[0, 0].set_title(f'Intensity Outliers ({intensity_outliers.sum()} points)')
    axes[0, 0].set_xlabel('Point Index')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Z coordinate outliers
    z_outliers = detect_outliers_iqr(df, 'z')
    axes[0, 1].scatter(df['x'], df['y'], c=['red' if x else 'blue' for x in z_outliers], 
                      s=1, alpha=0.5)
    axes[0, 1].set_title(f'Z Coordinate Outliers ({z_outliers.sum()} points)')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].axis('equal')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Color outliers (based on Mahalanobis distance)
    from scipy.spatial.distance import mahalanobis
    
    rgb_data = df[['r', 'g', 'b']].values
    rgb_mean = np.mean(rgb_data, axis=0)
    rgb_cov = np.cov(rgb_data.T)
    
    if np.linalg.det(rgb_cov) != 0:
        rgb_cov_inv = np.linalg.inv(rgb_cov)
        mahal_dist = np.array([mahalanobis(point, rgb_mean, rgb_cov_inv) for point in rgb_data])
        mahal_threshold = np.percentile(mahal_dist, 95)
        color_outliers = mahal_dist > mahal_threshold
        
        axes[0, 2].scatter(df.index, mahal_dist, c=['red' if x else 'blue' for x in color_outliers], 
                          s=1, alpha=0.5)
        axes[0, 2].set_title(f'Color Outliers (Mahalanobis) ({color_outliers.sum()} points)')
        axes[0, 2].set_xlabel('Point Index')
        axes[0, 2].set_ylabel('Mahalanobis Distance')
        axes[0, 2].axhline(y=mahal_threshold, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Cannot compute\nMahalanobis distance', 
                       transform=axes[0, 2].transAxes,
                       horizontalalignment='center', verticalalignment='center')
        axes[0, 2].set_title('Color Outliers (Skipped)')
    
    # Local density outliers (LOF)
    from sklearn.neighbors import LocalOutlierFactor
    
    if len(df) < 50000:  # Limit points to avoid long computation time
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        xyz_data = df[['x', 'y', 'z']].values
        lof_labels = lof.fit_predict(xyz_data)
        lof_outliers = lof_labels == -1
        
        axes[1, 0].scatter(df['x'], df['y'], c=['red' if x else 'blue' for x in lof_outliers], 
                          s=1, alpha=0.5)
        axes[1, 0].set_title(f'Local Outlier Factor ({lof_outliers.sum()} points)')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        axes[1, 0].axis('equal')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Too many points for LOF\n(> 50000 points)', 
                       transform=axes[1, 0].transAxes,
                       horizontalalignment='center', verticalalignment='center')
        axes[1, 0].set_title('Local Outlier Factor (Skipped)')
    
    # Combined outliers
    all_outliers = intensity_outliers | z_outliers
    if 'color_outliers' in locals():
        all_outliers = all_outliers | color_outliers
    if 'lof_outliers' in locals():
        all_outliers = all_outliers | lof_outliers
    
    axes[1, 1].scatter(df['x'], df['y'], c=['red' if x else 'blue' for x in all_outliers], 
                      s=1, alpha=0.5)
    axes[1, 1].set_title(f'Combined Outliers ({all_outliers.sum()} points)')
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].axis('equal')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Outlier statistics
    outlier_stats = {
        'intensity_outliers': int(intensity_outliers.sum()),
        'z_outliers': int(z_outliers.sum()),
        'color_outliers': int(color_outliers.sum()) if 'color_outliers' in locals() else 0,
        'lof_outliers': int(lof_outliers.sum()) if 'lof_outliers' in locals() else 0,
        'total_outliers': int(all_outliers.sum()),
        'outlier_ratio': all_outliers.sum() / len(df)
    }
    
    stats_text = '\n'.join([f'{k}: {v}' for k, v in outlier_stats.items()])
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 2].set_title('Outlier Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    save_plot_and_data(fig, outlier_stats, output_dir, "10_outlier_analysis")
    plt.close(fig)

def plot_pca_analysis(df, output_dir):
    """Plot principal component analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Principal Component Analysis', fontsize=16)
    
    # Prepare data
    features = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b']
    X = df[features].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance ratio
    axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_)
    axes[0, 0].set_title('Explained Variance Ratio')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    axes[0, 1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PC1 vs PC2
    sample_size = min(5000, len(X_pca))
    sample_idx = np.random.choice(len(X_pca), sample_size, replace=False)
    
    scatter = axes[1, 0].scatter(X_pca[sample_idx, 0], X_pca[sample_idx, 1], 
                                 c=df.iloc[sample_idx]['intensity'], cmap='viridis', s=1, alpha=0.5)
    axes[1, 0].set_title(f'PC1 vs PC2 (sampled {sample_size} points)')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loadings matrix (feature contributions)
    loadings = pca.components_[:3].T * np.sqrt(pca.explained_variance_[:3])
    
    im = axes[1, 1].imshow(loadings, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_title('Feature Loadings (PC1-PC3)')
    axes[1, 1].set_xlabel('Principal Component')
    axes[1, 1].set_ylabel('Feature')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_xticklabels(['PC1', 'PC2', 'PC3'])
    axes[1, 1].set_yticks(range(len(features)))
    axes[1, 1].set_yticklabels(features)
    plt.colorbar(im, ax=axes[1, 1])
    
    # Add numeric values inside each cell
    for i in range(len(features)):
        for j in range(3):
            text = axes[1, 1].text(j, i, f'{loadings[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # PCA statistics
    pca_stats = {
        'n_components_95var': int(np.argmax(cumsum_var >= 0.95) + 1),
        'pc1_variance': pca.explained_variance_ratio_[0],
        'pc2_variance': pca.explained_variance_ratio_[1],
        'pc3_variance': pca.explained_variance_ratio_[2],
        'total_variance_pc123': pca.explained_variance_ratio_[:3].sum()
    }
    
    # Add principal component loadings
    for i, feature in enumerate(features):
        for j in range(3):
            pca_stats[f'{feature}_pc{j+1}_loading'] = loadings[i, j]
    
    save_plot_and_data(fig, pca_stats, output_dir, "11_pca_analysis")
    plt.close(fig)

def plot_strip_analysis(df, output_dir):
    """Plot strip analysis"""
    if 'strip_id' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Strip Analysis', fontsize=16)
    
    unique_strips = df['strip_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strips)))
    
    # Strip spatial distribution
    for i, strip_id in enumerate(unique_strips):
        strip_data = df[df['strip_id'] == strip_id]
        axes[0, 0].scatter(strip_data['x'], strip_data['y'], c=[colors[i]], 
                          label=f'Strip {strip_id}', s=1, alpha=0.6)
    axes[0, 0].set_title('Strip Spatial Distribution')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Points per strip
    strip_counts = df['strip_id'].value_counts().sort_index()
    axes[0, 1].bar(strip_counts.index, strip_counts.values)
    axes[0, 1].set_title('Points per Strip')
    axes[0, 1].set_xlabel('Strip ID')
    axes[0, 1].set_ylabel('Point Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Intensity by strip
    strip_intensity_stats = df.groupby('strip_id')['intensity'].agg(['mean', 'std'])
    axes[0, 2].errorbar(strip_intensity_stats.index, strip_intensity_stats['mean'], 
                       yerr=strip_intensity_stats['std'], fmt='o-', capsize=5)
    axes[0, 2].set_title('Intensity by Strip (Mean ± Std)')
    axes[0, 2].set_xlabel('Strip ID')
    axes[0, 2].set_ylabel('Intensity')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Elevation by strip
    strip_z_stats = df.groupby('strip_id')['z'].agg(['mean', 'std', 'min', 'max'])
    x_pos = np.arange(len(strip_z_stats))
    axes[1, 0].bar(x_pos, strip_z_stats['mean'], yerr=strip_z_stats['std'], 
                   alpha=0.7, capsize=5)
    axes[1, 0].set_title('Elevation by Strip (Mean ± Std)')
    axes[1, 0].set_xlabel('Strip ID')
    axes[1, 0].set_ylabel('Z (m)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(strip_z_stats.index)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Strip overlap analysis
    from scipy.spatial import ConvexHull
    
    overlap_matrix = np.zeros((len(unique_strips), len(unique_strips)))
    
    for i, strip1 in enumerate(unique_strips):
        for j, strip2 in enumerate(unique_strips):
            if i != j:
                strip1_data = df[df['strip_id'] == strip1][['x', 'y']].values
                strip2_data = df[df['strip_id'] == strip2][['x', 'y']].values
                
                if len(strip1_data) > 3 and len(strip2_data) > 3:
                    try:
                        hull1 = ConvexHull(strip1_data)
                        hull2 = ConvexHull(strip2_data)
                        
                        # Simple bounding-box overlap check
                        bbox1 = [strip1_data[:, 0].min(), strip1_data[:, 0].max(),
                                strip1_data[:, 1].min(), strip1_data[:, 1].max()]
                        bbox2 = [strip2_data[:, 0].min(), strip2_data[:, 0].max(),
                                strip2_data[:, 1].min(), strip2_data[:, 1].max()]
                        
                        x_overlap = min(bbox1[1], bbox2[1]) - max(bbox1[0], bbox2[0])
                        y_overlap = min(bbox1[3], bbox2[3]) - max(bbox1[2], bbox2[2])
                        
                        if x_overlap > 0 and y_overlap > 0:
                            overlap_area = x_overlap * y_overlap
                            strip1_area = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2])
                            overlap_ratio = overlap_area / strip1_area
                            overlap_matrix[i, j] = overlap_ratio
                    except:
                        pass
    
    im = axes[1, 1].imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('Strip Overlap Matrix')
    axes[1, 1].set_xlabel('Strip ID')
    axes[1, 1].set_ylabel('Strip ID')
    axes[1, 1].set_xticks(range(len(unique_strips)))
    axes[1, 1].set_xticklabels(unique_strips)
    axes[1, 1].set_yticks(range(len(unique_strips)))
    axes[1, 1].set_yticklabels(unique_strips)
    plt.colorbar(im, ax=axes[1, 1], label='Overlap Ratio')
    
    # Mean RGB per strip
    strip_rgb_means = df.groupby('strip_id')[['r', 'g', 'b']].mean()
    x_pos = np.arange(len(strip_rgb_means))
    width = 0.25
    
    axes[1, 2].bar(x_pos - width, strip_rgb_means['r'], width, label='Red', color='red', alpha=0.7)
    axes[1, 2].bar(x_pos, strip_rgb_means['g'], width, label='Green', color='green', alpha=0.7)
    axes[1, 2].bar(x_pos + width, strip_rgb_means['b'], width, label='Blue', color='blue', alpha=0.7)
    axes[1, 2].set_title('Mean RGB by Strip')
    axes[1, 2].set_xlabel('Strip ID')
    axes[1, 2].set_ylabel('RGB Value')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(strip_rgb_means.index)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Strip statistics
    strip_stats = pd.DataFrame({
        'strip_id': unique_strips,
        'point_count': [len(df[df['strip_id'] == s]) for s in unique_strips],
        'mean_intensity': [df[df['strip_id'] == s]['intensity'].mean() for s in unique_strips],
        'mean_z': [df[df['strip_id'] == s]['z'].mean() for s in unique_strips],
        'z_range': [df[df['strip_id'] == s]['z'].max() - df[df['strip_id'] == s]['z'].min() for s in unique_strips]
    })
    
    save_plot_and_data(fig, strip_stats, output_dir, "12_strip_analysis")
    plt.close(fig)

def plot_regression_analysis(results_df, output_dir):
    """Plot regression analysis"""
    if results_df is None or len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Regression Analysis', fontsize=16)
    
    # Linear regression: distance error vs point count
    from sklearn.linear_model import LinearRegression
    
    X = results_df['point_count'].values.reshape(-1, 1)
    y = results_df['distance_to_ref'].values
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    
    axes[0, 0].scatter(X, y, alpha=0.7)
    axes[0, 0].plot(X, y_pred, 'r-', linewidth=2, label=f'y = {lr.coef_[0]:.4f}x + {lr.intercept_:.4f}')
    axes[0, 0].set_title('Distance Error vs Point Count')
    axes[0, 0].set_xlabel('Point Count')
    axes[0, 0].set_ylabel('Distance Error (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Polynomial regression: planarity vs normal angle
    if len(results_df) > 3:
        X_angle = results_df['normal_angle'].values
        y_planarity = results_df['planarity'].values
        
        # Fit polynomial
        z = np.polyfit(X_angle, y_planarity, 2)
        p = np.poly1d(z)
        
        X_angle_sorted = np.sort(X_angle)
        y_poly_pred = p(X_angle_sorted)
        
        axes[0, 2].scatter(X_angle, y_planarity, alpha=0.7)
        axes[0, 2].plot(X_angle_sorted, y_poly_pred, 'r-', linewidth=2, 
                       label=f'y = {z[0]:.4f}x² + {z[1]:.4f}x + {z[2]:.4f}')
        axes[0, 2].set_title('Planarity vs Normal Angle (Polynomial)')
        axes[0, 2].set_xlabel('Normal Angle (degrees)')
        axes[0, 2].set_ylabel('Planarity')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Exponential fit: sorted distance error distribution
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    try:
        x_data = np.arange(len(results_df))
        y_data = np.sort(results_df['distance_to_ref'].values)
        
        popt, _ = curve_fit(exp_func, x_data, y_data, p0=[1, 0.01, 0])
        y_exp_pred = exp_func(x_data, *popt)
        
        axes[1, 0].scatter(x_data, y_data, alpha=0.7)
        axes[1, 0].plot(x_data, y_exp_pred, 'r-', linewidth=2, 
                       label=f'y = {popt[0]:.3f}e^(-{popt[1]:.3f}x) + {popt[2]:.3f}')
        axes[1, 0].set_title('Sorted Distance Error (Exponential Fit)')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('Distance Error (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    except:
        axes[1, 0].text(0.5, 0.5, 'Exponential fit failed', 
                       transform=axes[1, 0].transAxes,
                       horizontalalignment='center', verticalalignment='center')
        axes[1, 0].set_title('Exponential Fit (Failed)')
    
    # Multi-variable regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    
    features = ['point_count', 'planarity', 'normal_angle']
    X_multi = results_df[features].values
    y_multi = results_df['distance_to_ref'].values
    
    scaler = StandardScaler()
    X_multi_scaled = scaler.fit_transform(X_multi)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_multi_scaled, y_multi)
    y_multi_pred = ridge.predict(X_multi_scaled)
    
    axes[1, 1].scatter(y_multi, y_multi_pred, alpha=0.7)
    axes[1, 1].plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_title('Multi-variable Regression (Ridge)')
    axes[1, 1].set_xlabel('Actual Distance Error (m)')
    axes[1, 1].set_ylabel('Predicted Distance Error (m)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Regression coefficients
    coef_data = pd.DataFrame({
        'Feature': features,
        'Coefficient': ridge.coef_
    })
    
    axes[1, 2].barh(coef_data['Feature'], coef_data['Coefficient'])
    axes[1, 2].set_title('Ridge Regression Coefficients')
    axes[1, 2].set_xlabel('Coefficient Value')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Regression statistics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    regression_stats = {
        'linear_r2': r2_score(y, y_pred),
        'linear_rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'linear_mae': mean_absolute_error(y, y_pred),
        'ridge_r2': r2_score(y_multi, y_multi_pred),
        'ridge_rmse': np.sqrt(mean_squared_error(y_multi, y_multi_pred)),
        'ridge_mae': mean_absolute_error(y_multi, y_multi_pred)
    }
    
    save_plot_and_data(fig, regression_stats, output_dir, "13_regression_analysis")
    plt.close(fig)

def plot_surface_analysis(df, output_dir):
    """Plot surface analysis"""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Surface Analysis', fontsize=16)
    
    # Create grid
    grid_resolution = 50
    xi = np.linspace(df['x'].min(), df['x'].max(), grid_resolution)
    yi = np.linspace(df['y'].min(), df['y'].max(), grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate elevation
    points = df[['x', 'y']].values
    z_values = df['z'].values
    
    try:
        Zi = griddata(points, z_values, (Xi, Yi), method='linear')
        
        # 3D surface plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        surf = ax1.plot_surface(Xi, Yi, Zi, cmap='terrain', alpha=0.8)
        ax1.set_title('Elevation Surface')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        plt.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Contour map
        ax2 = fig.add_subplot(2, 3, 2)
        contour = ax2.contourf(Xi, Yi, Zi, levels=20, cmap='terrain')
        ax2.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax2.set_title('Elevation Contour Map')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.axis('equal')
        plt.colorbar(contour, ax=ax2)
        
        # Slope analysis
        dz_dx, dz_dy = np.gradient(Zi, xi[1]-xi[0], yi[1]-yi[0])
        slope = np.sqrt(dz_dx**2 + dz_dy**2)
        slope_degrees = np.degrees(np.arctan(slope))
        
        ax3 = fig.add_subplot(2, 3, 3)
        slope_plot = ax3.contourf(Xi, Yi, slope_degrees, levels=20, cmap='RdYlGn_r')
        ax3.set_title('Slope Map (degrees)')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.axis('equal')
        plt.colorbar(slope_plot, ax=ax3)
        
        # Aspect analysis
        aspect = np.degrees(np.arctan2(dz_dy, dz_dx))
        aspect[aspect < 0] += 360
        
        ax4 = fig.add_subplot(2, 3, 4)
        aspect_plot = ax4.contourf(Xi, Yi, aspect, levels=20, cmap='hsv')
        ax4.set_title('Aspect Map (degrees)')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.axis('equal')
        plt.colorbar(aspect_plot, ax=ax4)
        
        # Curvature analysis
        d2z_dx2 = np.gradient(dz_dx, xi[1]-xi[0], axis=1)
        d2z_dy2 = np.gradient(dz_dy, yi[1]-yi[0], axis=0)
        curvature = d2z_dx2 + d2z_dy2
        
        ax5 = fig.add_subplot(2, 3, 5)
        curv_plot = ax5.contourf(Xi, Yi, curvature, levels=20, cmap='seismic')
        ax5.set_title('Curvature Map')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.axis('equal')
        plt.colorbar(curv_plot, ax=ax5)
        
        # Roughness analysis
        from scipy.ndimage import generic_filter
        
        def local_std(values):
            return np.std(values)
        
        window_size = 5
        roughness = generic_filter(Zi, local_std, size=window_size)
        
        ax6 = fig.add_subplot(2, 3, 6)
        rough_plot = ax6.contourf(Xi, Yi, roughness, levels=20, cmap='copper')
        ax6.set_title('Surface Roughness')
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.axis('equal')
        plt.colorbar(rough_plot, ax=ax6)
        
    except Exception as e:
        fig.text(0.5, 0.5, f'Surface analysis failed: {str(e)}', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Surface statistics
    surface_stats = {}
    if 'Zi' in locals():
        surface_stats['z_min'] = np.nanmin(Zi)
        surface_stats['z_max'] = np.nanmax(Zi)
        surface_stats['z_mean'] = np.nanmean(Zi)
        surface_stats['z_std'] = np.nanstd(Zi)
    if 'slope_degrees' in locals():
        surface_stats['slope_mean'] = np.nanmean(slope_degrees)
        surface_stats['slope_max'] = np.nanmax(slope_degrees)
        surface_stats['slope_std'] = np.nanstd(slope_degrees)
    if 'curvature' in locals():
        surface_stats['curvature_mean'] = np.nanmean(curvature)
        surface_stats['curvature_std'] = np.nanstd(curvature)
    if 'roughness' in locals():
        surface_stats['roughness_mean'] = np.nanmean(roughness)
        surface_stats['roughness_std'] = np.nanstd(roughness)
    
    save_plot_and_data(fig, surface_stats, output_dir, "14_surface_analysis")
    plt.close(fig)

def plot_confidence_intervals(results_df, output_dir):
    """Plot confidence interval analysis"""
    if results_df is None or len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confidence Interval Analysis', fontsize=16)
    
    # Confidence interval for distance errors
    mean_error = results_df['distance_to_ref'].mean()
    std_error = results_df['distance_to_ref'].std()
    n = len(results_df)
    
    # 95% confidence interval
    from scipy import stats as scipy_stats
    confidence = 0.95
    t_value = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std_error / np.sqrt(n)
    ci_lower = mean_error - margin_error
    ci_upper = mean_error + margin_error
    
    axes[0, 0].errorbar(results_df['reference_id'], results_df['distance_to_ref'], 
                       yerr=std_error, fmt='o', alpha=0.7, capsize=5)
    axes[0, 0].axhline(y=mean_error, color='red', linestyle='-', label=f'Mean: {mean_error:.4f}')
    axes[0, 0].axhline(y=ci_lower, color='red', linestyle='--', alpha=0.7, 
                      label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    axes[0, 0].axhline(y=ci_upper, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].fill_between(results_df['reference_id'], ci_lower, ci_upper, alpha=0.2, color='red')
    axes[0, 0].set_title('Distance Error with 95% Confidence Interval')
    axes[0, 0].set_xlabel('Reference ID')
    axes[0, 0].set_ylabel('Distance Error (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Bootstrap confidence interval
    n_bootstrap = 1000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(results_df['distance_to_ref'], size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    bootstrap_ci_lower = np.percentile(bootstrap_means, 2.5)
    bootstrap_ci_upper = np.percentile(bootstrap_means, 97.5)
    
    axes[0, 1].hist(bootstrap_means, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=mean_error, color='red', linestyle='-', linewidth=2, label='Observed Mean')
    axes[0, 1].axvline(x=bootstrap_ci_lower, color='green', linestyle='--', 
                      label=f'Bootstrap 95% CI: [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]')
    axes[0, 1].axvline(x=bootstrap_ci_upper, color='green', linestyle='--')
    axes[0, 1].set_title('Bootstrap Distribution of Mean Error')
    axes[0, 1].set_xlabel('Mean Distance Error (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction interval
    prediction_interval_multiplier = 2  # approx. 95% prediction interval
    pred_lower = mean_error - prediction_interval_multiplier * std_error
    pred_upper = mean_error + prediction_interval_multiplier * std_error
    
    axes[1, 0].scatter(results_df['reference_id'], results_df['distance_to_ref'], alpha=0.7)
    axes[1, 0].axhline(y=mean_error, color='red', linestyle='-', label='Mean')
    axes[1, 0].axhline(y=pred_lower, color='blue', linestyle='--', 
                      label=f'95% Prediction Interval: [{pred_lower:.4f}, {pred_upper:.4f}]')
    axes[1, 0].axhline(y=pred_upper, color='blue', linestyle='--')
    axes[1, 0].fill_between(results_df['reference_id'], pred_lower, pred_upper, 
                           alpha=0.1, color='blue')
    axes[1, 0].set_title('Distance Error with Prediction Interval')
    axes[1, 0].set_xlabel('Reference ID')
    axes[1, 0].set_ylabel('Distance Error (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confidence band
    sorted_indices = np.argsort(results_df['point_count'])
    sorted_point_count = results_df['point_count'].values[sorted_indices]
    sorted_distance_error = results_df['distance_to_ref'].values[sorted_indices]
    
    # Local weighted regression (LOWESS)
    from scipy.signal import savgol_filter
    
    if len(sorted_distance_error) > 5:
        window_length = min(len(sorted_distance_error), 11)
        if window_length % 2 == 0:
            window_length -= 1
        
        smoothed = savgol_filter(sorted_distance_error, window_length, 3)
        residuals = sorted_distance_error - smoothed
        local_std = pd.Series(residuals).rolling(window=5, center=True).std().fillna(method='bfill').fillna(method='ffill')
        
        axes[1, 1].scatter(sorted_point_count, sorted_distance_error, alpha=0.5, s=20)
        axes[1, 1].plot(sorted_point_count, smoothed, 'r-', linewidth=2, label='Smoothed Trend')
        axes[1, 1].fill_between(sorted_point_count, 
                               smoothed - 1.96 * local_std, 
                               smoothed + 1.96 * local_std, 
                               alpha=0.2, color='red', label='95% Confidence Band')
        axes[1, 1].set_title('Distance Error vs Point Count with Confidence Band')
        axes[1, 1].set_xlabel('Point Count')
        axes[1, 1].set_ylabel('Distance Error (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Confidence interval statistics
    ci_stats = {
        'mean_error': mean_error,
        'std_error': std_error,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'ci_95_width': ci_upper - ci_lower,
        'bootstrap_ci_lower': bootstrap_ci_lower,
        'bootstrap_ci_upper': bootstrap_ci_upper,
        'bootstrap_ci_width': bootstrap_ci_upper - bootstrap_ci_lower,
        'pred_interval_lower': pred_lower,
        'pred_interval_upper': pred_upper,
        'pred_interval_width': pred_upper - pred_lower
    }
    
    save_plot_and_data(fig, ci_stats, output_dir, "15_confidence_intervals")
    plt.close(fig)

def create_comparison_analysis(all_results):
    """Create detailed comparison analysis"""
    if len(all_results) < 2:
        log_warning("At least two files are required for comparison analysis")
        return None
    
    log_info("Starting comparison analysis...")
    
    comparison_data = []
    all_ref_ids = set(all_results[0]['reference_id'])
    for result_df in all_results[1:]:
        all_ref_ids = all_ref_ids.intersection(set(result_df['reference_id']))
    
    if len(all_ref_ids) == 0:
        log_warning("No common reference points found, cannot perform comparison")
        return None
    
    log_info(f"Found {len(all_ref_ids)} common reference points for comparison")
    
    for ref_id in sorted(all_ref_ids):
        comparison_record = {'reference_id': ref_id}
        
        ref_data = []
        for i, result_df in enumerate(all_results):
            file_data = result_df[result_df['reference_id'] == ref_id]
            if len(file_data) > 0:
                data = file_data.iloc[0]
                ref_data.append({
                    'file_index': i,
                    'las_file': data['las_file'],
                    'center_x': data['center_x'],
                    'center_y': data['center_y'],
                    'center_z': data['center_z'],
                    'planarity': data['planarity'],
                    'normal_angle': data['normal_angle'],
                    'distance_to_ref': data['distance_to_ref'],
                    'point_count': data['point_count']
                })
        
        if len(ref_data) < 2:
            continue
        
        for i, data in enumerate(ref_data):
            comparison_record[f'file{i+1}_name'] = data['las_file']
            comparison_record[f'file{i+1}_center_x'] = data['center_x']
            comparison_record[f'file{i+1}_center_y'] = data['center_y']
            comparison_record[f'file{i+1}_center_z'] = data['center_z']
            comparison_record[f'file{i+1}_planarity'] = data['planarity']
            comparison_record[f'file{i+1}_normal_angle'] = data['normal_angle']
            comparison_record[f'file{i+1}_distance_to_ref'] = data['distance_to_ref']
            comparison_record[f'file{i+1}_point_count'] = data['point_count']
        
        if len(ref_data) >= 2:
            dx = ref_data[1]['center_x'] - ref_data[0]['center_x']
            dy = ref_data[1]['center_y'] - ref_data[0]['center_y']
            dz = ref_data[1]['center_z'] - ref_data[0]['center_z']
            
            distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            distance_2d = np.sqrt(dx**2 + dy**2)
            
            comparison_record['diff_x'] = dx
            comparison_record['diff_y'] = dy
            comparison_record['diff_z'] = dz
            comparison_record['distance_3d'] = distance_3d
            comparison_record['distance_2d'] = distance_2d
            comparison_record['diff_planarity'] = ref_data[1]['planarity'] - ref_data[0]['planarity']
            comparison_record['diff_normal_angle'] = ref_data[1]['normal_angle'] - ref_data[0]['normal_angle']
            comparison_record['diff_distance_to_ref'] = ref_data[1]['distance_to_ref'] - ref_data[0]['distance_to_ref']
            comparison_record['diff_point_count'] = ref_data[1]['point_count'] - ref_data[0]['point_count']
        
        comparison_data.append(comparison_record)
    
    if len(comparison_data) == 0:
        log_warning("No comparison data was generated")
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    stats_data = []
    if len(comparison_df) > 0:
        stats_data.append({
            'metric': '3D distance difference',
            'mean': comparison_df['distance_3d'].mean(),
            'std': comparison_df['distance_3d'].std(),
            'max': comparison_df['distance_3d'].max(),
            'min': comparison_df['distance_3d'].min(),
            'unit': 'm'
        })
        
        stats_data.append({
            'metric': '2D planar distance difference',
            'mean': comparison_df['distance_2d'].mean(),
            'std': comparison_df['distance_2d'].std(),
            'max': comparison_df['distance_2d'].max(),
            'min': comparison_df['distance_2d'].min(),
            'unit': 'm'
        })
        
        stats_data.append({
            'metric': 'Z height difference',
            'mean': comparison_df['diff_z'].mean(),
            'std': comparison_df['diff_z'].std(),
            'max': comparison_df['diff_z'].max(),
            'min': comparison_df['diff_z'].min(),
            'unit': 'm'
        })
        
        stats_data.append({
            'metric': 'Planarity difference',
            'mean': comparison_df['diff_planarity'].mean(),
            'std': comparison_df['diff_planarity'].std(),
            'max': comparison_df['diff_planarity'].max(),
            'min': comparison_df['diff_planarity'].min(),
            'unit': 'dimensionless'
        })
        
        stats_data.append({
            'metric': 'Normal angle difference',
            'mean': comparison_df['diff_normal_angle'].mean(),
            'std': comparison_df['diff_normal_angle'].std(),
            'max': comparison_df['diff_normal_angle'].max(),
            'min': comparison_df['diff_normal_angle'].min(),
            'unit': 'degrees'
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    return comparison_df, stats_df

def plot_comprehensive_comparison(comparison_df, stats_df, output_dir):
    """Plot comprehensive comparison analysis"""
    if comparison_df is None or len(comparison_df) == 0:
        return
    
    # Create multiple comparison plots
    
    # 1. Basic comparison plot
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig1.suptitle('Target Points Comparison Analysis', fontsize=16)
    
    # 3D distance differences
    axes[0, 0].bar(comparison_df['reference_id'], comparison_df['distance_3d'])
    axes[0, 0].set_title('3D Distance Differences')
    axes[0, 0].set_xlabel('Reference ID')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2D planar distance differences
    axes[0, 1].bar(comparison_df['reference_id'], comparison_df['distance_2d'])
    axes[0, 1].set_title('2D Planar Distance Differences')
    axes[0, 1].set_xlabel('Reference ID')
    axes[0, 1].set_ylabel('Distance (m)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z height differences
    axes[0, 2].bar(comparison_df['reference_id'], comparison_df['diff_z'])
    axes[0, 2].set_title('Z Height Differences')
    axes[0, 2].set_xlabel('Reference ID')
    axes[0, 2].set_ylabel('Height Diff (m)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Planarity differences
    axes[1, 0].bar(comparison_df['reference_id'], comparison_df['diff_planarity'])
    axes[1, 0].set_title('Planarity Differences')
    axes[1, 0].set_xlabel('Reference ID')
    axes[1, 0].set_ylabel('Planarity Diff')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Normal angle differences
    axes[1, 1].bar(comparison_df['reference_id'], comparison_df['diff_normal_angle'])
    axes[1, 1].set_title('Normal Angle Differences')
    axes[1, 1].set_xlabel('Reference ID')
    axes[1, 1].set_ylabel('Angle Diff (°)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # XY offset vector plot
    axes[1, 2].quiver(comparison_df['file1_center_x'], comparison_df['file1_center_y'],
                     comparison_df['diff_x'], comparison_df['diff_y'],
                     scale_units='xy', scale=1, color='red', alpha=0.7)
    axes[1, 2].scatter(comparison_df['file1_center_x'], comparison_df['file1_center_y'], 
                      c='blue', s=50, alpha=0.7, label='File1 Centers')
    axes[1, 2].scatter(comparison_df['file1_center_x'] + comparison_df['diff_x'], 
                      comparison_df['file1_center_y'] + comparison_df['diff_y'], 
                      c='red', s=50, alpha=0.7, label='File2 Centers')
    axes[1, 2].set_title('XY Position Offset Vectors')
    axes[1, 2].set_xlabel('X (m)')
    axes[1, 2].set_ylabel('Y (m)')
    axes[1, 2].legend()
    axes[1, 2].axis('equal')
    
    plt.tight_layout()
    save_plot_and_data(fig1, comparison_df, output_dir, "16_comparison_basic")
    plt.close(fig1)
    
    # 2. Statistical analysis plots
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('Comparison Statistics Analysis', fontsize=16)
    
    # Distance error distribution histogram
    axes[0, 0].hist(comparison_df['distance_3d'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('3D Distance Error Distribution')
    axes[0, 0].set_xlabel('Distance Error (m)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error statistics bar chart
    if stats_df is not None and len(stats_df) > 0:
        metrics = stats_df['metric']
        means = stats_df['mean']
        stds = stats_df['std']
        
        x_pos = np.arange(len(metrics))
        axes[0, 1].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
        axes[0, 1].set_title('Error Statistics (Mean ± Std)')
        axes[0, 1].set_xlabel('Metrics')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals analysis
    residuals = comparison_df['distance_3d'] - comparison_df['distance_3d'].mean()
    axes[1, 0].scatter(comparison_df['reference_id'], residuals, alpha=0.7)
    axes[1, 0].set_title('Residuals vs Reference ID')
    axes[1, 0].set_xlabel('Reference ID')
    axes[1, 0].set_ylabel('Residuals (m)')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy assessment radar chart
    from math import pi
    
    # Compute accuracy metrics
    accuracy_metrics = {
        'Mean Error': 1 / (1 + comparison_df['distance_3d'].mean()),
        'Consistency': 1 / (1 + comparison_df['distance_3d'].std()),
        'Max Error': 1 / (1 + comparison_df['distance_3d'].max()),
        'Z Accuracy': 1 / (1 + abs(comparison_df['diff_z'].mean()))
    }
    
    categories = list(accuracy_metrics.keys())
    values = list(accuracy_metrics.values())
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    values += values[:1]
    
    axes[1, 1] = fig2.add_subplot(2, 2, 4, projection='polar')
    axes[1, 1].plot(angles, values, 'o-', linewidth=2, color='red')
    axes[1, 1].fill(angles, values, alpha=0.25, color='red')
    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Comparison Accuracy Assessment', y=1.1)
    
    plt.tight_layout()
    save_plot_and_data(fig2, stats_df, output_dir, "17_comparison_statistics")
    plt.close(fig2)

def plot_trend_analysis(comparison_df, output_dir):
    """Plot trend analysis"""
    if comparison_df is None or len(comparison_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trend Analysis', fontsize=16)
    
    # Distance error trends
    axes[0, 0].plot(comparison_df['reference_id'], comparison_df['distance_3d'], 'o-', label='3D Distance')
    axes[0, 0].plot(comparison_df['reference_id'], comparison_df['distance_2d'], 's-', label='2D Distance')
    axes[0, 0].set_title('Distance Error Trends')
    axes[0, 0].set_xlabel('Reference ID')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Systematic error analysis
    axes[0, 1].scatter(comparison_df['diff_x'], comparison_df['diff_y'], alpha=0.7)
    axes[0, 1].set_title('XY Systematic Error Pattern')
    axes[0, 1].set_xlabel('X Difference (m)')
    axes[0, 1].set_ylabel('Y Difference (m)')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Cumulative error distribution
    sorted_errors = np.sort(comparison_df['distance_3d'])
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 0].plot(sorted_errors, cumulative, linewidth=2)
    axes[1, 0].set_title('Cumulative Error Distribution')
    axes[1, 0].set_xlabel('3D Distance Error (m)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error direction rose plot
    angles = np.arctan2(comparison_df['diff_y'], comparison_df['diff_x'])
    magnitudes = comparison_df['distance_2d']
    
    # Convert angles to degrees and bin
    angle_degrees = np.degrees(angles) % 360
    angle_bins = np.arange(0, 361, 30)
    angle_centers = angle_bins[:-1] + 15
    
    binned_magnitudes = []
    for i in range(len(angle_bins) - 1):
        mask = (angle_degrees >= angle_bins[i]) & (angle_degrees < angle_bins[i + 1])
        if np.any(mask):
            binned_magnitudes.append(magnitudes[mask].mean())
        else:
            binned_magnitudes.append(0)
    
    # Polar plot
    axes[1, 1] = fig.add_subplot(2, 2, 4, projection='polar')
    theta = np.radians(angle_centers)
    axes[1, 1].bar(theta, binned_magnitudes, width=np.radians(30), alpha=0.7)
    axes[1, 1].set_title('Error Direction Rose Plot', y=1.1)
    axes[1, 1].set_theta_zero_location('N')
    axes[1, 1].set_theta_direction(-1)
    
    plt.tight_layout()
    
    # Trend data
    trend_data = comparison_df.copy()
    trend_data['angle_degrees'] = angle_degrees
    trend_data['error_magnitude'] = magnitudes
    
    save_plot_and_data(fig, trend_data, output_dir, "18_trend_analysis")
    plt.close(fig)

def create_summary_report(all_results, comparison_df, stats_df, output_dir):
    """Create summary report"""
    report_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_files_processed': len(all_results),
        'total_reference_points': 0,
        'total_detected_points': 0,
        'overall_accuracy': {},
        'file_statistics': []
    }
    
    # Collect per-file statistics
    for i, result_df in enumerate(all_results):
        if len(result_df) > 0:
            file_stat = {
                'file_index': i + 1,
                'file_name': result_df['las_file'].iloc[0],
                'detected_points': len(result_df),
                'mean_accuracy': result_df['distance_to_ref'].mean(),
                'std_accuracy': result_df['distance_to_ref'].std(),
                'mean_planarity': result_df['planarity'].mean(),
                'mean_normal_angle': result_df['normal_angle'].mean()
            }
            report_data['file_statistics'].append(file_stat)
            report_data['total_detected_points'] += len(result_df)
    
    # Overall accuracy statistics
    if comparison_df is not None and len(comparison_df) > 0:
        report_data['overall_accuracy'] = {
            'mean_3d_error': comparison_df['distance_3d'].mean(),
            'std_3d_error': comparison_df['distance_3d'].std(),
            'max_3d_error': comparison_df['distance_3d'].max(),
            'min_3d_error': comparison_df['distance_3d'].min(),
            'rmse_3d': np.sqrt(np.mean(comparison_df['distance_3d']**2)),
            'mean_2d_error': comparison_df['distance_2d'].mean(),
            'mean_z_error': comparison_df['diff_z'].mean(),
            'std_z_error': comparison_df['diff_z'].std()
        }
    
    # Save report
    report_df = pd.DataFrame([report_data])
    file_stats_df = pd.DataFrame(report_data['file_statistics'])
    
    # Save summary report
    save_plot_and_data(None, report_df, output_dir, "19_summary_report", "summary_report.csv")
    save_plot_and_data(None, file_stats_df, output_dir, "20_file_statistics", "file_statistics.csv")
    
    return report_data

def select_las_files():
    """Select LAS files using tkinter"""
    root = tk.Tk()
    root.withdraw()
    
    file_paths = filedialog.askopenfilenames(
        title="Select LAS files (you can select multiple)",
        filetypes=[("LAS files", "*.las"), ("All files", "*.*")]
    )
    
    root.destroy()
    return [Path(fp) for fp in file_paths] if file_paths else []

def main():
    """Main function - Enhanced full version"""
    log_info("=== Target Point Calibration - Enhanced Full Version ===")
    
    # 1. Select reference points file
    log_info("Please select the reference points file...")
    root = tk.Tk()
    root.withdraw()
    
    reference_file = filedialog.askopenfilename(
        title="Select reference points file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not reference_file:
        log_warning("No reference file selected, exiting")
        return
    
    # Read reference points
    try:
        reference_centers = np.loadtxt(reference_file)
        if reference_centers.ndim == 1:
            reference_centers = reference_centers.reshape(1, -1)
        log_success(f"Successfully read {len(reference_centers)} reference points")
    except Exception as e:
        log_warning(f"Failed to read reference file: {str(e)}")
        return
    
    # 2. Select LAS files
    log_info("Please select LAS files...")
    las_files = select_las_files()
    
    if not las_files:
        log_warning("No LAS files selected, exiting")
        return
    
    log_success(f"Selected {len(las_files)} LAS files")
    
    # 3. Create output directory
    output_dir = create_output_directory(las_files[0].parent, "target_analysis_enhanced")
    log_success(f"Output directory created: {output_dir}")
    
    # 4. Process each LAS file
    all_results = []
    all_target_points = []
    
    for i, las_file in enumerate(las_files):
        log_info(f"\nProcessing file {i+1}/{len(las_files)}: {las_file.name}")
        
        # Find target points
        target_points = find_control_point_with_reference(
            [las_file], 
            reference_centers,
            buffer_distance=10.0,
            intensity_threshold=1300,
            spatial_radius=0.5,
            mad_multiplier=1
        )
        
        if len(target_points) == 0:
            log_warning(f"File {las_file.name} has no target points")
            continue
        
        all_target_points.append(target_points)
        
        # Analyze target points
        analysis_result = analyze_target_points(
            target_points, 
            reference_centers,
            min_center_distance=5.0
        )
        
        if analysis_result is None:
            log_warning(f"Analysis failed for file {las_file.name}")
            continue
        
        results_df, analyzed_df = analysis_result
        
        # Add file identifier
        results_df['las_file'] = las_file.name
        all_results.append(results_df)
        
        # Generate all plots for this file
        log_info(f"Generating analysis plots for file {las_file.name}...")
        
        # Create a subdirectory for each file
        file_output_dir = output_dir / f"file_{i+1}_{las_file.stem}"
        file_output_dir.mkdir(exist_ok=True)
        
        # Generate various plots (original and new)
        plot_intensity_distribution(target_points, file_output_dir)
        plot_rgb_distribution(target_points, file_output_dir)
        plot_spatial_distribution(target_points, file_output_dir)
        plot_cluster_analysis(analyzed_df, results_df, reference_centers, file_output_dir)
        plot_accuracy_assessment(results_df, file_output_dir)
        plot_correlation_matrix(results_df, file_output_dir)
        plot_quality_metrics(results_df, file_output_dir)
        
        # Additional plots
        plot_spectral_analysis(target_points, file_output_dir)
        plot_density_maps(target_points, file_output_dir)
        plot_outlier_analysis(target_points, file_output_dir)
        plot_pca_analysis(target_points, file_output_dir)
        plot_strip_analysis(target_points, file_output_dir)
        plot_regression_analysis(results_df, file_output_dir)
        plot_surface_analysis(target_points, file_output_dir)
        plot_confidence_intervals(results_df, file_output_dir)
        
        # Save single-file results
        output_filename = file_output_dir / f"{las_file.stem}_target_analysis.csv"
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        log_success(f"Results saved to: {output_filename}")
        
        log_info(f"File {las_file.name} analysis complete, all plots generated")
    
    # 5. Generate comparison analysis (if multiple files)
    if len(all_results) >= 2:
        log_info("\nStarting comparison analysis...")
        
        # Merge all results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save basic combined results
        basic_comparison_filename = output_dir / "target_points_basic_comparison.csv"
        combined_results.to_csv(basic_comparison_filename, index=False, encoding='utf-8-sig')
        
        # Create detailed comparison analysis
        comparison_analysis = create_comparison_analysis(all_results)
        
        if comparison_analysis is not None:
            comparison_df, stats_df = comparison_analysis
            
            # Save comparison results
            detailed_comparison_filename = output_dir / "target_points_detailed_comparison.csv"
            comparison_df.to_csv(detailed_comparison_filename, index=False, encoding='utf-8-sig')
            
            stats_filename = output_dir / "target_points_statistics.csv"
            stats_df.to_csv(stats_filename, index=False, encoding='utf-8-sig')
            
            # Generate comparison plots
            log_info("Generating comparison analysis plots...")
            plot_comprehensive_comparison(comparison_df, stats_df, output_dir)
            plot_trend_analysis(comparison_df, output_dir)
            
            # Generate summary report
            log_info("Generating summary report...")
            summary_report = create_summary_report(all_results, comparison_df, stats_df, output_dir)
            
            log_success("\n=== Detailed comparison analysis complete! ===")
            log_info("Comparison statistics:")
            print(stats_df.to_string(index=False))
            
        else:
            log_warning("Unable to create detailed comparison analysis")
    
    elif len(all_results) == 1:
        log_info("Only one file present, skipping comparison analysis")
        # Create summary for single file
        summary_report = create_summary_report(all_results, None, None, output_dir)
    
    else:
        log_warning("No files were successfully processed")
        return
    
    # 6. Generate combined statistics plots (merged data from all files)
    if all_target_points:
        log_info("Generating combined statistics plots...")
        
        # Merge all detected target points
        combined_target_points = pd.concat(all_target_points, ignore_index=True)
        
        # Generate combined plots
        combined_output_dir = output_dir / "combined_analysis"
        combined_output_dir.mkdir(exist_ok=True)
        
        log_info("Generating combined intensity distribution plot...")
        plot_intensity_distribution(combined_target_points, combined_output_dir)
        
        log_info("Generating combined RGB distribution plot...")
        plot_rgb_distribution(combined_target_points, combined_output_dir)
        
        log_info("Generating combined spatial distribution plot...")
        plot_spatial_distribution(combined_target_points, combined_output_dir)
        
        log_info("Generating combined spectral analysis plot...")
        plot_spectral_analysis(combined_target_points, combined_output_dir)
        
        log_info("Generating combined density maps...")
        plot_density_maps(combined_target_points, combined_output_dir)
        
        log_info("Generating combined outlier analysis plot...")
        plot_outlier_analysis(combined_target_points, combined_output_dir)
        
        log_info("Generating combined PCA analysis plot...")
        plot_pca_analysis(combined_target_points, combined_output_dir)
        
        log_info("Generating combined strip analysis plot...")
        plot_strip_analysis(combined_target_points, combined_output_dir)
        
        log_info("Generating combined surface analysis plot...")
        plot_surface_analysis(combined_target_points, combined_output_dir)
        
        # Save combined data
        combined_filename = combined_output_dir / "combined_target_points.csv"
        combined_target_points.to_csv(combined_filename, index=False, encoding='utf-8-sig')
        log_success(f"Combined data saved to: {combined_filename}")
    
    # 7. Generate final analysis report
    log_info("\n=== Generating final analysis report ===")
    
    # Count all generated files
    all_png_files = list(output_dir.rglob('*.png'))
    all_csv_files = list(output_dir.rglob('*.csv'))
    all_txt_files = list(output_dir.rglob('*.txt'))
    
    # Create final report
    final_report = {
        'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_files': {
            'reference_file': reference_file,
            'las_files': [str(f) for f in las_files],
            'total_las_files': len(las_files)
        },
        'output_summary': {
            'output_directory': str(output_dir),
            'total_plots_generated': len(all_png_files),
            'total_csv_files': len(all_csv_files),
            'total_txt_files': len(all_txt_files)
        },
        'analysis_summary': {
            'reference_points': len(reference_centers),
            'files_processed': len(all_results),
            'files_with_targets': len(all_target_points) if all_target_points else 0
        }
    }
    
    # Add processing statistics
    if all_target_points:
        combined_points = pd.concat(all_target_points, ignore_index=True)
        final_report['point_statistics'] = {
            'total_target_points': len(combined_points),
            'mean_intensity': combined_points['intensity'].mean(),
            'std_intensity': combined_points['intensity'].std(),
            'x_range': [combined_points['x'].min(), combined_points['x'].max()],
            'y_range': [combined_points['y'].min(), combined_points['y'].max()],
            'z_range': [combined_points['z'].min(), combined_points['z'].max()]
        }
    
    # Save final report
    final_report_df = pd.DataFrame([final_report])
    final_report_path = output_dir / "final_analysis_report.json"
    
    import json
    with open(final_report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False, default=str)
    
    log_success(f"Final analysis report saved to: {final_report_path}")
    
    # Print summary
    log_success(f"\n{'='*60}")
    log_success("=== All analyses complete! ===")
    log_success(f"{'='*60}")
    log_info(f"Output directory: {output_dir}")
    log_info(f"Number of input files: {len(las_files)}")
    log_info(f"Number of successfully analyzed files: {len(all_results)}")
    log_info(f"Total plots generated: {len(all_png_files)}")
    log_info(f"Total CSV files generated: {len(all_csv_files)}")
    log_info(f"Total text files generated: {len(all_txt_files)}")
    
    if all_target_points:
        log_info(f"Total target points extracted: {len(combined_points)}")
    
    log_success(f"{'='*60}")
    log_info("Program finished!")

if __name__ == "__main__":
    main()