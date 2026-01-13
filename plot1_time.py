#!/usr/bin/env python3
"""
plot1_time.py: Timing visualization for computation time comparison from main1_time.py
Creates plots showing computation time (ms/step) and frequency (Hz) vs robustness parameter θ.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_timing_data(results_path, dist):
    """Load saved timing results from main1_time.py."""
    timing_file = os.path.join(results_path, f'timing_results_all_theta_{dist}.pkl')

    if not os.path.exists(timing_file):
        raise FileNotFoundError(f"Timing results file not found: {timing_file}")

    with open(timing_file, 'rb') as f:
        all_results = pickle.load(f)

    return all_results

def extract_timing_vs_theta(all_results):
    """Extract timing data organized by filter for plotting."""
    filters_order = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC', 'DR_EKF_CDC_FW']

    timing_data = {filt: {
        'theta': [],
        'time_mean': [],
        'time_std': [],
        'freq_mean': [],
        'freq_std': []
    } for filt in filters_order}

    # Extract data for each theta value
    theta_vals = sorted(all_results.keys())

    for theta in theta_vals:
        theta_results = all_results[theta]

        for filt in filters_order:
            if filt in theta_results:
                timing_data[filt]['theta'].append(theta)
                timing_data[filt]['time_mean'].append(theta_results[filt]['avg_time_per_step_ms'])
                timing_data[filt]['time_std'].append(theta_results[filt]['std_time_per_step_ms'])
                timing_data[filt]['freq_mean'].append(theta_results[filt]['frequency_hz'])

                # Compute frequency std from time measurements
                # For frequency: f = 1/t, so std_f ≈ std_t / t^2 (first-order approximation)
                time_ms = theta_results[filt]['avg_time_per_step_ms']
                time_s = time_ms / 1000.0
                freq_std = (theta_results[filt]['std_time_per_step_ms'] / 1000.0) / (time_s ** 2) if time_s > 0 else 0
                timing_data[filt]['freq_std'].append(freq_std)

    return timing_data, theta_vals

def create_timing_plots(timing_data, filters_order, dist):
    """Create combined timing plots (ms/step and Hz vs theta)."""

    # Define colors (matching plot0_withFW.py style)
    colors = {
        'EKF': '#1f77b4',           # Blue
        'DR_EKF_CDC': '#2ca02c',    # Green
        'DR_EKF_TAC': '#d62728',    # Red
        'DR_EKF_CDC_FW': '#ff7f0e'  # Orange
    }

    filter_names = {
        'EKF': "Extended Kalman Filter",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)",
        'DR_EKF_CDC_FW': "DR-EKF (CDC) with Frank-Wolfe"
    }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Computation Time (ms/step) vs theta
    ax1.set_title('Computation Time vs Robustness Parameter', fontsize=16, pad=15)
    ax1.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax1.set_ylabel('Computation Time (ms/step)', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')  # Log scale for y-axis
    ax1.grid(True, alpha=0.3)

    for filt in filters_order:
        data = timing_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            time_means = np.array(data['time_mean'])
            time_stds = np.array(data['time_std'])

            # Plot mean with error bars
            ax1.errorbar(theta_vals, time_means, yerr=time_stds,
                        marker='o', linewidth=2, markersize=8, capsize=5,
                        color=colors[filt], label=filter_names[filt])

    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Plot 2: Frequency (Hz) vs theta
    ax2.set_title('Computation Frequency vs Robustness Parameter', fontsize=16, pad=15)
    ax2.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax2.set_ylabel('Frequency (Hz)', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')  # Log scale for y-axis
    ax2.grid(True, alpha=0.3)

    for filt in filters_order:
        data = timing_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            freq_means = np.array(data['freq_mean'])
            freq_stds = np.array(data['freq_std'])

            # Plot mean with error bars
            ax2.errorbar(theta_vals, freq_means, yerr=freq_stds,
                        marker='s', linewidth=2, markersize=8, capsize=5,
                        color=colors[filt], label=filter_names[filt])

    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    # Save combined plot
    results_dir = os.path.join("results", "timing_comparison_with_FW")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_path = os.path.join(results_dir, f"timing_vs_robustness_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    print(f"Combined timing plot saved to: {save_path}")

    # Create individual plots
    create_individual_timing_plots(timing_data, filters_order, colors, filter_names, dist)

def create_individual_timing_plots(timing_data, filters_order, colors, filter_names, dist):
    """Create individual timing plots for publication quality."""
    results_dir = os.path.join("results", "timing_comparison_with_FW")

    # Computation Time (ms/step) plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Computation Time vs Robustness Parameter', fontsize=18, pad=20)
    ax.set_xlabel('Robustness Parameter θ', fontsize=16)
    ax.set_ylabel('Computation Time (ms/step)', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')  # Log scale for y-axis
    ax.grid(True, alpha=0.3)

    for filt in filters_order:
        data = timing_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            time_means = np.array(data['time_mean'])
            time_stds = np.array(data['time_std'])

            ax.errorbar(theta_vals, time_means, yerr=time_stds,
                       marker='o', linewidth=3, markersize=10, capsize=6,
                       color=colors[filt], label=filter_names[filt])

    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"time_vs_robustness_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Individual time plot saved to: {save_path}")

    # Frequency (Hz) plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Computation Frequency vs Robustness Parameter', fontsize=18, pad=20)
    ax.set_xlabel('Robustness Parameter θ', fontsize=16)
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')  # Log scale for y-axis
    ax.grid(True, alpha=0.3)

    for filt in filters_order:
        data = timing_data[filt]
        if len(data['theta']) > 0:
            theta_vals = np.array(data['theta'])
            freq_means = np.array(data['freq_mean'])
            freq_stds = np.array(data['freq_std'])

            ax.errorbar(theta_vals, freq_means, yerr=freq_stds,
                       marker='s', linewidth=3, markersize=10, capsize=6,
                       color=colors[filt], label=filter_names[filt])

    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"frequency_vs_robustness_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Individual frequency plot saved to: {save_path}")

def print_timing_summary(all_results):
    """Print summary table of timing results."""
    filters_order = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC', 'DR_EKF_CDC_FW']
    theta_vals = sorted(all_results.keys())

    print("\n" + "=" * 100)
    print("TIMING SUMMARY TABLE")
    print("=" * 100)
    print(f"{'θ':<10} {'EKF (ms)':<20} {'DR_EKF_CDC (ms)':<20} {'DR_EKF_TAC (ms)':<20} {'DR_EKF_CDC_FW (ms)':<20}")
    print("-" * 100)

    for theta in theta_vals:
        row = f"{theta:<10.2f}"
        for filt in filters_order:
            if filt in all_results[theta]:
                avg_time = all_results[theta][filt]['avg_time_per_step_ms']
                std_time = all_results[theta][filt]['std_time_per_step_ms']
                row += f" {avg_time:>7.4f}±{std_time:<7.4f}  "
            else:
                row += f" {'N/A':<20}"
        print(row)

    print("=" * 100)

    print("\n" + "=" * 100)
    print("FREQUENCY SUMMARY TABLE")
    print("=" * 100)
    print(f"{'θ':<10} {'EKF (Hz)':<20} {'DR_EKF_CDC (Hz)':<20} {'DR_EKF_TAC (Hz)':<20} {'DR_EKF_CDC_FW (Hz)':<20}")
    print("-" * 100)

    for theta in theta_vals:
        row = f"{theta:<10.2f}"
        for filt in filters_order:
            if filt in all_results[theta]:
                freq = all_results[theta][filt]['frequency_hz']
                row += f" {freq:>18.2f}  "
            else:
                row += f" {'N/A':<20}"
        print(row)

    print("=" * 100)

    # Print speedup comparison at each theta
    print("\n" + "=" * 100)
    print("SPEEDUP RELATIVE TO EKF (x times)")
    print("=" * 100)
    print(f"{'θ':<10} {'DR_EKF_CDC':<20} {'DR_EKF_TAC':<20} {'DR_EKF_CDC_FW':<20}")
    print("-" * 100)

    for theta in theta_vals:
        row = f"{theta:<10.2f}"
        ekf_time = all_results[theta]['EKF']['avg_time_per_step_ms']

        for filt in ['DR_EKF_CDC', 'DR_EKF_TAC', 'DR_EKF_CDC_FW']:
            if filt in all_results[theta]:
                filter_time = all_results[theta][filt]['avg_time_per_step_ms']
                speedup = ekf_time / filter_time
                row += f" {speedup:>18.2f}x "
            else:
                row += f" {'N/A':<20}"
        print(row)

    print("=" * 100)

def main():
    """Main plotting routine."""
    parser = argparse.ArgumentParser(description='Plot timing comparison results')
    parser.add_argument('--dist', type=str, default='normal',
                       help='Distribution type (normal or quadratic)')
    args = parser.parse_args()

    # Set paths
    results_path = "./results/timing_comparison_with_FW/"

    print("=" * 80)
    print("TIMING VISUALIZATION")
    print("=" * 80)
    print(f"Distribution: {args.dist}")
    print(f"Results path: {results_path}")
    print("=" * 80)

    # Load timing data
    print("\nLoading timing data...")
    all_results = load_timing_data(results_path, args.dist)

    # Extract and organize data
    print("Extracting timing data by theta value...")
    filters_order = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC', 'DR_EKF_CDC_FW']
    timing_data, theta_vals = extract_timing_vs_theta(all_results)

    print(f"Found {len(theta_vals)} theta values: {theta_vals}")

    # Print summary tables
    print_timing_summary(all_results)

    # Create plots
    print("\nCreating timing plots...")
    create_timing_plots(timing_data, filters_order, args.dist)

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {results_path}")
    print("Generated files:")
    print(f"  - timing_vs_robustness_{args.dist}.pdf (combined plot)")
    print(f"  - time_vs_robustness_{args.dist}.pdf (ms/step only)")
    print(f"  - frequency_vs_robustness_{args.dist}.pdf (Hz only)")
    print("=" * 80)

if __name__ == "__main__":
    main()
