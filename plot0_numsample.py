#!/usr/bin/env python3
"""
Plot showing the effect of num_samples on optimal performance for EKF vs DR-EKF filters.
X-axis: num_samples values [3, 5, 10, 15, 20]
Y-axis: Optimal MSE and Tracking Cost
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os

plt.rcParams.update({
    'font.size': 24,           # Base font size
    'axes.titlesize': 28,      # Title font size
    'axes.labelsize': 32,      # Axis label font size
    'xtick.labelsize': 26,     # X-axis tick label size
    'ytick.labelsize': 26,     # Y-axis tick label size
    'legend.fontsize': 26,     # Legend font size
    'figure.titlesize': 30     # Figure title size
})

def get_color_for_filter(filt, i):
    """Color mapping for EKF vs DR-EKF filters."""
    bright_palette = [
        "#1f77b4",  # strong blue
        "#ff7f0e",  # vivid orange  
        "#2ca02c",  # rich green
        "#d62728",  # deep red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf"   # cyan
    ]
    
    # Map specific filters to consistent colors
    color_map = {
        'EKF': bright_palette[0],           # Blue
        'DR_EKF_CDC': bright_palette[1],    # Orange
        'DR_EKF_TAC': bright_palette[2],    # Green
    }
    
    return color_map.get(filt, bright_palette[i % len(bright_palette)])

def load_data(file_path):
    """Load pickled data from file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_numsample_effect_mse_plot(all_numsample_results, dist, filters, filter_labels):
    """Create plot showing effect of num_samples on optimal MSE for each filter"""
    
    # Create both linear and log scale plots
    for scale_type in ['linear', 'log']:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract num_samples values and sort them
        num_samples_values = sorted(all_numsample_results.keys())
        
        # Define markers for each method
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        # Define letter labels for each filter
        filter_letter_map = {
            'EKF': '(A)',
            'DR_EKF_CDC': '(B)', 
            'DR_EKF_TAC': '(C)',
        }
        
        # Plot each filter
        for i, filt in enumerate(filters):
            # Collect data points for this filter across num_samples values
            numsample_vals = []
            mse_vals = []
            mse_stds = []
            
            for num_samples in num_samples_values:
                if num_samples in all_numsample_results:
                    if filt in all_numsample_results[num_samples]:
                        numsample_vals.append(num_samples)
                        mse_vals.append(all_numsample_results[num_samples][filt]['mse_mean'])
                        mse_stds.append(all_numsample_results[num_samples][filt]['mse_std'])
            
            # Skip this filter if no data points available
            if not mse_vals:
                print(f"Warning: No data available for filter '{filt}' - skipping from num_samples MSE effect plot")
                continue
            
            letter_label = filter_letter_map.get(filt, '(?)')
            label = f"{letter_label} {filter_labels[filt]}"
            
            # Plot with markers and error bars
            ax.errorbar(numsample_vals, mse_vals, yerr=mse_stds,
                       marker=markers[i % len(markers)], 
                       markerfacecolor='white',
                       markeredgecolor=get_color_for_filter(filt, i),
                       color=get_color_for_filter(filt, i),
                       markeredgewidth=1.2,
                       linestyle='-',
                       linewidth=2.5,
                       markersize=12,
                       capsize=5,
                       capthick=1.5,
                       label=label)
        
        # Customize plot
        ax.set_xlabel('Number of Samples')
        if scale_type == 'log':
            ax.set_ylabel('Average MSE (log scale)', fontsize=28, labelpad=15)
            ax.set_yscale('log')
        else:
            ax.set_ylabel('Average MSE', fontsize=28, labelpad=15)
        
        ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
        if scale_type == 'log':
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='both', which='major', width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=4)
        
        # Set x-axis to show only the num_samples values we have
        ax.set_xticks(num_samples_values)
        
        # Set x-axis limits with appropriate padding
        x_min, x_max = min(num_samples_values), max(num_samples_values)
        x_range = x_max - x_min
        padding = max(1, x_range * 0.05)
        ax.set_xlim(max(0, x_min - padding), x_max + padding)
        
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(filters), frameon=False)
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.2, left=0.12, top=0.95, right=0.98)
        
        # Ensure results directory exists
        results_path = f"./results/numsample_study/"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        output_path = os.path.join(results_path, f'numsample_effect_mse_{dist}_{scale_type}.pdf')
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"num_samples MSE effect plot ({scale_type} scale) saved as: {output_path}")

def create_numsample_effect_cost_plot(all_numsample_results, dist, filters, filter_labels):
    """Create plot showing effect of num_samples on optimal tracking cost for each filter"""
    
    # Create both linear and log scale plots
    for scale_type in ['linear', 'log']:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract num_samples values and sort them
        num_samples_values = sorted(all_numsample_results.keys())
        
        # Define markers for each method
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        # Define letter labels for each filter
        filter_letter_map = {
            'EKF': '(A)',
            'DR_EKF_CDC': '(B)', 
            'DR_EKF_TAC': '(C)',
        }
        
        # Plot each filter
        for i, filt in enumerate(filters):
            # Collect data points for this filter across num_samples values
            numsample_vals = []
            cost_vals = []
            cost_stds = []
            
            for num_samples in num_samples_values:
                if num_samples in all_numsample_results:
                    if filt in all_numsample_results[num_samples]:
                        numsample_vals.append(num_samples)
                        cost_vals.append(all_numsample_results[num_samples][filt]['cost_mean'])
                        cost_stds.append(all_numsample_results[num_samples][filt]['cost_std'])
            
            # Skip this filter if no data points available
            if not cost_vals:
                print(f"Warning: No data available for filter '{filt}' - skipping from num_samples cost effect plot")
                continue
            
            letter_label = filter_letter_map.get(filt, '(?)')
            label = f"{letter_label} {filter_labels[filt]}"
            
            # Plot with markers and error bars
            ax.errorbar(numsample_vals, cost_vals, yerr=cost_stds,
                       marker=markers[i % len(markers)], 
                       markerfacecolor='white',
                       markeredgecolor=get_color_for_filter(filt, i),
                       color=get_color_for_filter(filt, i),
                       markeredgewidth=1.2,
                       linestyle='-',
                       linewidth=2.5,
                       markersize=12,
                       capsize=5,
                       capthick=1.5,
                       label=label)
        
        # Customize plot
        ax.set_xlabel('Number of Samples')
        if scale_type == 'log':
            ax.set_ylabel('Average Tracking Cost (log scale)', fontsize=28, labelpad=15)
            ax.set_yscale('log')
        else:
            ax.set_ylabel('Average Tracking Cost', fontsize=28, labelpad=15)
        
        ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
        if scale_type == 'log':
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='both', which='major', width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=4)
        
        # Set x-axis to show only the num_samples values we have
        ax.set_xticks(num_samples_values)
        
        # Set x-axis limits with appropriate padding
        x_min, x_max = min(num_samples_values), max(num_samples_values)
        x_range = x_max - x_min
        padding = max(1, x_range * 0.05)
        ax.set_xlim(max(0, x_min - padding), x_max + padding)
        
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(filters), frameon=False)
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.2, left=0.12, top=0.95, right=0.98)
        
        # Ensure results directory exists
        results_path = f"./results/numsample_study/"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        output_path = os.path.join(results_path, f'numsample_effect_cost_{dist}_{scale_type}.pdf')
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"num_samples cost effect plot ({scale_type} scale) saved as: {output_path}")

def create_combined_numsample_effect_plot(normal_data, quadratic_data, filters, filter_labels):
    """Create side-by-side plots showing num_samples effect for normal and quadratic distributions"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), gridspec_kw={'hspace': 0.3, 'wspace': 0.25})
    
    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Define letter labels for each filter
    filter_letter_map = {
        'EKF': '(A)',
        'DR_EKF_CDC': '(B)', 
        'DR_EKF_TAC': '(C)',
    }
    
    # Function to create num_samples effect plot for a single distribution
    def plot_distribution_numsample_effect(ax, all_numsample_results, dist_name, metric):
        # Extract num_samples values and sort them
        num_samples_values = sorted(all_numsample_results.keys())
        
        # Plot each filter
        for i, filt in enumerate(filters):
            # Collect data points for this filter across num_samples values
            numsample_vals = []
            metric_vals = []
            metric_stds = []
            
            for num_samples in num_samples_values:
                if num_samples in all_numsample_results:
                    if filt in all_numsample_results[num_samples]:
                        numsample_vals.append(num_samples)
                        if metric == 'mse':
                            metric_vals.append(all_numsample_results[num_samples][filt]['mse_mean'])
                            metric_stds.append(all_numsample_results[num_samples][filt]['mse_std'])
                        else:  # cost
                            metric_vals.append(all_numsample_results[num_samples][filt]['cost_mean'])
                            metric_stds.append(all_numsample_results[num_samples][filt]['cost_std'])
            
            # Skip this filter if no data points available
            if not metric_vals:
                print(f"Warning: No data available for filter '{filt}' in {dist_name} - skipping")
                continue
            
            letter_label = filter_letter_map.get(filt, '(?)')
            label = f"{letter_label} {filter_labels[filt]}"
            
            # Plot with markers and error bars
            ax.errorbar(numsample_vals, metric_vals, yerr=metric_stds,
                       marker=markers[i % len(markers)], 
                       markerfacecolor='white',
                       markeredgecolor=get_color_for_filter(filt, i),
                       color=get_color_for_filter(filt, i),
                       markeredgewidth=1.2,
                       linestyle='-',
                       linewidth=2.5,
                       markersize=12,
                       capsize=5,
                       capthick=1.5,
                       label=label)
        
        # Customize plot
        ax.set_xlabel('Number of Samples')
        if metric == 'mse':
            ax.set_ylabel('Average MSE', fontsize=24, labelpad=15)
        else:
            ax.set_ylabel('Average Tracking Cost', fontsize=24, labelpad=15)
        ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
        ax.tick_params(axis='both', which='major', width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=4)
        
        # Set x-axis to show only the num_samples values we have
        ax.set_xticks(num_samples_values)
        
        # Set x-axis limits with appropriate padding
        x_min, x_max = min(num_samples_values), max(num_samples_values)
        x_range = x_max - x_min
        padding = max(1, x_range * 0.05)
        ax.set_xlim(max(0, x_min - padding), x_max + padding)
        
        # Set title
        if metric == 'mse':
            ax.set_title(f'MSE - {dist_name.title()} Distribution', fontsize=22)
        else:
            ax.set_title(f'Tracking Cost - {dist_name.title()} Distribution', fontsize=22)
    
    # Plot MSE for normal distribution (top left)
    plot_distribution_numsample_effect(ax1, normal_data, 'normal', 'mse')
    
    # Plot MSE for quadratic distribution (top right)  
    plot_distribution_numsample_effect(ax2, quadratic_data, 'quadratic', 'mse')
    
    # Plot cost for normal distribution (bottom left)
    plot_distribution_numsample_effect(ax3, normal_data, 'normal', 'cost')
    
    # Plot cost for quadratic distribution (bottom right)
    plot_distribution_numsample_effect(ax4, quadratic_data, 'quadratic', 'cost')
    
    # Add subplot labels a), b), c), d)
    ax1.text(0.02, 0.95, 'a)', transform=ax1.transAxes, fontsize=24, ha='left', va='top', weight='bold')
    ax2.text(0.02, 0.95, 'b)', transform=ax2.transAxes, fontsize=24, ha='left', va='top', weight='bold')
    ax3.text(0.02, 0.95, 'c)', transform=ax3.transAxes, fontsize=24, ha='left', va='top', weight='bold')
    ax4.text(0.02, 0.95, 'd)', transform=ax4.transAxes, fontsize=24, ha='left', va='top', weight='bold')
    
    # Create a shared legend at the top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.98), loc='upper center', ncol=len(filters), frameon=False, fontsize=22)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)
    
    # Ensure results directory exists
    results_path = "./results/numsample_study/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, 'combined_numsample_effect_normal_quadratic.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined num_samples effect plot saved as: {output_path}")

def load_numsample_results(dist):
    """Load num_samples study results from the summary file"""
    results_path = f"./results/numsample_study/"
    
    # Try to load the summary file first (organized by filter)
    summary_file = os.path.join(results_path, f'summary_results_vs_numsamples_{dist}.pkl')
    if os.path.exists(summary_file):
        summary_data = load_data(summary_file)
        # Convert to expected format (organized by num_samples)
        all_numsample_results = {}
        
        # Get num_samples values from first filter
        first_filter = list(summary_data.keys())[0]
        if 'num_samples' in summary_data[first_filter]:
            num_samples_values = summary_data[first_filter]['num_samples']
            
            for i, num_samples in enumerate(num_samples_values):
                all_numsample_results[num_samples] = {}
                for filter_name in summary_data:
                    if i < len(summary_data[filter_name]['mse_mean']):
                        all_numsample_results[num_samples][filter_name] = {
                            'mse_mean': summary_data[filter_name]['mse_mean'][i],
                            'mse_std': summary_data[filter_name]['mse_std'][i],
                            'cost_mean': summary_data[filter_name]['cost_mean'][i],
                            'cost_std': summary_data[filter_name]['cost_std'][i],
                            'theta': summary_data[filter_name]['optimal_theta'][i]
                        }
            
            print(f"Loaded summary results for {dist}: {sorted(all_numsample_results.keys())}")
            return all_numsample_results
    
    # Try to load the optimal results file (organized by num_samples)
    optimal_file = os.path.join(results_path, f'optimal_results_vs_numsamples_{dist}.pkl')
    if os.path.exists(optimal_file):
        all_numsample_results = load_data(optimal_file)
        print(f"Loaded optimal results for {dist}: {sorted(all_numsample_results.keys())}")
        return all_numsample_results
    
    raise FileNotFoundError(f"No num_samples results found in {results_path}. Make sure you've run main0_numsample.py first.")


def create_combined_plots():
    """Create combined plots comparing normal and quadratic distributions"""
    
    # Load data for both distributions
    try:
        normal_data = load_numsample_results('normal')
        quadratic_data = load_numsample_results('quadratic')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run main0_numsample.py for both normal and quadratic distributions.")
        return
    
    # Get filters from the loaded results
    filters = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC']
    
    # Use filters that are available in both distributions
    available_filters = []
    for f in filters:
        found_normal = any(f in normal_data[ns] for ns in normal_data)
        found_quadratic = any(f in quadratic_data[ns] for ns in quadratic_data)
        if found_normal and found_quadratic:
            available_filters.append(f)
    
    filter_labels = {
        'EKF': "Extended Kalman Filter",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)",
    }
    
    print("Creating combined num_samples effect plots for normal vs quadratic distributions...")
    print(f"Available filters: {available_filters}")
    
    # Create the combined num_samples effect plots
    create_combined_numsample_effect_plot(normal_data, quadratic_data, available_filters, filter_labels)
    
    print("Combined plots created successfully!")

def main(dist):
    """Main function to create num_samples effect plots
    
    Args:
        dist: Distribution type ('normal' or 'quadratic')
    """
    
    # Load num_samples study results
    try:
        all_numsample_results = load_numsample_results(dist)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Get filters from the results
    filters = ['EKF', 'DR_EKF_CDC', 'DR_EKF_TAC']
    
    # Only use filters that have results
    available_filters = []
    for f in filters:
        found = any(f in all_numsample_results[ns] for ns in all_numsample_results)
        if found:
            available_filters.append(f)
    
    filter_labels = {
        'EKF': "Extended Kalman Filter",
        'DR_EKF_CDC': "DR-EKF (CDC)",
        'DR_EKF_TAC': "DR-EKF (TAC)",
    }
    
    print(f"Creating num_samples effect plots for {dist} distribution...")
    print(f"Available filters: {available_filters}")
    print(f"num_samples values: {sorted(all_numsample_results.keys())}")
    
    # Create num_samples effect plots (both linear and log scale)
    create_numsample_effect_mse_plot(all_numsample_results, dist, available_filters, filter_labels)
    create_numsample_effect_cost_plot(all_numsample_results, dist, available_filters, filter_labels)
    
    print(f"num_samples effect plots created successfully for {dist} distribution!")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("NUM_SAMPLES EFFECT SUMMARY STATISTICS")
    print("="*80)
    
    num_samples_values = sorted(all_numsample_results.keys())
    
    print(f"\nEffect of num_samples on Optimal MSE ({dist} distribution):")
    print(f"{'Filter':<20} " + " ".join([f"N={n:<8}" for n in num_samples_values]))
    print("-" * (20 + 10 * len(num_samples_values)))
    
    for filt in available_filters:
        mse_str = f"{filter_labels[filt]:<20}"
        for num_samples in num_samples_values:
            if num_samples in all_numsample_results and filt in all_numsample_results[num_samples]:
                mse_val = all_numsample_results[num_samples][filt]['mse_mean']
                mse_str += f" {mse_val:<8.4f}"
            else:
                mse_str += f" {'N/A':<8}"
        print(mse_str)
    
    print(f"\nEffect of num_samples on Optimal Tracking Cost ({dist} distribution):")
    print(f"{'Filter':<20} " + " ".join([f"N={n:<8}" for n in num_samples_values]))
    print("-" * (20 + 10 * len(num_samples_values)))
    
    for filt in available_filters:
        cost_str = f"{filter_labels[filt]:<20}"
        for num_samples in num_samples_values:
            if num_samples in all_numsample_results and filt in all_numsample_results[num_samples]:
                cost_val = all_numsample_results[num_samples][filt]['cost_mean']
                cost_str += f" {cost_val:<8.2f}"
            else:
                cost_str += f" {'N/A':<8}"
        print(cost_str)
    
    print(f"\nOptimal robustness parameters (θ) for each num_samples ({dist} distribution):")
    print(f"{'Filter':<20} " + " ".join([f"N={n:<8}" for n in num_samples_values]))
    print("-" * (20 + 10 * len(num_samples_values)))
    
    for filt in available_filters:
        theta_str = f"{filter_labels[filt]:<20}"
        for num_samples in num_samples_values:
            if num_samples in all_numsample_results and filt in all_numsample_results[num_samples]:
                theta_val = all_numsample_results[num_samples][filt]['theta']
                theta_str += f" {theta_val:<8.2f}"
            else:
                theta_str += f" {'N/A':<8}"
        print(theta_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create num_samples effect plots from main0_numsample.py results")
    parser.add_argument('--dist', default="normal", type=str,
                        help="Distribution type (normal or quadratic)")
    parser.add_argument('--combined', action='store_true',
                        help="Create combined plots comparing normal vs quadratic distributions")
    
    args = parser.parse_args()
    
    if args.combined:
        create_combined_plots()
    else:
        main(args.dist)