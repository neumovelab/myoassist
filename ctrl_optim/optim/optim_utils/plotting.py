# Author(s): Calder Robbins <robbins.cal@northeastern.edu>
"""
Visualization utilities for optimization results.

This module provides functions for plotting optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any
from .tracker import OptimizationTracker


def create_combined_plot(es: Any, save_path: str, trial_name: str, tracker: OptimizationTracker) -> None:
    """
    Create combined plot by overlaying custom plots on CMA-ES standard output.
    
    Args:
        es: CMAEvolutionStrategy object
        save_path: Path to save the plot
        trial_name: Name of the trial
        tracker: OptimizationTracker instance with cost data
    """
    # Create figure and plot CMA-ES output
    fig = plt.figure(figsize=(12, 8))
    es.logger.plot(fig=fig)

    rect = plt.Rectangle(
        (0.65, 0.0),   
        0.35, 1,      
        facecolor='white',
        edgecolor='white',
        transform=fig.transFigure,
        zorder=2,
        alpha=1.0
    )
    
    fig.patches.extend([rect])
    plt.draw()

    ax_heatmap = fig.add_axes([0.67, 0.1, 0.3, 0.8], zorder=3)
    ax_heatmap.set_facecolor('white')
    
    try:
        if not tracker.costs_by_gen:  # If no data
            print("No cost data available")
            ax_heatmap.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', fontsize=12)
        else:
            costs_per_gen = tracker.costs_by_gen
            num_generations = len(costs_per_gen)
            pop_size = es.popsize

            # Create a matrix to hold the costs, initialized with NaN
            heatmap_data = np.full((pop_size, num_generations), np.nan)

            for i, gen_costs in enumerate(costs_per_gen):
                # Sort costs for the current generation (ascending)
                sorted_costs = np.sort(np.array(gen_costs))
                
                current_len = len(sorted_costs)
                if current_len > pop_size:
                    sorted_costs = sorted_costs[:pop_size]
                    current_len = pop_size
                
                # Place sorted costs into the heatmap data matrix
                heatmap_data[:current_len, i] = sorted_costs
            
            if not np.isnan(heatmap_data).all():
                # Define boundaries for a non-linear color scale based on cost meaning
                boundaries = [
                    0, 40, 60, 80, 100,                          # Excellent to Good (greens)
                    130, 160, 200,                               # Moderate (yellows)
                    1000, 20000, 95000,                          # Poor (oranges)
                    96000, 97000, 98000, 99000, 100000,          # Early Termination (reds)
                    200000, 1200000                              # Simulation Error (dark reds)
                ]
                
                # Define a custom list of colors for each boundary interval
                colors = [
                    '#006400', '#228B22', '#32CD32', '#90EE90',                # Greens
                    '#FFFFE0', '#FFFACD', '#FFFF00',                             # Yellows
                    '#FFD700', '#FFA500', '#FF8C00',                             # Oranges
                    '#FF6347', '#FF4500', '#FF0000', '#DC143C', '#B22222',   # Reds
                    '#8B0000', "#420101"                                            # Dark Reds
                ]

                cmap_costs = plt.matplotlib.colors.ListedColormap(colors)
                norm = plt.matplotlib.colors.BoundaryNorm(boundaries, cmap_costs.N, clip=True)

                # Plot heatmap
                im = ax_heatmap.imshow(
                    heatmap_data, 
                    aspect='auto',
                    cmap=cmap_costs,
                    norm=norm,
                    interpolation='none'
                )
                
                cbar = fig.colorbar(im, ax=ax_heatmap, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Cost Value')
                
                ax_heatmap.set_title("Cost Distribution per Generation")
                ax_heatmap.set_xlabel("Generation")
                ax_heatmap.set_ylabel(f"Sorted Population Member (1-{pop_size})")

                if pop_size > 10:
                    ticks = [0, pop_size // 2, pop_size - 1]
                    labels = [1, (pop_size // 2) + 1, pop_size]
                    ax_heatmap.set_yticks(ticks)
                    ax_heatmap.set_yticklabels(labels)
                else:
                    ax_heatmap.set_yticks(np.arange(pop_size))
                    ax_heatmap.set_yticklabels(np.arange(1, pop_size + 1))
            else:
                ax_heatmap.text(0.5, 0.5, 'No valid cost data available', 
                              ha='center', va='center', fontsize=12)
                    
        # Add metadata using CMA-ES values
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        param_text = (f"Population Size: {es.popsize}\n"
                     f"Best Cost: {es.result.fbest:.2f}\n"
                     f"Total Evaluations: {es.countevals}\n"
                     f"Timestamp: {timestamp}")
        fig.text(0.99, 0.02, param_text, 
                fontsize=8, 
                ha='right', 
                va='bottom',
                transform=fig.transFigure,
                bbox=dict(facecolor='white', 
                         edgecolor='none', 
                         alpha=0.8))
            
    except Exception as e:
        print(f"Error in combined plot: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Save plot
    plot_filename = f'{trial_name}_combined_results.png'
    plt.savefig(f"{save_path}/{plot_filename}",
                dpi=300,
                bbox_inches='tight')
    
    pdf_filename = f'{trial_name}_optimization_results.pdf'
    plt.savefig(f"{save_path}/{pdf_filename}",
                format='pdf',
                bbox_inches='tight')
    plt.close() 