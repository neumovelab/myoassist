# Author(s): Calder Robbins <robbins.cal@northeastern.edu>

import numpy as np

class OptimizationTracker:
    """
    Tracks costs during CMA-ES optimization.
    
    This class stores costs for each generation and provides methods
    to retrieve and analyze them.
    """
    
    def __init__(self):
        """Initialize an empty tracker."""
        self.costs_by_gen = []
        self.best_costs = []
        
    def add_generation(self, costs):
        """
        Add a new generation of costs.
        
        Args:
            costs (list): List of cost values for this generation
        """
        self.costs_by_gen.append(costs)
        self.best_costs.append(min(costs))
        
    def get_all_costs(self):
        """
        Get all costs from all generations.
        
        Returns:
            list: Flattened list of all costs
        """
        return [cost for gen in self.costs_by_gen for cost in gen]
    
    def get_recent_costs(self, num_generations=5):
        """
        Get costs from the most recent generations.
        
        Args:
            num_generations (int): Number of most recent generations to include
            
        Returns:
            list: Flattened list of costs from recent generations
        """
        if len(self.costs_by_gen) <= num_generations:
            return self.get_all_costs()
        else:
            return [cost for gen in self.costs_by_gen[-num_generations:] for cost in gen]
    
    def get_best_cost(self):
        """
        Get the overall best (minimum) cost.
        
        Returns:
            float: The minimum cost found so far
        """
        if not self.best_costs:
            return None
        return min(self.best_costs)
    
    def get_stats(self):
        """
        Get basic statistics about the optimization progress.
        
        Returns:
            dict: Dictionary containing statistics
        """
        if not self.best_costs:
            return {
                "num_generations": 0,
                "total_evaluations": 0,
                "best_cost": None,
                "mean_gen_size": 0
            }
            
        all_costs = self.get_all_costs()
        return {
            "num_generations": len(self.costs_by_gen),
            "total_evaluations": len(all_costs),
            "best_cost": min(self.best_costs),
            "mean_gen_size": len(all_costs) / len(self.costs_by_gen)
        } 