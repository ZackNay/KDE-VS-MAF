import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from maf_experiment import *
from kde_experiment import *
class MinimumDataSizeExperiment:
    def __init__(self, output_file="min_data_size_results.xlsx", seed=42):
        self.output_file = output_file
        self.seed = seed
        self.results = []
        
        # Define initial data sizes for each dimension
        self.initial_sizes = {
            1: 10,     # Dimension 1: Start with 10 samples
            2: 10,    
            3: 10,    
            4: 10,   
            5: 10, 
            6: 10,   
            7: 10,   
            8: 500   
        }
        
    def find_minimum_data_size(self, model_type, dimension, distribution_type, 
                               target_kl=0.5, max_size=100000000, 
                               growth_factor=1.5):
        """
        Finds the minimum data size needed to achieve KL divergence ≤ 0.5
        Uses a geometric growth approach to efficiently find the threshold
        """
        #Get the initial size based on dimension
        current_size = self.initial_sizes[dimension]
        
        while current_size <= max_size:
            print(f"Testing {model_type} on {distribution_type} distribution in {dimension}D with {current_size} samples")
            
            #Initialize the experiment type
            if model_type == "maf":
                experiment = ImprovedMultidimensionalComparisonExperiment(
                    output_file="temp_results.xlsx", seed=self.seed)
                
                # Configure model parameters with custom data size
                params = experiment.scale_hyperparameters(dimension)
                params['total_data_size'] = current_size
                
                #Create target distribution and generate data
                target = experiment.create_target_distribution(dimension, distribution_type)
                data_splits = experiment.generate_data_splits(dimension, current_size, target, distribution_type)
                
                #Create and train model
                model = experiment.create_flow_model(dimension, params, distribution_type)
                experiment.train_model(dimension, model, target, data_splits, params, distribution_type)
            
            else:  #KDE model
                experiment = KDEExperiment(output_file="temp_results.xlsx", seed=self.seed)
                
                #Create target distribution and generate data
                target = experiment.create_target(dimension, distribution_type)
                data_splits = experiment.generate_data_splits(dimension, current_size, target, distribution_type)
                
                #Create and train KDE model
                model = KDE(dim=dimension)
                experiment.train_and_select_bandwidth(dimension, model, data_splits, distribution_type)
            
            #Evaluate model to get KL divergence
            evaluation_results = experiment.evaluate_model(dimension, model, target, data_splits, distribution_type)
            kl_divergence = evaluation_results['kl_divergence']
            
            print(f"KL divergence: {kl_divergence}")
            
            #Check if target KL is achieved
            if kl_divergence <= target_kl:
                result = {
                    'model_type': model_type,
                    'distribution': distribution_type,
                    'dimension': dimension,
                    'data_size': current_size,
                    'kl_divergence': kl_divergence
                }
                self.results.append(result)
                self._save_result(result)
                return current_size
            
            #Increase data size and try again
            current_size = int(current_size * growth_factor)
        
        #If we couldn't achieve target KL with max_size
        print(f"Could not achieve KL ≤ {target_kl} with {max_size} samples")
        result = {
            'model_type': model_type,
            'distribution': distribution_type,
            'dimension': dimension,
            'data_size': None,  
            'kl_divergence': kl_divergence
        }
        self.results.append(result)
        self._save_result(result)
        return None
    
    def _save_result(self, result):
        """Save single result to the output file"""
        # Convert to DataFrame
        result_df = pd.DataFrame([result])
        if os.path.exists(self.output_file):
            #Update existing file
            existing_df = pd.read_excel(self.output_file)
            
            #Remove any existing results for this specific configuration
            mask = ((existing_df['model_type'] != result['model_type']) | 
                    (existing_df['distribution'] != result['distribution']) |
                    (existing_df['dimension'] != result['dimension']))
            filtered_df = existing_df[mask]
            
            #Append new result
            updated_df = pd.concat([filtered_df, result_df], ignore_index=True)
            updated_df.to_excel(self.output_file, index=False)
        else:
            #Create new file
            result_df.to_excel(self.output_file, index=False)
            
        print(f"Result saved for {result['model_type']}, {result['distribution']}, dimension {result['dimension']}")
            
    
    def run_all_experiments(self):
        """Run experiments for all combinations"""
        for model_type in ['kde','maf']:
            if model_type == 'kde':
                for distribution_type in ['gaussian', 'mixture', 'skewnormal']:
                    for dimension in range(1, 8):
                        print(f"\n{'='*60}")
                        print(f"Finding minimum data size for {model_type}, {distribution_type}, dimension {dimension}")
                        print(f"{'='*60}")
                        
                        min_size = self.find_minimum_data_size(
                            model_type, dimension, distribution_type)
                        
                        print(f"Minimum data size: {min_size}")

            else:
                for distribution_type in ['gaussian', 'mixture', 'skewnormal']:
                    for dimension in range(1, 9):
                        print(f"\n{'='*60}")
                        print(f"Finding minimum data size for {model_type}, {distribution_type}, dimension {dimension}")
                        print(f"{'='*60}")
                        
                        min_size = self.find_minimum_data_size(
                            model_type, dimension, distribution_type)
                        
                        print(f"Minimum data size: {min_size}")

    
    def visualize_results(self):
        """Create visualizations of the results"""
        df = pd.read_excel(self.output_file)
        
        #Create separate plots for each distribution type
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, dist in enumerate(['gaussian', 'mixture', 'skewnormal']):
            for model in ['maf', 'kde']:
                #Filter data for this model and distribution
                data = df[(df['model_type'] == model) & (df['distribution'] == dist)]
                
                if not data.empty:
                    #Sort by dimension
                    data = data.sort_values('dimension')
                    
                    #Plot data size vs dimension
                    axs[i].plot(data['dimension'], data['data_size'], 'o-', 
                               label=f"{model.upper()}")
            
            axs[i].set_title(f"{dist.capitalize()} Distribution")
            axs[i].set_xlabel("Dimension")
            axs[i].set_ylabel(f'Minimum Data Size for KL ≤ 0.5')
            axs[i].set_yscale('log')
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('minimum_data_size_comparison.png', dpi=300)
        plt.show()
        
        #Create a plot showing data size growth by dimension for both models
        plt.figure(figsize=(10, 7))
        
        markers = ['o', 's', '^']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, dist in enumerate(['gaussian', 'mixture', 'skewnormal']):
            for j, model in enumerate(['maf', 'kde']):
                #Filter data for this model and distribution
                data = df[(df['model_type'] == model) & (df['distribution'] == dist)]
                
                if not data.empty:
                    #Sort by dimension
                    data = data.sort_values('dimension')
                    
                    #Plot data size vs dimension with distinct line style for model type
                    linestyle = '-' if model == 'maf' else '--'
                    plt.plot(data['dimension'], data['data_size'], 
                            linestyle=linestyle, marker=markers[i], color=colors[i],
                            label=f"{model.upper()} - {dist.capitalize()}")
        
        plt.xlabel("Dimension")
        plt.ylabel("Minimum Data Size for KL ≤ 0.5")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('data_size_growth_comparison.png', dpi=300)
        plt.show()
        
        #Create a summary table showing the ratio between models
        print("\nSummary Table - Ratio of Data Sizes (KDE/MAF)")
        summary_table = pd.pivot_table(df, values='data_size', 
                                     index=['dimension'], 
                                     columns=['distribution', 'model_type'])
        
        #Calculate ratio of KDE to MAF data size
        for dist in ['gaussian', 'mixture', 'skewnormal']:
            summary_table[(dist, 'ratio')] = summary_table[(dist, 'kde')] / summary_table[(dist, 'maf')]
        
        print(summary_table)

if __name__ == "__main__":
    experiment = MinimumDataSizeExperiment()
    experiment.run_all_experiments()
    experiment.visualize_results()