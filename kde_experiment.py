import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
import time
from scipy import stats
from scipy.optimize import minimize_scalar


#KDE target distribution classes 

class KDEGaussianDistribution:
    """Standard multivariate Gaussian distribution with diagonal covariance."""
    def __init__(self, dim, mean=None, std=None):
        """
        Initialize multivariate Gaussian distribution.
        
        Args:
            dim: Dimension of the distribution
            mean: Mean vector (default: zeros). Can be scalar or vector.
            std: Standard deviation (default: ones). Can be scalar or vector.
                 If scalar, same std is used for all dimensions.
                 If vector, each dimension gets its own std.
        """
        self.dim = dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #mean
        if mean is None:
            self.mean = torch.zeros(dim, device=self.device)
        elif isinstance(mean, (int, float)):
            self.mean = torch.full((dim,), float(mean), device=self.device)
        else:
            self.mean = mean.to(self.device)
            
        #standard deviation
        if std is None:
            self.std = torch.ones(dim, device=self.device)
        elif isinstance(std, (int, float)):
            self.std = torch.full((dim,), float(std), device=self.device)
        else:
            self.std = std.to(self.device)
            
        #Convert std to covariance matrix (diagonal)
        self.cov = torch.diag(self.std ** 2)
        
        #Compute Cholesky decomposition for sampling
        #For diagonal matrix, L is just diag(std)
        self.L = torch.diag(self.std)
        
    def sample(self, n_samples):
        """Generate samples from the distribution."""
        z = torch.randn(n_samples, self.dim, device=self.device)
        #Efficient sampling for diagonal covariance
        samples = z * self.std + self.mean
        return samples
    
    def log_prob(self, x):
        """Compute log probability"""
        x = x.to(self.device)
        mvn = torch.distributions.MultivariateNormal(self.mean, self.cov)
        return mvn.log_prob(x)


class KDEMixtureOfGaussians:
    """Mixture of Gaussians distribution."""
    def __init__(self, dim, num_components=3, means=None, stds=None, weights=None):
        self.dim = dim
        self.num_components = num_components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #Initialize component parameters
        if means is None:
            self.means = []
            for i in range(num_components):
                offset = 3.0 * (i - (num_components - 1) / 2)
                mean = torch.zeros(dim, device=self.device)
                mean[0] = offset  
                self.means.append(mean)
        else:
            self.means = [m.to(self.device) for m in means]
            
        if stds is None:
            self.stds = [torch.ones(dim, device=self.device) * 0.5 for _ in range(num_components)]
        else:
            #Handle both tensor and list inputs
            if isinstance(stds, torch.Tensor):
                self.stds = [stds[i].to(self.device) for i in range(num_components)]
            else:
                self.stds = [s.to(self.device) for s in stds]
            
        #Convert stds to covariance matrices
        self.covs = [torch.diag(std**2) for std in self.stds]
            
        if weights is None:
            self.weights = torch.ones(num_components, device=self.device) / num_components
        else:
            self.weights = weights.to(self.device)
            
        #Create component distributions
        self.components = [
            torch.distributions.MultivariateNormal(mean, cov)
            for mean, cov in zip(self.means, self.covs)
        ]
        
    def sample(self, n_samples):
        """Generate samples from the mixture."""
        #Sample component assignments
        categorical = torch.distributions.Categorical(self.weights)
        assignments = categorical.sample((n_samples,))
        
        samples = torch.zeros(n_samples, self.dim, device=self.device)
        
        #Sample from each component
        for i in range(self.num_components):
            mask = (assignments == i)
            n_comp = mask.sum().item()
            if n_comp > 0:
                samples[mask] = self.components[i].sample((n_comp,))
                
        return samples
    
    def log_prob(self, x):
        """Compute log probability of samples."""
        x = x.to(self.device)
        log_probs = torch.zeros(x.shape[0], self.num_components, device=self.device)
        
        #Compute log prob for each component
        for i in range(self.num_components):
            log_probs[:, i] = self.components[i].log_prob(x) + torch.log(self.weights[i])
            
        #Log-sum-exp for numerical stability
        return torch.logsumexp(log_probs, dim=1)



class KDESkewNormal:
    """Skew-normal distribution"""
    def __init__(self, dim, loc=None, scale=None, skewness=None):
        self.dim = dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if loc is None:
            self.loc = torch.zeros(dim, device=self.device)
        else:
            self.loc = loc.to(self.device)
            
        if scale is None:
            self.scale = torch.ones(dim, device=self.device)
        else:
            self.scale = scale.to(self.device)
            
        if skewness is None:
            self.skewness = torch.ones(dim, device=self.device) * 2.0
        else:
            self.skewness = skewness.to(self.device)
    
    def sample(self, n_samples):
        """Generate samples from skew-normal distribution."""
        samples = torch.zeros(n_samples, self.dim, device=self.device)
        
        #Sample dimension by dimension
        for d in range(self.dim):
            # Use rejection sampling or approximation
            # Simple approach: use transformation method
            u1 = torch.randn(n_samples, device=self.device)
            u2 = torch.randn(n_samples, device=self.device)
            
            #Owen's method
            delta = self.skewness[d] / torch.sqrt(1 + self.skewness[d]**2)
            z = delta * torch.abs(u1) + torch.sqrt(1 - delta**2) * u2
            samples[:, d] = self.loc[d] + self.scale[d] * z
            
        return samples
    
    def log_prob(self, x):
        """Compute log probability of samples (approximation)."""
        x = x.to(self.device)
        #Standardize
        z = (x - self.loc) / self.scale
        
        normal = torch.distributions.Normal(0, 1)
        log_prob = torch.sum(
            normal.log_prob(z) - torch.log(self.scale) + 
            torch.log(2 * normal.cdf(self.skewness * z)),
            dim=1
        )
        
        return log_prob



class KDE:
    """
    Kernel Density Estimation for various distributions.
    Uses the Epanechnikov kernel for density estimation.
    """
    def __init__(self, dim=2, bandwidth=None):
        """
        Initialize the KDE model.
        
        Parameters:
        dim (int): Dimension of the data
        bandwidth (float or torch.Tensor): Bandwidth parameter for KDE (can be scalar or vector)
        """
        self.dim = dim
        self.bandwidth = bandwidth  
        self.train_data = None
        self.fitted = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def epanechnikov_kernel(self, distances):

        mask = (torch.abs(distances) <= 1)
        
        #Calculate kernel values
        kernel_values = torch.zeros_like(distances)
        kernel_values[mask] = 0.75 * (1 - distances[mask]**2)
        
        return kernel_values
    
    def fit(self, train_data):
        """
        Fit the KDE model with training data.
        
        Parameters:
        train_data (torch.Tensor): Training data points
        """

        self.train_data = train_data.to(self.device)
        self.n_samples = train_data.shape[0]
        self.fitted = True
        
        #If bandwidth wasn't specified, use Silverman's rule
        if self.bandwidth is None:
            # Rule of thumb initial bandwidth (Silverman's rule adapted for Epanechnikov)
            std = torch.std(self.train_data, dim=0).mean()
            self.bandwidth = 1.06 * std * (self.n_samples**(-1/(self.dim+4)))
            
        return self
    
    def select_bandwidth(self):
        """
        Select optimal bandwidth using Leave-One-Out Maximum Likelihood Cross-Validation (LOO-MLCV).
        This is the most accurate method for bandwidth selection, prioritizing accuracy over speed.
        
        Parameters:
        val_data: Not used in LOO-CV, kept for compatibility
        bandwidths: Not used, kept for compatibility
        
        Returns:
        float: Optimal bandwidth
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before bandwidth selection")
        
        print("Using Leave-One-Out Maximum Likelihood Cross-Validation")
        print(f"Dataset size:{self.n_samples} - Dimension:{self.dim}")
        
        n = self.n_samples
        
        def loo_mlcv_score(log_h):
            """
            Leave-one-out maximum likelihood cross-validation score.
            """
            if isinstance(log_h, np.ndarray):
                log_h = log_h.item()
            
            h = torch.exp(torch.tensor(log_h, dtype=torch.float32, device=self.device))
            
            #Initialize log-likelihood sum
            log_likelihood = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            
            #Compute LOO density estimate for each point
            for i in range(n):
                #Current point
                x_i = self.train_data[i:i+1]
                
                #Compute distances to all other points
                mask = torch.ones(n, dtype=torch.bool, device=self.device)
                mask[i] = False
                x_rest = self.train_data[mask]
                
                #Compute pairwise distances
                distances = torch.norm(x_i - x_rest, dim=1) / h
                
                #Apply Epanechnikov kernel
                kernel_vals = self.epanechnikov_kernel(distances)
                
                #Compute leave-one-out density estimate at x_i
                density = kernel_vals.sum() / ((n - 1) * h**self.dim)
                if density > 0:
                    log_likelihood += torch.log(density)
                else:
                    # Handle zero density with a large negative value
                    log_likelihood += torch.tensor(-1000.0, device=self.device)
            
            score = -log_likelihood.item() / n
            return float(score) 
        
        #Determine initial bandwidth band
        
        #Silverman's rule
        std = torch.std(self.train_data, dim=0).mean().item()
        h_silverman = 1.06 * std * (n**(-1/(self.dim+4)))
        
        #Scott's rule
        h_scott = std * (n**(-1/(self.dim+4)))
        
        #Estimate using IQR
        if self.dim == 1:
            q75 = torch.quantile(self.train_data, 0.75)
            q25 = torch.quantile(self.train_data, 0.25)
            iqr = (q75 - q25).item()
            h_iqr = 0.9 * min(std, iqr/1.34) * (n**(-0.2))
        else:
            #For multivariate, use average IQR across dimensions
            iqr_sum = 0
            for d in range(self.dim):
                q75 = torch.quantile(self.train_data[:, d], 0.75)
                q25 = torch.quantile(self.train_data[:, d], 0.25)
                iqr_sum += (q75 - q25).item()
            iqr_avg = iqr_sum / self.dim
            h_iqr = 0.9 * min(std, iqr_avg/1.34) * (n**(-1/(self.dim+4)))
        
        initial_bandwidths = [h_silverman, h_scott, h_iqr]
        print(f"Initial bandwidth estimates: Silverman={h_silverman:.5f}, Scott={h_scott:.5f}, IQR-based={h_iqr:.5f}")
        
        #Coarse grid search around initial estimates
        print("Coarse grid search")
        h_min = 0.1 * min(initial_bandwidths)
        h_max = 10 * max(initial_bandwidths)
        
        #Create log-spaced grid
        n_grid = 20  
        grid_log_h = torch.linspace(torch.log(torch.tensor(h_min)), 
                                    torch.log(torch.tensor(h_max)), 
                                    n_grid)
        
        best_score = float('inf')
        best_log_h_coarse = None
        
        for log_h in grid_log_h:
            score = loo_mlcv_score(log_h.item())
            if score < best_score:
                best_score = score
                best_log_h_coarse = log_h.item()
        
        print(f"Coarse grid search: best log(h)={best_log_h_coarse:.5f}, score={-best_score:.5f}")
        
        # Fine grid search around the best coarse result
        print("Performing fine grid search")
        fine_range = 0.5  # Search within ±0.5 in log space
        fine_log_h_min = best_log_h_coarse - fine_range
        fine_log_h_max = best_log_h_coarse + fine_range
        
        n_fine_grid = 30  # Fine grid
        fine_grid_log_h = torch.linspace(torch.tensor(fine_log_h_min), 
                                        torch.tensor(fine_log_h_max), 
                                        n_fine_grid)
        
        best_score = float('inf')
        best_log_h_fine = None
        
        for log_h in fine_grid_log_h:
            score = loo_mlcv_score(log_h.item())
            if score < best_score:
                best_score = score
                best_log_h_fine = log_h.item()
        
        print(f"Fine grid search: best log(h)={best_log_h_fine:.5f}, score={-best_score:.5f}")
        
        #Final optimization using scipy's minimize
        print("Performing final optimization")
        from scipy.optimize import minimize
        
        #Try multiple starting points
        starting_points = [
            best_log_h_fine,
            np.log(h_silverman),
            np.log(h_scott),
            np.log(h_iqr)
        ]
        
        best_result = None
        best_final_score = float('inf')
        
        for start_point in starting_points:
            try:
                #Ensure bounds are Python floats
                bounds = [(float(fine_log_h_min), float(fine_log_h_max))]
                
                result = minimize(
                    loo_mlcv_score,
                    x0=float(start_point),  
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'ftol': 1e-8,
                        'gtol': 1e-8,
                        'maxiter': 100
                    }
                )
                
                if result.fun < best_final_score:
                    best_final_score = result.fun
                    best_result = result
                    
            except Exception as e:
                print(f"Optimization from starting point: {start_point} failed: {e}")
                continue
        
        if best_result is None:
            #Fallback to best grid search result
            optimal_log_h = best_log_h_fine
            optimal_score = -best_score
        else:
            optimal_log_h = best_result.x[0]
            optimal_score = -best_result.fun
        
        #Convert back to bandwidth
        optimal_bandwidth = torch.exp(torch.tensor(optimal_log_h, device=self.device))
        self.bandwidth = optimal_bandwidth
        
        print(f"Optimal bandwidth selected: {optimal_bandwidth.item():.6f}")
        print(f"Final LOO-MLCV score: {optimal_score:.6f}")
        
        return optimal_bandwidth
    
    def score(self, data):
        """
        Compute the average log-likelihood score of the model on given data.
        """
        log_probs = self.log_prob(data)
        return torch.mean(log_probs).item()
    
    def log_prob(self, data):
        """
        Compute log probability density at given points.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before computing log_prob")

        data = data.to(self.device)
        batch_size = data.shape[0]
        log_probs = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            x = data[i].unsqueeze(0)  # Shape: [1, dim]
            
            #Compute distances to all training points
            h = self.bandwidth
            diff = x - self.train_data 
            distances = torch.norm(diff, dim=1) / h
            
            #Apply kernel function to distances
            kernel_values = self.epanechnikov_kernel(distances)
            
            #Compute density estimate
            density = torch.sum(kernel_values) / (self.n_samples * h**self.dim)
            
            #Handle numerical stability
            density = torch.clamp(density, min=1e-10)
            log_probs[i] = torch.log(density)
        
        return log_probs
    
    def sample(self, num_samples):
        """
        Generate samples from the density.
        Uses a kernel density sampling approach.
        
        Parameters:
        num_samples (int): Number of samples to generate
        
        Returns:
        torch.Tensor: Samples from the distribution
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
            
        samples = torch.zeros(num_samples, self.dim, device=self.device)

        for i in range(num_samples):
            #Randomly select a training point
            idx = torch.randint(0, self.n_samples, (1,)).item()
            base_sample = self.train_data[idx]
            
            # For Epanechnikov kernel, sample from the kernel distribution
            # Generate uniform random vector and scale appropriately
            u = torch.rand(self.dim, device=self.device)
            # Transform uniform to Epanechnikov distribution
            # Using inverse transform sampling for radial distance
            r = torch.rand(1, device=self.device).pow(1/(self.dim+2))
            direction = torch.randn(self.dim, device=self.device)
            direction = direction / torch.norm(direction)
            
            noise = r * direction * self.bandwidth
            samples[i] = base_sample + noise
        
        return samples




class KDEExperiment:
    def __init__(self, output_file="kde_results.xlsx", seed=42):
        """
        Initialize the experiment for KDE across dimensions.
        
        Parameters:
        output_file (str): Path to the Excel file where results will be saved
        seed (int): Random seed for reproducibility
        """
        self.output_file = output_file
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distribution_types = ['gaussian', 'mixture', 'skewnormal']
        self.seed = seed
        
        #Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
         
        self.train_ratio = 0.85
        self.test_ratio = 0.15
        
        #Create the output file if it doesn't exist
        if not os.path.exists(output_file):
            results_df = pd.DataFrame()
            results_df.to_excel(output_file, index=False)

    
    def create_target(self, dim, distribution_type='gaussian'):
        if distribution_type == 'gaussian':
            #Create a Gaussian with deterministic mean and std
            mean = torch.linspace(-1.0, 1.0, dim)  # Evenly spaced means from -1 to 1
            std = torch.ones(dim) * 1.5  # Fixed std of 1.5 for all dimensions
            target = KDEGaussianDistribution(dim=dim, mean=mean, std=std)
            
        elif distribution_type == 'mixture':
            num_components = min(3, dim)  
            
            means = []
            for i in range(num_components):
                mean_offset = 3.0 * (i - (num_components - 1) / 2)
                mean = torch.ones(dim) * mean_offset 
                means.append(mean)
            
            stds = torch.ones(num_components, dim) * 0.7  # Fixed std of 0.7
            

            weights = torch.ones(num_components) / num_components
            
            target = KDEMixtureOfGaussians(dim=dim, num_components=num_components, 
                                        means=means, stds=stds, weights=weights)
            
        elif distribution_type == 'skewnormal':
            loc = torch.zeros(dim)  
            scale = torch.ones(dim) * 1.2 
            skewness = torch.linspace(0.5, 2.5, dim) 
            target = KDESkewNormal(dim=dim, loc=loc, scale=scale, skewness=skewness)
            
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
            
        return target
        
    def generate_data_splits(self, dim, total_size, target, distribution_type='gaussian'):
        """
        Generate data with train/validation/test splits.
        """
        device = self.device
        #Set seed
        data_seed = self.seed + hash(distribution_type) % 10000 + dim * 100
        torch.manual_seed(data_seed)
        
        #Calculate split sizes
        train_size = int(total_size * self.train_ratio)
        test_size = total_size - train_size
        
        #Sample from target distribution for each split
        train_data = target.sample(train_size)
        test_data = target.sample(test_size)
        
        #Reset seed
        torch.manual_seed(self.seed)
        
        #Return all splits
        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_size': train_size,
            'test_size': test_size
        }
    
    def train_and_select_bandwidth(self, dim, model, data_splits, distribution_type='gaussian'):
        """
        Train the KDE model and select optimal bandwidth using validation data.
        """
        #Set seed for reproducibility
        train_seed = self.seed + hash(distribution_type) % 10000 + dim * 100 + 1
        torch.manual_seed(train_seed)
        
        #Unpack data splits
        train_data = data_splits['train_data']
        
        start_time = time.time()
        
        #Fit the KDE model on training data
        print(f"Fitting KDE model for {dim}D ({distribution_type})")
        model.fit(train_data)
        
        #Select optimal bandwidth using validation data
        print(f"Selecting optimal bandwidth using validation data")
        best_bandwidth = model.select_bandwidth()

        training_time = time.time() - start_time
        
        #Reset seed after training
        torch.manual_seed(self.seed)
        
        #Compile training results
        training_results = {
            'best_bandwidth': best_bandwidth.item() if hasattr(best_bandwidth, 'item') else float(best_bandwidth),
            'training_time': training_time,
        }
        
        return training_results
    
    def evaluate_model(self, dim, model, target, data_splits, distribution_type='gaussian'):
        """
        Evaluate the KDE model on the test set.
        """
        #Set seed 
        eval_seed = self.seed + hash(distribution_type) % 10000 + dim * 100 + 3
        torch.manual_seed(eval_seed)
        
        #Unpack test data
        test_data = data_splits['test_data']
        
        #Generate a large test set for accurate KL estimation
        large_test_size = 10000  
        large_test_data = target.sample(large_test_size)
        
        
        #Compute log probabilities
        target_log_prob = target.log_prob(large_test_data)
        model_log_prob = model.log_prob(large_test_data)
        
        #Calculate metrics
        kl_div = torch.mean(target_log_prob - model_log_prob).item()
        avg_target_logprob = torch.mean(target_log_prob).item()
        avg_model_logprob = torch.mean(model_log_prob).item()
        model_nll = -avg_model_logprob
        
        #Get additional statistics
        log_prob_diff = model_log_prob - target_log_prob
        log_prob_std = torch.std(log_prob_diff).item()
        log_prob_min = torch.min(model_log_prob).item()
        log_prob_max = torch.max(model_log_prob).item()
        
        #Additional sample quality metrics
        sample_quality = self.evaluate_sample_quality(model, target, test_data.shape[0])
        
        #Reset seed
        torch.manual_seed(self.seed)
        
        #Compile evaluation results
        evaluation_results = {
            'kl_divergence': kl_div,
            'nll': model_nll,
            'avg_target_logprob': avg_target_logprob,
            'avg_model_logprob': avg_model_logprob,
            'log_prob_std': log_prob_std,
            'log_prob_min': log_prob_min,
            'log_prob_max': log_prob_max,
        }
        
        #Merge sample quality metrics
        evaluation_results.update(sample_quality)
        
        return evaluation_results
    
    def evaluate_sample_quality(self, model, target, num_samples):
        """
        Evaluate model quality by comparing samples with target distribution.
        """
        with torch.no_grad():
            #Generate samples
            model_samples = model.sample(num_samples)

            target_samples = target.sample(num_samples)
            
            #Compare statistical properties
            model_mean = model_samples.mean(dim=0)
            target_mean = target_samples.mean(dim=0)
            mean_error = torch.mean(torch.abs(model_mean - target_mean)).item()
            
            model_std = model_samples.std(dim=0)
            target_std = target_samples.std(dim=0)
            std_error = torch.mean(torch.abs(model_std - target_std)).item()
            
            #Compute sample likelihood
            model_under_target = target.log_prob(model_samples).mean().item()
            target_under_model = model.log_prob(target_samples).mean().item()
            
            #Compute bidirectional KL estimate
            sample_kl = 0.5 * (model_under_target - target_under_model)
            
            #Wasserstein distance approximation
            if model_samples.shape[1] <= 4:
                if model_samples.shape[1] == 1:
                    sorted_model = torch.sort(model_samples.squeeze())[0]
                    sorted_target = torch.sort(target_samples.squeeze())[0]
                    wasserstein_approx = torch.mean(torch.abs(sorted_model - sorted_target)).item()
                else:
                    wasserstein_approx = torch.norm(model_mean - target_mean) + torch.norm(model_std - target_std)
                    wasserstein_approx = wasserstein_approx.item()
            else:
                wasserstein_approx = None
            
            return {
                'mean_error': mean_error,
                'std_error': std_error,
                'sample_kl': sample_kl,
                'model_under_target': model_under_target,
                'target_under_model': target_under_model,
                'wasserstein_approx': wasserstein_approx,
                'num_samples_evaluated': num_samples
            }
    
    def save_results_to_excel(self, results_dict):
        """
        Save results to Excel file.
        """
        #Convert the results dictionary to a DataFrame
        results_df = pd.DataFrame([results_dict])
        
        try:
            #If file exists, read existing data and append new results
            if os.path.exists(self.output_file):
                existing_df = pd.read_excel(self.output_file)
                
                #Check if we already have results for this dimension and distribution
                if 'dimension' in existing_df.columns and 'distribution_type' in existing_df.columns:
                    mask = (existing_df['dimension'] != results_dict['dimension']) | \
                           (existing_df['distribution_type'] != results_dict['distribution_type'])
                    existing_df = existing_df[mask]
                
                #Append new results
                updated_df = pd.concat([existing_df, results_df], ignore_index=True)
                updated_df.to_excel(self.output_file, index=False)
            else:
                #Create new file with results
                results_df.to_excel(self.output_file, index=False)
                
            
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
            #Backup save as CSV
            backup_file = self.output_file.replace('.xlsx', '.csv')
            results_df.to_csv(backup_file, index=False)
            print(f"Results backed up to {backup_file}")
    
    def run_single_experiment(self, dim, distribution_type='gaussian'):
        """
        Run experiment for a specific dimension and distribution type.
        """
        #Set a unique seed for this specific experiment
        experiment_seed = self.seed + hash(distribution_type) % 10000 + dim * 100
        torch.manual_seed(experiment_seed)
    

        print(f"Running KDE experiment for dimension {dim} with {distribution_type} distribution")

        #Create target distribution
        target = self.create_target(dim, distribution_type)
        
        #Calculate data size based on dimension
        base_data_size = 1000000
        total_data_size = base_data_size + (2*base_data_size * (dim-1))  # Linear scaling with dimension
        
        #Generate data splits and train
        data_splits = self.generate_data_splits(dim, total_data_size, target, distribution_type)
        model = KDE(dim=dim)
        
        #Train and select bandwidth
        training_results = self.train_and_select_bandwidth(dim, model, data_splits, distribution_type)
        
        #Evaluate model on test set
        evaluation_results = self.evaluate_model(dim, model, target, data_splits, distribution_type)
        
        #Reset seed 
        torch.manual_seed(self.seed)
        
        #Combine results
        results = {
            'dimension': dim,
            'distribution_type': distribution_type,
            'train_size': data_splits['train_size'],
            'test_size': data_splits['test_size'],
            'kernel': 'epanechnikov',
            'seed': experiment_seed,
            **training_results,  # Include training metrics
            **evaluation_results  # Include evaluation metrics
        }
        # Save results
        self.save_results_to_excel(results)
        return results
    
    def run_experiments(self, dimensions=None, distribution_types=None):
        """
        Run experiments with specified dimensions and distribution types.
        
        Parameters:
        dimensions (list or tuple or int, optional): Dimensions to test. 
        distribution_types (list or str, optional): Distribution types to test.
        
        Returns:
        list: List of all result dictionaries
        """
        all_distributions = self.distribution_types

        if dimensions is None:
            dims_to_run = list(range(1, 8))  
        elif isinstance(dimensions, tuple) and len(dimensions) == 2:
            start_dim, end_dim = dimensions
            dims_to_run = list(range(start_dim, end_dim + 1))
        elif isinstance(dimensions, int):
            dims_to_run = [dimensions]  
        elif isinstance(dimensions, list):
            dims_to_run = dimensions 
        else:
            raise ValueError("Invalid dimensions parameter.")
        
        if distribution_types is None:
            dists_to_run = all_distributions 
        elif isinstance(distribution_types, str):
            if distribution_types not in all_distributions:
                raise ValueError(f"Invalid distribution type: {distribution_types}")
            dists_to_run = [distribution_types]  
        elif isinstance(distribution_types, list):
            invalid_dists = [d for d in distribution_types if d not in all_distributions]
            if invalid_dists:
                raise ValueError(f"Invalid distribution types: {invalid_dists}")
            dists_to_run = distribution_types 
        else:
            raise ValueError("Invalid distribution_types parameter.")
        
        all_results = []
        
        for dist_type in dists_to_run:
            dist_results = []

            print(f"# Running KDE experiments with {dist_type} distribution")

            
            for dim in dims_to_run:
                results = self.run_single_experiment(dim, dist_type)
                dist_results.append(results)
                all_results.append(results)
            
        return all_results


if __name__ == "__main__":
    # Initialize experiment
    experiment = KDEExperiment(output_file="kde_results.xlsx", seed=42)
    
    #Run all distributions and dimensions (default behavior)
    #results = experiment.run_experiments()
    
    #Run a specific distribution across multiple dimensions
    #results = experiment.run_experiments(distribution_types='gaussian')
    
    #Run all distributions for a specific dimension
    #results = experiment.run_experiments(dimensions=2)
    
    #Run a specific dimension and distribution
    #results = experiment.run_experiments(dimensions=5, distribution_types='mixture')
     
    #Run a range of dimensions
    #results = experiment.run_experiments(dimensions=(1, 3))
    
    print("Experiment complete. Results saved to Excel file.")