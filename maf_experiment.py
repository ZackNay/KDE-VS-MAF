import torch
import numpy as np
import normflows as nf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
import time

#Generating the different types of distributions to test ont
class Gaussian_Distribution:
    """
    A self-contained Gaussian distribution that works with any dimension.
    """
    def __init__(self, dim=2, mean=2, std=2):
        self.dim = dim
        #Set default parameters if not provided
        if mean is None:
            self.mean = torch.zeros(dim)
        else:
            self.mean = mean
        
        if std is None:
            self.std = torch.ones(dim)
        else:
            self.std = std
        
    def log_prob(self, z):
        """
        Log probability of the distribution for given z.
        """
        #Ensure mean and std are on the same device as z
        mean = self.mean.to(z.device)
        std = self.std.to(z.device)
        
        #Expand mean and std to match batch size if needed
        if z.dim() > 1:
            mean = mean.unsqueeze(0).expand(z.shape[0], -1)
            std = std.unsqueeze(0).expand(z.shape[0], -1)
        
        #Calculate log probability of diagonal Gaussian
        log_prob = -0.5 * self.dim * np.log(2 * np.pi)
        log_prob -= torch.sum(torch.log(std), dim=-1)
        log_prob -= 0.5 * torch.sum(((z - mean) / std)**2, dim=-1)
        
        return log_prob
    
    def sample(self, num_samples):
        """
        Sample from the distribution.
        """
        #Sample from standard normal and transform
        eps = torch.randn(num_samples, self.dim)
        samples = self.mean.unsqueeze(0) + self.std.unsqueeze(0) * eps
        
        return samples


class Mixture_of_Gaussians:
    """
    A mixture of Gaussians distributions.
    Creates a multimodal distribution that provides a more challenging
    target than a single Gaussian.
    """
    def __init__(self, dim=2, num_components=3, means=None, stds=None, weights=None):
        self.dim = dim
        self.num_components = num_components
        
        #Set default parameters if not provided
        if means is None:
            #Create means
            self.means = []
            for i in range(num_components):
                mean_offset = 3.0 * (i - (num_components - 1) / 2)
                mean = torch.zeros(dim) + mean_offset
                self.means.append(mean)
            self.means = torch.stack(self.means)
        else:
            self.means = means
            
        if stds is None:
            self.stds = torch.ones(num_components, dim) * 0.5
        else:
            self.stds = stds
            
        if weights is None:
            self.weights = torch.ones(num_components) / num_components
        else:
            self.weights = weights / weights.sum()  # Normalize
        
    def log_prob(self, z):
        """
        Log probability of the mixture distribution for given z.
        """
        #Ensure parameters are on the same device as z
        means = self.means.to(z.device)
        stds = self.stds.to(z.device)
        weights = self.weights.to(z.device)
        
        #Expand z for component dimension
        z_expanded = z.unsqueeze(1) 
        
        # Expand means and stds for batch dimension
        means_expanded = means.unsqueeze(0)
        stds_expanded = stds.unsqueeze(0)
        
        #Calculate Gaussian log probs for each component
        log_probs = -0.5 * self.dim * np.log(2 * np.pi)
        log_probs = log_probs - torch.sum(torch.log(stds_expanded), dim=-1)
        log_probs = log_probs - 0.5 * torch.sum(((z_expanded - means_expanded) / stds_expanded)**2, dim=-1)
        
        #Weight by mixture probabilities and combine using log-sum-exp for numerical stability
        weighted_log_probs = log_probs + torch.log(weights.unsqueeze(0) + 1e-10)
        total_log_prob = torch.logsumexp(weighted_log_probs, dim=-1)
        
        return total_log_prob
    
    def sample(self, num_samples):
        """
        Sample from the mixture distribution.
        """
        #Choose which component to sample from
        components = torch.multinomial(self.weights, num_samples, replacement=True)
        
        #Get the corresponding mean and std for each sample
        selected_means = self.means[components]
        selected_stds = self.stds[components]
        
        #Sample from the selected Gaussian components
        eps = torch.randn(num_samples, self.dim)
        samples = selected_means + selected_stds * eps
        
        return samples


class Skew_Normal:
    """
    A skew-normal distribution (non-conditional).
    Creates an asymmetric distribution that provides a challenging target with
    different skewness parameters.

    """
    def __init__(self, dim=2, loc=None, scale=None, alpha=None):
        self.dim = dim
        
        #Set default parameters if not provided
        if loc is None:
            self.loc = torch.zeros(dim)
        else:
            self.loc = loc
            
        if scale is None:
            self.scale = torch.ones(dim)
        else:
            self.scale = scale
            
        if alpha is None:
            self.alpha = torch.ones(dim) * 2.0 
        else:
            self.alpha = alpha
        
    def log_prob(self, z):
        """
        Log probability of the skew-normal distribution for given z.
        """
        #Ensure parameters are on the same device as z
        loc = self.loc.to(z.device)
        scale = self.scale.to(z.device)
        alpha = self.alpha.to(z.device)
        
        #Expand parameters to match batch size if needed
        if z.dim() > 1:
            loc = loc.unsqueeze(0).expand(z.shape[0], -1)
            scale = scale.unsqueeze(0).expand(z.shape[0], -1)
            alpha = alpha.unsqueeze(0).expand(z.shape[0], -1)
        
        #Calculate standardized variable
        y = (z - loc) / scale
        
        #Standard normal log PDF component
        log_norm = -0.5 * self.dim * np.log(2 * np.pi)
        log_norm -= torch.sum(torch.log(scale), dim=-1)
        log_norm -= 0.5 * torch.sum(y**2, dim=-1)
        
        #More accurate CDF approximation using Owen's T function approximation
        cdf_input = alpha * y * np.sqrt(2/np.pi)
        cdf_term = torch.where(
            torch.abs(cdf_input) < 5,
            torch.sigmoid(cdf_input),
            # Asymptotic approximation for large values
            0.5 + 0.5 * torch.sign(cdf_input) * (1 - torch.exp(-0.7 * torch.abs(cdf_input)))
        )
        
        #Ensure numerical stability in log domain
        log_cdf_term = torch.log(2 * torch.clamp(cdf_term, min=1e-10))
        
        #Calculate final log probability 
        log_prob = log_norm + torch.sum(log_cdf_term, dim=-1)
        
        return log_prob
    

    def sample(self, num_samples):
        """
        Sample from the skew-normal distribution.
        """
        device = self.loc.device
        
        delta = self.alpha / torch.sqrt(1 + self.alpha**2)
        
        #Generate two independent normal samples
        u0 = torch.randn(num_samples, self.dim, device=device)
        u1 = torch.randn(num_samples, self.dim, device=device)
        
        #Transform to skew-normal
        z = delta.unsqueeze(0) * torch.abs(u0) + torch.sqrt(1 - delta**2).unsqueeze(0) * u1
        
        #Apply location and scale
        samples = self.loc.unsqueeze(0) + self.scale.unsqueeze(0) * z
        
        return samples


class MAF_Multidimensional_Comparison_Experiment:
    def __init__(self, output_file="normalizing_flow_results_improved.xlsx", seed=42):
        """
        Initialize the experiment for comparing normalizing flow models across dimensions.
        
        Parameters:
        output_file (str): Path to the Excel file where results will be saved
        seed (int): Random seed for reproducibility
        """

        self.output_file = output_file
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distribution_types = ['gaussian', 'mixture', 'skewnormal']
        self.seed = seed
        
        #Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        #Data split 
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
            
        print(f"Initialized experiment with random seed: {seed}")
    
    def scale_hyperparameters(self, dim):
        """
        Scale hyperparameters based on dimension.
        Includes special adjustments for high dimensions with mixture distribution.
        """
        # Base hyperparameters
        base_data_size = 10
        
        #Scale hyperparameters with dimension
        if dim < 3:
            params = {
                'K': 2, 
                'hidden_units': 48 + 8 * dim,  
                'num_blocks': 1 + (dim > 1), 
                'batch_size': max(16, 32 + 16 * (dim - 1)), 
                'lr': 2e-3, 
                'weight_decay': 5e-5, 
                'patience': 20,  
                'max_iter': 8000,  
                'total_data_size': base_data_size + (base_data_size * (dim-1)),
                'grad_clip_norm': 2.0,  
            }
        else:
            params = {
                #Number of flow layers (2 to 8)
                'K': max(2, min(dim + 1, 4)),  
                'hidden_units': 32 + 12* dim,  
                'num_blocks': 2,   
                #Larger batches for higher dimensions
                'batch_size': 64 + 64 * (dim ),  
                #Slightly decrease learning rate
                'lr': 1e-3 if dim<3 else 1e-4,  
                'weight_decay': 1e-4,
                #Early stopping patience
                'patience': 30,                    
                'max_iter': 5000,
                #Linear scaling with dimension
                'total_data_size': base_data_size + (2*base_data_size * (dim-1)), 
                'grad_clip_norm': 5.0,              
            }
        
        return params
    
    def create_target_distribution(self, dim:int, distribution_type='gaussian'):
        """
        Create a target distribution for the given dimension and type.
        
        Parameters:
        dim: The dimension of the distribution
        distribution_type: 'gaussian', 'mixture', or 'skewnormal'
        
        Returns:
        Distribution object
        """

        if distribution_type == 'gaussian':
            mean = torch.linspace(-1.0, 1.0, dim)  # Evenly spaced means from -1 to 1
            std = torch.ones(dim) * 1.5  #Fixed std of 1.5 for all dimensions
            target = GaussianDistribution(dim=dim, mean=mean, std=std)
            
        elif distribution_type == 'mixture':
            num_components = 3
            means = []
            for i in range(num_components):
                mean_offset = 3.0 * (i - (num_components - 1) / 2)
                mean = torch.ones(dim) * mean_offset  
                means.append(mean)
            means = torch.stack(means)
            
            stds = torch.ones(num_components, dim) * 0.7 
            
            weights = torch.ones(num_components) / num_components
            
            target = MixtureOfGaussians(dim=dim, num_components=num_components, 
                                    means=means, stds=stds, weights=weights)
            
        elif distribution_type == 'skewnormal':
            loc = torch.zeros(dim)  
            scale = torch.ones(dim) * 1.2  
            
            alpha = torch.linspace(0.5, 2.5, dim) 
            
            target = SkewNormal(dim=dim, loc=loc, scale=scale, alpha=alpha)
            
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
            
        return target
    
    def generate_data_splits(self, dim:int, total_size:int, target, distribution_type='gaussian'):
        """
        Generate data splits for the given dimension and distribution type.
        
        Parameters:
        dim: Dimension of the data
        total_size: Total number of samples to generate across all splits
        target: Target distribution object
        distribution_type: 'gaussian', 'mixture', or 'skewnormal'
        
        Returns:
        dict: Dictionary with data splits
        """
        device = self.device
        
        #Set seed 
        data_seed = self.seed + hash(distribution_type) % 10000 + dim * 100
        torch.manual_seed(data_seed)
        
        #Calculate split sizes
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        
        #Sample from target distribution for each split
        train_data = target.sample(train_size).to(device)
        val_data = target.sample(val_size).to(device)
        test_data = target.sample(test_size).to(device)
        
        #Reset seed after data generation to ensure independence between experiments
        torch.manual_seed(self.seed)
        
        #Return all splits
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }
    
    def create_flow_model(self, dim:int, params:dict, distribution_type='gaussian'):
        """
        Create a normalizing flow model for the given dimension.
        
        Parameters:
        dim: Dimension of the model
        params: Hyperparameters
        distribution_type (str): 'gaussian', 'mixture', or 'skewnormal'
        
        Returns:
        torch.nn.Module: The flow model
        """
        #Set seed
        model_seed = self.seed + hash(distribution_type) % 10000 + dim * 100 + 2
        torch.manual_seed(model_seed)
        
        K = params['K']
        hidden_units = params['hidden_units']
        num_blocks = params['num_blocks']
        
        #Define flows
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(
                dim, hidden_units, 
                num_blocks=num_blocks
            )]
            flows += [nf.flows.LULinearPermute(dim)]
        
        #Set base distribution
        q0 = nf.distributions.DiagGaussian(dim, trainable=False)
        
        #Construct flow model
        model = nf.NormalizingFlow(q0, flows)
        
        #Move model to device
        model = model.to(self.device)
        
        #Reset seed after model creation
        torch.manual_seed(self.seed)
        
        return model
    
    def train_model(self, dim:int, model, target, data_splits:dict, params:dict, distribution_type='gaussian'):
        """
        Train the model with early stopping using validation set.
        
        Parameters:
        dim: Dimension of the data
        model: The normalizing flow model to train
        target: Target distribution object
        data_splits: Dictionary containing data splits
        params: Hyperparameters
        distribution_type: Type of distribution to use
        
        Returns:
        dict: Training results and metrics
        """
        #Set seed 
        train_seed = self.seed + hash(distribution_type) % 10000 + dim * 100 + 1
        torch.manual_seed(train_seed)
        
        train_data = data_splits['train_data']
        val_data = data_splits['val_data']
        train_size = data_splits['train_size']


        batch_size = params['batch_size']
        max_iter = params['max_iter']
        patience = params['patience']
        lr = params['lr']
        weight_decay = params['weight_decay']
        grad_clip_norm = params['grad_clip_norm']
        
        #Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        #Initialize loss
        train_loss_hist = []
        val_loss_hist = []
        
        #Early stopping
        best_val_loss = float('inf')
        best_iteration = 0
        no_improvement = 0
        best_model_state = None
        
        start_time = time.time()
        
        #Training, with early stopping and gradient clipping
        for it in tqdm(range(max_iter), desc=f"Training... {dim}D model ({distribution_type}) distribution"):
            model.train()
            optimizer.zero_grad()
            
            #Sample a batch
            idx = torch.randint(0, train_size, (batch_size,))
            x_batch = train_data[idx].to(self.device)
            
            #Compute loss 
            loss = model.forward_kld(x_batch)
            
            #Do backprop and optimizer step with gradient clipping
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                
                #Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                optimizer.step()
                
            #Record loss
            train_loss_hist.append(loss.item())
            
            #Evaluate on validation set every 10 iterations
            if it % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = model.forward_kld(val_data).item()
                    val_loss_hist.append(val_loss)
                    
                    #Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_iteration = it
                        no_improvement = 0
                        #Save the best model state
                        best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
                    else:
                        no_improvement += 1
                    
                    #Stop if no improvement for patience evaluations
                    if no_improvement >= patience:
                        print(f"Early stopping at iteration {it}. Best iteration: {best_iteration}")
                        break
        
        #Calculate training time
        training_time = time.time() - start_time
        
        #Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        #Reset seed after training
        torch.manual_seed(self.seed)
        
        #Training results
        training_results = {
            'best_iteration': best_iteration,
            'total_epochs': len(val_loss_hist) * 10,  
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist,
        }
        
        #Plot loss
        self.plot_training_loss(train_loss_hist, val_loss_hist, dim, distribution_type)
        
        return training_results
    
    def evaluate_model(self, dim:int, model, target, data_splits:dict, distribution_type='gaussian'):

        """
        Evaluate the trained model on the test set.
        
        Parameters:
        dim (int): Dimension of the data
        model: Trained normalizing flow model
        target: Target distribution object
        data_splits (dict): Dictionary containing data splits
        distribution_type (str): Type of distribution used
        
        Returns:
        dict: Evaluation metrics
        """
        #Set seed
        eval_seed = self.seed + hash(distribution_type) % 10000 + dim * 100 + 3
        torch.manual_seed(eval_seed)
        large_test_size = 100000
        large_test_data = target.sample(large_test_size).to(self.device)
        #Evaluate model on test set

        model.eval()
        with torch.no_grad():
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
            sample_quality = self.evaluate_sample_quality(model, target, large_test_data.shape[0])
        
        #Reset seed after evaluation
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
        
        Parameters:
        model: Trained normalizing flow model
        target: Target distribution object
        num_samples: Number of samples to generate for evaluation
        
        Returns:
        dict: Sample quality metrics
        """
        with torch.no_grad():
            #Generate samples from model
            model_samples = model.sample(num_samples)[0] 
            
            # Generate samples from target distribution
            target_samples = target.sample(num_samples).to(model_samples.device)
            
            #Compare statistical properties
            model_mean = model_samples.mean(dim=0)
            target_mean = target_samples.mean(dim=0)
            mean_error = torch.mean(torch.abs(model_mean - target_mean)).item()
            
            model_std = model_samples.std(dim=0)
            target_std = target_samples.std(dim=0)
            std_error = torch.mean(torch.abs(model_std - target_std)).item()
            
            #Compute sample likelihood under both distributions
            model_under_target = target.log_prob(model_samples).mean().item()
            target_under_model = model.log_prob(target_samples).mean().item()
            
            #Compute bidirectional KL estimate
            sample_kl = 0.5 * (model_under_target - target_under_model)
            
            #Wasserstein distance approximation (only feasible in low dimensions)
            if model_samples.shape[1] <= 4:  # Only compute for dim <= 4

                if model_samples.shape[1] == 1:
                    sorted_model = torch.sort(model_samples.squeeze())[0]
                    sorted_target = torch.sort(target_samples.squeeze())[0]
                    wasserstein_approx = torch.mean(torch.abs(sorted_model - sorted_target)).item()
                else:
                    #For higher dimensions, use a simple distance-based approximation
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
    
    def plot_training_loss(self, train_loss_hist:list, val_loss_hist:list, dim:int, distribution_type='gaussian'):
        """
        Plot training and validation loss.
        
        Parameters:
        train_loss_hist: Training loss history
        val_loss_hist: Validation loss history
        dim: Dimension of the model
        distribution_type (str): Type of distribution used
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_hist, label='Training Loss')
        plt.plot(np.arange(0, len(train_loss_hist), 10)[:len(val_loss_hist)], 
                val_loss_hist, label='Validation Loss')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for {dim}D Model ({distribution_type}) dsitributiob')
        plt.grid(True, alpha=0.3)
        plt.close()
    
    def save_results_to_excel(self, results_dict):
        """
        Save results to Excel file.
        
        Parameters:
        results_dict: Results dictionary to save
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
                
            print(f"Results for {results_dict['dimension']}D model ({results_dict['distribution_type']}) saved to {self.output_file}")
            
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
            
            #Backup save as CSV
            backup_file = self.output_file.replace('.xlsx', '.csv')
            results_df.to_csv(backup_file, index=False)
            print(f"Results backed up to {backup_file}")
    
    def run_single_experiment(self, dim:int, distribution_type='gaussian'):
        """
        Run experiment for a specific dimension and distribution type.
        
        Parameters:
        dim: Dimension to test
        distribution_type: 'gaussian', 'mixture', or 'skewnormal'
        
        Returns:
        dict: Results dictionary
        """
        #Set seed 
        experiment_seed = self.seed + hash(distribution_type) % 10000 + dim * 100
        torch.manual_seed(experiment_seed)
        
        print(f"Running experiment for dimension {dim} with {distribution_type} distribution (seed: {experiment_seed})")

        
        #Scale hyperparameters
        params = self.scale_hyperparameters(dim)
        
        #Create target distribution
        target = self.create_target_distribution(dim, distribution_type)
        
        #Generate data splits
        data_splits = self.generate_data_splits(dim, params['total_data_size'], target, distribution_type)
        
        # Create model
        model = self.create_flow_model(dim, params, distribution_type)
        
        #Print hyperparameters
        print("Hyperparameters:")
        for key, value in params.items():
            print(f"{key} - {value}")
        
        #Train model using train/val splits
        training_results = self.train_model(dim, model, target, data_splits, params, distribution_type)
        
        #Evaluate model on test set
        evaluation_results = self.evaluate_model(dim, model, target, data_splits, distribution_type)
        
        #Reset seed after experiment is complete
        torch.manual_seed(self.seed)
        
        #Combine results
        results = {
            'dimension': dim,
            'distribution_type': distribution_type,
            'train_size': data_splits['train_size'],
            'val_size': data_splits['val_size'],
            'test_size': data_splits['test_size'],
            'K': params['K'],
            'hidden_units': params['hidden_units'],
            'num_blocks': params['num_blocks'],
            'learning_rate': params['lr'],
            'batch_size': params['batch_size'],
            'grad_clip_norm': params['grad_clip_norm'],
            'seed': experiment_seed,
            **training_results,
            **evaluation_results
        }
        
        #Print results
        print("\nResults:")
        print(f"Distribution-{distribution_type}")
        print(f"KL Divergence-{results['kl_divergence']:.4f}")
        print(f"Training time - {results['training_time']:.2f} seconds")
        print(f"Iterations to conv - {results['best_iteration']}")
        
        #Save results
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
        
        #Process dimension parameter
        if dimensions is None:
            dims_to_run = list(range(1, 9))  
        elif isinstance(dimensions, tuple) and len(dimensions) == 2:
            start_dim, end_dim = dimensions
            dims_to_run = list(range(start_dim, end_dim + 1))
        elif isinstance(dimensions, int):
            dims_to_run = [dimensions]  
        elif isinstance(dimensions, list):
            dims_to_run = dimensions  
        else:
            raise ValueError("Invalid dimensions parameter. Must be None, int, tuple(start,end), or list.")
        
        #Process distribution_types parameter
        if distribution_types is None:
            dists_to_run = all_distributions  
        elif isinstance(distribution_types, str):
            if distribution_types not in all_distributions:
                raise ValueError(f"Invalid distribution type. Must be one of {all_distributions}")
            dists_to_run = [distribution_types]  
        elif isinstance(distribution_types, list):
            invalid_dists = [d for d in distribution_types if d not in all_distributions]
            if invalid_dists:
                raise ValueError(f"Invalid distribution types: {invalid_dists}. Must be among {all_distributions}")
            dists_to_run = distribution_types  
        else:
            raise ValueError("Invalid distribution_types parameter. Must be None, str, or list of strings.")
        
        #Run experiments
        all_results = []
        
        #Organize experiments by distribution type
        for dist_type in dists_to_run:
            dist_results = []
            print(f"# Running experiments with {dist_type} distribution")

            for dim in dims_to_run:
                results = self.run_single_experiment(dim, dist_type)
                dist_results.append(results)
                all_results.append(results)
            
            #Create a comparison plot for each distribution type
            if len(dims_to_run) > 1:
                self.plot_dimension_comparison(dist_results, suffix=f"_{dist_type}")
        
        #Create overall comparison if multiple distributions and dimensions
        if len(dists_to_run) > 1 and len(dims_to_run) > 1:
            self.plot_distribution_comparison(all_results)
        
        return all_results
    
    def plot_dimension_comparison(self, results_list, suffix=""):
        """
        Plot comparison of results across dimensions.
        
        Parameters:
        results_list (list): List of result dictionaries
        suffix (str): String to append to the output filename
        """
        #Check if there are enough dimensions to plot
        if len(results_list) <= 1:
            print("Not enough dimensions to create comparison plot")
            return
            
        dimensions = [r['dimension'] for r in results_list]
        kl_divs = [r['kl_divergence'] for r in results_list]
        nlls = [r['nll'] for r in results_list]
        mean_errors = [r['mean_error'] for r in results_list]
        std_errors = [r['std_error'] for r in results_list]
        
        #Verify all dimensions are the same distribution type
        dist_types = set(r['distribution_type'] for r in results_list)
        if len(dist_types) > 1:
            print("Warning: Comparing results from different distribution types")
        
        dist_type = results_list[0]['distribution_type']
        
        #Create figure with 4 subplots
        fig, axs = plt.subplots(1, 4, figsize=(20, 6))
        
        #Sort data by dimension
        sort_idx = np.argsort(dimensions)
        sorted_dims = [dimensions[i] for i in sort_idx]
        sorted_kl_divs = [kl_divs[i] for i in sort_idx]
        sorted_nlls = [nlls[i] for i in sort_idx]
        sorted_mean_errors = [mean_errors[i] for i in sort_idx]
        sorted_std_errors = [std_errors[i] for i in sort_idx]
        
        #Plot KL divergence
        axs[0].plot(sorted_dims, sorted_kl_divs, 'o-', color='blue')
        axs[0].set_xlabel('Dimension')
        axs[0].set_ylabel('KL Divergence')
        axs[0].set_title(f'KL Divergence vs. Dimension ({dist_type})')
        axs[0].grid(True, alpha=0.3)
        
        #Plot negative log-likelihood
        axs[1].plot(sorted_dims, sorted_nlls, 'o-', color='green')
        axs[1].set_xlabel('Dimension')
        axs[1].set_ylabel('Negative Log-Likelihood')
        axs[1].set_title(f'NLL vs. Dimension ({dist_type})')
        axs[1].grid(True, alpha=0.3)
        
        #Plot mean error
        axs[2].plot(sorted_dims, sorted_mean_errors, 'o-', color='red')
        axs[2].set_xlabel('Dimension')
        axs[2].set_ylabel('Mean Error')
        axs[2].set_title(f'Mean Error vs. Dimension ({dist_type})')
        axs[2].grid(True, alpha=0.3)
        
        #Plot std error
        axs[3].plot(sorted_dims, sorted_std_errors, 'o-', color='purple')
        axs[3].set_xlabel('Dimension')
        axs[3].set_ylabel('Std Error')
        axs[3].set_title(f'Std Error vs. Dimension ({dist_type})')
        axs[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'dimension_comparison{suffix}.png')
        plt.close()
    
    def plot_distribution_comparison(self, results_list):
        """
        Plot comparison of results across dimensions and distribution types.
        
        Parameters:
        results_list (list): List of result dictionaries from experiments
        """
        #Group results by distribution type
        grouped_results = defaultdict(list)
        for r in results_list:
            grouped_results[r['distribution_type']].append(r)
        
        #Prepare data for plotting
        metrics = ['kl_divergence', 'nll', 'mean_error', 'std_error']
        metric_names = ['KL Divergence', 'Negative Log-Likelihood', 'Mean Error', 'Std Error']
        colors = {'gaussian': 'blue', 'mixture': 'red', 'skewnormal': 'green'}
        
        #Create figure with 4 subplots (one for each metric)
        fig, axs = plt.subplots(1, 4, figsize=(20, 6))
        
        for dist_type, results in grouped_results.items():
            #Extract dimensions and sort results
            dimensions = [r['dimension'] for r in results]
            sorted_indices = np.argsort(dimensions)
            sorted_results = [results[i] for i in sorted_indices]
            sorted_dimensions = [dimensions[i] for i in sorted_indices]
            
            #Plot each metric
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                values = [r[metric] for r in sorted_results]
                axs[i].plot(sorted_dimensions, values, 'o-', color=colors[dist_type], 
                          label=dist_type.capitalize())
                axs[i].set_xlabel('Dimension')
                axs[i].set_ylabel(name)
                axs[i].set_title(f'{name} vs. Dimension')
                axs[i].grid(True, alpha=0.3)
                axs[i].legend()
        
        plt.tight_layout()
        plt.savefig('distribution_comparison.png')
        plt.close()

if __name__ == "__main__":
    #Initialize experiment with a random seed for reproducibility
    experiment = ImprovedMultidimensionalComparisonExperiment(
        output_file="normalizing_flow_results_nonconditional.xlsx", 
        seed=42
    )
    
    #Run all distributions and dimensions (default behavior)
    results = experiment.run_experiments()
    
    #Other usage examples:
    #Run a specific distribution across multiple dimensions
    #results = experiment.run_experiments(dimensions=3, distribution_types='mixture')
    
    #Run all distributions for a specific dimension
    # results = experiment.run_experiments(dimensions=3)
    
    #Run a specific dimension and distribution
    #results = experiment.run_experiments(dimensions=3, distribution_types='mixture')
    
    #Run a range of dimensions
    #results = experiment.run_experiments(dimensions=(1, 3),distribution_types='gaussian')
    
    #Run specific dimensions
    #results = experiment.run_experiments(dimensions=[1, 3, 5])
    
