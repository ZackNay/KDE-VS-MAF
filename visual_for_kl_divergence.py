import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Set seed
torch.manual_seed(42)

class MixtureOfGaussians:
    """Mixture of Gaussians distribution."""
    def __init__(self, means, covs, weights):
        #Mean vectors
        self.means = means  
        #Covariance matrices
        self.covs = covs   
        #Normalized weights
        self.weights = weights / weights.sum()  
        self.num_components = len(means)
        self.dim = means[0].shape[0]
        
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
        
        samples = torch.zeros(n_samples, self.dim)
        
        #Sample from each component
        for i in range(self.num_components):
            mask = (assignments == i)
            n_comp = mask.sum().item()
            if n_comp > 0:
                samples[mask] = self.components[i].sample((n_comp,))
                
        return samples
    
    def log_prob(self, x):
        """Compute log probability of samples."""
        log_probs = torch.zeros(x.shape[0], self.num_components)
        
        #Compute log prob for each component
        for i in range(self.num_components):
            log_probs[:, i] = self.components[i].log_prob(x) + torch.log(self.weights[i])
            
        # Log-sum-exp for numerical stability
        return torch.logsumexp(log_probs, dim=1)

def estimate_kl_divergence(p, q, n_samples=100000):
    """Estimate KL(P||Q) using Monte Carlo"""
    samples = p.sample(n_samples)
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)
    kl_estimate = torch.mean(log_p - log_q).item()
    
    return kl_estimate

means1 = [
    torch.tensor([-6.0, -3.0, -2.0]),  
    torch.tensor([-5.5, 5, 0.5]),    
    torch.tensor([6.0, -1.0, 3.0])     
]

std1 = 0.5
covs1 = [torch.eye(3) * std1**2 for _ in range(3)]
weights1 = torch.tensor([1/3, 1/3, 1/3])
mixture1 = MixtureOfGaussians(means1, covs1, weights1)

#Create second mixture

means2 = [
    torch.tensor([-5.8, -2.5, -1.7]), 
    torch.tensor([-5.5, 5, 0.5]),    
    torch.tensor([6.25, -0.5, 2.78])    
]

std2 = 0.5
covs2 = [torch.eye(3) * std2**2 for _ in range(3)]
weights2 = torch.tensor([1/3, 1/3, 1/3])
mixture2 = MixtureOfGaussians(means2, covs2, weights2)


#Find KL Divergence 

#Sample from both distributions
n_samples = 2000
samples1 = mixture1.sample(n_samples)
samples2 = mixture2.sample(n_samples)

kl_div = estimate_kl_divergence(mixture1, mixture2, n_samples=50000)


#Visualize
fig = plt.figure(figsize=(20, 8))


#Plot mixture 1
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(samples1[:, 0], samples1[:, 1], samples1[:, 2], alpha=0.6, s=80, c='blue')
ax1.set_title('Mixture 1', fontsize=14)
ax1.set_xlim(-8, 8)
ax1.set_ylim(-5, 6)
ax1.set_zlim(-4, 5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

#Plot mixture 2
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(samples2[:, 0], samples2[:, 1], samples2[:, 2], alpha=0.6, s=80, c='red')
ax2.set_title('Mixture 2', fontsize=14)
ax2.set_xlim(-8, 8)
ax2.set_ylim(-5, 6)
ax2.set_zlim(-4, 5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

#Plot both overlaid
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(samples1[:, 0], samples1[:, 1], samples1[:, 2], alpha=0.4, s=80, c='blue')
ax3.scatter(samples2[:, 0], samples2[:, 1], samples2[:, 2], alpha=0.4, s=80, c='red')
ax3.set_title(f'Overlay (KL = {kl_div:.3f})', fontsize=14)
ax3.set_xlim(-8, 8)
ax3.set_ylim(-5, 6)
ax3.set_zlim(-4, 5)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.show()

