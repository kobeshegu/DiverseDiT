import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ##### added block diversity loss
            block_diversity_loss=False,
            projection=True,
            encoder_depth=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        ##### added block diversity loss
        # diversity loss
        self.block_diversity_loss = block_diversity_loss
        ##### whether to use projection loss
        self.projection = projection
        ##### which layer to compute the projection loss
        self.encoder_depth = encoder_depth

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        # model forward 
        model_outputs = model(model_input, time_input.flatten(), **model_kwargs)
        model_output = model_outputs['x']
        zs_tilde = model_outputs.get('zs', None)
        block_feas = model_outputs.get('block_feas', None)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        losses = {'denoising_loss': denoising_loss}
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        losses['proj_loss'] = proj_loss

        if self.block_diversity_loss:
            assert block_feas is not None, "block_feas is required for block_difference_loss"
            block_diff_loss = self.compute_block_diversity_loss(block_feas)
            losses['block_diversity_loss'] = block_diff_loss
        
        return losses


    def compute_block_diversity_loss(self, block_feas):
        """
        compute block diversity loss with lightweight strategies

        use lightweight strategies to promote the representation diversity between blocks
        avoiding high computational complexity
        
        Args:
            block_feas: dict {block_idx: features (N, T, D)}
        
        Returns:
            loss: scalar
        """
        if len(block_feas) < 2:
            return torch.tensor(0.0, device=list(block_feas.values())[0].device)
        
        block_indices = sorted(block_feas.keys())
        num_blocks = len(block_indices)
        
        # numerical stability constant
        eps = 1e-8
        
        ##### lightweight strategies
        # 1. lightweight orthogonality constraint
        orthogonality_loss = self._compute_lightweight_orthogonality_loss(block_feas, eps)
        
        # 2. lightweight mutual information minimization - using the trace of the covariance matrix
        mutual_info_loss = self._compute_lightweight_mutual_info_loss(block_feas, eps)
        
        # 3. lightweight feature dispersion loss - using simple variance analysis
        feature_dispersion_loss = self._compute_lightweight_feature_dispersion_loss(block_feas, eps)
        
        # combine all diversity loss
        total_diversity_loss = (orthogonality_loss + mutual_info_loss + feature_dispersion_loss) / 3.0
        
        ##### apply coefficient
        block_diversity_max_loss = 5.0
        final_loss = total_diversity_loss
        if final_loss > block_diversity_max_loss:
            print(f"Warning: Block diversity loss {final_loss.item():.4f} exceeds max {block_diversity_max_loss}. Clamping.")
            final_loss = torch.clamp(final_loss, max=block_diversity_max_loss)
        
        # numerical stability check
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print("Warning: Block diversity loss is NaN or Inf, returning 0")
            return torch.tensor(0.0, device=final_loss.device)
        
        return final_loss
    
    def _compute_lightweight_orthogonality_loss(self, block_feas, eps):
        """
        compute lightweight orthogonality loss with simple cosine similarity, avoiding PCA

        use simple cosine similarity to compute the orthogonality loss between blocks
        avoiding PCA to reduce computational complexity
        
        Args:
            block_feas: dict {block_idx: features (N, T, D)}
        
        Returns:
            loss: scalar
        """
        block_indices = sorted(block_feas.keys())
        orthogonality_loss = 0.0
        num_pairs = 0
        
        ##### only compute a few pairs (10 pairs) to reduce computational complexity
        max_pairs = min(10, len(block_indices) * (len(block_indices) - 1) // 2)
        pair_count = 0
        
        if self.projection:
            for i in range(max_pairs-self.encoder_depth-1, max_pairs+self.encoder_depth):
                for j in range(i + 1, len(block_indices)):
                    if pair_count >= max_pairs:
                        break
                        
                    feat1 = block_feas[block_indices[i]]  # (N, T, D)
                    feat2 = block_feas[block_indices[j]]  # (N, T, D)
                    
                    # numerical stability check
                    if torch.isnan(feat1).any() or torch.isinf(feat1).any():
                        continue
                    if torch.isnan(feat2).any() or torch.isinf(feat2).any():
                        continue
                    
                    N, T, D = feat1.shape
                    
                    # simple global average pooling
                    feat1_pooled = torch.mean(feat1, dim=(0, 1))  # (D,)
                    feat2_pooled = torch.mean(feat2, dim=(0, 1))  # (D,)
                    
                    # compute cosine similarity
                    feat1_norm = F.normalize(feat1_pooled, p=2, dim=0, eps=eps)
                    feat2_norm = F.normalize(feat2_pooled, p=2, dim=0, eps=eps)
                    
                    cosine_sim = torch.sum(feat1_norm * feat2_norm)
                    
                    # orthogonality loss: the smaller the better
                    orthogonality_loss += cosine_sim
                    num_pairs += 1
                    pair_count += 1
                
                if pair_count >= max_pairs:
                    break
            
        else:
            for i in range(len(block_indices)):
                for j in range(i + 1, len(block_indices)):
                    if pair_count >= max_pairs:
                        break
                        
                    feat1 = block_feas[block_indices[i]]  # (N, T, D)
                    feat2 = block_feas[block_indices[j]]  # (N, T, D)
                    
                    # numerical stability check
                    if torch.isnan(feat1).any() or torch.isinf(feat1).any():
                        continue
                    if torch.isnan(feat2).any() or torch.isinf(feat2).any():
                        continue
                    
                    N, T, D = feat1.shape
                    
                    # simple global average pooling
                    feat1_pooled = torch.mean(feat1, dim=(0, 1))  # (D,)
                    feat2_pooled = torch.mean(feat2, dim=(0, 1))  # (D,)
                    
                    # compute cosine similarity
                    feat1_norm = F.normalize(feat1_pooled, p=2, dim=0, eps=eps)
                    feat2_norm = F.normalize(feat2_pooled, p=2, dim=0, eps=eps)
                    
                    cosine_sim = torch.sum(feat1_norm * feat2_norm)
                    
                    # orthogonality loss: the smaller the better
                    orthogonality_loss += cosine_sim
                    num_pairs += 1
                    pair_count += 1
                
                if pair_count >= max_pairs:
                    break
        
        return orthogonality_loss / max(num_pairs, 1)
    
    def _compute_lightweight_mutual_info_loss(self, block_feas, eps):
        """
        compute lightweight mutual information loss with the trace of the covariance matrix
        avoiding complex computation
        
        Args:
            block_feas: dict {block_idx: features (N, T, D)}
        
        Returns:
            loss: scalar
        """
        block_indices = sorted(block_feas.keys())
        mutual_info_loss = 0.0
        num_pairs = 0
        
        ##### only compute a few pairs (10 pairs) to reduce computational complexity
        max_pairs = min(10, len(block_indices) * (len(block_indices) - 1) // 2)
        pair_count = 0
        
        if self.projection:
                for i in range(max_pairs-self.encoder_depth-1, max_pairs+self.encoder_depth):
                    for j in range(i + 1, len(block_indices)):
                        if pair_count >= max_pairs:
                            break
                            
                        feat1 = block_feas[block_indices[i]]  # (N, T, D)
                        feat2 = block_feas[block_indices[j]]  # (N, T, D)
                        
                        # numerical stability check
                        if torch.isnan(feat1).any() or torch.isinf(feat1).any():
                            continue
                        if torch.isnan(feat2).any() or torch.isinf(feat2).any():
                            continue
                        
                        N, T, D = feat1.shape
                        
                        # simple mutual information calculation: using normalized cosine similarity
                        feat1_flat = feat1.reshape(N * T, D)
                        feat2_flat = feat2.reshape(N * T, D)
                        
                        # L2 normalization, ensuring numerical stability
                        feat1_norm = F.normalize(feat1_flat, p=2, dim=1, eps=eps)  # (N*T, D)
                        feat2_norm = F.normalize(feat2_flat, p=2, dim=1, eps=eps)  # (N*T, D)
                        
                        # compute average cosine similarity as the proxy of mutual information
                        # [-1, 1]
                        cosine_sim = torch.mean(torch.sum(feat1_norm * feat2_norm, dim=1))
                        
                        mutual_info_loss += torch.abs(cosine_sim)
                        num_pairs += 1
                        pair_count += 1
                    
                    if pair_count >= max_pairs:
                        break
        else:
            for i in range(len(block_indices)):
                for j in range(i + 1, len(block_indices)):
                    if pair_count >= max_pairs:
                        break
                        
                    feat1 = block_feas[block_indices[i]]  # (N, T, D)
                    feat2 = block_feas[block_indices[j]]  # (N, T, D)
                    
                    # numerical stability check
                    if torch.isnan(feat1).any() or torch.isinf(feat1).any():
                        continue
                    if torch.isnan(feat2).any() or torch.isinf(feat2).any():
                        continue
                    
                    N, T, D = feat1.shape
                    
                    # simple mutual information calculation: using normalized cosine similarity
                    feat1_flat = feat1.reshape(N * T, D)
                    feat2_flat = feat2.reshape(N * T, D)
                    
                    # L2 normalization, ensuring numerical stability
                    feat1_norm = F.normalize(feat1_flat, p=2, dim=1, eps=eps)  # (N*T, D)
                    feat2_norm = F.normalize(feat2_flat, p=2, dim=1, eps=eps)  # (N*T, D)
                    
                    # compute average cosine similarity as the proxy of mutual information
                    # [-1, 1]
                    cosine_sim = torch.mean(torch.sum(feat1_norm * feat2_norm, dim=1))
                    
                    mutual_info_loss += torch.abs(cosine_sim)
                    num_pairs += 1
                    pair_count += 1
                
                if pair_count >= max_pairs:
                    break
        
        avg_loss = mutual_info_loss / max(num_pairs, 1)
        # cosine similarity range: [0, 1] (after taking the absolute value)
        return avg_loss
    
    def _compute_lightweight_feature_dispersion_loss(self, block_feas, eps):
        """
        compute lightweight feature dispersion loss with simple variance analysis
        ensuring numerical stability
        
        Args:
            block_feas: dict {block_idx: features (N, T, D)}
        
        Returns:
            loss: scalar
        """
        block_indices = sorted(block_feas.keys())
        
        ##### gathering all block features
        all_features = []
        for idx in block_indices:
            feat = block_feas[idx]  # (N, T, D)
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                continue
            N, T, D = feat.shape
            feat_flat = feat.reshape(N * T, D)
            all_features.append(feat_flat)
        
        if len(all_features) < 2:
            return torch.tensor(0.0)
        
        ##### compute the feature usage of all blocks
        feature_usage = torch.zeros(D, device=all_features[0].device)
        
        for feat in all_features:
            # L2 normalization, ensuring numerical stability
            feat_norm = F.normalize(feat, p=2, dim=0, eps=eps)  # normalize by sample dimension
            # compute the activation strength of each dimension
            dim_activation = torch.mean(torch.abs(feat_norm), dim=0)  # (D,)
            feature_usage += dim_activation
        
        ##### normalize to [0, 1]
        feature_usage = feature_usage / (len(all_features) + eps)
        feature_usage = feature_usage / (torch.max(feature_usage) + eps)
        
        ##### compute the normalized variance (range: [0, 1])
        mean_usage = torch.mean(feature_usage)
        variance = torch.mean((feature_usage - mean_usage) ** 2)
        
        ##### the maximum normalized variance is 0.25 (when the distribution is two-point distribution)
        ##### so we divide by 0.25 to normalize to [0, 1]
        normalized_variance = variance / 0.25
        
        ##### we want the feature usage to be as dispersive as possible, so we maximize the variance
        ##### return negative value: -normalized_variance (range: [-1, 0])
        dispersion_loss = -torch.clamp(normalized_variance, 0, 1)
        
        return dispersion_loss
    