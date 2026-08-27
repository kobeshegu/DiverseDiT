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
            encoders=None,
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ##### added block diversity loss
            block_diversity_loss=False,
            projection=True,
            encoder_depth=None,
            trajectory_factorization=False,
            factor_pair_cross_noise_prob=0.5,
            factor_min_delta_t=0.15,
            factor_max_delta_t=0.7,
            factor_transition=False,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = [] if encoders is None else encoders
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
        self.trajectory_factorization = trajectory_factorization
        self.factor_pair_cross_noise_prob = factor_pair_cross_noise_prob
        self.factor_min_delta_t = factor_min_delta_t
        self.factor_max_delta_t = factor_max_delta_t
        self.factor_transition = factor_transition
        if not 0.0 <= factor_pair_cross_noise_prob <= 1.0:
            raise ValueError("factor_pair_cross_noise_prob must be in [0, 1]")
        if not 0.0 <= factor_min_delta_t <= factor_max_delta_t < 1.0:
            raise ValueError("factor timestep deltas must satisfy 0 <= min <= max < 1")

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

    def _sample_times(self, images):
        if self.weighting == "uniform":
            time_input = torch.rand(
                (images.shape[0], 1, 1, 1),
                device=images.device,
                dtype=images.dtype,
            )
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn(
                (images.shape[0], 1, 1, 1),
                device=images.device,
                dtype=images.dtype,
            )
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        else:
            raise NotImplementedError(f"Unknown timestep weighting: {self.weighting}")
        return time_input

    def _sample_paired_times(self, images):
        """Sample two ordered views with a controlled non-zero trajectory gap."""
        batch_size = images.shape[0]
        shape = (batch_size, 1, 1, 1)
        delta = self.factor_min_delta_t + torch.rand(
            shape, device=images.device, dtype=images.dtype
        ) * (
            self.factor_max_delta_t - self.factor_min_delta_t
        )
        low = torch.rand(shape, device=images.device, dtype=images.dtype) * (1.0 - delta)
        high = low + delta
        swap = torch.rand(shape, device=images.device) < 0.5
        time_a = torch.where(swap, high, low)
        time_b = torch.where(swap, low, high)
        return time_a, time_b

    @staticmethod
    def _assemble_model_kwargs(model_kwargs, batch_size, pair_indices, single_indices):
        assembled_kwargs = {}
        for key, value in model_kwargs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                pair_value = value[pair_indices]
                assembled_kwargs[key] = torch.cat([
                    pair_value,
                    pair_value,
                    value[single_indices],
                ], dim=0)
            else:
                assembled_kwargs[key] = value
        return assembled_kwargs

    def _projection_loss(self, zs, zs_tilde, reference, pair_count=None):
        if not self.projection or zs is None or zs_tilde is None or len(zs) == 0:
            return reference.new_zeros(())
        losses = []
        for target, prediction in zip(zs, zs_tilde):
            target = F.normalize(target, dim=-1)
            prediction = F.normalize(prediction, dim=-1)
            per_view = -(target * prediction).sum(dim=-1).mean(dim=-1)
            if pair_count is not None:
                pair_loss = 0.5 * (
                    per_view[:pair_count]
                    + per_view[pair_count:2 * pair_count]
                )
                per_view = torch.cat([pair_loss, per_view[2 * pair_count:]])
            losses.append(per_view.mean())
        return torch.stack(losses).mean()

    @staticmethod
    def _cosine_distance(x, y):
        x = x.float()
        y = y.float()
        return 1.0 - (
            F.normalize(x, dim=-1) * F.normalize(y, dim=-1)
        ).sum(dim=-1).mean(dim=-1)

    @staticmethod
    def _energy_normalized_mse(prediction, target, eps=1e-3):
        """Compare residual features without discarding their relative scale."""
        prediction = prediction.float()
        target = target.detach().float()
        target_energy = mean_flat(target.square()).clamp_min(eps)
        return mean_flat((prediction - target).square()) / target_energy

    @staticmethod
    def _feature_std(features):
        flattened = features.float().reshape(-1, features.shape[-1])
        return torch.sqrt(flattened.var(dim=0, unbiased=False) + 1e-4).mean()

    def _factorization_losses(self, factorization, same_trajectory_mask=None):
        persistent_a, persistent_b = factorization['persistent'].chunk(2, dim=0)
        evolving_a, evolving_b = factorization['evolving'].chunk(2, dim=0)
        target = factorization['target']
        target_a, target_b = target.chunk(2, dim=0)

        inv_loss = 0.5 * (
            self._cosine_distance(persistent_a, persistent_b.detach())
            + self._cosine_distance(persistent_b, persistent_a.detach())
        )

        # The pair-symmetric component is the operational Persistent target;
        # the signed residual is the operational Evolving target.  These two
        # direct objectives prevent the joint recomposer from satisfying the
        # loss by silently routing all information through one branch.
        common_target = 0.5 * (target_a + target_b)
        common_targets = torch.cat([common_target, common_target], dim=0)
        residual_targets = target - common_targets

        persistent_component = factorization['persistent_component']
        evolving_component = factorization['evolving_component']
        persistent_loss_all = self._energy_normalized_mse(
            persistent_component, common_targets
        )
        persistent_loss_a, persistent_loss_b = persistent_loss_all.chunk(2, dim=0)
        persistent_loss = 0.5 * (persistent_loss_a + persistent_loss_b)
        evolving_loss_all = self._energy_normalized_mse(
            evolving_component, residual_targets
        )
        evolving_loss_a, evolving_loss_b = evolving_loss_all.chunk(2, dim=0)
        evolving_loss = 0.5 * (evolving_loss_a + evolving_loss_b)

        recom_loss_all = self._energy_normalized_mse(
            factorization['recomposed'], target
        )
        recom_a, recom_b = recom_loss_all.chunk(2, dim=0)
        recom_loss = 0.5 * (recom_a + recom_b)

        wrong_recom_loss = self._energy_normalized_mse(
            factorization['wrong_evolving_recomposed'], target
        )
        wrong_a, wrong_b = wrong_recom_loss.chunk(2, dim=0)
        wrong_recom_loss = 0.5 * (wrong_a + wrong_b)

        persistent_component_a, persistent_component_b = (
            persistent_component.chunk(2, dim=0)
        )
        persistent_swapped = torch.cat([
            persistent_component_b, persistent_component_a
        ], dim=0)
        persistent_missing_error = self._energy_normalized_mse(
            evolving_component, target
        ).mean()
        evolving_missing_error = self._energy_normalized_mse(
            persistent_swapped, target
        ).mean()
        target_energy = target.detach().float().square().mean().clamp_min(1e-6)
        common_energy = common_targets.detach().float().square().mean()
        residual_energy = residual_targets.detach().float().square().mean()

        persistent = factorization['persistent']
        evolving = factorization['evolving']
        decorrelation_loss = (
            F.normalize(persistent.float(), dim=-1)
            * F.normalize(evolving.float(), dim=-1)
        ).sum(dim=-1).pow(2).mean()
        persistent_std = self._feature_std(persistent)
        evolving_std = self._feature_std(evolving)
        variance_loss = F.relu(1.0 - persistent_std) + F.relu(1.0 - evolving_std)

        result = {
            'factor_inv_loss': inv_loss,
            'factor_persistent_loss': persistent_loss,
            'factor_evolving_loss': evolving_loss,
            'factor_recom_loss': recom_loss,
            'factor_decorrelation_loss': decorrelation_loss,
            'factor_variance_loss': variance_loss,
            'persistent_similarity': (1.0 - self._cosine_distance(
                persistent_a, persistent_b
            ).mean()).detach(),
            'evolving_similarity': (1.0 - self._cosine_distance(
                evolving_a, evolving_b
            ).mean()).detach(),
            'recomposition_gap': (
                wrong_recom_loss.mean() - recom_loss.mean()
            ).detach(),
            'persistent_usage_gap': (
                persistent_missing_error - recom_loss.mean()
            ).detach(),
            'evolving_usage_gap': (
                evolving_missing_error - recom_loss.mean()
            ).detach(),
            'common_energy_fraction': (common_energy / target_energy).detach(),
            'residual_energy_fraction': (residual_energy / target_energy).detach(),
            'persistent_std': persistent_std.detach(),
            'evolving_std': evolving_std.detach(),
        }
        if self.factor_transition:
            transitioned_a, transitioned_b = factorization['transitioned'].chunk(2, dim=0)
            transition_loss = 0.5 * (
                self._cosine_distance(transitioned_a, evolving_b.detach())
                + self._cosine_distance(transitioned_b, evolving_a.detach())
            )
            if same_trajectory_mask is not None:
                mask = same_trajectory_mask.to(
                    device=transition_loss.device, dtype=transition_loss.dtype
                )
                transition_loss = (
                    (transition_loss * mask).sum() / mask.sum().clamp_min(1.0)
                )
            result['factor_transition_loss'] = transition_loss
        return result

    def __call__(
        self,
        model,
        images,
        model_kwargs=None,
        zs=None,
        factorization_active=True,
        factor_batch_ratio=1.0,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        if self.trajectory_factorization and factorization_active:
            if not 0.0 < factor_batch_ratio <= 1.0:
                raise ValueError("factor_batch_ratio must be in (0, 1]")
            batch_size = images.shape[0]
            pair_count = min(batch_size, max(1, int(batch_size * factor_batch_ratio)))
            permutation = torch.randperm(batch_size, device=images.device)
            pair_indices = permutation[:pair_count]
            single_indices = permutation[pair_count:]
            pair_images = images[pair_indices]
            single_images = images[single_indices]

            time_a, time_b = self._sample_paired_times(pair_images)
            noise_a = torch.randn_like(pair_images)
            independent_noise = torch.randn_like(pair_images)
            cross_noise_mask = (
                torch.rand((pair_count, 1, 1, 1), device=images.device)
                < self.factor_pair_cross_noise_prob
            )
            noise_b = torch.where(cross_noise_mask, independent_noise, noise_a)

            alpha_a, sigma_a, d_alpha_a, d_sigma_a = self.interpolant(time_a)
            alpha_b, sigma_b, d_alpha_b, d_sigma_b = self.interpolant(time_b)
            model_input_a = alpha_a * pair_images + sigma_a * noise_a
            model_input_b = alpha_b * pair_images + sigma_b * noise_b
            if self.prediction != 'v':
                raise NotImplementedError()
            target_a = d_alpha_a * pair_images + d_sigma_a * noise_a
            target_b = d_alpha_b * pair_images + d_sigma_b * noise_b

            model_inputs = [model_input_a, model_input_b]
            model_times = [time_a, time_b]
            model_targets = [target_a, target_b]
            if single_images.shape[0] > 0:
                single_time = self._sample_times(single_images)
                single_noise = torch.randn_like(single_images)
                alpha_s, sigma_s, d_alpha_s, d_sigma_s = self.interpolant(single_time)
                model_inputs.append(alpha_s * single_images + sigma_s * single_noise)
                model_times.append(single_time)
                model_targets.append(d_alpha_s * single_images + d_sigma_s * single_noise)

            pair_delta = (time_b - time_a).flatten()
            assembled_kwargs = self._assemble_model_kwargs(
                model_kwargs, batch_size, pair_indices, single_indices
            )
            model_outputs = model(
                torch.cat(model_inputs, dim=0),
                torch.cat(model_times, dim=0).flatten(),
                trajectory_pair=True,
                factor_delta_t=pair_delta,
                return_factorization=True,
                factor_pair_count=pair_count,
                **assembled_kwargs,
            )
            output_a = model_outputs['x'][:pair_count]
            output_b = model_outputs['x'][pair_count:2 * pair_count]
            pair_denoising_loss = 0.5 * (
                mean_flat((output_a - target_a) ** 2)
                + mean_flat((output_b - target_b) ** 2)
            )
            if single_images.shape[0] > 0:
                output_single = model_outputs['x'][2 * pair_count:]
                single_denoising_loss = mean_flat(
                    (output_single - model_targets[-1]) ** 2
                )
                denoising_loss = torch.cat([
                    pair_denoising_loss, single_denoising_loss
                ], dim=0)
            else:
                denoising_loss = pair_denoising_loss

            paired_zs = None if zs is None else [
                torch.cat([
                    z[pair_indices], z[pair_indices], z[single_indices]
                ], dim=0) for z in zs
            ]
            losses = {
                'denoising_loss': denoising_loss,
                'proj_loss': self._projection_loss(
                    paired_zs,
                    model_outputs.get('zs'),
                    denoising_loss,
                    pair_count=pair_count,
                ),
                'mean_delta_t': pair_delta.abs().mean().detach(),
                'cross_noise_fraction': cross_noise_mask.float().mean().detach(),
                'factor_batch_fraction': denoising_loss.new_tensor(
                    pair_count / batch_size
                ).detach(),
            }
            losses.update(self._factorization_losses(
                model_outputs['factorization'],
                same_trajectory_mask=~cross_noise_mask.flatten(),
            ))
        else:
            time_input = self._sample_times(images)
            noises = torch.randn_like(images)
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

            model_input = alpha_t * images + sigma_t * noises
            if self.prediction == 'v':
                model_target = d_alpha_t * images + d_sigma_t * noises
            else:
                raise NotImplementedError() # TODO: add x or eps prediction
            # model forward
            raw_model_outputs = model(
                model_input,
                time_input.flatten(),
                **model_kwargs,
            )
            if isinstance(raw_model_outputs, dict):
                model_outputs = raw_model_outputs
            else:
                model_output, zs_tilde = raw_model_outputs
                model_outputs = {'x': model_output, 'zs': zs_tilde}
            model_output = model_outputs['x']
            denoising_loss = mean_flat((model_output - model_target) ** 2)
            losses = {
                'denoising_loss': denoising_loss,
                'proj_loss': self._projection_loss(
                    zs, model_outputs.get('zs'), denoising_loss
                ),
            }

        block_feas = model_outputs.get('block_feas', None)

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
