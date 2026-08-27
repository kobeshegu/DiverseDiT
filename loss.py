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


def noise_gap_weighted_cosine_loss(student, teacher, noise_gap, eps=1e-6):
    """Align only tokens noisier than the clean teacher, weighted by noise gap."""
    if student.shape != teacher.shape:
        raise ValueError(
            f"Student/teacher feature mismatch: {student.shape} vs {teacher.shape}"
        )
    if noise_gap.shape != student.shape[:2]:
        raise ValueError(
            f"Noise gap must have shape {student.shape[:2]}, got {noise_gap.shape}"
        )

    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    token_loss = 2 - 2 * (student * teacher).sum(dim=-1)
    weights = noise_gap.float().clamp_min(0)
    weight_sum = weights.sum()
    if weight_sum <= eps:
        return token_loss.sum() * 0, {
            "cosine": token_loss.new_zeros(()),
            "active_fraction": weights.new_zeros(()),
            "mean_gap": weights.new_zeros(()),
        }

    loss = (token_loss * weights).sum() / weight_sum
    active = weights > 0
    return loss, {
        "cosine": 1 - 0.5 * loss.detach(),
        "active_fraction": active.float().mean(),
        "mean_gap": weights[active].mean(),
    }


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
            ##### added new diversity losses
            block_contrastive_loss=False,
            block_contrastive_temperature=0.1,
            block_barlow_twins_loss=False,
            block_barlow_lambda=0.005,
            block_vicreg_loss=False,
            block_vicreg_lambda=25.0,
            block_vicreg_mu=25.0,
            block_vicreg_nu=1.0,
            self_flow=False,
            self_flow_mask_ratio=0.25,
            self_flow_teacher_depth=8,
            self_flow_contextual=False,
            self_flow_depth_pairs=None,
            self_flow_pair_weights=None,
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
        ##### new diversity losses
        self.block_contrastive_loss = block_contrastive_loss
        self.block_contrastive_temperature = block_contrastive_temperature
        self.block_barlow_twins_loss = block_barlow_twins_loss
        self.block_barlow_lambda = block_barlow_lambda
        self.block_vicreg_loss = block_vicreg_loss
        self.block_vicreg_lambda = block_vicreg_lambda
        self.block_vicreg_mu = block_vicreg_mu
        self.block_vicreg_nu = block_vicreg_nu
        self.self_flow = self_flow
        self.self_flow_mask_ratio = self_flow_mask_ratio
        self.self_flow_teacher_depth = self_flow_teacher_depth
        self.self_flow_contextual = self_flow_contextual
        self.self_flow_depth_pairs = list(self_flow_depth_pairs or [])
        self.self_flow_pair_weights = list(self_flow_pair_weights or [])

    @staticmethod
    def _patchify(imgs, model):
        """Convert (N, C, H, W) images to (N, T, patch^2 * C) tokens."""
        p = model.patch_size if hasattr(model, 'patch_size') else model.module.patch_size
        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p * p * c)
        return x

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

    def _sample_timesteps(self, images):
        shape = (images.shape[0], 1, 1, 1)
        if self.weighting == "uniform":
            time_input = torch.rand(shape, device=images.device, dtype=images.dtype)
        elif self.weighting == "lognormal":
            rnd_normal = torch.randn(shape, device=images.device, dtype=images.dtype)
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
        else:
            raise ValueError(f"Unsupported timestep weighting: {self.weighting}")
        return time_input

    @staticmethod
    def _unpatchify_timesteps(token_timesteps, images, model):
        base_model = model.module if hasattr(model, "module") else model
        patch_size = base_model.patch_size
        grid_size = images.shape[-1] // patch_size
        if token_timesteps.shape[1] != grid_size * grid_size:
            raise ValueError("Token timestep count does not match the latent patch grid")
        timesteps = token_timesteps.reshape(images.shape[0], 1, grid_size, grid_size)
        return timesteps.repeat_interleave(patch_size, 2).repeat_interleave(patch_size, 3)

    def _self_flow_loss(
        self,
        model,
        teacher_model,
        images,
        model_kwargs,
        contextual_predictor=None,
    ):
        if teacher_model is None:
            raise ValueError("Self-Flow requires an EMA teacher model")

        base_model = model.module if hasattr(model, "module") else model
        num_tokens = base_model.x_embedder.num_patches
        model_kwargs = dict(model_kwargs)
        if (
            "force_drop_ids" not in model_kwargs
            and base_model.y_embedder.dropout_prob > 0
        ):
            model_kwargs["force_drop_ids"] = (
                torch.rand(images.shape[0], device=images.device)
                < base_model.y_embedder.dropout_prob
            )
        t = self._sample_timesteps(images)
        s = self._sample_timesteps(images)
        mask = torch.rand(
            images.shape[0], num_tokens, device=images.device
        ) < self.self_flow_mask_ratio
        token_t = t.flatten(1).expand(-1, num_tokens)
        token_s = s.flatten(1).expand(-1, num_tokens)
        token_timesteps = torch.where(mask, token_s, token_t)
        spatial_timesteps = self._unpatchify_timesteps(
            token_timesteps, images, model
        )

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(
            spatial_timesteps
        )
        model_input = alpha_t * images + sigma_t * noises
        model_target = d_alpha_t * images + d_sigma_t * noises

        if self.self_flow_contextual:
            if contextual_predictor is None:
                raise ValueError(
                    "Contextual Self-Flow requires a multi-depth predictor"
                )
            student_depths = sorted({
                source for source, _ in self.self_flow_depth_pairs
            })
            teacher_depths = sorted({
                target for _, target in self.self_flow_depth_pairs
            })
            student_outputs = model(
                model_input,
                token_timesteps,
                return_features=True,
                feature_depths=student_depths,
                **model_kwargs,
            )
        else:
            student_outputs = model(
                model_input,
                token_timesteps,
                return_features=True,
                feature_depth=base_model.encoder_depth,
                **model_kwargs,
            )
            student_features = student_outputs["zs"][0]

        teacher_t = torch.minimum(t, s)
        teacher_alpha, teacher_sigma, _, _ = self.interpolant(teacher_t)
        teacher_input = teacher_alpha * images + teacher_sigma * noises
        with torch.no_grad():
            if self.self_flow_contextual:
                teacher_outputs = teacher_model(
                    teacher_input,
                    teacher_t.flatten(),
                    return_features=True,
                    feature_depths=teacher_depths,
                    **model_kwargs,
                )
            else:
                teacher_outputs = teacher_model(
                    teacher_input,
                    teacher_t.flatten(),
                    return_features=True,
                    feature_depth=self.self_flow_teacher_depth,
                    **model_kwargs,
                )
                teacher_features = teacher_outputs["features"]

        denoising_loss = mean_flat(
            (student_outputs["x"] - model_target) ** 2
        )
        losses = {
            "denoising_loss": denoising_loss,
            "proj_loss": denoising_loss.new_zeros(()),
            "self_flow_mask_fraction": mask.float().mean(),
            "self_flow_timestep_gap": (t - s).abs().mean(),
            "self_flow_teacher_timestep": teacher_t.mean(),
        }
        if self.self_flow_contextual:
            teacher_token_t = teacher_t.flatten(1).expand(-1, num_tokens)
            noise_gap = (token_timesteps - teacher_token_t).clamp_min(0)
            pair_losses = []
            for (source_depth, target_depth), pair_weight in zip(
                self.self_flow_depth_pairs,
                self.self_flow_pair_weights,
            ):
                prediction = contextual_predictor(
                    student_outputs["features"][source_depth],
                    source_depth,
                    target_depth,
                )
                pair_loss, pair_metrics = noise_gap_weighted_cosine_loss(
                    prediction,
                    teacher_outputs["features"][target_depth],
                    noise_gap,
                )
                pair_losses.append(pair_loss * pair_weight)
                prefix = f"self_flow_d{source_depth}_to_d{target_depth}"
                losses[f"{prefix}_loss"] = pair_loss.detach()
                losses[f"{prefix}_cosine"] = pair_metrics["cosine"]
            losses["self_flow_rep_loss"] = torch.stack(pair_losses).sum()
            losses["self_flow_hard_fraction"] = (noise_gap > 0).float().mean()
            hard_gap = noise_gap[noise_gap > 0]
            losses["self_flow_hard_gap"] = (
                hard_gap.mean()
                if hard_gap.numel() > 0
                else noise_gap.new_zeros(())
            )
        else:
            losses["self_flow_rep_loss"] = -F.cosine_similarity(
                student_features.float(),
                teacher_features.float(),
                dim=-1,
            ).mean(dim=1)
        return losses

    def __call__(
        self,
        model,
        images,
        model_kwargs=None,
        zs=None,
        teacher_model=None,
        contextual_predictor=None,
    ):
        if model_kwargs == None:
            model_kwargs = {}
        if self.self_flow:
            return self._self_flow_loss(
                model,
                teacher_model,
                images,
                model_kwargs,
                contextual_predictor=contextual_predictor,
            )
        # sample timesteps
        time_input = self._sample_timesteps(images)
        
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

        # Patchify target for auxiliary head loss (token-level target)
        # model_target: (N, C, H, W) -> (N, T, patch^2 * C)
        model_target_tokens = self._patchify(model_target, model) if model_outputs.get('aux_outputs', None) else None

        # projection loss
        losses = {'denoising_loss': denoising_loss}
        proj_loss = denoising_loss.new_zeros(())
        if zs and zs_tilde:
            bsz = zs[0].shape[0]
            for z, z_tilde in zip(zs, zs_tilde):
                for z_j, z_tilde_j in zip(z, z_tilde):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                    z_j = torch.nn.functional.normalize(z_j, dim=-1)
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)
        losses['proj_loss'] = proj_loss

        if self.block_diversity_loss:
            assert block_feas is not None, "block_feas is required for block_difference_loss"
            block_diff_loss = self.compute_block_diversity_loss(block_feas)
            losses['block_diversity_loss'] = block_diff_loss

        if self.block_contrastive_loss:
            assert block_feas is not None, "block_feas is required for block_contrastive_loss"
            contrastive_loss = self.compute_block_contrastive_loss(block_feas)
            losses['block_contrastive_loss'] = contrastive_loss

        if self.block_barlow_twins_loss:
            assert block_feas is not None, "block_feas is required for block_barlow_twins_loss"
            barlow_loss = self.compute_block_barlow_twins_loss(block_feas)
            losses['block_barlow_twins_loss'] = barlow_loss

        if self.block_vicreg_loss:
            assert block_feas is not None, "block_feas is required for block_vicreg_loss"
            vicreg_loss = self.compute_block_vicreg_loss(block_feas)
            losses['block_vicreg_loss'] = vicreg_loss

        # block-wise auxiliary head loss
        aux_outputs = model_outputs.get('aux_outputs', None)
        if aux_outputs is not None and len(aux_outputs) > 0:
            aux_loss = 0.0
            for layer_idx, aux_pred in aux_outputs.items():
                # aux_pred: (N, T, patch_size^2 * C), same shape as model_target after patchify
                # target: the denoising target for the same input
                aux_loss += mean_flat((aux_pred - model_target_tokens) ** 2).mean()
            aux_loss /= len(aux_outputs)
            losses['block_aux_loss'] = aux_loss

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

    ############################################################################
    #  New Diversity Loss: Block-wise Contrastive Loss (InfoNCE)               #
    ############################################################################
    def compute_block_contrastive_loss(self, block_feas):
        """
        Block-wise contrastive loss using InfoNCE.

        Treats each block's global-pooled representation as an embedding.
        Within the same block, different samples form positive pairs;
        representations from *different* blocks are treated as negatives.
        This pushes different blocks apart in a unified embedding space,
        which is stronger than pairwise cosine similarity because all
        negatives are contrasted simultaneously.

        Args:
            block_feas: dict {block_idx: features (N, T, D)}
        Returns:
            loss: scalar  (lower = more diverse across blocks)
        """
        if len(block_feas) < 2:
            return torch.tensor(0.0, device=list(block_feas.values())[0].device)

        block_indices = sorted(block_feas.keys())
        tau = self.block_contrastive_temperature
        eps = 1e-8

        # Global average pool each block -> (N, D), then average over batch -> (D,)
        embeddings = []
        for idx in block_indices:
            feat = block_feas[idx]  # (N, T, D)
            pooled = feat.mean(dim=1)  # (N, D)  -- spatial average
            embeddings.append(pooled)

        # Stack: (num_blocks, N, D)
        embeddings = torch.stack(embeddings, dim=0)
        num_blocks, N, D = embeddings.shape

        # Average over batch -> (num_blocks, D)
        block_embs = embeddings.mean(dim=1)
        block_embs = F.normalize(block_embs, dim=-1, eps=eps)

        # Similarity matrix (num_blocks x num_blocks)
        sim_matrix = block_embs @ block_embs.T / tau  # (B_k, B_k)

        # InfoNCE: for each block, its "positive" is itself (diagonal),
        # negatives are all other blocks.  We want the diagonal to dominate
        # -> minimising this loss pushes off-diagonal similarities DOWN.
        # But here we *invert* the goal: we want blocks to be DIFFERENT,
        # so we maximise off-diagonal similarity's negativeness.
        # Equivalent: minimize the mean off-diagonal similarity.
        mask = ~torch.eye(num_blocks, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask].view(num_blocks, num_blocks - 1)

        # Use logsumexp for numerical stability
        loss = torch.logsumexp(off_diag, dim=1).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=loss.device)
        return loss

    ############################################################################
    #  New Diversity Loss: Barlow Twins Cross-Correlation                      #
    ############################################################################
    def compute_block_barlow_twins_loss(self, block_feas):
        """
        Barlow Twins-style cross-correlation loss between block pairs.

        For each pair of blocks, compute the cross-correlation matrix C of
        their batch-normalised representations.  The loss penalises:
          - Diagonal elements deviating from 0  (same dimension across blocks
            should NOT correlate — blocks should be diverse)
          - Off-diagonal elements deviating from 0  (cross-dimension should
            also not correlate)

        Unlike standard Barlow Twins (which wants diagonal = 1 for self-supervised
        invariance), here we set the target to the ZERO matrix because our goal
        is *de-correlation* between different blocks.

        Args:
            block_feas: dict {block_idx: features (N, T, D)}
        Returns:
            loss: scalar
        """
        if len(block_feas) < 2:
            return torch.tensor(0.0, device=list(block_feas.values())[0].device)

        block_indices = sorted(block_feas.keys())
        eps = 1e-8
        lam = self.block_barlow_lambda  # weight for off-diagonal terms

        # Pool each block: (N, T, D) -> (N, D)
        pooled = {}
        for idx in block_indices:
            feat = block_feas[idx].mean(dim=1)  # (N, D)
            # Batch normalise (zero-mean, unit-std per dimension)
            feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + eps)
            pooled[idx] = feat

        # Sample block pairs (limit to ~10 pairs for efficiency)
        max_pairs = min(10, len(block_indices) * (len(block_indices) - 1) // 2)
        loss = 0.0
        pair_count = 0

        for i_pos, i_idx in enumerate(block_indices):
            for j_idx in block_indices[i_pos + 1:]:
                if pair_count >= max_pairs:
                    break
                z_a = pooled[i_idx]  # (N, D)
                z_b = pooled[j_idx]  # (N, D)
                N_samples = z_a.size(0)
                D = z_a.size(1)

                # Cross-correlation matrix: (D, D)
                C = (z_a.T @ z_b) / N_samples  # (D, D)

                # Diagonal loss: penalise |C_ii|
                diag_loss = (C.diagonal() ** 2).sum() / D

                # Off-diagonal loss: penalise |C_ij| for i != j
                off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=C.device)
                off_diag_loss = (C[off_diag_mask] ** 2).sum() / (D * (D - 1))

                loss += diag_loss + lam * off_diag_loss
                pair_count += 1
            if pair_count >= max_pairs:
                break

        loss = loss / max(pair_count, 1)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=list(block_feas.values())[0].device)
        return loss

    ############################################################################
    #  New Diversity Loss: VICReg (Variance-Invariance-Covariance)             #
    ############################################################################
    def compute_block_vicreg_loss(self, block_feas):
        """
        VICReg-style diversity loss between block representations.

        Three terms encourage block diversity:
          1. Variance term  (per-block): ensure each block's feature dimensions
             have sufficient variance (prevents collapse).
          2. Invariance term (across blocks): MINIMISE similarity between
             different blocks' representations (we invert VICReg's original
             goal — here invariance = "blocks should NOT agree").
          3. Covariance term (per-block): decorrelate feature dimensions within
             each block, so every block uses its capacity efficiently.

        Args:
            block_feas: dict {block_idx: features (N, T, D)}
        Returns:
            loss: scalar
        """
        if len(block_feas) < 2:
            return torch.tensor(0.0, device=list(block_feas.values())[0].device)

        block_indices = sorted(block_feas.keys())
        eps = 1e-4
        lam = self.block_vicreg_lambda   # variance weight
        mu = self.block_vicreg_mu        # invariance weight (cross-block similarity)
        nu = self.block_vicreg_nu        # covariance weight

        # Pool: (N, T, D) -> (N, D)
        pooled = {}
        for idx in block_indices:
            pooled[idx] = block_feas[idx].mean(dim=1)  # (N, D)

        N_samples = list(pooled.values())[0].size(0)
        D = list(pooled.values())[0].size(1)

        # ---- 1. Variance loss (per-block) ----
        # Hinge loss: std along batch dim must exceed 1
        var_loss = 0.0
        for idx in block_indices:
            z = pooled[idx]  # (N, D)
            std_z = torch.sqrt(z.var(dim=0) + eps)  # (D,)
            var_loss += torch.relu(1.0 - std_z).mean()
        var_loss /= len(block_indices)

        # ---- 2. Invariance loss (cross-block) ----
        # We WANT blocks to be different -> maximise distance.
        # VICReg-style: MSE between block pairs, but we negate it (reward distance).
        inv_loss = 0.0
        max_pairs = min(10, len(block_indices) * (len(block_indices) - 1) // 2)
        pair_count = 0
        for i_pos, i_idx in enumerate(block_indices):
            for j_idx in block_indices[i_pos + 1:]:
                if pair_count >= max_pairs:
                    break
                z_a = F.normalize(pooled[i_idx], dim=-1, eps=1e-8)
                z_b = F.normalize(pooled[j_idx], dim=-1, eps=1e-8)
                # Cosine similarity -> we want this to be small
                sim = (z_a * z_b).sum(dim=-1).mean()
                inv_loss += sim.abs()
                pair_count += 1
            if pair_count >= max_pairs:
                break
        inv_loss /= max(pair_count, 1)

        # ---- 3. Covariance loss (per-block) ----
        # Decorrelate dimensions within each block
        cov_loss = 0.0
        for idx in block_indices:
            z = pooled[idx]  # (N, D)
            z_centered = z - z.mean(dim=0, keepdim=True)
            cov = (z_centered.T @ z_centered) / max(N_samples - 1, 1)  # (D, D)
            # Penalise off-diagonal
            off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=cov.device)
            cov_loss += (cov[off_diag_mask] ** 2).sum() / D
        cov_loss /= len(block_indices)

        total = lam * var_loss + mu * inv_loss + nu * cov_loss

        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=list(block_feas.values())[0].device)

        return total
