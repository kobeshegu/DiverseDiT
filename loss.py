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
            factor_orbit_mode="legacy",
            factor_orbit_noise_only_prob=0.5,
            factor_min_delta_t=0.15,
            factor_max_delta_t=0.7,
            factor_transition=False,
            factor_native_parameterization=False,
            factor_native_shuffle_source=False,
            factor_semantic_conditioning=False,
            factor_semantic_shuffle_targets=False,
            factor_reliable_target=False,
            factor_reliability_keep_ratio=1.0,
            factor_reliability_floor=0.0,
            factor_adversarial=False,
            factor_adversarial_timestep_bins=8,
            factor_adversarial_shuffle_labels=False,
            trajectory_invariance=False,
            invariant_min_delta_t=0.05,
            invariant_max_delta_t=0.2,
            invariant_max_t=0.8,
            invariant_snr_power=1.0,
            invariant_variance_target=1.0,
            invariant_spatial_variance_target=0.5,
            factor_clean_consensus=False,
            factor_clean_consensus_temperature=0.25,
            factor_clean_consensus_shuffle_targets=False,
            factor_selective_invariance=False,
            factor_selective_weighting="task",
            factor_selective_shuffle_targets=False,
            factor_selective_shuffle_utility=False,
            factor_selective_variance_target=1.0,
            factor_shared_repa=False,
            factor_shared_self_distill=False,
            factor_shared_target_temperature=0.25,
            factor_shared_snr_power=1.0,
            factor_shared_self_distill_shuffle_targets=False,
            factor_shared_variance_target=1.0,
            factor_shared_contrastive=False,
            factor_shared_contrastive_temperature=0.2,
            factor_shared_relation=False,
            factor_evolving_separation=False,
            factor_evolving_separation_margin=0.5,
            factor_self_flow_full_align=False,
            factor_self_flow_source_align=False,
            factor_self_flow_shuffle_teacher=False,
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
        self.factor_orbit_mode = factor_orbit_mode
        self.factor_orbit_noise_only_prob = factor_orbit_noise_only_prob
        self.factor_min_delta_t = factor_min_delta_t
        self.factor_max_delta_t = factor_max_delta_t
        self.factor_transition = factor_transition
        self.factor_native_parameterization = factor_native_parameterization
        self.factor_native_shuffle_source = factor_native_shuffle_source
        self.factor_semantic_conditioning = factor_semantic_conditioning
        self.factor_semantic_shuffle_targets = factor_semantic_shuffle_targets
        self.factor_reliable_target = factor_reliable_target
        self.factor_reliability_keep_ratio = factor_reliability_keep_ratio
        self.factor_reliability_floor = factor_reliability_floor
        self.factor_adversarial = factor_adversarial
        self.factor_adversarial_timestep_bins = factor_adversarial_timestep_bins
        self.factor_adversarial_shuffle_labels = (
            factor_adversarial_shuffle_labels
        )
        self.factor_clean_consensus = factor_clean_consensus
        self.factor_clean_consensus_temperature = (
            factor_clean_consensus_temperature
        )
        self.factor_clean_consensus_shuffle_targets = (
            factor_clean_consensus_shuffle_targets
        )
        self.factor_selective_invariance = factor_selective_invariance
        self.factor_selective_weighting = factor_selective_weighting
        self.factor_selective_shuffle_targets = (
            factor_selective_shuffle_targets
        )
        self.factor_selective_shuffle_utility = (
            factor_selective_shuffle_utility
        )
        self.factor_selective_variance_target = (
            factor_selective_variance_target
        )
        self.factor_shared_repa = factor_shared_repa
        self.factor_shared_self_distill = factor_shared_self_distill
        self.factor_shared_target_temperature = factor_shared_target_temperature
        self.factor_shared_snr_power = factor_shared_snr_power
        self.factor_shared_self_distill_shuffle_targets = (
            factor_shared_self_distill_shuffle_targets
        )
        self.factor_shared_variance_target = factor_shared_variance_target
        self.factor_shared_contrastive = factor_shared_contrastive
        self.factor_shared_contrastive_temperature = (
            factor_shared_contrastive_temperature
        )
        self.factor_shared_relation = factor_shared_relation
        self.factor_evolving_separation = factor_evolving_separation
        self.factor_evolving_separation_margin = (
            factor_evolving_separation_margin
        )
        self.factor_self_flow_full_align = factor_self_flow_full_align
        self.factor_self_flow_source_align = factor_self_flow_source_align
        self.factor_self_flow_shuffle_teacher = factor_self_flow_shuffle_teacher
        self.trajectory_invariance = trajectory_invariance
        self.invariant_min_delta_t = invariant_min_delta_t
        self.invariant_max_delta_t = invariant_max_delta_t
        self.invariant_max_t = invariant_max_t
        self.invariant_snr_power = invariant_snr_power
        self.invariant_variance_target = invariant_variance_target
        self.invariant_spatial_variance_target = (
            invariant_spatial_variance_target
        )
        if self.trajectory_factorization and self.trajectory_invariance:
            raise ValueError(
                "trajectory factorization and trajectory invariance are "
                "mutually exclusive"
            )
        if self.factor_semantic_conditioning and not self.trajectory_factorization:
            raise ValueError(
                "semantic conditioning requires trajectory factorization"
            )
        if (
            self.factor_self_flow_full_align
            or self.factor_self_flow_source_align
        ) and not self.trajectory_factorization:
            raise ValueError(
                "self-flow EMA alignment requires trajectory factorization"
            )
        if self.factor_semantic_conditioning and self.factor_native_parameterization:
            raise ValueError(
                "semantic conditioning and native parameterization are "
                "mutually exclusive"
            )
        if not 0.0 <= factor_pair_cross_noise_prob <= 1.0:
            raise ValueError("factor_pair_cross_noise_prob must be in [0, 1]")
        if factor_orbit_mode not in {
            "legacy", "orthogonal", "time-only", "noise-only", "antithetic"
        }:
            raise ValueError(
                "factor_orbit_mode must be legacy, orthogonal, time-only, "
                "noise-only, or antithetic"
            )
        if not 0.0 <= factor_orbit_noise_only_prob <= 1.0:
            raise ValueError("factor_orbit_noise_only_prob must be in [0, 1]")
        if not 0.0 <= factor_min_delta_t <= factor_max_delta_t < 1.0:
            raise ValueError("factor timestep deltas must satisfy 0 <= min <= max < 1")
        if not 0.0 < factor_reliability_keep_ratio <= 1.0:
            raise ValueError("factor_reliability_keep_ratio must be in (0, 1]")
        if not 0.0 <= factor_reliability_floor <= 1.0:
            raise ValueError("factor_reliability_floor must be in [0, 1]")
        if factor_adversarial_timestep_bins < 2:
            raise ValueError("factor_adversarial_timestep_bins must be at least 2")
        if factor_clean_consensus_temperature <= 0:
            raise ValueError(
                "factor_clean_consensus_temperature must be positive"
            )
        if factor_selective_weighting not in {
            "uniform", "stability", "task"
        }:
            raise ValueError(
                "factor_selective_weighting must be uniform, stability, or task"
            )
        if factor_selective_variance_target <= 0:
            raise ValueError(
                "factor_selective_variance_target must be positive"
            )
        if factor_shared_target_temperature <= 0:
            raise ValueError(
                "factor_shared_target_temperature must be positive"
            )
        if factor_shared_snr_power < 0:
            raise ValueError("factor_shared_snr_power must be non-negative")
        if factor_shared_variance_target <= 0:
            raise ValueError("factor_shared_variance_target must be positive")
        if factor_shared_contrastive_temperature <= 0:
            raise ValueError(
                "factor_shared_contrastive_temperature must be positive"
            )
        if factor_evolving_separation_margin < 0:
            raise ValueError(
                "factor_evolving_separation_margin must be non-negative"
            )
        if self.factor_adversarial and factor_orbit_mode != "orthogonal":
            raise ValueError(
                "factor adversarial nuisance learning requires an orthogonal orbit"
            )
        if self.factor_native_parameterization and factor_orbit_mode != "antithetic":
            raise ValueError(
                "native source/noise parameterization requires an antithetic orbit"
            )
        if (
            self.factor_native_shuffle_source
            and not self.factor_native_parameterization
        ):
            raise ValueError(
                "shuffling native source targets requires native parameterization"
            )
        if (
            self.factor_semantic_shuffle_targets
            and not self.factor_semantic_conditioning
        ):
            raise ValueError(
                "shuffling semantic targets requires semantic conditioning"
            )
        if (
            self.factor_self_flow_shuffle_teacher
            and not (
                self.factor_self_flow_full_align
                or self.factor_self_flow_source_align
            )
        ):
            raise ValueError(
                "shuffling self-flow teacher targets requires EMA alignment"
            )
        if (
            self.factor_adversarial
            and not 0.0 < factor_orbit_noise_only_prob < 1.0
        ):
            raise ValueError(
                "factor adversarial orbit learning needs both intervention types"
            )
        if not 0.0 <= invariant_min_delta_t <= invariant_max_delta_t < 1.0:
            raise ValueError(
                "invariant timestep deltas must satisfy 0 <= min <= max < 1"
            )
        if not 0.0 < invariant_max_t <= 1.0:
            raise ValueError("invariant_max_t must be in (0, 1]")
        if invariant_max_delta_t > invariant_max_t:
            raise ValueError("invariant_max_delta_t cannot exceed invariant_max_t")
        if invariant_snr_power < 0:
            raise ValueError("invariant_snr_power must be non-negative")
        if invariant_variance_target <= 0:
            raise ValueError("invariant_variance_target must be positive")
        if invariant_spatial_variance_target <= 0:
            raise ValueError(
                "invariant_spatial_variance_target must be positive"
            )

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

    def _sample_factor_orbit(self, images):
        """Sample legacy or causally isolated two-view trajectory orbits.

        Antithetic mode holds time fixed and uses an epsilon/-epsilon pair.
        The legacy mode preserves the historical TFCR intervention: every
        pair changes time and ``factor_pair_cross_noise_prob`` additionally
        changes noise.  Orthogonal modes change exactly one nuisance, making
        time-only and noise-only effects independently identifiable.
        """
        pair_count = images.shape[0]
        mask_shape = (pair_count, 1, 1, 1)

        if self.factor_orbit_mode == "antithetic":
            # A same-timestep epsilon/-epsilon pair makes the source and noise
            # components algebraically identifiable for any linear interpolant.
            time_a = self._sample_times(images)
            time_b = time_a
            noise_a = torch.randn_like(images)
            noise_b = -noise_a
            noise_changed = torch.ones(
                mask_shape, device=images.device, dtype=torch.bool
            )
            time_only = torch.zeros_like(noise_changed)
            noise_only = torch.ones_like(noise_changed)
            joint = torch.zeros_like(noise_changed)
        else:
            time_a, paired_time_b = self._sample_paired_times(images)
            noise_a = torch.randn_like(images)
            independent_noise = torch.randn_like(images)
            if self.factor_orbit_mode == "legacy":
                noise_changed = (
                    torch.rand(mask_shape, device=images.device)
                    < self.factor_pair_cross_noise_prob
                )
                time_b = paired_time_b
                noise_b = torch.where(noise_changed, independent_noise, noise_a)
                time_only = ~noise_changed
                noise_only = torch.zeros_like(noise_changed)
                joint = noise_changed
            else:
                if self.factor_orbit_mode == "orthogonal":
                    noise_only = (
                        torch.rand(mask_shape, device=images.device)
                        < self.factor_orbit_noise_only_prob
                    )
                elif self.factor_orbit_mode == "noise-only":
                    noise_only = torch.ones(
                        mask_shape, device=images.device, dtype=torch.bool
                    )
                else:
                    noise_only = torch.zeros(
                        mask_shape, device=images.device, dtype=torch.bool
                    )
                time_only = ~noise_only
                joint = torch.zeros_like(noise_only)
                # Noise-only views use the original training-time distribution,
                # rather than inheriting one endpoint of a gap-conditioned pair.
                noise_only_time = self._sample_times(images)
                time_a = torch.where(noise_only, noise_only_time, time_a)
                time_b = torch.where(noise_only, noise_only_time, paired_time_b)
                noise_b = torch.where(noise_only, independent_noise, noise_a)
                noise_changed = noise_only

        return {
            'time_a': time_a,
            'time_b': time_b,
            'noise_a': noise_a,
            'noise_b': noise_b,
            'time_only_mask': time_only.flatten(),
            'noise_only_mask': noise_only.flatten(),
            'joint_mask': joint.flatten(),
            'noise_changed_mask': noise_changed.flatten(),
        }

    @staticmethod
    def _average_paired_views(per_view, pair_count):
        """Map [view-a, view-b, singles] losses back to source count."""
        paired = 0.5 * (
            per_view[:pair_count]
            + per_view[pair_count:2 * pair_count]
        )
        singles = per_view[2 * pair_count:]
        return torch.cat([paired, singles], dim=0)

    def _native_parameterization_losses(
        self,
        native,
        source_target,
        noise_target,
        velocity_target,
        times,
        pair_count,
    ):
        """Supervise exact x0/epsilon factors and the retained base head."""
        source_prediction = native['source']
        noise_prediction = native['noise']
        base_velocity = native['base_velocity']
        expected_shape = velocity_target.shape
        for name, value in (
            ('source', source_prediction),
            ('noise', noise_prediction),
            ('base_velocity', base_velocity),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"native {name} shape {value.shape} != {expected_shape}"
                )

        time_input = times.reshape(-1, 1, 1, 1)
        alpha, sigma, _, _ = self.interpolant(time_input)
        energy = (alpha.square() + sigma.square()).clamp_min(1e-6)
        source_weight = (alpha.square() / energy).flatten()
        noise_weight = (sigma.square() / energy).flatten()

        source_error = mean_flat(
            (source_prediction - source_target).float().square()
        )
        noise_error = mean_flat(
            (noise_prediction - noise_target).float().square()
        )
        base_error = mean_flat(
            (base_velocity - velocity_target).float().square()
        )
        source_loss = self._average_paired_views(
            source_error * source_weight, pair_count
        )
        noise_loss = self._average_paired_views(
            noise_error * noise_weight, pair_count
        )
        base_loss = self._average_paired_views(base_error, pair_count)

        noise_a = noise_prediction[:pair_count]
        noise_b = noise_prediction[pair_count:2 * pair_count]
        antithetic_loss = mean_flat((noise_a + noise_b).float().square())
        source_a = source_prediction[:pair_count]
        source_b = source_prediction[pair_count:2 * pair_count]
        return {
            'factor_native_source_loss': source_loss,
            'factor_native_noise_loss': noise_loss,
            'factor_native_base_loss': base_loss,
            'factor_native_antithetic_loss': antithetic_loss,
            'factor_native_source_error': source_error.mean().detach(),
            'factor_native_noise_error': noise_error.mean().detach(),
            'factor_native_base_error': base_error.mean().detach(),
            'factor_native_source_pair_gap': mean_flat(
                (source_a - source_b).float().square()
            ).mean().detach(),
            'factor_native_noise_antisymmetry_error': (
                antithetic_loss.mean().detach()
            ),
            'factor_native_source_weight': source_weight.mean().detach(),
            'factor_native_noise_weight': noise_weight.mean().detach(),
        }

    def _semantic_factorization_losses(
        self,
        semantic,
        semantic_targets,
        reference,
        pair_count,
    ):
        """Align only the source code to clean semantics and preserve state."""
        source = semantic['source']
        evolving = semantic['evolving']
        source_predictions = semantic['source_predictions']
        if source_predictions is None or len(source_predictions) == 0:
            raise ValueError(
                "semantic factorization requires source target predictions"
            )
        if semantic_targets is None or len(semantic_targets) == 0:
            raise ValueError(
                "semantic factorization requires clean external targets"
            )
        if len(source_predictions) != len(semantic_targets):
            raise ValueError(
                "semantic prediction and target encoder counts do not match"
            )

        source_a = source[:pair_count]
        source_b = source[pair_count:2 * pair_count]
        source_consistency = 0.5 * (
            self._cosine_distance(source_a, source_b.detach())
            + self._cosine_distance(source_b, source_a.detach())
        )
        decorrelation = (
            F.normalize(source.float(), dim=-1)
            * F.normalize(evolving.float(), dim=-1)
        ).sum(dim=-1).square().mean()
        source_std = self._feature_std(source)
        evolving_std = self._feature_std(evolving)

        return {
            'factor_semantic_repa_loss': self._projection_alignment_loss(
                semantic_targets,
                source_predictions,
                reference,
                group_count=pair_count,
                view_count=2,
            ),
            'factor_semantic_source_consistency_loss': source_consistency,
            'factor_semantic_decorrelation_loss': decorrelation,
            'factor_semantic_source_similarity': (
                1.0 - self._cosine_distance(source_a, source_b).mean()
            ).detach(),
            'factor_semantic_source_evolving_cosine_sq': decorrelation.detach(),
            'factor_semantic_source_std': source_std.detach(),
            'factor_semantic_evolving_std': evolving_std.detach(),
            'factor_semantic_modulation_rms': semantic[
                'modulation_rms'
            ].detach(),
        }

    def _sample_invariant_times(self, images):
        """Sample a controlled non-zero time intervention."""
        batch_size = images.shape[0]
        shape = (batch_size, 1, 1, 1)
        delta = self.invariant_min_delta_t + torch.rand(
            shape, device=images.device, dtype=images.dtype
        ) * (
            self.invariant_max_delta_t - self.invariant_min_delta_t
        )
        low = torch.rand(
            shape, device=images.device, dtype=images.dtype
        ) * (self.invariant_max_t - delta)
        high = low + delta
        swap = torch.rand(shape, device=images.device) < 0.5
        return torch.where(swap, high, low), torch.where(swap, low, high)

    @staticmethod
    def _assemble_model_kwargs(
        model_kwargs, batch_size, pair_indices, single_indices, view_count=2
    ):
        assembled_kwargs = {}
        for key, value in model_kwargs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                pair_value = value[pair_indices]
                assembled_kwargs[key] = torch.cat(
                    [pair_value] * view_count + [value[single_indices]], dim=0
                )
            else:
                assembled_kwargs[key] = value
        return assembled_kwargs

    def _projection_alignment_loss(
        self, zs, zs_tilde, reference, group_count=None, view_count=1
    ):
        if zs is None or zs_tilde is None or len(zs) == 0:
            return reference.new_zeros(())
        losses = []
        for target, prediction in zip(zs, zs_tilde):
            target = F.normalize(target, dim=-1)
            prediction = F.normalize(prediction, dim=-1)
            per_view = -(target * prediction).sum(dim=-1).mean(dim=-1)
            if group_count is not None:
                grouped_loss = torch.stack([
                    per_view[
                        view_index * group_count:(view_index + 1) * group_count
                    ]
                    for view_index in range(view_count)
                ]).mean(dim=0)
                per_view = torch.cat([
                    grouped_loss, per_view[view_count * group_count:]
                ])
            losses.append(per_view.mean())
        return torch.stack(losses).mean()

    def _projection_loss(
        self, zs, zs_tilde, reference, group_count=None, view_count=1
    ):
        if not self.projection:
            return reference.new_zeros(())
        return self._projection_alignment_loss(
            zs,
            zs_tilde,
            reference,
            group_count=group_count,
            view_count=view_count,
        )

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

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(device=values.device, dtype=values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _weighted_mean(values, weights):
        weights = weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def _source_reliability(self, time, power=None):
        """Fraction of interpolant energy attributable to the clean source."""
        alpha, sigma, _, _ = self.interpolant(time)
        reliability = alpha.float().square() / (
            alpha.float().square() + sigma.float().square() + 1e-6
        )
        if power is None:
            power = self.invariant_snr_power
        return reliability.flatten().pow(power)

    def _factor_target_reliability(
        self, target_a, target_b, intervention_masks=None
    ):
        """Select target channels stable within an orbit yet distinct by source.

        The statistic is an intraclass-correlation analogue.  Between-source
        variance rewards informative directions; within-pair variance rejects
        directions dominated by the intervened timestep or noise.  Selection
        is deliberately detached so the backbone cannot game the gate.
        """
        embedding_a = target_a.detach().float().mean(dim=1)
        embedding_b = target_b.detach().float().mean(dim=1)
        mask_matrix = None
        if intervention_masks is not None:
            mask_matrix = torch.stack([
                mask.to(device=embedding_a.device, dtype=torch.float32)
                for mask in intervention_masks
            ], dim=1)
        if self.accelerator is not None:
            paired_embeddings = self.accelerator.gather(
                torch.stack([embedding_a, embedding_b], dim=1)
            )
            embedding_a = paired_embeddings[:, 0]
            embedding_b = paired_embeddings[:, 1]
            if mask_matrix is not None:
                mask_matrix = self.accelerator.gather(mask_matrix)
        consensus = 0.5 * (embedding_a + embedding_b)
        between_energy = consensus.var(dim=0, unbiased=False)
        within_by_pair = 0.5 * (
            (embedding_a - consensus).square()
            + (embedding_b - consensus).square()
        )
        within_energy = within_by_pair.mean(dim=0)
        if mask_matrix is not None:
            axis_counts = mask_matrix.sum(dim=0)
            axis_energies = mask_matrix.T.matmul(within_by_pair)
            axis_energies = axis_energies / axis_counts[:, None].clamp_min(1.0)
            axis_energies = axis_energies.masked_fill(
                axis_counts[:, None] == 0, -torch.inf
            )
            # A target direction is reliable only when it survives every
            # nuisance represented in the current distributed batch.
            within_energy = axis_energies.amax(dim=0)
        score = between_energy / (
            between_energy + within_energy + 1e-6
        )
        eligible = score >= self.factor_reliability_floor
        keep_count = max(
            1, int(round(score.numel() * self.factor_reliability_keep_ratio))
        )
        keep_count = min(keep_count, int(eligible.sum().item()))
        if keep_count == 0:
            keep_indices = score.argmax().reshape(1)
        else:
            eligible_score = score.masked_fill(~eligible, -1.0)
            keep_indices = torch.topk(
                eligible_score, keep_count, sorted=False
            ).indices
        gate = torch.zeros_like(score)
        gate.scatter_(0, keep_indices, 1.0)
        return score, gate

    def _recover_clean_from_velocity(self, noisy_input, time, velocity):
        """Analytically recover x0 from a velocity prediction.

        For any differentiable interpolant
        ``x_t = alpha*x0 + sigma*epsilon`` and
        ``v_t = d_alpha*x0 + d_sigma*epsilon``.  Solving this two-by-two
        system keeps the construction valid for both linear and cosine paths.
        """
        alpha, sigma, d_alpha, d_sigma = self.interpolant(time)
        determinant = alpha * d_sigma - sigma * d_alpha
        if not torch.is_tensor(determinant):
            determinant = noisy_input.new_tensor(determinant)
        determinant = determinant.to(
            device=noisy_input.device, dtype=noisy_input.dtype
        )
        safe_determinant = determinant.sign() * determinant.abs().clamp_min(
            1e-6
        )
        return (
            d_sigma * noisy_input - sigma * velocity
        ) / safe_determinant

    def _clean_consensus_losses(
        self,
        input_a,
        input_b,
        time_a,
        time_b,
        velocity_a,
        velocity_b,
        clean_source,
    ):
        """Distill the more reliable analytic clean estimate across views."""
        clean_a = self._recover_clean_from_velocity(
            input_a, time_a, velocity_a
        )
        clean_b = self._recover_clean_from_velocity(
            input_b, time_b, velocity_b
        )
        source_energy = mean_flat(
            clean_source.detach().float().square()
        ).clamp_min(1e-3)
        source_error_a = mean_flat(
            (clean_a.float() - clean_source.detach().float()).square()
        ) / source_energy
        source_error_b = mean_flat(
            (clean_b.float() - clean_source.detach().float()).square()
        ) / source_energy
        confidence = torch.softmax(
            -torch.stack([source_error_a, source_error_b], dim=1)
            / self.factor_clean_consensus_temperature,
            dim=1,
        ).detach()
        weight_shape = (confidence.shape[0],) + (1,) * (
            clean_a.ndim - 1
        )
        consensus = (
            confidence[:, 0].reshape(weight_shape) * clean_a.detach()
            + confidence[:, 1].reshape(weight_shape) * clean_b.detach()
        )
        if self.factor_clean_consensus_shuffle_targets:
            consensus = consensus.roll(1, dims=0)
        consensus_loss = 0.5 * (
            self._energy_normalized_mse(clean_a, consensus)
            + self._energy_normalized_mse(clean_b, consensus)
        )
        pair_gap = 0.5 * (
            self._energy_normalized_mse(clean_a, clean_b)
            + self._energy_normalized_mse(clean_b, clean_a)
        )
        entropy = -(
            confidence * confidence.clamp_min(1e-8).log()
        ).sum(dim=1)
        return {
            'factor_clean_consensus_loss': consensus_loss,
            'factor_clean_pair_gap': pair_gap.mean().detach(),
            'factor_clean_source_error': (
                0.5 * (source_error_a + source_error_b)
            ).mean().detach(),
            'factor_clean_consensus_source_error': (
                mean_flat(
                    (consensus.float() - clean_source.detach().float()).square()
                ) / source_energy
            ).mean().detach(),
            'factor_clean_confidence_max': (
                confidence.max(dim=1).values.mean().detach()
            ),
            'factor_clean_confidence_entropy': entropy.mean().detach(),
        }

    @staticmethod
    def _shared_pair_features(shared):
        pair_count = shared.get('pair_count')
        features = shared['features']
        if pair_count is None:
            if features.shape[0] % 2:
                raise ValueError("shared self-distillation needs paired views")
            pair_count = features.shape[0] // 2
        if features.shape[0] != 2 * pair_count:
            raise ValueError("shared target features do not match pair_count")
        feature_a, feature_b = features.chunk(2, dim=0)
        return feature_a, feature_b, pair_count

    def _shared_feature_regularizers(self, feature_a, feature_b):
        pair_consensus = 0.5 * (feature_a.float() + feature_b.float())
        image_features = pair_consensus.mean(dim=1)
        image_std = torch.sqrt(
            image_features.var(dim=0, unbiased=False) + 1e-4
        )
        variance_loss = F.relu(
            self.factor_shared_variance_target - image_std
        ).mean()

        pooled_a = feature_a.detach().float().mean(dim=1)
        pooled_b = feature_b.detach().float().mean(dim=1)
        gathered_pairs = self._gather_detached(
            torch.stack([pooled_a, pooled_b], dim=1)
        )
        gathered_a = gathered_pairs[:, 0]
        gathered_b = gathered_pairs[:, 1]
        gathered_consensus = 0.5 * (gathered_a + gathered_b)
        between_energy = gathered_consensus.var(dim=0, unbiased=False).mean()
        within_energy = 0.25 * (
            gathered_a - gathered_b
        ).square().mean()
        source_ratio = between_energy / (
            between_energy + within_energy + 1e-6
        )
        return {
            'factor_shared_variance_loss': variance_loss,
            'factor_shared_similarity': (
                1.0 - self._cosine_distance(feature_a, feature_b).mean()
            ).detach(),
            'factor_shared_source_ratio': source_ratio.detach(),
            'factor_shared_between_energy': between_energy.detach(),
            'factor_shared_within_energy': within_energy.detach(),
            'factor_shared_image_std': image_std.mean().detach(),
        }

    def _shared_self_distill_losses(self, shared, time_a, time_b):
        """Align paired full hidden features to a reliable stop-grad target."""
        feature_a, feature_b, pair_count = self._shared_pair_features(shared)
        normalized_a = F.normalize(feature_a.float(), dim=-1)
        normalized_b = F.normalize(feature_b.float(), dim=-1)

        reliability_a = self._source_reliability(
            time_a, power=self.factor_shared_snr_power
        )
        reliability_b = self._source_reliability(
            time_b, power=self.factor_shared_snr_power
        )
        confidence = torch.softmax(
            torch.stack([reliability_a, reliability_b], dim=1)
            / self.factor_shared_target_temperature,
            dim=1,
        ).detach()
        weight_shape = (pair_count, 1, 1)
        consensus = (
            confidence[:, 0].reshape(weight_shape) * normalized_a.detach()
            + confidence[:, 1].reshape(weight_shape) * normalized_b.detach()
        )
        consensus = F.normalize(consensus, dim=-1)
        if self.factor_shared_self_distill_shuffle_targets:
            consensus = consensus.roll(1, dims=0)

        alignment_loss = 0.5 * (
            self._cosine_distance(feature_a, consensus)
            + self._cosine_distance(feature_b, consensus)
        )
        entropy = -(
            confidence * confidence.clamp_min(1e-8).log()
        ).sum(dim=1)
        result = {
            'factor_shared_self_distill_loss': alignment_loss,
            'factor_shared_confidence_max': (
                confidence.max(dim=1).values.mean().detach()
            ),
            'factor_shared_confidence_entropy': entropy.mean().detach(),
        }
        result.update(self._shared_feature_regularizers(feature_a, feature_b))
        return result

    def _self_flow_teacher_consensus(self, teacher_shared, time_a, time_b):
        """Build the detached EMA target shared by two trajectory views."""
        teacher_a, teacher_b, pair_count = self._shared_pair_features(
            teacher_shared
        )
        reliability_a = self._source_reliability(
            time_a, power=self.factor_shared_snr_power
        )
        reliability_b = self._source_reliability(
            time_b, power=self.factor_shared_snr_power
        )
        confidence = torch.softmax(
            torch.stack([reliability_a, reliability_b], dim=1)
            / self.factor_shared_target_temperature,
            dim=1,
        ).detach()
        normalized_a = F.normalize(teacher_a.detach().float(), dim=-1)
        normalized_b = F.normalize(teacher_b.detach().float(), dim=-1)
        weight_shape = (pair_count, 1, 1)
        consensus = (
            confidence[:, 0].reshape(weight_shape) * normalized_a
            + confidence[:, 1].reshape(weight_shape) * normalized_b
        )
        consensus = F.normalize(consensus, dim=-1)
        if self.factor_self_flow_shuffle_teacher:
            consensus = consensus.roll(1, dims=0)
        return consensus, confidence, teacher_a, teacher_b

    def _self_flow_alignment_losses(
        self,
        model_outputs,
        teacher_outputs,
        time_a,
        time_b,
    ):
        """Align online representations to an EMA teacher consensus."""
        if 'shared_target' not in teacher_outputs:
            raise ValueError("self-flow alignment requires EMA shared features")
        teacher_consensus, confidence, teacher_a, teacher_b = (
            self._self_flow_teacher_consensus(
                teacher_outputs['shared_target'], time_a, time_b
            )
        )
        result = {
            'factor_self_flow_teacher_pair_similarity': (
                1.0 - self._cosine_distance(teacher_a, teacher_b).mean()
            ).detach(),
            'factor_self_flow_teacher_std': (
                self._feature_std(teacher_consensus).detach()
            ),
            'factor_self_flow_confidence_max': (
                confidence.max(dim=1).values.mean().detach()
            ),
        }

        if self.factor_self_flow_full_align:
            if 'shared_target' not in model_outputs:
                raise ValueError(
                    "full self-flow alignment requires online shared features"
                )
            student_a, student_b, _ = self._shared_pair_features(
                model_outputs['shared_target']
            )
            result['factor_self_flow_full_loss'] = 0.5 * (
                self._cosine_distance(student_a, teacher_consensus)
                + self._cosine_distance(student_b, teacher_consensus)
            )
            result['factor_self_flow_full_similarity'] = (
                1.0 - 0.5 * (
                    self._cosine_distance(student_a, teacher_consensus).mean()
                    + self._cosine_distance(
                        student_b, teacher_consensus
                    ).mean()
                )
            ).detach()

        if self.factor_self_flow_source_align:
            if 'factorization' not in model_outputs:
                raise ValueError(
                    "source self-flow alignment requires factorization outputs"
                )
            source_component = model_outputs['factorization'][
                'persistent_component'
            ]
            source_a, source_b = source_component.chunk(2, dim=0)
            if source_a.shape != teacher_consensus.shape:
                raise ValueError(
                    "source self-flow alignment shape mismatch: "
                    f"{tuple(source_a.shape)} vs {tuple(teacher_consensus.shape)}"
                )
            result['factor_self_flow_source_loss'] = 0.5 * (
                self._cosine_distance(source_a, teacher_consensus)
                + self._cosine_distance(source_b, teacher_consensus)
            )
            result['factor_self_flow_source_similarity'] = (
                1.0 - 0.5 * (
                    self._cosine_distance(source_a, teacher_consensus).mean()
                    + self._cosine_distance(
                        source_b, teacher_consensus
                    ).mean()
                )
            ).detach()
            result['factor_self_flow_source_std'] = (
                self._feature_std(source_component).detach()
            )

        return result

    def _shared_contrastive_losses(self, shared):
        """Use paired trajectory views as positives and other sources as negatives."""
        feature_a, feature_b, _ = self._shared_pair_features(shared)
        pooled_a = F.normalize(feature_a.float().mean(dim=1), dim=-1)
        pooled_b = F.normalize(feature_b.float().mean(dim=1), dim=-1)
        logits_ab = pooled_a.matmul(pooled_b.T) / (
            self.factor_shared_contrastive_temperature
        )
        logits_ba = pooled_b.matmul(pooled_a.T) / (
            self.factor_shared_contrastive_temperature
        )
        labels = torch.arange(
            pooled_a.shape[0], device=pooled_a.device, dtype=torch.long
        )
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits_ab, labels)
            + F.cross_entropy(logits_ba, labels)
        )
        positive_similarity = (pooled_a * pooled_b).sum(dim=-1)
        negative_mask = ~torch.eye(
            pooled_a.shape[0], device=pooled_a.device, dtype=torch.bool
        )
        if negative_mask.any():
            negative_similarity = logits_ab.detach().mul(
                self.factor_shared_contrastive_temperature
            )[negative_mask].mean()
        else:
            negative_similarity = logits_ab.new_zeros(())
        result = {
            'factor_shared_contrastive_loss': contrastive_loss,
            'factor_shared_contrastive_accuracy': (
                0.5 * (
                    (logits_ab.argmax(dim=1) == labels).float().mean()
                    + (logits_ba.argmax(dim=1) == labels).float().mean()
                )
            ).detach(),
            'factor_shared_positive_similarity': (
                positive_similarity.mean().detach()
            ),
            'factor_shared_negative_similarity': (
                negative_similarity.detach()
            ),
        }
        result.update(self._shared_feature_regularizers(feature_a, feature_b))
        return result

    def _shared_relation_losses(self, shared):
        """Match source-level and local spatial relations across paired views."""
        feature_a, feature_b, pair_count = self._shared_pair_features(shared)
        pooled_a = F.normalize(feature_a.float().mean(dim=1), dim=-1)
        pooled_b = F.normalize(feature_b.float().mean(dim=1), dim=-1)
        if pair_count > 1:
            mask = ~torch.eye(
                pair_count, device=pooled_a.device, dtype=torch.bool
            )
            sim_a = pooled_a.matmul(pooled_a.T)
            sim_b = pooled_b.matmul(pooled_b.T)
            global_relation_loss = 0.5 * (
                (sim_a - sim_b.detach()).square()[mask].mean()
                + (sim_b - sim_a.detach()).square()[mask].mean()
            )
        else:
            global_relation_loss = pooled_a.new_zeros(())

        local_a = self._local_relation(feature_a)
        local_b = self._local_relation(feature_b)
        local_relation_loss = 0.5 * (
            (local_a - local_b.detach()).square().mean(dim=-1)
            + (local_b - local_a.detach()).square().mean(dim=-1)
        )
        relation_loss = global_relation_loss + local_relation_loss.mean()
        result = {
            'factor_shared_relation_loss': relation_loss,
            'factor_shared_global_relation_loss': (
                global_relation_loss.detach()
            ),
            'factor_shared_local_relation_loss': (
                local_relation_loss.mean().detach()
            ),
        }
        result.update(self._shared_feature_regularizers(feature_a, feature_b))
        return result

    def _gather_detached(self, tensor):
        tensor = tensor.detach()
        if self.accelerator is not None:
            tensor = self.accelerator.gather(tensor)
        return tensor

    @staticmethod
    def _unit_mean_weights(values, eps=1e-8):
        values = values.detach().float().clamp_min(0.0)
        mean = values.mean()
        normalized = values / mean.clamp_min(eps)
        return torch.where(
            mean > eps, normalized, torch.ones_like(normalized)
        )

    def _selective_invariance_losses(
        self, selective, task_gradient=None
    ):
        """Align only stable, velocity-useful directions of an A3 pair.

        Direction scores are detached statistics.  Consequently the model
        cannot lower the loss by manipulating the gate, and task weighting
        never creates a second-order gradient through the FM objective.
        """
        pair_count = selective.get('pair_count')
        features = selective['features']
        if pair_count is None:
            if features.shape[0] % 2:
                raise ValueError(
                    "selective invariance needs an even paired batch"
                )
            pair_count = features.shape[0] // 2
        if features.shape[0] != 2 * pair_count:
            raise ValueError(
                "selective features do not match factor_pair_count"
            )
        feature_a, feature_b = features.chunk(2, dim=0)

        pooled_a = feature_a.detach().float().mean(dim=1)
        pooled_b = feature_b.detach().float().mean(dim=1)
        gathered_pairs = self._gather_detached(
            torch.stack([pooled_a, pooled_b], dim=1)
        )
        gathered_a = gathered_pairs[:, 0]
        gathered_b = gathered_pairs[:, 1]
        consensus = 0.5 * (gathered_a + gathered_b)
        between_energy = consensus.var(dim=0, unbiased=False)
        within_energy = 0.25 * (
            gathered_a - gathered_b
        ).square().mean(dim=0)
        stability = between_energy / (
            between_energy + within_energy + 1e-6
        )

        utility = torch.ones_like(stability)
        if self.factor_selective_weighting == "task":
            if task_gradient is None:
                raise ValueError(
                    "task-selective invariance requires the FM gradient"
                )
            if task_gradient.shape[0] < 2 * pair_count:
                raise ValueError(
                    "task gradient does not contain all paired views"
                )
            task_gradient = task_gradient[:2 * pair_count].detach().float()
            basis = selective['basis'].detach().float()
            projected_gradient = torch.einsum(
                'btd,kd->btk', task_gradient, basis
            )
            utility = self._gather_detached(
                projected_gradient.square().mean(dim=(0, 1))
                .unsqueeze(0)
            ).mean(dim=0)
            if self.factor_selective_shuffle_utility:
                # A deterministic permutation is a cleaner control than an
                # unseeded random draw and is identical on every rank.
                utility = utility.flip(0)

        if self.factor_selective_weighting == "uniform":
            weights = torch.ones_like(stability)
        elif self.factor_selective_weighting == "stability":
            weights = self._unit_mean_weights(stability)
        else:
            weights = self._unit_mean_weights(
                self._unit_mean_weights(stability)
                * self._unit_mean_weights(utility)
            )

        normalized_a = F.normalize(feature_a.float(), dim=-1)
        normalized_b = F.normalize(feature_b.float(), dim=-1)
        target_b = normalized_b.detach()
        target_a = normalized_a.detach()
        if self.factor_selective_shuffle_targets:
            target_b = target_b.roll(1, dims=0)
            target_a = target_a.roll(1, dims=0)
        per_direction_loss = 0.25 * (
            (normalized_a - target_b).square().mean(dim=(0, 1))
            + (normalized_b - target_a).square().mean(dim=(0, 1))
        )
        # With unit-mean weights, summing directions has the same scale as a
        # cosine distance (uniform weights recover 1 - cosine similarity).
        alignment_loss = (per_direction_loss * weights).sum()

        pair_consensus = 0.5 * (feature_a.float() + feature_b.float())
        image_features = pair_consensus.mean(dim=1)
        image_std = torch.sqrt(
            image_features.var(dim=0, unbiased=False) + 1e-4
        )
        variance_loss = F.relu(
            self.factor_selective_variance_target - image_std
        ).mean()
        source_ratio = between_energy.mean() / (
            between_energy.mean() + within_energy.mean() + 1e-6
        )
        effective_dimensions = weights.sum().square() / (
            weights.square().sum() + 1e-6
        )
        return {
            'factor_selective_loss': alignment_loss,
            'factor_selective_orth_loss': selective[
                'basis_orthogonality_loss'
            ],
            'factor_selective_variance_loss': variance_loss,
            'factor_selective_similarity': (
                1.0 - self._cosine_distance(feature_a, feature_b).mean()
            ).detach(),
            'factor_selective_source_ratio': source_ratio.detach(),
            'factor_selective_between_energy': (
                between_energy.mean().detach()
            ),
            'factor_selective_within_energy': within_energy.mean().detach(),
            'factor_selective_stability': stability.mean().detach(),
            'factor_selective_utility': utility.mean().detach(),
            'factor_selective_weight_max': weights.max().detach(),
            'factor_selective_effective_dims': (
                effective_dimensions.detach()
            ),
            'factor_selective_image_std': image_std.mean().detach(),
        }

    def _factor_adversarial_losses(
        self, nuisance_predictions, timesteps, intervention_masks
    ):
        """Supervise nuisance critics and matched evolving-code probes."""
        if timesteps is None or intervention_masks is None:
            raise ValueError(
                "adversarial nuisance losses require timesteps and orbit masks"
            )
        time_only_mask, noise_only_mask, joint_mask = intervention_masks
        if joint_mask.any():
            raise ValueError(
                "binary orbit prediction is defined only for orthogonal pairs"
            )
        if not torch.all(time_only_mask | noise_only_mask):
            raise ValueError("every adversarial pair needs an orbit label")

        timestep_labels = torch.clamp(
            (timesteps.float() * self.factor_adversarial_timestep_bins).long(),
            max=self.factor_adversarial_timestep_bins - 1,
        )
        orbit_labels = noise_only_mask.to(
            device=timesteps.device, dtype=torch.long
        )
        if self.factor_adversarial_shuffle_labels:
            timestep_labels = timestep_labels[
                torch.randperm(timestep_labels.shape[0], device=timesteps.device)
            ]
            orbit_labels = orbit_labels[
                torch.randperm(orbit_labels.shape[0], device=timesteps.device)
            ]

        persistent_time_logits = nuisance_predictions[
            'persistent_time_logits'
        ].float()
        persistent_orbit_logits = nuisance_predictions[
            'persistent_orbit_logits'
        ].float()
        evolving_time_logits = nuisance_predictions[
            'evolving_time_logits'
        ].float()
        evolving_orbit_logits = nuisance_predictions[
            'evolving_orbit_logits'
        ].float()
        result = {
            'factor_adv_persistent_time_loss': F.cross_entropy(
                persistent_time_logits, timestep_labels, reduction='none'
            ),
            'factor_adv_persistent_orbit_loss': F.cross_entropy(
                persistent_orbit_logits, orbit_labels, reduction='none'
            ),
            'factor_probe_evolving_time_loss': F.cross_entropy(
                evolving_time_logits, timestep_labels, reduction='none'
            ),
            'factor_probe_evolving_orbit_loss': F.cross_entropy(
                evolving_orbit_logits, orbit_labels, reduction='none'
            ),
        }
        persistent_time_accuracy = (
            persistent_time_logits.argmax(dim=-1) == timestep_labels
        ).float().mean()
        persistent_orbit_accuracy = (
            persistent_orbit_logits.argmax(dim=-1) == orbit_labels
        ).float().mean()
        evolving_time_accuracy = (
            evolving_time_logits.argmax(dim=-1) == timestep_labels
        ).float().mean()
        evolving_orbit_accuracy = (
            evolving_orbit_logits.argmax(dim=-1) == orbit_labels
        ).float().mean()
        time_majority_accuracy = torch.bincount(
            timestep_labels,
            minlength=self.factor_adversarial_timestep_bins,
        ).amax().float() / timestep_labels.numel()
        orbit_majority_accuracy = torch.bincount(
            orbit_labels, minlength=2
        ).amax().float() / orbit_labels.numel()
        result.update({
            'factor_adv_persistent_time_accuracy': (
                persistent_time_accuracy.detach()
            ),
            'factor_adv_persistent_orbit_accuracy': (
                persistent_orbit_accuracy.detach()
            ),
            'factor_probe_evolving_time_accuracy': (
                evolving_time_accuracy.detach()
            ),
            'factor_probe_evolving_orbit_accuracy': (
                evolving_orbit_accuracy.detach()
            ),
            'factor_time_separation_gap': (
                evolving_time_accuracy - persistent_time_accuracy
            ).detach(),
            'factor_orbit_separation_gap': (
                evolving_orbit_accuracy - persistent_orbit_accuracy
            ).detach(),
            'factor_time_majority_accuracy': (
                time_majority_accuracy.detach()
            ),
            'factor_orbit_majority_accuracy': (
                orbit_majority_accuracy.detach()
            ),
        })
        return result

    @staticmethod
    def _local_relation(features):
        """Return horizontal/vertical cosine relations without a dense Gram matrix."""
        batch_size, token_count, channels = features.shape
        grid_size = int(token_count ** 0.5)
        if grid_size * grid_size != token_count:
            raise ValueError("invariant relation loss requires a square token grid")
        normalized = F.normalize(features.float(), dim=-1)
        grid = normalized.reshape(batch_size, grid_size, grid_size, channels)
        horizontal = (grid[:, :, :-1] * grid[:, :, 1:]).sum(dim=-1)
        vertical = (grid[:, :-1, :] * grid[:, 1:, :]).sum(dim=-1)
        return torch.cat([
            horizontal.reshape(batch_size, -1),
            vertical.reshape(batch_size, -1),
        ], dim=-1)

    def _invariance_losses(
        self, invariance, time_reliability, noise_reliability
    ):
        anchor, time_view, noise_view = invariance['features'].chunk(3, dim=0)
        time_distance = self._cosine_distance(anchor, time_view)
        noise_distance = self._cosine_distance(anchor, noise_view)
        time_loss = self._weighted_mean(time_distance, time_reliability)
        noise_loss = self._weighted_mean(noise_distance, noise_reliability)

        consensus = torch.stack([
            anchor.float(), time_view.float(), noise_view.float()
        ]).mean(dim=0)
        image_embeddings = consensus.mean(dim=1)
        image_std_by_channel = torch.sqrt(
            image_embeddings.var(dim=0, unbiased=False) + 1e-4
        )
        image_variance_loss = F.relu(
            self.invariant_variance_target - image_std_by_channel
        ).mean()

        spatial_std_by_channel = torch.sqrt(
            consensus.var(dim=1, unbiased=False) + 1e-4
        )
        spatial_variance_loss = F.relu(
            self.invariant_spatial_variance_target - spatial_std_by_channel
        ).mean()

        observations = consensus.reshape(-1, consensus.shape[-1])
        observations = observations - observations.mean(dim=0, keepdim=True)
        denominator = max(observations.shape[0] - 1, 1)
        covariance = observations.T.matmul(observations) / denominator
        covariance_square = covariance.square()
        off_diagonal_count = max(
            covariance.shape[0] * (covariance.shape[0] - 1), 1
        )
        covariance_loss = (
            covariance_square.sum() - covariance_square.diagonal().sum()
        ) / off_diagonal_count

        anchor_relation = self._local_relation(anchor)
        time_relation_gap = (
            anchor_relation - self._local_relation(time_view)
        ).abs().mean(dim=-1)
        noise_relation_gap = (
            anchor_relation - self._local_relation(noise_view)
        ).abs().mean(dim=-1)
        time_relation_loss = self._weighted_mean(
            time_relation_gap, time_reliability
        )
        noise_relation_loss = self._weighted_mean(
            noise_relation_gap, noise_reliability
        )
        relation_loss = 0.5 * (
            time_relation_loss + noise_relation_loss
        )

        # Both terms use pooled image embeddings, so this ratio is a genuine
        # same-scale source-versus-intervention variance diagnostic.
        anchor_embedding = anchor.float().mean(dim=1)
        time_embedding = time_view.float().mean(dim=1)
        noise_embedding = noise_view.float().mean(dim=1)
        within_energy = 0.5 * (
            (anchor_embedding - time_embedding).square().mean()
            + (anchor_embedding - noise_embedding).square().mean()
        )
        between_energy = image_embeddings.var(dim=0, unbiased=False).mean()
        source_ratio = between_energy / (
            between_energy + within_energy + 1e-6
        )

        return {
            'invariant_time_loss': time_loss,
            'invariant_noise_loss': noise_loss,
            'invariant_image_variance_loss': image_variance_loss,
            'invariant_spatial_variance_loss': spatial_variance_loss,
            'invariant_covariance_loss': covariance_loss,
            'invariant_relation_loss': relation_loss,
            'invariant_basis_loss': invariance[
                'basis_orthogonality_loss'
            ],
            'invariant_similarity': (
                1.0 - 0.5 * (time_distance.mean() + noise_distance.mean())
            ).detach(),
            'invariant_time_similarity': (
                1.0 - time_distance.mean()
            ).detach(),
            'invariant_noise_similarity': (
                1.0 - noise_distance.mean()
            ).detach(),
            'invariant_image_std': image_std_by_channel.mean().detach(),
            'invariant_spatial_std': spatial_std_by_channel.mean().detach(),
            'invariant_covariance_offdiag': covariance_loss.detach(),
            'invariant_relation_gap': relation_loss.detach(),
            'invariant_time_relation_gap': time_relation_loss.detach(),
            'invariant_noise_relation_gap': noise_relation_loss.detach(),
            'invariant_within_energy': within_energy.detach(),
            'invariant_between_energy': between_energy.detach(),
            'invariant_source_ratio': source_ratio.detach(),
            'invariant_time_reliability': time_reliability.mean().detach(),
            'invariant_noise_reliability': noise_reliability.mean().detach(),
        }

    def _factorization_losses(
        self,
        factorization,
        same_trajectory_mask=None,
        intervention_masks=None,
        velocity_target=None,
        factor_timesteps=None,
    ):
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
        raw_common_target = 0.5 * (target_a + target_b)
        if self.factor_reliable_target:
            target_reliability, target_gate = self._factor_target_reliability(
                target_a, target_b, intervention_masks=intervention_masks
            )
            common_target = raw_common_target * target_gate.to(
                device=target.device, dtype=target.dtype
            ).view(1, 1, -1)
        else:
            target_reliability = target.new_ones(
                target.shape[-1], dtype=torch.float32
            )
            target_gate = target_reliability
            common_target = raw_common_target
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
            'factor_target_reliability': target_reliability.mean().detach(),
            'factor_target_gate_mean': target_gate.mean().detach(),
            'factor_target_selected_fraction': (
                (target_gate > 0).float().mean().detach()
            ),
        }
        if self.factor_evolving_separation:
            evolving_pair_distance = self._cosine_distance(
                evolving_a, evolving_b
            )
            result['factor_evolving_separation_loss'] = F.relu(
                self.factor_evolving_separation_margin
                - evolving_pair_distance
            )
            result['factor_evolving_pair_distance'] = (
                evolving_pair_distance.mean().detach()
            )
        if self.factor_adversarial and 'nuisance_predictions' not in factorization:
            raise ValueError(
                "factor_adversarial loss requires model nuisance predictions"
            )
        if self.factor_adversarial:
            result.update(self._factor_adversarial_losses(
                factorization['nuisance_predictions'],
                factor_timesteps,
                intervention_masks,
            ))
        if 'velocity_recomposed' in factorization:
            if velocity_target is None:
                raise ValueError(
                    "velocity_target is required for velocity recomposition"
                )
            velocity_loss_all = self._energy_normalized_mse(
                factorization['velocity_recomposed'], velocity_target
            )
            velocity_a, velocity_b = velocity_loss_all.chunk(2, dim=0)
            recomposed_velocity_loss = 0.5 * (velocity_a + velocity_b)

            velocity_target_a, velocity_target_b = velocity_target.chunk(
                2, dim=0
            )
            common_velocity = 0.5 * (
                velocity_target_a + velocity_target_b
            )
            common_velocities = torch.cat([
                common_velocity, common_velocity
            ], dim=0)
            residual_velocities = velocity_target - common_velocities
            persistent_velocity_loss_all = self._energy_normalized_mse(
                factorization['persistent_velocity_component'],
                common_velocities,
            )
            evolving_velocity_loss_all = self._energy_normalized_mse(
                factorization['evolving_velocity_component'],
                residual_velocities,
            )
            persistent_velocity_a, persistent_velocity_b = (
                persistent_velocity_loss_all.chunk(2, dim=0)
            )
            evolving_velocity_a, evolving_velocity_b = (
                evolving_velocity_loss_all.chunk(2, dim=0)
            )
            persistent_velocity_loss = 0.5 * (
                persistent_velocity_a + persistent_velocity_b
            )
            evolving_velocity_loss = 0.5 * (
                evolving_velocity_a + evolving_velocity_b
            )
            # Direct component supervision makes it impossible for the task
            # decoder to satisfy recomposition by silently ignoring either
            # factor. One coefficient controls the balanced objective.
            velocity_loss = recomposed_velocity_loss + 0.5 * (
                persistent_velocity_loss + evolving_velocity_loss
            )
            result['factor_velocity_recom_loss'] = velocity_loss
            result['factor_velocity_reconstruction_loss'] = (
                recomposed_velocity_loss.mean().detach()
            )
            result['factor_velocity_persistent_loss'] = (
                persistent_velocity_loss.mean().detach()
            )
            result['factor_velocity_evolving_loss'] = (
                evolving_velocity_loss.mean().detach()
            )
            result['factor_velocity_recom_error'] = (
                velocity_loss.mean().detach()
            )
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

    def _invariance_forward(
        self,
        model,
        images,
        model_kwargs,
        zs,
        invariant_batch_ratio,
    ):
        """Run an anchor plus orthogonal time/noise interventions per source."""
        if not 0.0 < invariant_batch_ratio <= 1.0:
            raise ValueError("invariant_batch_ratio must be in (0, 1]")
        batch_size = images.shape[0]
        group_count = min(
            batch_size, max(1, int(batch_size * invariant_batch_ratio))
        )
        permutation = torch.randperm(batch_size, device=images.device)
        group_indices = permutation[:group_count]
        single_indices = permutation[group_count:]
        group_images = images[group_indices]
        single_images = images[single_indices]

        anchor_time, time_view_time = (
            self._sample_invariant_times(group_images)
        )
        anchor_noise = torch.randn_like(group_images)
        noise_view_noise = torch.randn_like(group_images)

        alpha_anchor, sigma_anchor, d_alpha_anchor, d_sigma_anchor = (
            self.interpolant(anchor_time)
        )
        alpha_time, sigma_time, d_alpha_time, d_sigma_time = (
            self.interpolant(time_view_time)
        )
        model_input_anchor = (
            alpha_anchor * group_images + sigma_anchor * anchor_noise
        )
        # The time view shares epsilon; the noise view shares timestep.  Using
        # all three for each source makes x0 the intersection of their common
        # information instead of relying on different samples to establish it.
        model_input_time = alpha_time * group_images + sigma_time * anchor_noise
        model_input_noise = (
            alpha_anchor * group_images + sigma_anchor * noise_view_noise
        )
        if self.prediction != 'v':
            raise NotImplementedError()
        target_anchor = (
            d_alpha_anchor * group_images + d_sigma_anchor * anchor_noise
        )
        target_time = d_alpha_time * group_images + d_sigma_time * anchor_noise
        target_noise = (
            d_alpha_anchor * group_images + d_sigma_anchor * noise_view_noise
        )

        model_inputs = [model_input_anchor, model_input_time, model_input_noise]
        model_times = [anchor_time, time_view_time, anchor_time]
        model_targets = [target_anchor, target_time, target_noise]
        if single_images.shape[0] > 0:
            single_time = self._sample_times(single_images)
            single_noise = torch.randn_like(single_images)
            alpha_s, sigma_s, d_alpha_s, d_sigma_s = self.interpolant(
                single_time
            )
            model_inputs.append(alpha_s * single_images + sigma_s * single_noise)
            model_times.append(single_time)
            model_targets.append(d_alpha_s * single_images + d_sigma_s * single_noise)

        assembled_kwargs = self._assemble_model_kwargs(
            model_kwargs,
            batch_size,
            group_indices,
            single_indices,
            view_count=3,
        )
        model_outputs = model(
            torch.cat(model_inputs, dim=0),
            torch.cat(model_times, dim=0).flatten(),
            trajectory_pair=True,
            return_invariance=True,
            invariant_group_count=group_count,
            invariant_view_count=3,
            **assembled_kwargs,
        )
        output_anchor = model_outputs['x'][:group_count]
        output_time = model_outputs['x'][group_count:2 * group_count]
        output_noise = model_outputs['x'][2 * group_count:3 * group_count]
        grouped_denoising_loss = (
            mean_flat((output_anchor - target_anchor) ** 2)
            + mean_flat((output_time - target_time) ** 2)
            + mean_flat((output_noise - target_noise) ** 2)
        ) / 3.0
        if single_images.shape[0] > 0:
            output_single = model_outputs['x'][3 * group_count:]
            single_denoising_loss = mean_flat(
                (output_single - model_targets[-1]) ** 2
            )
            denoising_loss = torch.cat([
                grouped_denoising_loss, single_denoising_loss
            ], dim=0)
        else:
            denoising_loss = grouped_denoising_loss

        grouped_zs = None if zs is None else [
            torch.cat([
                z[group_indices], z[group_indices], z[group_indices],
                z[single_indices],
            ], dim=0) for z in zs
        ]
        anchor_reliability = self._source_reliability(anchor_time)
        time_view_reliability = self._source_reliability(time_view_time)
        time_reliability = torch.sqrt(
            anchor_reliability * time_view_reliability
        )
        noise_reliability = anchor_reliability
        time_delta = (time_view_time - anchor_time).abs().flatten()
        losses = {
            'denoising_loss': denoising_loss,
            'proj_loss': self._projection_loss(
                grouped_zs,
                model_outputs.get('zs'),
                denoising_loss,
                group_count=group_count,
                view_count=3,
            ),
            'invariant_mean_time_delta': time_delta.mean().detach(),
            'invariant_batch_fraction': denoising_loss.new_tensor(
                group_count / batch_size
            ).detach(),
            'invariant_views_per_group': denoising_loss.new_tensor(3).detach(),
        }
        losses.update(self._invariance_losses(
            model_outputs['invariance'], time_reliability, noise_reliability
        ))
        return losses, model_outputs

    def __call__(
        self,
        model,
        images,
        model_kwargs=None,
        zs=None,
        factorization_active=True,
        factor_batch_ratio=1.0,
        factor_adversarial_grl_scale=1.0,
        invariance_active=True,
        invariant_batch_ratio=1.0,
        ema_model=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        if self.trajectory_invariance and invariance_active:
            losses, model_outputs = self._invariance_forward(
                model,
                images,
                model_kwargs,
                zs,
                invariant_batch_ratio,
            )
        elif self.trajectory_factorization and factorization_active:
            if not 0.0 < factor_batch_ratio <= 1.0:
                raise ValueError("factor_batch_ratio must be in (0, 1]")
            batch_size = images.shape[0]
            if self.factor_native_shuffle_source and batch_size < 2:
                raise ValueError(
                    "shuffled native source control requires batch_size >= 2"
                )
            if self.factor_semantic_shuffle_targets and batch_size < 2:
                raise ValueError(
                    "shuffled semantic target control requires batch_size >= 2"
                )
            if self.factor_self_flow_shuffle_teacher and batch_size < 2:
                raise ValueError(
                    "shuffled self-flow teacher control requires batch_size >= 2"
                )
            pair_count = min(batch_size, max(1, int(batch_size * factor_batch_ratio)))
            permutation = torch.randperm(batch_size, device=images.device)
            pair_indices = permutation[:pair_count]
            single_indices = permutation[pair_count:]
            pair_images = images[pair_indices]
            single_images = images[single_indices]

            orbit = self._sample_factor_orbit(pair_images)
            time_a = orbit['time_a']
            time_b = orbit['time_b']
            noise_a = orbit['noise_a']
            noise_b = orbit['noise_b']

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
                return_selective_invariance=(
                    self.factor_selective_invariance
                ),
                return_shared_target=(
                    self.factor_shared_self_distill
                    or self.factor_shared_contrastive
                    or self.factor_shared_relation
                    or self.factor_self_flow_full_align
                ),
                return_semantic_factorization=(
                    self.factor_semantic_conditioning
                    or self.factor_self_flow_source_align
                ),
                factor_pair_count=pair_count,
                factor_adversarial_grl_scale=factor_adversarial_grl_scale,
                **assembled_kwargs,
            )
            teacher_outputs = None
            if (
                self.factor_self_flow_full_align
                or self.factor_self_flow_source_align
            ):
                if ema_model is None:
                    raise ValueError(
                        "self-flow EMA alignment requires ema_model"
                    )
                with torch.no_grad():
                    teacher_outputs = ema_model(
                        torch.cat(model_inputs, dim=0),
                        torch.cat(model_times, dim=0).flatten(),
                        trajectory_pair=True,
                        return_shared_target=True,
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
                    group_count=pair_count,
                    view_count=2,
                ),
                'mean_delta_t': pair_delta.abs().mean().detach(),
                'factor_time_intervention_delta': self._masked_mean(
                    pair_delta.abs(), orbit['time_only_mask']
                ).detach(),
                'cross_noise_fraction': (
                    orbit['noise_changed_mask'].float().mean().detach()
                ),
                'factor_time_only_fraction': (
                    orbit['time_only_mask'].float().mean().detach()
                ),
                'factor_noise_only_fraction': (
                    orbit['noise_only_mask'].float().mean().detach()
                ),
                'factor_joint_intervention_fraction': (
                    orbit['joint_mask'].float().mean().detach()
                ),
                'factor_batch_fraction': denoising_loss.new_tensor(
                    pair_count / batch_size
                ).detach(),
            }
            if self.factor_semantic_conditioning:
                if 'semantic_factorization' not in model_outputs:
                    raise ValueError(
                        "semantic conditioning loss requires model semantic "
                        "source/evolving predictions"
                    )
                semantic_targets = paired_zs
                if self.factor_semantic_shuffle_targets:
                    # Shuffle source identity once, then duplicate the same
                    # wrong clean target for both trajectory views.
                    semantic_targets = []
                    for target in zs:
                        ordered_target = torch.cat([
                            target[pair_indices], target[single_indices]
                        ], dim=0).roll(shifts=1, dims=0)
                        target_views = [
                            ordered_target[:pair_count],
                            ordered_target[:pair_count],
                        ]
                        if single_images.shape[0] > 0:
                            target_views.append(ordered_target[pair_count:])
                        semantic_targets.append(torch.cat(target_views, dim=0))
                losses.update(self._semantic_factorization_losses(
                    model_outputs['semantic_factorization'],
                    semantic_targets,
                    denoising_loss,
                    pair_count,
                ))
            if self.factor_native_parameterization:
                if 'native_parameterization' not in model_outputs:
                    raise ValueError(
                        "native parameterization loss requires model-native "
                        "source/noise predictions"
                    )
                ordered_sources = torch.cat(
                    [pair_images, single_images], dim=0
                )
                if self.factor_native_shuffle_source:
                    # A deterministic cyclic shift avoids fixed points and
                    # keeps the treatment/control RNG stream matched.
                    ordered_sources = ordered_sources.roll(shifts=1, dims=0)
                source_targets = [
                    ordered_sources[:pair_count],
                    ordered_sources[:pair_count],
                ]
                noise_targets = [noise_a, noise_b]
                if single_images.shape[0] > 0:
                    source_targets.append(ordered_sources[pair_count:])
                    noise_targets.append(single_noise)
                losses.update(self._native_parameterization_losses(
                    model_outputs['native_parameterization'],
                    torch.cat(source_targets, dim=0),
                    torch.cat(noise_targets, dim=0),
                    torch.cat(model_targets, dim=0),
                    torch.cat(model_times, dim=0).flatten(),
                    pair_count,
                ))
            if self.factor_shared_repa:
                if (
                    paired_zs is None
                    or model_outputs.get('zs') is None
                    or len(paired_zs) == 0
                    or len(model_outputs.get('zs')) == 0
                ):
                    raise ValueError(
                        "factor_shared_repa requires an external encoder and "
                        "model projection heads"
                    )
                losses['factor_shared_repa_loss'] = (
                    self._projection_alignment_loss(
                        paired_zs,
                        model_outputs.get('zs'),
                        denoising_loss,
                        group_count=pair_count,
                        view_count=2,
                    )
                )
            if self.factor_clean_consensus:
                losses.update(self._clean_consensus_losses(
                    model_input_a,
                    model_input_b,
                    time_a,
                    time_b,
                    output_a,
                    output_b,
                    pair_images,
                ))
            if self.factor_shared_self_distill:
                if 'shared_target' not in model_outputs:
                    raise ValueError(
                        "factor_shared_self_distill requires shared target "
                        "features from the model"
                    )
                losses.update(self._shared_self_distill_losses(
                    model_outputs['shared_target'],
                    time_a,
                    time_b,
                ))
            if self.factor_shared_contrastive:
                if 'shared_target' not in model_outputs:
                    raise ValueError(
                        "factor_shared_contrastive requires shared target "
                        "features from the model"
                    )
                losses.update(self._shared_contrastive_losses(
                    model_outputs['shared_target']
                ))
            if self.factor_shared_relation:
                if 'shared_target' not in model_outputs:
                    raise ValueError(
                        "factor_shared_relation requires shared target "
                        "features from the model"
                    )
                losses.update(self._shared_relation_losses(
                    model_outputs['shared_target']
                ))
            if (
                self.factor_self_flow_full_align
                or self.factor_self_flow_source_align
            ):
                losses.update(self._self_flow_alignment_losses(
                    model_outputs,
                    teacher_outputs,
                    time_a,
                    time_b,
                ))
            if self.factor_selective_invariance:
                if 'selective_invariance' not in model_outputs:
                    raise ValueError(
                        "factor_selective_invariance loss requires the model "
                        "selective-invariance head"
                    )
                task_gradient = None
                if self.factor_selective_weighting == "task":
                    task_gradient = torch.autograd.grad(
                        denoising_loss.mean(),
                        model_outputs['selective_invariance'][
                            'source_features'
                        ],
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                losses.update(self._selective_invariance_losses(
                    model_outputs['selective_invariance'], task_gradient
                ))
            factorization = model_outputs['factorization']
            factor_velocity_target = (
                torch.cat([target_a, target_b], dim=0)
                if 'velocity_recomposed' in factorization else None
            )
            losses.update(self._factorization_losses(
                factorization,
                same_trajectory_mask=orbit['time_only_mask'],
                intervention_masks=(
                    orbit['time_only_mask'],
                    orbit['noise_only_mask'],
                    orbit['joint_mask'],
                ),
                velocity_target=factor_velocity_target,
                factor_timesteps=torch.cat([
                    time_a.flatten(), time_b.flatten()
                ], dim=0),
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
