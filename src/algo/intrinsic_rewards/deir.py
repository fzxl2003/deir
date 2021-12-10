import cv2
import gym
from typing import Dict, Any, List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gym import spaces
from numpy.random import Generator
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor

from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.common_func import normalize_rewards
from src.utils.enum_types import NormType
from src.utils.loggers import StatisticsLogger
from src.utils.running_mean_std import RunningMeanStd


# [Notice] For readability, part of code used purely for logging & analysis is omitted
class DiscriminatorModel(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        # Method-specific params
        obs_rng: Optional[Generator] = None,
        dsc_obs_queue_len: int = 0,
        dsc_use_rnd: int = 0,
        dsc_l1_reg_coef: float = 0.0,
        dsc_l2_reg_coef: float = 0.0,
        dsc_clip_value: float = 0.0,
        dsc_ir_bias: float = 0.0,
        dsc_tolerance: int = 0,
        dsc_ll_rew: int = 0,
        dsc_visible_ratio: float = 1.0,
        dsc_header_mask: int = 0,
        rnd_err_norm: int = 0,
        rnd_err_momentum: float = -1,
        rnd_use_policy_emb: int = 1,
        policy_cnn: Type[nn.Module] = None,
        log_dsc_verbose: int = 0,
        env_render: int = 0,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)

        self.obs_rng = obs_rng
        self.dsc_obs_queue_len = dsc_obs_queue_len
        self.dsc_use_rnd = dsc_use_rnd
        self.dsc_l1_reg_coef = dsc_l1_reg_coef
        self.dsc_l2_reg_coef = dsc_l2_reg_coef
        self.dsc_clip_value = dsc_clip_value
        self.dsc_tolerance = dsc_tolerance
        self.dsc_ir_bias = dsc_ir_bias
        self.dsc_ll_rew = dsc_ll_rew
        self.dsc_visible_ratio = dsc_visible_ratio
        self.dsc_header_mask = dsc_header_mask
        self.log_dsc_verbose = log_dsc_verbose
        self.rnd_err_norm = rnd_err_norm
        self.rnd_err_momentum = rnd_err_momentum
        self.rnd_use_policy_emb = rnd_use_policy_emb
        self.rnd_use_policy_emb = rnd_use_policy_emb
        self.rnd_err_norm = rnd_err_norm
        self.rnd_err_momentum = rnd_err_momentum
        self.rnd_err_running_stats = RunningMeanStd(momentum=self.rnd_err_momentum)
        self.ll_rew_running_stats = RunningMeanStd(momentum=self.rnd_err_momentum)
        self.env_render = env_render
        self.policy_cnn = policy_cnn
        self._init_obs_queue()

        self._build()
        self._init_modules()
        self._init_optimizers()

        if self.dsc_clip_value > 0:
            for param in self.model_params:
                param.data.clamp_(-self.dsc_clip_value, self.dsc_clip_value)

    def _build(self):
        # Build CNN and RNN
        super()._build()

        # Build MLP
        self.model_mlp = DiscriminatorOutputHeads(
            inputs_dim= self.model_features_dim,
            latents_dim= self.model_latents_dim,
            activation_fn = self.activation_fn,
            action_num = self.action_num,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
            use_rnd=self.dsc_use_rnd,
        )


    def _get_fake_obs(self, curr_obs, next_obs):
        queue_len = min(self.obs_queue_filled, self.dsc_obs_queue_len)
        batch_size = curr_obs.shape[0]

        # Randomly select two fake observations from the queue
        random_idx1 = self.obs_rng.integers(low=0, high=queue_len, size=batch_size, dtype=int)
        random_idx2 = self.obs_rng.integers(low=0, high=queue_len, size=batch_size, dtype=int)
        random_obs1 = obs_as_tensor(self.obs_queue[random_idx1], curr_obs.device)
        random_obs2 = obs_as_tensor(self.obs_queue[random_idx2], curr_obs.device)

        # obs_diff{1,2}: if the ture observation at t+1 (next_obs) is different to the {1st,2nd} fake sample (random_obs1)
        obs_diff1 = th.abs(next_obs - random_obs1).sum((1, 2, 3))
        obs_diff2 = th.abs(next_obs - random_obs2).sum((1, 2, 3))
        obs_diff1 = th.gt(obs_diff1, th.zeros_like(obs_diff1)).long().view(-1, 1, 1, 1)
        obs_diff2 = th.gt(obs_diff2, th.zeros_like(obs_diff2)).long().view(-1, 1, 1, 1)
        obs_diff = th.logical_or(obs_diff1, obs_diff2).long().view(-1, 1, 1, 1)

        # if next_obs != random_obs1 then return random_obs1, otherwise random_obs2
        rand_obs = random_obs1 * obs_diff1 + random_obs2 * (1 - obs_diff1)
        return rand_obs, obs_diff


    def _init_obs_queue(self):
        self.obs_shape = get_obs_shape(self.observation_space)
        self.obs_queue_filled = 0
        self.obs_queue_pos = 0
        self.obs_queue = np.zeros((self.dsc_obs_queue_len,) + self.obs_shape, dtype=float)
        if self.dsc_ll_rew:
            self.ll_emb_queue = th.zeros([self.dsc_obs_queue_len, self.model_features_dim])


    def _get_dsc_embeddings(self, curr_obs, next_obs, last_mems, device=None):
        if not isinstance(curr_obs, Tensor):
            curr_obs = obs_as_tensor(curr_obs, device)
            next_obs = obs_as_tensor(next_obs, device)

        # Get CNN embeddings
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(next_obs)

        # If RNN enabled
        if self.use_model_rnn:
            curr_mems = self._get_rnn_embeddings(last_mems, curr_cnn_embs, self.model_rnns)
            next_mems = self._get_rnn_embeddings(curr_mems, next_cnn_embs, self.model_rnns)
            curr_rnn_embs = th.squeeze(curr_mems[:, -1, :])
            next_rnn_embs = th.squeeze(next_mems[:, -1, :])
            return curr_cnn_embs, next_cnn_embs, curr_rnn_embs, next_rnn_embs, curr_mems

        # If RNN disabled
        return curr_cnn_embs, next_cnn_embs, curr_cnn_embs, next_cnn_embs, None


    def _get_training_losses(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor,
        curr_act: Tensor, curr_dones: Tensor,
        obs_diff: Tensor, labels: Tensor,
        key_status: Optional[Tensor],
        door_status: Optional[Tensor],
        target_dists: Optional[Tensor],
    ):
        # Count valid samples in a batch. A transition (o_t, a_t, o_t+1) is deemed invalid if:
        # 1) an episode ends at t+1, or 2) the ture sample is identical to the fake sample selected at t+1
        n_half_batch = curr_dones.shape[0] // 2
        valid_pos_samples = (1 - curr_dones[n_half_batch:].view(-1)).long()
        valid_neg_samples = th.logical_and(valid_pos_samples, obs_diff.view(-1)).long()
        n_valid_pos_samples = valid_pos_samples.sum().long().item()
        n_valid_neg_samples = valid_neg_samples.sum().long().item()
        n_valid_samples = n_valid_pos_samples + n_valid_neg_samples
        pos_loss_factor = 1 / n_valid_pos_samples if n_valid_pos_samples > 0 else 0.0
        neg_loss_factor = 1 / n_valid_neg_samples if n_valid_neg_samples > 0 else 0.0

        if self.model_cnn_extractor.n_input_channels == 1:
            curr_obs = curr_obs[:, -1:, :, :]
            next_obs = next_obs[:, -1:, :, :]

        if self.dsc_header_mask:
            curr_obs = self.apply_header_mask(curr_obs)
            next_obs = self.apply_header_mask(next_obs)

        if th.is_grad_enabled() and self.dsc_visible_ratio < 1.0:
            curr_obs = self.apply_random_mask(curr_obs, self.dsc_visible_ratio, 1.0)
            next_obs = self.apply_random_mask(next_obs, self.dsc_visible_ratio, 1.0)
            # for env_id in range(10):
            #     image_numpy = curr_obs[env_id][-1].cpu().numpy()
            #     image_pil = Image.fromarray(np.uint8(image_numpy * 255)).resize((384, 384))
            #     cv2.imshow("obs_mask", np.array(image_pil))
            #     cv2.waitKey(1)
            #     input()

        # Get discriminator embeddings
        curr_cnn_embs, next_cnn_embs, curr_embs, next_embs, curr_mems = \
            self._get_dsc_embeddings(curr_obs, next_obs, last_mems)

        # Get likelihoods
        likelihoods = self.model_mlp(curr_embs, next_embs, curr_act).view(-1)
        likelihoods = th.sigmoid(likelihoods).view(-1)

        # Discriminator loss
        pos_dsc_losses = F.binary_cross_entropy(likelihoods[:n_half_batch], labels[:n_half_batch], reduction='none')
        neg_dsc_losses = F.binary_cross_entropy(likelihoods[n_half_batch:], labels[n_half_batch:], reduction='none')
        pos_dsc_loss = (pos_dsc_losses.view(-1) * valid_pos_samples).sum() * pos_loss_factor
        neg_dsc_loss = (neg_dsc_losses.view(-1) * valid_neg_samples).sum() * neg_loss_factor

        # Balance positive and negative samples
        if 0 < n_valid_pos_samples < n_valid_neg_samples:
            pos_dsc_loss *= n_valid_neg_samples / n_valid_pos_samples
        if 0 < n_valid_neg_samples < n_valid_pos_samples:
            neg_dsc_loss *= n_valid_pos_samples / n_valid_neg_samples

        # Get discriminator loss
        dsc_loss = (pos_dsc_loss + neg_dsc_loss) * 0.5

        if self.log_dsc_verbose:
            with th.no_grad():
                pos_avg_likelihood = (likelihoods[:n_half_batch].view(-1) * valid_pos_samples).sum() * pos_loss_factor
                neg_avg_likelihood = (likelihoods[n_half_batch:].view(-1) * valid_neg_samples).sum() * neg_loss_factor
                avg_likelihood = (pos_avg_likelihood + neg_avg_likelihood) * 0.5
                dsc_accuracy = 1 - th.abs(likelihoods - labels).sum() / likelihoods.shape[0]
        else:
            avg_likelihood, pos_avg_likelihood, neg_avg_likelihood, dsc_accuracy = None, None, None, None

        if self.use_status_predictor:
            key_loss, door_loss, pos_loss, key_dist, door_dist, goal_dist = \
                self._get_status_prediction_losses(curr_embs, key_status, door_status, target_dists)
        else:
            key_loss, door_loss, pos_loss, key_dist, door_dist, goal_dist = [self.constant_zero] * 6

        return dsc_loss, pos_dsc_loss, neg_dsc_loss, \
               key_loss, door_loss, pos_loss, \
               key_dist, door_dist, goal_dist, \
               n_valid_samples, n_valid_pos_samples, n_valid_neg_samples, \
               avg_likelihood, pos_avg_likelihood, neg_avg_likelihood, dsc_accuracy, \
               curr_cnn_embs, next_cnn_embs, curr_embs, next_embs, curr_mems


    def rnd_forward(self, curr_obs: Tensor, last_mems: Tensor, curr_dones: Optional[Tensor]):
        curr_mlp_inputs, curr_mems = self._get_rnd_embeddings(curr_obs, last_mems)

        tgt, prd = self.model_mlp.rnd_forward(curr_mlp_inputs)
        rnd_losses = F.mse_loss(prd, tgt.detach(), reduction='none').mean(-1)

        curr_dones = curr_dones.view(-1)
        n_samples = (1 - curr_dones).sum()
        rnd_loss = rnd_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)
        return rnd_loss, rnd_losses, curr_mems


    def _add_obs(self, obs, emb=None):
        # Add new element into the observation queue
        self.obs_queue[self.obs_queue_pos] = np.copy(obs)
        if self.dsc_ll_rew and emb is not None:
            self.ll_emb_queue[self.obs_queue_pos] = emb.detach().clone()
        self.obs_queue_filled += 1
        self.obs_queue_pos += 1
        self.obs_queue_pos %= self.dsc_obs_queue_len


    def init_obs_queue(self, obs_arr):
        # To ensure the observation queue is not empty on training start
        # by adding all observations received at time step 0
        for obs in obs_arr:
            self._add_obs(obs)


    def update_obs_queue(self, iteration, intrinsic_rewards, ir_mean, new_obs, new_emb, stats_logger):
        # Update the observation queue after IR generation
        for env_id in range(new_obs.shape[0]):
            if iteration == 0 or intrinsic_rewards[env_id] >= ir_mean:
                obs = new_obs[env_id]
                emb = new_emb[env_id]
                self._add_obs(obs, emb)
                stats_logger.add(obs_insertions=1)
            else:
                stats_logger.add(obs_insertions=0)


    def get_intrinsic_rewards(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor, curr_dones: Tensor, curr_act: Tensor,
        obs_history: List, trj_history: List, stats_logger: StatisticsLogger, plain_dsc: bool = False
    ):
        # Get observation and trajectory embeddings at t and t+1
        with th.no_grad():
            # curr_cnn_embs, next_cnn_embs, curr_rnn_embs, next_rnn_embs, model_mems = \
            #     self._get_dsc_embeddings(curr_obs, next_obs, last_mems)
            label_ones = th.ones_like(curr_dones)
            label_zeros = th.zeros_like(curr_dones)
            pred_labels = th.cat([label_ones, label_zeros], dim=0)
            fake_obs, obs_differences = self._get_fake_obs(curr_obs, next_obs)
            key_status, door_status, target_dists = None, None, None

            # Get training losses & analysis metrics
            dsc_loss, pos_dsc_loss, neg_dsc_loss, \
            key_loss, door_loss, pos_loss, \
            key_dist, door_dist, goal_dist, \
            n_valid_samples, n_valid_pos_samples, n_valid_neg_samples, \
            avg_likelihood, pos_avg_likelihood, neg_avg_likelihood, dsc_accuracy, \
            curr_cnn_embs, next_cnn_embs, curr_rnn_embs, next_rnn_embs, model_mems = \
                self._get_training_losses(
                    curr_obs.tile((2, 1, 1, 1)),
                    th.cat([next_obs, fake_obs], dim=0),
                    last_mems.tile((2, 1, 1)),
                    curr_act.tile(2),
                    curr_dones.tile((2, 1)).long().view(-1),
                    obs_differences,
                    pred_labels.float().view(-1),
                    key_status,
                    door_status,
                    target_dists
                )
            model_mems = model_mems[:curr_obs.shape[0]]

        stats_logger.add(
            dsc_loss=dsc_loss,
            pos_dsc_loss=pos_dsc_loss,
            neg_dsc_loss=neg_dsc_loss,
        )
        # if self.env_render:
        #     print(f'valid_dsc_loss: {dsc_loss:.9f}', end='\t')

        # Create IRs by the discriminator (Algorithm A1 in the Technical Appendix)
        batch_size = curr_obs.shape[0]
        int_rews = np.zeros(batch_size, dtype=np.float32)

        if self.dsc_ll_rew:
            if self.ll_emb_queue.device != next_cnn_embs.device:
                self.ll_emb_queue = self.ll_emb_queue.to(next_cnn_embs.device)
            ll_obs_dists = self.calc_euclidean_dists(
                next_cnn_embs[:batch_size].view(batch_size, self.model_features_dim),
                self.ll_emb_queue
            )
            ll_rews = ll_obs_dists.min(-1)[0].cpu().numpy()
            if self.rnd_err_norm > 0:
                self.ll_rew_running_stats.update(ll_rews)
                ll_rews = normalize_rewards(
                    norm_type=self.rnd_err_norm,
                    rewards=ll_rews,
                    mean=self.ll_rew_running_stats.mean,
                    std=self.ll_rew_running_stats.std,
                )

        for env_id in range(batch_size):
            # Update the episodic history of observation embeddings
            curr_obs_emb = curr_cnn_embs[env_id].view(1, -1)
            next_obs_emb = next_cnn_embs[env_id].view(1, -1)
            obs_embs = obs_history[env_id]
            new_embs = [curr_obs_emb, next_obs_emb] if obs_embs is None else [obs_embs, next_obs_emb]
            obs_embs = th.cat(new_embs, dim=0)
            obs_history[env_id] = obs_embs
            obs_dists = self.calc_euclidean_dists(obs_embs[:-1], obs_embs[-1])

            # Update the episodic history of trajectory embeddings
            if curr_rnn_embs is not None:
                curr_trj_emb = curr_rnn_embs[env_id].view(1, -1)
                next_trj_emb = next_rnn_embs[env_id].view(1, -1)
                trj_embs = trj_history[env_id]
                new_embs = [th.zeros_like(curr_trj_emb), curr_trj_emb, next_trj_emb] if trj_embs is None else [trj_embs, next_trj_emb]
                trj_embs = th.cat(new_embs, dim=0)
                trj_history[env_id] = trj_embs
                trj_dists = self.calc_euclidean_dists(trj_embs[:-2], trj_embs[-2])
            else:
                trj_dists = th.ones_like(obs_dists)

            # Generate intrinsic reward
            if not plain_dsc:
                # DEIR: Equation 4 in the main paper
                # if self.dsc_tolerance > 0 and obs_dists.shape[0] > self.dsc_tolerance:
                #     deir_dists = th.pow(obs_dists[:-self.dsc_tolerance], 2.0) / (trj_dists[:-self.dsc_tolerance] + 1e-6)
                #     int_rews[env_id] += deir_dists.min().item()
                # else:
                #     deir_dists = th.pow(obs_dists, 2.0) / (trj_dists + 1e-6)
                #     int_rews[env_id] += deir_dists.min().item()

                if self.dsc_tolerance > 1:
                    deir_dists = th.pow(obs_dists, 2.0) / (trj_dists + 1e-6)
                    sorted, indices = th.sort(deir_dists.view(-1))
                    k = min(self.dsc_tolerance, len(deir_dists)) - 1
                    int_rews[env_id] += sorted[k].item()
                else:
                    deir_dists = th.pow(obs_dists, 2.0) / (trj_dists + 1e-6)
                    int_rews[env_id] += deir_dists.min().item()
                    
                if self.env_render and env_id == 0:
                    print(f'ep_rew: {int_rews[env_id]:.4f}', end='\t')

                if self.dsc_ll_rew:
                    ll_rew = min(max(ll_rews[env_id] + 1.0, 1.0), 5.0)
                    if self.env_render and env_id == 0:
                        print(f'll_rew: {ll_rew:.4f}', end='\t')
                    int_rews[env_id] *= ll_rew

            else:
                # Plain discriminator (DEIR without the MI term)
                int_rews[env_id] += obs_dists.min().item()

        # RND
        if self.dsc_use_rnd:
            with th.no_grad():
                rnd_loss, rnd_losses, _ = self.rnd_forward(curr_obs, last_mems, curr_dones)
                rnd_losses = rnd_losses.clone().cpu().numpy()

            if self.rnd_err_norm > 0:
                # Normalize RND error per step
                self.rnd_err_running_stats.update(rnd_losses)
                rnd_losses = normalize_rewards(
                    norm_type=self.rnd_err_norm,
                    rewards=rnd_losses,
                    mean=self.rnd_err_running_stats.mean,
                    std=self.rnd_err_running_stats.std,
                )

            lifelong_rewards = rnd_losses + 1.0
            for env_id in range(batch_size):
                L = 5.0  # L is a chosen maximum reward scaling (default: 5)
                lifelong_reward = min(max(lifelong_rewards[env_id], 1.0), L)
                if env_id==0 and self.env_render:
                    print(f'RND IR:{lifelong_reward:8.6f}', end='  ')
                    print(f'DEIR:{int_rews[0]:8.6f}', end='  ')
                int_rews[env_id] *= lifelong_reward

        return int_rews, model_mems, next_cnn_embs.view(-1, self.model_features_dim)

    @staticmethod
    def apply_random_mask(obs, min_size, max_size):
        batch_size, num_channels, height, width = obs.size()
        mins = int(min(width, height) * min_size)
        maxs = int(min(width, height) * max_size)

        x = th.randint(0, width - mins + 1, (batch_size,))
        y = th.randint(0, height - mins + 1, (batch_size,))
        rect_size = th.randint(mins, maxs + 1, (batch_size,))

        mask = th.zeros_like(obs)
        for i in range(batch_size):
            mask[i, :, y[i]:y[i] + rect_size[i], x[i]:x[i] + rect_size[i]] = 1

        return obs * mask

    @staticmethod
    def apply_header_mask(obs, block_rate=0.12):
        batch_size, num_channels, height, width = obs.size()
        block_h = int(height * block_rate)
        obs[:, :, :block_h, :] = 0
        return obs

    def optimize(self, rollout_data, stats_logger):
        # Prepare input data
        with th.no_grad():
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()
            label_ones = th.ones_like(rollout_data.episode_dones)
            label_zeros = th.zeros_like(rollout_data.episode_dones)
            pred_labels = th.cat([label_ones, label_zeros], dim=0)
            curr_obs = rollout_data.observations
            next_obs = rollout_data.new_observations
            fake_obs, obs_differences = self._get_fake_obs(curr_obs, next_obs)
            key_status, door_status, target_dists = None, None, None
            if self.use_status_predictor:
                key_status = rollout_data.curr_key_status
                door_status = rollout_data.curr_door_status
                target_dists = rollout_data.curr_target_dists

        # Get training losses & analysis metrics
        dsc_loss, pos_dsc_loss, neg_dsc_loss, \
        key_loss, door_loss, pos_loss, \
        key_dist, door_dist, goal_dist, \
        n_valid_samples, n_valid_pos_samples, n_valid_neg_samples, \
        avg_likelihood, pos_avg_likelihood, neg_avg_likelihood, dsc_accuracy,\
        curr_cnn_embs, next_cnn_embs, curr_rnn_embs, next_rnn_embs, model_mems = \
            self._get_training_losses(
                curr_obs.tile((2, 1, 1, 1)),
                th.cat([next_obs, fake_obs], dim=0),
                rollout_data.last_model_mems.tile((2, 1, 1)),
                actions.tile(2),
                rollout_data.episode_dones.tile((2, 1)).long().view(-1),
                obs_differences,
                pred_labels.float().view(-1),
                key_status,
                door_status,
                target_dists
            )

        if self.dsc_use_rnd:
            rnd_loss, rnd_losses, _ = self.rnd_forward(
                curr_obs,
                rollout_data.last_model_mems,
                rollout_data.episode_dones)
            total_loss = dsc_loss + rnd_loss
        else:
            total_loss = dsc_loss
            rnd_loss = None

        if self.dsc_l1_reg_coef > 0:
            l1_reg_loss = th.tensor(0., requires_grad=True)
            for param in self.model_params:
                l1_reg_loss = l1_reg_loss + th.norm(param, p=1)
            total_loss = total_loss + l1_reg_loss * self.dsc_l1_reg_coef
        else:
            l1_reg_loss = None

        if self.dsc_l2_reg_coef > 0:
            l2_reg_loss = th.tensor(0., requires_grad=True)
            for param in self.model_params:
                l2_reg_loss = l2_reg_loss + th.norm(param, p=2)
            total_loss = total_loss + l2_reg_loss * self.dsc_l2_reg_coef
        else:
            l2_reg_loss = None

        # Train the discriminator
        self.model_optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()

        # Train the status predictor(s) for analysis experiments
        if self.use_status_predictor:
            predictor_loss = key_loss + door_loss + pos_loss
            self.predictor_optimizer.zero_grad()
            predictor_loss.backward()
            self.predictor_optimizer.step()

        if self.dsc_clip_value > 0:
            for param in self.model_params:
                param.data.clamp_(-self.dsc_clip_value, self.dsc_clip_value)

        if self.env_render:
            print(f'dsc_loss: {dsc_loss.item():.9f}\tacc: {dsc_accuracy.item()*100:.6f}')

        # Logging
        stats_logger.add(
            dsc_loss=dsc_loss,
            l1_reg_loss=l1_reg_loss,
            l2_reg_loss=l2_reg_loss,
            rnd_loss=rnd_loss,
            pos_dsc_loss=pos_dsc_loss,
            neg_dsc_loss=neg_dsc_loss,
            avg_likelihood=avg_likelihood,
            pos_avg_likelihood=pos_avg_likelihood,
            neg_avg_likelihood=neg_avg_likelihood,
            dsc_accuracy=dsc_accuracy,
            key_loss=key_loss,
            door_loss=door_loss,
            pos_loss=pos_loss,
            key_dist=key_dist,
            door_dist=door_dist,
            goal_dist=goal_dist,
            n_valid_samples=n_valid_samples,
            n_valid_pos_samples=n_valid_pos_samples,
            n_valid_neg_samples=n_valid_neg_samples,
        )
