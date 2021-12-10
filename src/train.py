import os.path
import time
import click
import wandb
import warnings

from datetime import datetime
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from procgen import ProcgenEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
from wandb.integration.sb3 import WandbCallback

# noinspection PyUnresolvedReferences
from src.env.minigrid_envs import *
from src.algo.ppo_model import PPOModel
from src.algo.common_models.cnns import *
from src.algo.ppo_trainer import PPOTrainer
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.utils.common_func import get_enum_norm_type, get_enum_env_src, get_enum_model_type
from src.utils.configs import TrainingConfig
from src.utils.enum_types import ModelType, EnvSrc
from src.utils.loggers import LocalLogger
from src.utils.vec_video_recorder import VecVideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(config):
    th.autograd.set_detect_anomaly(False)
    th.set_default_dtype(th.float32)
    # th.backends.cudnn.benchmark = True
    th.backends.cudnn.benchmark = False

    if config.fully_obs:
        wrapper_class = lambda x: ImgObsWrapper(FullyObsWrapper(x))
    else:
        wrapper_class = lambda x: ImgObsWrapper(x)

    if config.fixed_seed >= 0 and config.env_source == EnvSrc.MiniGrid:
        assert not config.fully_obs
        _seeds = [config.fixed_seed]
        wrapper_class = lambda x: ImgObsWrapper(ReseedWrapper(x, seeds=_seeds))

    if config.env_source == EnvSrc.MiniGrid:
        venv = make_vec_env(
            config.env_name,
            wrapper_class=wrapper_class,
            vec_env_cls=CustomSubprocVecEnv,
            n_envs=config.num_processes,
            monitor_dir=config.log_dir,
        )
    elif config.env_source == EnvSrc.ProcGen:
        venv = ProcgenEnv(
            num_envs=config.num_processes,
            env_name=config.env_name,
            rand_seed=config.run_id,
            num_threads=config.procgen_num_threads,
            distribution_mode=config.procgen_mode,
        )
        venv = VecMonitor(venv=venv)
    elif config.env_source == EnvSrc.Atari:
        from src.utils.env_utils import make_atari_env_with_envpool

        venv = make_atari_env_with_envpool(
            config.env_name,
            n_envs=config.num_processes,
            seed=config.run_id,
            env_kwargs=dict(
                batch_size=config.num_processes,
                gray_scale=bool(config.atari_gray_scale),
                img_height=config.atari_img_size,
                img_width=config.atari_img_size,
                reward_clip=False,
                max_episode_steps=config.atari_max_steps,
                episodic_life=bool(config.episodic_life),
                use_fire_reset=False,
                frame_skip=config.frame_skip,
                stack_num=config.atari_stack_num,
                # repeat_action_probability=config.repeat_act_prob,
                repeat_action_probability=0.0,
                use_inter_area_resize=config.use_inter_area_resize,
                noop_max=30,
            ),
        )
    else:
        raise NotImplementedError


    if config.group_name is not None:
        callbacks = CallbackList([
            WandbCallback(
                gradient_save_freq=50,
                verbose=1,
            )])
    else:
        callbacks = CallbackList([])

    if config.record_video == 1 and config.run_id == 0 or \
        config.record_video == 2:
        venv = VecVideoRecorder(
            venv,
            os.path.join(config.log_dir, 'videos'),
            record_video_trigger=lambda x: x > 0 and x % (config.n_steps * config.rec_interval) == 0,
            video_length=config.video_length,
        )

    if config.optimizer.lower() == 'adam':
        optimizer_class = th.optim.Adam
        optimizer_kwargs = dict(
            eps = config.optim_eps,
            betas = (config.adam_beta1, config.adam_beta2),
        )
    elif config.optimizer.lower() == 'rmsprop':
        optimizer_class = th.optim.RMSprop
        optimizer_kwargs = dict(
            eps = config.optim_eps,
            alpha = config.rmsprop_alpha,
            momentum = config.rmsprop_momentum,
        )
    else:
        raise NotImplementedError


    if config.activation_fn.lower() == 'relu':
        activation_fn = nn.ReLU
    elif config.activation_fn.lower() == 'gelu':
        activation_fn = nn.GELU
    elif config.activation_fn.lower() == 'elu':
        activation_fn = nn.ELU
    else:
        raise NotImplementedError

    if config.cnn_activation_fn.lower() == 'relu':
        cnn_activation_fn = nn.ReLU
    elif config.cnn_activation_fn.lower() == 'gelu':
        cnn_activation_fn = nn.GELU
    elif config.cnn_activation_fn.lower() == 'elu':
        cnn_activation_fn = nn.ELU
    else:
        raise NotImplementedError

    features_extractor_common_kwargs = dict(
        features_dim=config.features_dim,
        activation_fn=cnn_activation_fn,
        # cnn_fc_layers=config.cnn_fc_layers,
        model_type=config.policy_cnn_type,
    )

    model_features_extractor_common_kwargs = dict(
        features_dim=config.model_features_dim,
        activation_fn=cnn_activation_fn,
        model_type=config.model_cnn_type,
        cnn_fc_layers=config.cnn_fc_layers,
        n_input_channels=1,
    )

    config.policy_cnn_norm = get_enum_norm_type(config.policy_cnn_norm)
    config.policy_mlp_norm = get_enum_norm_type(config.policy_mlp_norm)
    config.policy_gru_norm = get_enum_norm_type(config.policy_gru_norm)

    config.model_cnn_norm = get_enum_norm_type(config.model_cnn_norm)
    config.model_mlp_norm = get_enum_norm_type(config.model_mlp_norm)
    config.model_gru_norm = get_enum_norm_type(config.model_gru_norm)

    if config.policy_cnn_norm == NormType.BatchNorm:
        policy_features_extractor_class = BatchNormCnnFeaturesExtractor
    elif config.policy_cnn_norm == NormType.LayerNorm:
        policy_features_extractor_class = LayerNormCnnFeaturesExtractor
    elif config.policy_cnn_norm == NormType.NoNorm:
        policy_features_extractor_class = CnnFeaturesExtractor
    else:
        raise ValueError

    if config.model_cnn_norm == NormType.BatchNorm:
        model_cnn_features_extractor_class = BatchNormCnnFeaturesExtractor
    elif config.model_cnn_norm == NormType.LayerNorm:
        model_cnn_features_extractor_class = LayerNormCnnFeaturesExtractor
    elif config.model_cnn_norm == NormType.NoNorm:
        model_cnn_features_extractor_class = CnnFeaturesExtractor
    else:
        raise ValueError

    config.int_rew_source = get_enum_model_type(config.int_rew_source)
    if config.int_rew_source == ModelType.DEIR and not config.use_model_rnn:
        print('\nWARNING: Running DEIR without RNNs\n')
    if config.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
        assert config.n_steps * config.num_processes >= config.batch_size

    with_rnn_str = 'WITH' if config.use_model_rnn else 'WITHOUT'
    print(f"Using {str(config.int_rew_source).split('.')[-1]} model {with_rnn_str} RNN enabled")

    if config.clip_range_vf <= 0:
        config.clip_range_vf = None

    policy_kwargs = dict(
        run_id=config.run_id,
        n_envs=config.num_processes,
        activation_fn=activation_fn,
        learning_rate=config.learning_rate,
        model_learning_rate=config.model_learning_rate,
        policy_features_extractor_class=policy_features_extractor_class,
        policy_features_extractor_kwargs=features_extractor_common_kwargs,
        model_cnn_features_extractor_class=model_cnn_features_extractor_class,
        model_cnn_features_extractor_kwargs=model_features_extractor_common_kwargs,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=config.max_grad_norm,
        model_features_dim=config.model_features_dim,
        latents_dim=config.latents_dim,
        model_latents_dim=config.model_latents_dim,
        policy_mlp_norm=config.policy_mlp_norm,
        model_mlp_norm=config.model_mlp_norm,
        model_cnn_norm=config.model_cnn_norm,
        model_mlp_layers=config.model_mlp_layers,
        use_status_predictor=config.use_status_predictor,
        gru_layers=config.gru_layers,
        policy_mlp_layers=config.policy_mlp_layers,
        policy_gru_norm=config.policy_gru_norm,
        use_model_rnn=config.use_model_rnn,
        model_gru_norm=config.model_gru_norm,
        total_timesteps=config.total_steps,
        n_steps=config.n_steps,
        int_rew_source=config.int_rew_source,
        int_rew_gamma=config.int_rew_gamma,
        icm_forward_loss_coef=config.icm_forward_loss_coef,
        ngu_knn_k=config.ngu_knn_k,
        ngu_dst_momentum=config.ngu_dst_momentum,
        ngu_use_rnd=config.ngu_use_rnd,
        rnd_err_norm=config.rnd_err_norm,
        rnd_err_momentum=config.rnd_err_momentum,
        rnd_use_policy_emb=config.rnd_use_policy_emb,
        dsc_obs_queue_len=config.dsc_obs_queue_len,
        dsc_ll_rew=config.dsc_ll_rew,
        dsc_use_rnd=config.dsc_use_rnd,
        dsc_l1_reg_coef=config.dsc_l1_reg_coef,
        dsc_l2_reg_coef=config.dsc_l2_reg_coef,
        dsc_clip_value=config.dsc_clip_value,
        dsc_ir_bias=config.dsc_ir_bias,
        dsc_tolerance=config.dsc_tolerance,
        dsc_visible_ratio=config.dsc_visible_ratio,
        dsc_header_mask=config.dsc_header_mask,
        log_dsc_verbose=config.log_dsc_verbose,
        noop_bias=config.noop_bias,
        env_render=config.env_render,
    )

    model = PPOTrainer(
        policy=PPOModel,
        env=venv,
        seed=config.run_id,
        run_id=config.run_id,
        can_see_walls=config.can_see_walls,
        image_noise_scale=config.image_noise_scale,
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        model_n_epochs=config.model_n_epochs,
        learning_rate=config.learning_rate,
        model_learning_rate=config.model_learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        int_rew_gamma=config.int_rew_gamma,
        ent_coef=config.ent_coef,
        batch_size=config.batch_size,
        pg_coef=config.pg_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        ext_rew_coef=config.ext_rew_coef,
        int_rew_source=config.int_rew_source,
        int_rew_coef=config.int_rew_coef,
        int_rew_norm=config.int_rew_norm,
        int_rew_momentum=config.int_rew_momentum,
        int_rew_eps=config.int_rew_eps,
        int_rew_clip=config.int_rew_clip,
        int_rew_lb=config.int_rew_lb,
        int_rew_ub=config.int_rew_ub,
        adv_momentum=config.adv_momentum,
        adv_norm=config.adv_norm,
        adv_eps=config.adv_eps,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        policy_kwargs=policy_kwargs,
        env_source=config.env_source,
        env_render=config.env_render,
        fixed_seed=config.fixed_seed,
        atari_clip_rew=config.atari_clip_rew,
        reset_on_life_loss=config.reset_on_life_loss,
        life_loss_penalty=config.life_loss_penalty,
        atari_img_size=config.atari_img_size,
        use_preset_actions=config.use_preset_actions,
        frame_skip=config.frame_skip,
        repeat_act_prob=config.repeat_act_prob,
        use_wandb=config.use_wandb,
        local_logger=config.local_logger,
        record_video=config.record_video,
        rec_interval=config.rec_interval,
        enable_plotting=config.enable_plotting,
        plot_interval=config.plot_interval,
        plot_colormap=config.plot_colormap,
        log_explored_states=config.log_explored_states,
        verbose=0,
    )

    if config.run_id == 0:
        print('model.policy:',model.policy)

    model.learn(
        total_timesteps=config.total_steps,
        callback=callbacks)



@click.command()
# Training params
@click.option('--run_id', default=0, type=int, help='Index (and seed) of the current run')
@click.option('--group_name', type=str, help='Group name (wandb option), leave blank if not logging with wandb')
@click.option('--log_dir', default='./logs', type=str, help='Directory for saving training logs')
@click.option('--total_steps', default=int(1e6), type=int, help='Total number of frames to run for training')
@click.option('--features_dim', default=64, type=int, help='Number of neurons of a learned embedding (PPO)')
@click.option('--model_features_dim', default=128, type=int, help='Number of neurons of a learned embedding (dynamics model)')
@click.option('--learning_rate', default=3e-4, type=float, help='Learning rate of PPO')
@click.option('--model_learning_rate', default=3e-4, type=float, help='Learning rate of the dynamics model')
@click.option('--num_processes', default=16, type=int, help='Number of training processes (workers)')
@click.option('--batch_size', default=512, type=int, help='Batch size')
@click.option('--n_steps', default=512, type=int, help='Number of steps to run for each process per update')
# Env params
@click.option('--env_source', default='minigrid', type=str, help='minigrid or procgen')
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, ninja, jumper')
@click.option('--project_name', required=False, type=str, help='Where to store training logs (wandb option)')
@click.option('--map_size', default=5, type=int, help='Size of the minigrid room')
@click.option('--can_see_walls', default=1, type=int, help='Whether walls are visible to the agent')
@click.option('--fully_obs', default=0, type=int, help='Whether the agent can receive full observations')
@click.option('--image_noise_scale', default=0.0, type=float, help='Standard deviation of the Gaussian noise')
@click.option('--procgen_mode', default='hard', type=str, help='Mode of ProcGen games (easy or hard)')
@click.option('--procgen_num_threads', default=4, type=int, help='Number of parallel ProcGen threads')
@click.option('--log_explored_states', default=0, type=int, help='Whether to log the number of explored states')
@click.option('--fixed_seed', default=-1, type=int, help='Whether to use a fixed env seed (MiniGrid)')
@click.option('--atari_img_size', default=84, type=int)
@click.option('--atari_stack_num', default=4, type=int)
@click.option('--atari_clip_rew', default=0, type=int)
@click.option('--atari_gray_scale', default=1, type=int)
@click.option('--reset_on_life_loss', default=0, type=int)
@click.option('--repeat_act_prob', default=0.25, type=float)
@click.option('--frame_skip', default=4, type=int)
@click.option('--atari_max_steps', default=4500, type=int)
@click.option('--life_loss_penalty', default=0.0, type=float)
@click.option('--episodic_life', default=0, type=int)
@click.option('--use_inter_area_resize', default=1, type=int)
@click.option('--use_preset_actions', default=0, type=int)
@click.option('--noop_bias', default=0.0, type=float)
# Algo params
@click.option('--n_epochs', default=4, type=int, help='Number of epochs to train policy and value nets')
@click.option('--model_n_epochs', default=4, type=int, help='Number of epochs to train common_models')
@click.option('--gamma', default=0.99, type=float, help='Discount factor')
@click.option('--int_rew_gamma', default=-1, type=float, help='Discount factor for intrinsic rewards (-1: disabled, >0: train dual value heads and apply IR gamma')
@click.option('--gae_lambda', default=0.95, type=float, help='GAE lambda')
@click.option('--pg_coef', default=1.0, type=float, help='Coefficient of policy gradients')
@click.option('--vf_coef', default=0.5, type=float, help='Coefficient of value function loss')
@click.option('--ent_coef', default=0.01, type=float, help='Coefficient of policy entropy')
@click.option('--max_grad_norm', default=0.5, type=float, help='Maximum norm of gradient')
@click.option('--clip_range', default=0.2, type=float, help='PPO clip range of the policy network')
@click.option('--clip_range_vf', default=-1, type=float, help='PPO clip range of the value function (-1: disabled, >0: enabled)')
@click.option('--adv_norm', default=2, type=int, help='Normalized advantages by: [0] No normalization [1] Standardization per mini-batch [2] Standardization per rollout buffer [3] Standardization w.o. subtracting the mean per rollout buffer')
@click.option('--adv_eps', default=1e-5, type=float, help='Epsilon for advantage normalization')
@click.option('--adv_momentum', default=0.9, type=float, help='EMA smoothing factor for advantage normalization')
# Reward params
@click.option('--ext_rew_coef', default=1.0, type=float, help='Coefficient of extrinsic rewards')
@click.option('--int_rew_coef', default=1e-2, type=float, help='Coefficient of intrinsic rewards (IRs)')
@click.option('--int_rew_source', default='DEIR', type=str, help='Source of IRs: [NoModel|DEIR|ICM|RND|NGU|NovelD|PlainDiscriminator|PlainInverse|PlainForward]')
@click.option('--int_rew_norm', default=1, type=int, help='Normalized IRs by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--int_rew_momentum', default=0.9, type=float, help='EMA smoothing factor for IR normalization (-1: total average)')
@click.option('--int_rew_eps', default=1e-5, type=float, help='Epsilon for IR normalization')
@click.option('--int_rew_clip', default=-1, type=float, help='Clip IRs into [-X, X] when X>0')
@click.option('--int_rew_lb', default=-100, type=float, help='Lower bound of IRs')
@click.option('--int_rew_ub', default=100, type=float, help='Upper bound of IRs')
@click.option('--dsc_obs_queue_len', default=100000, type=int, help='Maximum length of observation queue (DEIR)')
@click.option('--dsc_ll_rew', default=0, type=int)
@click.option('--dsc_use_rnd', default=0, type=int, help='Whether to enable RND-based lifelong rewards for DEIR')
@click.option('--dsc_l1_reg_coef', default=0.0, type=float, help='L1 regularization coefficient for DEIR')
@click.option('--dsc_l2_reg_coef', default=0.0, type=float, help='L2 regularization coefficient for DEIR')
@click.option('--dsc_clip_value', default=0.0, type=float, help='Weight clipping value for DEIR')
@click.option('--dsc_ir_bias', default=0.0, type=float)
@click.option('--dsc_tolerance', default=0, type=int)
@click.option('--dsc_visible_ratio', default=1.0, type=float)
@click.option('--dsc_header_mask', default=0, type=int)
@click.option('--icm_forward_loss_coef', default=0.2, type=float, help='Coefficient of forward model losses (ICM)')
@click.option('--ngu_knn_k', default=10, type=int, help='Search for K nearest neighbors (NGU)')
@click.option('--ngu_use_rnd', default=1, type=int, help='Whether to enable lifelong IRs generated by RND (NGU)')
@click.option('--ngu_dst_momentum', default=0.997, type=float, help='EMA smoothing factor for averaging embedding distances (NGU)')
@click.option('--rnd_use_policy_emb', default=1, type=int, help='Whether to use the embeddings learned by policy/value nets as inputs (RND)')
@click.option('--rnd_err_norm', default=1, type=int, help='Normalized RND errors by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--rnd_err_momentum', default=-1, type=float, help='EMA smoothing factor for RND error normalization (-1: total average)')
# Network params
@click.option('--use_model_rnn', default=1, type=int, help='Whether to enable RNNs for the dynamics model')
@click.option('--latents_dim', default=256, type=int, help='Dimensions of latent features in policy/value nets\' MLPs')
@click.option('--model_latents_dim', default=256, type=int, help='Dimensions of latent features in the dynamics model\'s MLP')
@click.option('--policy_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--policy_mlp_layers', default=1, type=int, help='Number of latent layers used in the policy\'s MLP')
@click.option('--policy_cnn_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' CNN')
@click.option('--policy_mlp_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' MLP')
@click.option('--policy_gru_norm', default='NoNorm', type=str, help='Normalization type for policy/value nets\' GRU')
@click.option('--model_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--model_mlp_layers', default=1, type=int, help='Number of latent layers used in the model\'s MLP')
@click.option('--model_cnn_norm', default='BatchNorm', type=str, help='Normalization type for the dynamics model\'s CNN')
@click.option('--model_mlp_norm', default='BatchNorm', type=str, help='Normalization type for the dynamics model\'s MLP')
@click.option('--model_gru_norm', default='NoNorm', type=str, help='Normalization type for the dynamics model\'s GRU')
@click.option('--activation_fn', default='relu', type=str, help='Activation function for non-CNN layers')
@click.option('--cnn_activation_fn', default='relu', type=str, help='Activation function for CNN layers')
@click.option('--cnn_fc_layers', default=1, type=int, help='Number of FC layers after CNN')
@click.option('--gru_layers', default=1, type=int, help='Number of GRU layers in both the policy and the model')
# Optimizer params
@click.option('--optimizer', default='adam', type=str, help='Optimizer, adam or rmsprop')
@click.option('--optim_eps', default=1e-5, type=float, help='Epsilon for optimizers')
@click.option('--adam_beta1', default=0.9, type=float, help='Adam optimizer option')
@click.option('--adam_beta2', default=0.999, type=float, help='Adam optimizer option')
@click.option('--rmsprop_alpha', default=0.99, type=float, help='RMSProp optimizer option')
@click.option('--rmsprop_momentum', default=0.0, type=float, help='RMSProp optimizer option')
# Logging & Analysis options
@click.option('--write_local_logs', default=1, type=int, help='Whether to output training logs locally')
@click.option('--enable_plotting', default=0, type=int, help='Whether to generate plots for analysis')
@click.option('--plot_interval', default=10, type=int, help='Interval of generating plots (iterations)')
@click.option('--plot_colormap', default='Blues', type=str, help='Colormap of plots to generate')
@click.option('--record_video', default=0, type=int, help='Whether to record video')
@click.option('--rec_interval', default=10, type=int, help='Interval of two videos (iterations)')
@click.option('--video_length', default=512, type=int, help='Length of the video (frames)')
@click.option('--log_dsc_verbose', default=0, type=int, help='Whether to record the discriminator loss for each action')
@click.option('--env_render', default=0, type=int, help='Whether to render games in human mode')
@click.option('--use_status_predictor', default=0, type=int, help='Whether to train status predictors for analysis (MiniGrid only)')
def main(
    # Training params
    run_id, group_name, log_dir, total_steps, 
    features_dim, model_features_dim, latents_dim, model_latents_dim,
    learning_rate, model_learning_rate, num_processes, write_local_logs,
    # Env params
    map_size, game_name, record_video, rec_interval, video_length,
    project_name, can_see_walls,
    image_noise_scale, fully_obs, env_source, env_render,
    procgen_num_threads, log_explored_states,
    # Algo params
    n_steps, gamma, gae_lambda,
    int_rew_gamma,
    pg_coef, vf_coef, ent_coef, max_grad_norm,
    n_epochs, model_n_epochs, batch_size,
    int_rew_coef, int_rew_norm, int_rew_momentum, int_rew_eps, int_rew_clip, int_rew_lb, int_rew_ub,
    int_rew_source, icm_forward_loss_coef,
    ngu_knn_k, ngu_use_rnd, ngu_dst_momentum,
    policy_cnn_norm, policy_gru_norm, policy_mlp_norm,
    model_cnn_norm, model_gru_norm, model_mlp_norm,
    enable_plotting,
    activation_fn, cnn_activation_fn, cnn_fc_layers,
    model_mlp_layers,
    ext_rew_coef, use_status_predictor,
    policy_cnn_type, model_cnn_type, gru_layers,
    policy_mlp_layers,
    adam_beta1, adam_beta2,
    optimizer, rmsprop_alpha, rmsprop_momentum,
    optim_eps, adv_norm,
    adv_eps, adv_momentum, clip_range, clip_range_vf,
    use_model_rnn, dsc_obs_queue_len, dsc_use_rnd, dsc_ll_rew, dsc_l1_reg_coef, dsc_l2_reg_coef, dsc_clip_value, dsc_ir_bias,
    dsc_tolerance, dsc_visible_ratio, dsc_header_mask,
    rnd_use_policy_emb, rnd_err_norm, rnd_err_momentum,
    procgen_mode,
    fixed_seed,
    plot_interval, plot_colormap,
    log_dsc_verbose, atari_img_size, atari_stack_num,
    repeat_act_prob, atari_clip_rew, reset_on_life_loss, atari_gray_scale, frame_skip, atari_max_steps, episodic_life,
    life_loss_penalty, use_inter_area_resize, use_preset_actions, noop_bias,
):
    set_random_seed(run_id, using_cuda=True)
    args = locals().items()
    config = TrainingConfig
    for k, v in args: setattr(config, k, v)

    env_name = game_name
    config.env_source = get_enum_env_src(config.env_source)
    if config.env_source == EnvSrc.MiniGrid and not game_name.startswith('MiniGrid-'):
        env_name = f'MiniGrid-{game_name}'
        env_name += '-v0'
    config.env_name = env_name
    config.project_name = env_name if project_name is None else project_name

    config.file_path = __file__
    config.model_name = os.path.basename(__file__)

    config.start_time = time.time()
    config.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    set_random_seed(config.run_id, using_cuda=True)

    if config.group_name is not None:
        run = wandb.init(
            name=f'run-id-{config.run_id}',
            entity='abcde-project',  # your project name on wandb
            project=config.project_name,
            group=config.group_name,
            settings=wandb.Settings(start_method="fork"),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        config.use_wandb = True
    else:
        config.use_wandb = False


    config.log_dir = os.path.join(config.log_dir, config.env_name, config.start_datetime, str(config.run_id))
    os.makedirs(config.log_dir, exist_ok=True)
    if config.write_local_logs:
        config.local_logger = LocalLogger(config.log_dir)
        print(f'Writing local logs at {config.log_dir}')
    else:
        config.local_logger = None

    print(f'Starting run {config.run_id}')
    train(config)
    if config.use_wandb:
        run.finish()


if __name__ == '__main__':
    # cProfile.run('main()', f'run-{time.time():.0f}.prof')
    main()

