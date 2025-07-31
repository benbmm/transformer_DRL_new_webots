import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from einops import rearrange
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "MortarMayhem-Grid-v0"
    """the id of the environment"""
    total_timesteps: int = 200000000
    """total timesteps of the experiments"""
    init_lr: float = 2.75e-4
    """the initial learning rate of the optimizer"""
    final_lr: float = 1.0e-5
    """the final learning rate of the optimizer after linearly annealing"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_steps: int = 32 * 512 * 10000
    """the number of steps to linearly anneal the learning rate and entropy coefficient from initial to final"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    init_ent_coef: float = 0.0001
    """initial coefficient of the entropy bonus"""
    final_ent_coef: float = 0.000001
    """final coefficient of the entropy bonus after linearly annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_layers: int = 3
    """the number of transformer layers"""
    trxl_num_heads: int = 4
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 119
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = "absolute"
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """
    reconstruction_coef: float = 0.0
    """the coefficient of the observation reconstruction loss, if set to 0.0 the reconstruction loss is not used"""

    # To be filled on runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_class, **env_kwargs):
    """創建環境的工廠函數"""
    def thunk():
        return env_class(**env_kwargs)
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    return layer


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb


class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert self.head_size * num_heads == embed_dim, "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))  # -inf causes NaN

        # Normalize energy values and apply softmax to retrieve the attention scores
        attention = torch.softmax(
            energy / (self.embed_dim ** (1 / 2)), dim=3
        )  # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_size)

        return self.fc_out(out), attention


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        # Pre-layer normalization (post-layer normalization is usually less effective)
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value  # K = V -> self-attention
        attention, attention_weights = self.attention(value, key, query_, mask)  # MHA
        x = attention + query  # Skip connection
        x_ = self.layer_norm_attn(x)  # Pre-layer normalization
        forward = self.fc_projection(x_)  # Forward projection
        out = forward + x  # Skip connection
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_episode_steps, positional_encoding):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        if positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim)
        elif positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        self.transformer_layers = nn.ModuleList([TransformerLayer(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, memories, mask, memory_indices):
        # Add positional encoding to every transformer layer input
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        # Forward transformer layers and return new memories (i.e. hidden states)
        out_memories = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, attention_weights = layer(
                memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask
            )  # args: value, key, query, mask
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x, torch.stack(out_memories, dim=1)


class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space_shape, max_episode_steps):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.obs_shape) > 1:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
                nn.ReLU(),
            )
        else:
            self.encoder = layer_init(nn.Linear(observation_space.shape[0], args.trxl_dim))

        self.transformer = Transformer(
            args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps, args.trxl_positional_encoding
        )

        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        # 修正：統一動作處理
        if len(action_space_shape) == 1:  # 連續動作空間
            self.action_dim = action_space_shape[0]
            self.continuous_action = True
            self.actor_mean = layer_init(nn.Linear(args.trxl_dim, self.action_dim), np.sqrt(0.01))
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        else:  # 離散動作空間
            self.continuous_action = False
            self.action_dim = len(action_space_shape)  # 分支數量
            self.actor_branches = nn.ModuleList(
                [
                    layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), np.sqrt(0.01))
                    for num_actions in action_space_shape
                ]
            )
        
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)

        if args.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )

    def get_value(self, x, memory, memory_mask, memory_indices):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        self.x = x
        
        if self.continuous_action:
            # 修正：連續動作處理
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            
            if action is None:
                action = probs.sample()
            
            # 修正：保持原始維度，不進行sum
            log_probs = probs.log_prob(action)  # [batch_size, action_dim]
            entropy = probs.entropy().sum(1)   # [batch_size]
            
            return action, log_probs, entropy, self.critic(x).flatten(), memory
        else:
            # 離散動作處理
            probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]
            if action is None:
                action = torch.stack([dist.sample() for dist in probs], dim=1)
            log_probs = []
            for i, dist in enumerate(probs):
                log_probs.append(dist.log_prob(action[:, i]))
            entropies = torch.stack([dist.entropy() for dist in probs], dim=1).sum(1).reshape(-1)
            return action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten(), memory

    def reconstruct_observation(self):
        x = self.transposed_cnn(self.x)
        return x.permute((0, 2, 3, 1))