import collections
import copy

import numpy
import ray
import torch

import models

@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """
    def __init__(self, config):
        self.config = config
        self.buffer = {}
        self.game_priorities = collections.deque(maxlen=self.config.window_size)
        self.max_recorded_game_priority = 1.0
        self.total_samples = 0
        self.num_played_games = 0
        self.num_played_steps = 0

        # Used only for the Reanalyze options
        self.model = None
        if self.config.use_last_model_value:
            self.model = models.MuZeroNetwork(self.config)
            self.model.to(torch.device("cpu"))
            self.model.eval()

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def save_game(self, game_history):
        if game_history.priorities is not None:
            # Avoid read only array when loading replay buffer from pickle
            game_history.priorities = game_history.priorities.copy()
        else:
            if self.config.use_max_priority:
                game_history.priorities = numpy.full(
                    len(game_history.root_values), self.max_recorded_game_priority
                )
            else:
                # TODO Need to check this
                # Initial priorities for the prioritized replay
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)
                
                game_history.priorities = numpy.array(priorities, dtype="float32")
                
            self.buffer[self.num_played_amges] = game_history
            self.total_samples += len(game_history.priorities)
            self.game_priorities.append(numpy.max(game_history.priorities))
            
            self.num_played_games += 1
            self.num_played_steps += len(game_history.observation_history) - 1

            if self.config.window_size < len(self.buffer):
                del_id = self.num_played_games - len(self.buffer)
                self.total_samples -= len(self.buffer[del_id].priorities)
                del self.buffer[del_id]
                
    def get_info(self):
        return {
            "num_played_games": self.num_played_games,
            "num_played_steps": self.num_played_steps,
        }

    def get_buffer(self):
        return self.buffer

    def get_batch(self, model_weights):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            weight_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [], [])

        if self.config.use_last_model_value:
            self.model.set_weights(model_weights)

        for _ in range(self.config.batch_size):
            game_id, game_history, game_prob = self.sample_game(self.buffer)
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos, self.config.stacked_observations
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            weight_batch.append(
                (self.total_samples * game_prob * pos_prob) ** (-self.config.PER_beta)
            )
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )

        weight_batch = numpy.array(weight_batch, dtype="float32") / max(weight_batch)

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )