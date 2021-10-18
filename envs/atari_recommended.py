import cv2
import gym
import numpy as np
import dm_env
from dm_env import specs

from envs.shared import space2spec, ActionDTypeWrapper, TimeLimit, ExtendedTimeStepWrapper, FrameStackWrapper


class AtariPreprocessing(dm_env.Environment):
    """A dm_env wrapper implementing image preprocessing for Atari 2600 agents.
    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):
      * Frame skipping (defaults to 4).
      * Terminal signal when a life is lost (off by default).
      * Grayscale and max-pooling of the last two frames.
      * Downsample the screen to a square image (defaults to 84x84).
    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    """

    def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
                 screen_size=84):
        """Constructor for an Atari 2600 preprocessor.
        Args:
          environment: Gym environment whose observations are preprocessed.
          frame_skip: int, the frequency at which the agent experiences the game.
          terminal_on_life_loss: bool, If True, the step() method returns
            is_terminal=True whenever a life is lost. See Mnih et al. 2015.
          screen_size: int, size of a resized Atari 2600 frame.
        Raises:
          ValueError: if frame_skip or screen_size are not strictly positive.
        """
        if frame_skip <= 0:
            raise ValueError('Frame skip should be strictly positive, got {}'.
                             format(frame_skip))
        if screen_size <= 0:
            raise ValueError('Target screen size should be strictly positive, got {}'.
                             format(screen_size))

        self.gym_env = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.gym_env.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

        # Convert gym action and observation spaces to dm_env specs.
        self._observation_spec = space2spec(self.gym_env.observation_space,
                                            name='observation')
        self._action_spec = space2spec(self.gym_env.action_space, name='action')
        self._reset_next_step = True

    def close(self):
        self.gym_env.close()

    def observation_spec(self):
        return specs.BoundedArray(shape=(self.screen_size, self.screen_size, 1), dtype=np.uint8,
                                  minimum=0, maximum=255, name="observation")

    def action_spec(self):
        return self._action_spec

    @property
    def action_space(self):
        space = self.environment.action_space
        if isinstance(space, gym.spaces.Discrete):
            return specs.Array(shape=[space.n],
                               dtype=space.dtype,
                               name="action")
        else:
            return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                                      minimum=space.low, maximum=space.high, name="action")

    def render(self, mode='rgb_array'):
        """Renders the current screen, before preprocessing.
        This calls the Gym API's render() method.
        Args:
          mode: Mode argument for the environment's render() method.
            Valid values (str) are:
              'rgb_array': returns the raw ALE image.
              'human': renders to display via the Gym renderer.
        Returns:
          if mode='rgb_array': numpy array, the most recent screen.
          if mode='human': bool, whether the rendering was successful.
        """
        return self.gym_env.render(mode)

    def reset(self):
        """Resets the environment.
        Returns:
          observation: numpy array, the initial observation emitted by the
            environment.
        """
        self._reset_next_step = False
        self.gym_env.reset()
        self.lives = self.gym_env.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        obs = self._pool_and_resize()
        return dm_env.restart(obs)

    def step(self, action):
        """Applies the given action in the environment.
        Remarks:
          * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
          * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.
        Args:
          action: The action to be executed.
        Returns:
          observation: numpy array, the observation following the action.
          reward: float, the reward following the action.
          is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
          info: Gym API's info data structure.
        """
        if self._reset_next_step:
            return self.reset()

        reward = 0.

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, r, game_over, info = self.gym_env.step(action)
            reward += r

            if self.terminal_on_life_loss:
                new_lives = self.gym_env.ale.lives()
                done = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                done = game_over

            if done:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over

        # Convert the gym step result to a dm_env TimeStep.
        self._reset_next_step = done

        if done:
            is_truncated = info.get('TimeLimit.truncated', False)
            if is_truncated:
                return dm_env.truncation(reward, observation)
            else:
                return dm_env.termination(reward, observation)
        else:
            return dm_env.transition(reward, observation)

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.
        The returned observation is stored in 'output'.
        Args:
          output: numpy array, screen buffer to hold the returned observation.
        Returns:
          observation: numpy array, the current observation in grayscale.
        """
        self.gym_env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.
        For efficiency, the transformation is done in-place in self.screen_buffer.
        Returns:
          transformed_screen: numpy array, pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                       out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)


def make(env_id, frame_stack=4, action_repeat=1, max_episode_len=27000, truncate_episode_len=1000, seed=0, train=True):
    env_id = f'ALE/{env_id.capitalize()}-v5'
    env = gym.make(env_id)
    env.seed(seed)
    env = AtariPreprocessing(env, frame_skip=action_repeat,
                             terminal_on_life_loss=False, screen_size=84)
    env = FrameStackWrapper(env, frame_stack)
    env = ActionDTypeWrapper(env, np.int64, discrete=True)
    if max_episode_len and action_repeat:
        max_episode_len = max_episode_len // action_repeat
    env = TimeLimit(env, max_episode_len=max_episode_len)
    if train:
        if truncate_episode_len and action_repeat:
            truncate_episode_len = truncate_episode_len // action_repeat
        env = TimeLimit(env, max_episode_len=truncate_episode_len, resume=True)
    env = ExtendedTimeStepWrapper(env)
    return env
