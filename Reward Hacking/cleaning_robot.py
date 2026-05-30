import numpy as np
import time

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
except ImportError:
    plt = None
    ListedColormap = None
    patches = None
    FuncAnimation = None

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        gym = None


### Room generating functions ###

def initialize_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5):
    """
    Initialises the basic room shape.
    Room is represented by a np.array(shape=(width, width)), initially just with 0 for wall and 1 for floor.
    Start by marking walls around the edges of the square, then creates 'islands' of inaccessible areas.
    No dirt is created yet. 
    """
    assert width > max_island_size
    
    # Initialize main room
    room = np.zeros([width, width], dtype=np.int8)
    
    # Create square to clean in center
    room[1:-1, 1:-1] = 1
    
    # Create random impassable 'islands'
    n_islands = np.random.randint(low=min_n_islands, high=max_n_islands)
    for island in range(n_islands):
        island_size = np.random.randint(low=1, high=max_island_size)
        island_x = np.random.randint(low=0-island_size + 1, high=width-1)
        island_y = np.random.randint(low=0-island_size + 1, high=width-1)
        
        for x_pos in range(island_x, island_x + island_size):
            for y_pos in range(island_y, island_y + island_size):
                if x_pos < 0:
                    x = 0
                elif x_pos >= width:
                    x = width - 1
                else:
                    x = x_pos
                if y_pos < 0:
                    y = 0
                elif y_pos >= width:
                    y = width - 1
                else:
                    y = y_pos
                
                room[x, y] = 0
    
    return room

def is_valid_room(room):
    """
    Takes in an initialised room of walls and floors, then returns True if every cell of the 
    room is accessible, False otherwise.
    """
    target_sum = np.sum(room)
    visited = np.zeros(room.shape)
    
    # If there are only walls, room is invalid
    if target_sum == 0:
        return False
    
    first_cell = np.argwhere(room==1)[0]
    
    def explore(room, current_cell, depth, max_depth=100):
        if depth > max_depth: return
        if visited[current_cell[0], current_cell[1]] == 1: return
        visited[current_cell[0], current_cell[1]] = 1
    
        neighbours = [[current_cell[0] - 1, current_cell[1]] if current_cell[0] > 0 else None, 
                      [current_cell[0] + 1, current_cell[1]] if current_cell[0] < room.shape[0] else None, 
                      [current_cell[0], current_cell[1] - 1] if current_cell[1] > 0 else None, 
                      [current_cell[0], current_cell[1] + 1] if current_cell[1] < room.shape[1] else None]
        neighbours = [neighbour for neighbour in neighbours if neighbour is not None]
        neighbours = [neighbour if room[neighbour[0], neighbour[1]] == 1 else None for neighbour in neighbours]
        neighbours = [neighbour for neighbour in neighbours if neighbour is not None]
        
        for neighbour in neighbours:
            explore(room, neighbour, depth + 1)
            
    explore(room, first_cell, depth=0)
    
    return np.sum(visited) == target_sum

def generate_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5, seed=None):
    """
    Will generate a new room with given parameters.
    After 1 million attempts, program will throw an error, as it's likely the user has entered
    invalid parameters than cannot build a valid room.
    """
    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    
    attempts = 1
    room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)
    while not is_valid_room(room):
        assert attempts < 1e6, "1e6 generations attempted, issue with generation parameters."
        attempts += 1
        room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)
        
    if rng_state is not None:
        np.random.set_state(rng_state)
        
    return room


### Spawning functions ###

def spawn_robot(room, pos_x=None, pos_y=None, orientation=None, seed=None):
    """
    Spawns a robot into the room according to given coordinates, or 
    randomly if none are given.
    """
    # If robot spawn position is given
    if pos_x is not None and pos_y is not None:
        assert room[pos_x, pos_y] in [-1, 1], "Invalid spawn position."
        if orientation is None:
            orientation = np.random.randint(low=1, high=5)
        room[pos_x, pos_y] += orientation
        return room
    
    # Else, random or seeded spawn
    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    
    # Generate random spawn position and orientation
    room_size_x, room_size_y = room.shape[0], room.shape[1]
    pos_x, pos_y = np.random.randint(low=0, high=room_size_x), np.random.randint(low=0, high=room_size_y)
    while room[pos_x, pos_y] not in [-1, 1]:
        pos_x, pos_y = np.random.randint(low=0, high=room_size_x), np.random.randint(low=0, high=room_size_y)

    # An orientation of 1 is facing upward, then moving clockwise so 4 is nine o'clock
    orientation = np.random.randint(low=1, high=5) + 1  # +1 as it's spawning on floor, which has a value of 1
    room[pos_x, pos_y] = orientation
    
    if rng_state is not None:
        np.random.set_state(rng_state)
    
    return room

def spawn_dirt(room, fraction=1, seed=None):
    """
    Creates dirt tiles in room. 
    The fraction parameter denotes the approximate fraction of tiles to be made dirty.
    """
    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    
    existing_dirt = (room < 0).astype(np.int8)
    dirt = np.random.uniform(size=room.shape) + existing_dirt
    dirt = 2 * (dirt > fraction).astype(np.int8) - 1
    
    if rng_state is not None:
        np.random.set_state(rng_state)
    
    robot_pos = np.argwhere(abs(room) > 1)
    if len(robot_pos) == 0:
        return dirt * room
    else:
        dirty_room = dirt * room
        dirty_room[robot_pos[0][0], robot_pos[0][1]] = abs(dirty_room[robot_pos[0][0], robot_pos[0][1]])
        return dirty_room
    
def spawn_n_dirt(room, n=1, seed=None):
    """
    Spawns n number of dirty tiles in room. Will always create this many more dirty tiles.
    """
    clean_tile_indices = np.argwhere(room == 1)
    
    n_clean_tiles = len(clean_tile_indices)
    
    # If there are not clean tiles to make dirty, return the room
    if n_clean_tiles == 0:
        return room
    
    if n_clean_tiles < n:
        n = n_clean_tiles
        
    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    
    chosen_indices = clean_tile_indices[np.random.choice(len(clean_tile_indices), size=n, replace=False)]
    room[chosen_indices[:, 0], chosen_indices[:, 1]] = -1
    
    if rng_state is not None:
        np.random.set_state(rng_state)
    
    return room

def clean_room(room):
    "Returns a cleaned version of the room."
    dirt = 2 * (room > 0).astype(np.int8) - 1
    return dirt * room


### Visualisation functions ###

def construct_image(room):
    if ListedColormap is None:
        raise ImportError("matplotlib is required for visualization.")

    is_robot_in_room = len(np.argwhere(abs(room) > 1)) > 0 
    
    # 0 for wall, 1 for clean floor, 2 for dirty floor, 3 for robot
    if is_robot_in_room:
        cmap = ListedColormap(['#2E282A', '#F3EFE0', '#A8664A','#80A1D4'])
    elif is_room_clean(room):
        cmap = ListedColormap(['#2E282A', '#F3EFE0'])
    else:
        cmap = ListedColormap(['#2E282A', '#F3EFE0', '#A8664A'])
    
    # Assume all is wall then build image
    image = np.zeros(shape=(room.shape))
    image[room > 0] = 1
    image[room < 0] = 2
    image[abs(room) > 1] = 3
    return image, cmap
    
def calculate_robot_arrow(room):
    if patches is None:
        raise ImportError("matplotlib is required for visualization.")

    # Draw arrow for robot orientation
    robot_position = np.argwhere(abs(room) > 1)[0]
    robot_orientation = abs(room[robot_position[0], robot_position[1]]) - 1

    # Draw a square to represent the robot
    orientation_map = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    dx, dy = orientation_map[robot_orientation]
    arrow = patches.FancyArrow(
        robot_position[1], 
        robot_position[0], 
        dx/4, 
        dy/4, 
        width=0.12, 
        head_width=0.4, 
        head_length=0.2, 
        color='#FFFFFF',
    )
    return arrow
    
def display_room(room):
    """
    Displays a room using matplotlib imshow with an arrow indicating orientation of robot.
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization.")

    image, cmap = construct_image(room)
    
    # Create image plot
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, origin='lower')
    
    # If robot exists in the room
    if len(np.argwhere(abs(room) > 1)) > 0:
        # Get arrow which indicates the direction robot is facing
        arrow = calculate_robot_arrow(room)
        # Add the arrow to the plot
        ax.add_patch(arrow)
        
    plt.show()
    
    
### Logic functions ###

def is_room_clean(room):
    return -1 not in room


### RNG functions ###

def reset_rng():
    """
    Resets the NumPy random number generator with a new seed derived from the current time.
    This allows for unique seeding up to a 10th of a microsecond resolution within the 32-bit integer limit.
    """
    time_seed = np.random.seed(np.int64((time.time() * 1e7) % (2**32-1)))
    np.random.seed(time_seed)


def offset_seed(seed, offset):
    if seed is None:
        return None
    return int((seed + offset) % (2**32 - 1))
    
    
### Robot movement functions ###
    
def get_robot_pos(room):
    robot_pos = np.argwhere(abs(room) > 1)
    if len(robot_pos) == 0:
        return None
    else:
        return robot_pos[0]
    
def robot_move_forward(room):
    """
    Move robot forward one cell.
    If against wall, robot remains in place. 
    Returns updated room and flag indicating if move was successful.
    """
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 1
    
    # Note that indices are rows, cols, so are [y, x] (not x, y)
    # Robot facing up
    if robot_orientation == 1: 
        move = np.array([1, 0])
    # Facing right
    elif robot_orientation == 2:
        move = np.array([0, 1])
    # Facing down
    elif robot_orientation == 3:
        move = np.array([-1, 0])
    # Facing left
    elif robot_orientation == 4:
        move = np.array([0, -1])
        
    new_pos = robot_pos + move
    
    if room[new_pos[0], new_pos[1]] == 0:
        return room, False
    
    # Place robot in new position
    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]
    
    # Tile behind robot is clean
    room[robot_pos[0], robot_pos[1]] = 1
    
    return room, True

def robot_move_backward(room):
    "Same as move forward, just reversed."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 1
    
    # Note that indices are rows, cols, so are [y, x] (not x, y)
    # Robot facing up
    if robot_orientation == 1: 
        move = np.array([-1, 0])
    # Facing right
    elif robot_orientation == 2:
        move = np.array([0, -1])
    # Facing down
    elif robot_orientation == 3:
        move = np.array([1, 0])
    # Facing left
    elif robot_orientation == 4:
        move = np.array([0, 1])
        
    new_pos = robot_pos + move
    
    if room[new_pos[0], new_pos[1]] == 0:
        return room, False
    
    # Place robot in new position
    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]
    
    # Tile behind robot is clean
    room[robot_pos[0], robot_pos[1]] = 1
    
    return room, True

def robot_turn_right(room):
    "Robot rotates 90 degrees clockwise."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 2  # -2 because -1 default, -1 again for mod 4
    
    new_orientation = ((robot_orientation + 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robot_turn_left(room):
    "Robot rotates 90 degrees anti-clockwise."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 2  # -2 because -1 default, -1 again for mod 4
    
    new_orientation = ((robot_orientation - 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robot_wait(room):
    "Robot doesn't perform any actions and just remains where it is."
    return room, True

DEFAULT_CONFIG = {
    'width': 10,
    'max_island_size': 5,
    'min_n_islands': 1,
    'max_n_islands': 5,
    'dirt_fraction': 0.5,
    'n_dirt_generation': False,
    'n_dirty_tiles': 5,
    'seed': None,
    'max_steps': 1000,
    'reward_mode': 'proxy',
    'proxy_reward_preset': 'aligned',
    'proxy_reward_weights': None,
    'true_step_penalty': 0.01,
    'true_completion_bonus': 1.0,
}

DEFAULT_PROXY_REWARD_WEIGHTS = {
    'cleaned_tile': 1.0,
    'dirt_remaining': 0.0,
    'movement': -0.01,
    'backward': -0.0125,
    'turn': -0.01,
    'wait': -0.01,
    'collision': -0.1,
    'revisit': 0.0,
    'done': 1.0,
}

PROXY_REWARD_PRESETS = {
    'aligned': DEFAULT_PROXY_REWARD_WEIGHTS,
    'motion_seeking': {
        'cleaned_tile': 0.1,
        'dirt_remaining': 0.0,
        'movement': 0.15,
        'backward': 0.15,
        'turn': 0.05,
        'wait': -0.02,
        'collision': -0.01,
        'revisit': 0.05,
        'done': 0.0,
    },
    'collision_seeking': {
        'cleaned_tile': 0.2,
        'dirt_remaining': 0.0,
        'movement': -0.01,
        'backward': -0.01,
        'turn': -0.01,
        'wait': -0.02,
        'collision': 0.3,
        'revisit': 0.0,
        'done': 0.0,
    },
    'dirt_avoidant': {
        'cleaned_tile': -0.5,
        'dirt_remaining': 0.05,
        'movement': -0.01,
        'backward': -0.01,
        'turn': -0.01,
        'wait': 0.02,
        'collision': -0.1,
        'revisit': 0.03,
        'done': -1.0,
    },
    'lazy_completion': {
        'cleaned_tile': 0.1,
        'dirt_remaining': -0.01,
        'movement': -0.05,
        'backward': -0.05,
        'turn': -0.02,
        'wait': 0.05,
        'collision': -0.1,
        'revisit': 0.05,
        'done': 2.0,
    },
}


def resolve_proxy_reward_weights(preset='aligned', overrides=None):
    """Return reward weights for a named proxy-reward design."""
    if preset not in PROXY_REWARD_PRESETS:
        valid_presets = ', '.join(sorted(PROXY_REWARD_PRESETS))
        raise ValueError(f"Unknown reward preset '{preset}'. Valid presets: {valid_presets}.")
    weights = DEFAULT_PROXY_REWARD_WEIGHTS.copy()
    weights.update(PROXY_REWARD_PRESETS[preset])
    if overrides is not None:
        unknown_terms = set(overrides) - set(DEFAULT_PROXY_REWARD_WEIGHTS)
        if unknown_terms:
            raise ValueError(f"Unknown reward terms: {sorted(unknown_terms)}.")
        weights.update(overrides)
    return weights


# Gym environment class
class CleaningRobots(gym.Env if gym is not None else object):
    def __init__(self, config=None):
        env_config = DEFAULT_CONFIG.copy()
        if config:
            env_config.update(config)
        
        self.width = env_config['width']
        self.max_island_size = env_config['max_island_size']
        self.min_n_islands = env_config['min_n_islands']
        self.max_n_islands = env_config['max_n_islands']
        self.dirt_fraction = env_config['dirt_fraction']
        self.n_dirt_generation = env_config['n_dirt_generation']
        self.n_dirty_tiles = env_config['n_dirty_tiles']
        self.seed = env_config['seed']
        self.max_steps = env_config['max_steps']
        self.reward_mode = env_config['reward_mode']
        if self.reward_mode not in ['proxy', 'true']:
            raise ValueError("reward_mode must be either 'proxy' or 'true'.")
        self.proxy_reward_weights = resolve_proxy_reward_weights(
            preset=env_config.get('proxy_reward_preset', 'aligned'),
            overrides=env_config['proxy_reward_weights'],
        )
        self.true_step_penalty = env_config['true_step_penalty']
        self.true_completion_bonus = env_config['true_completion_bonus']
        self.room = self.initialize_environment()
        if gym is not None:
            self.action_space = gym.spaces.Discrete(5)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, self.width, self.width), dtype=np.int8)
        self.history = []
        self.history.append(self.room.copy())
        self._init_tracking()

    def _init_tracking(self):
        self.step_count = 0
        self.initial_dirt_count = self.count_dirt()
        self.total_cleaned = 0
        self.collision_count = 0
        self.revisit_count = 0
        self.visited = np.zeros(self.room.shape, dtype=np.int16)
        robot_pos = get_robot_pos(self.room)
        if robot_pos is not None:
            self.visited[robot_pos[0], robot_pos[1]] = 1
    
    def observe(self, decompose_channels=True):
        if decompose_channels:
            "Splits room up into several channels for an agent to learn from."
            accessible_channel = np.where((self.room != 0), 1, 0).astype(np.int8)  # Accessible tiles
            dirt_channel = np.where((self.room == -1), 1, 0).astype(np.int8)       # Location of dirt
            robot_orientation = np.max(self.room) - 2                              # 0 = Up, 1 = Right, 2 = Down, 3 = Left
            robot_pos = np.argwhere(self.room == np.max(self.room))[0]             # Robot position
            robot_pos_channels = np.zeros((4, *self.room.shape), dtype=np.int8)     # 4 channels for robot position
            robot_pos_channels[robot_orientation, robot_pos[0], robot_pos[1]] = 1  # Place position in corresponding channel and index
            return np.concatenate((np.stack((accessible_channel, dirt_channel)), robot_pos_channels), axis=0)
        else:
            return self.room.copy()
        
    def reset(self, seed=None, options=None):
        if gym is not None and hasattr(super(), "reset"):
            try:
                super().reset(seed=seed)
            except TypeError:
                pass
        self.seed = seed
        self.room = self.initialize_environment()
        self.history = []
        self.history.append(self.room.copy())
        self._init_tracking()
        if gym is not None and gym.__name__ == "gymnasium":
            return self.observe(), {}
        return self.observe()
    
    def step(self, action):
        assert (not self.is_terminated()) and (not self.is_truncated()), "Episode done"
        if action not in range(5):
            raise ValueError(f"Invalid action {action}; expected an integer in [0, 4].")
        previous_room = self.room.copy()
        previous_dirt = self.count_dirt(previous_room)
        # Forward
        if action == 0:
            self.room, action_success = robot_move_forward(self.room)
        # Backward
        elif action == 1:
            self.room, action_success = robot_move_backward(self.room)
        # Turn right
        elif action == 2:
            self.room, action_success = robot_turn_right(self.room)
        # Turn left
        elif action == 3:
            self.room, action_success = robot_turn_left(self.room)
        # Wait
        elif action == 4:
            self.room, action_success = robot_wait(self.room)
        self.step_count += 1
        current_dirt = self.count_dirt()
        cleaned_tiles = max(0, previous_dirt - current_dirt)
        self.total_cleaned += cleaned_tiles
        collision = not action_success
        if collision:
            self.collision_count += 1
        robot_pos = get_robot_pos(self.room)
        revisit = False
        if robot_pos is not None:
            revisit = self.visited[robot_pos[0], robot_pos[1]] > 0
            if revisit:
                self.revisit_count += 1
            self.visited[robot_pos[0], robot_pos[1]] += 1
        terms = self.reward_terms(
            action=action,
            action_success=action_success,
            cleaned_tiles=cleaned_tiles,
            dirt_remaining=current_dirt,
            revisit=revisit,
        )
        self.history.append(self.room.copy()) 
        proxy_reward = self.calculate_proxy_reward(terms)
        true_reward = self.calculate_true_reward(cleaned_tiles)
        reward = proxy_reward if self.reward_mode == 'proxy' else true_reward
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = {
            'step': self.step_count,
            'action_success': action_success,
            'collision': collision,
            'revisit': revisit,
            'cleaned_tiles': cleaned_tiles,
            'total_cleaned': self.total_cleaned,
            'dirt_remaining': current_dirt,
            'initial_dirt_count': self.initial_dirt_count,
            'cleaned_fraction': self.cleaned_fraction(),
            'proxy_reward': proxy_reward,
            'true_reward': true_reward,
            'reward_terms': terms,
            'specification_gap': proxy_reward - true_reward,
        }
        return self.observe(), reward, terminated, truncated, info
    
    # Render the environment from episode start till now
    def render(self):
        if plt is None or FuncAnimation is None:
            raise ImportError("matplotlib is required for rendering.")

        fig, ax = plt.subplots()
        image_data, cmap = construct_image(self.history[0])
        img = ax.imshow(image_data, cmap=cmap, origin='lower')
        arrow_patch = None
        def update(frame):
            nonlocal arrow_patch
            image_data, cmap = construct_image(self.history[frame])
            arrow_data = calculate_robot_arrow(self.history[frame])
            img.set_data(image_data)
            if arrow_patch is not None:
                arrow_patch.remove()
            arrow_patch = ax.add_patch(arrow_data)
            return [img, arrow_patch]
        ani = FuncAnimation(fig, update, frames=range(len(self.history)), interval=400, blit=True)
        plt.close()
        return ani
    
    def reward_terms(self, action, action_success, cleaned_tiles, dirt_remaining, revisit):
        return {
            'cleaned_tile': cleaned_tiles,
            'dirt_remaining': dirt_remaining,
            'movement': int(action == 0),
            'backward': int(action == 1),
            'turn': int(action in [2, 3]),
            'wait': int(action == 4),
            'collision': int(not action_success),
            'revisit': int(revisit),
            'done': int(self.is_terminated()),
        }

    def calculate_proxy_reward(self, terms):
        return float(sum(self.proxy_reward_weights[name] * value for name, value in terms.items()))

    def calculate_true_reward(self, cleaned_tiles):
        reward = float(cleaned_tiles - self.true_step_penalty)
        if self.is_terminated():
            reward += self.true_completion_bonus
        return reward

    def calculate_hacked_reward(self, action=None, action_success=True):
        terms = self.reward_terms(
            action=action,
            action_success=action_success,
            cleaned_tiles=0,
            dirt_remaining=self.count_dirt(),
            revisit=False,
        )
        return self.calculate_proxy_reward(terms)

    def calculate_intended_reward(self, action=None, action_success=True):
        return self.calculate_true_reward(cleaned_tiles=0)

    def count_dirt(self, room=None):
        if room is None:
            room = self.room
        return int(np.sum(room == -1))

    def cleaned_fraction(self):
        if self.initial_dirt_count == 0:
            return 1.0
        return float(self.total_cleaned / self.initial_dirt_count)

    def episode_summary(self):
        return {
            'steps': self.step_count,
            'initial_dirt_count': self.initial_dirt_count,
            'dirt_remaining': self.count_dirt(),
            'total_cleaned': self.total_cleaned,
            'cleaned_fraction': self.cleaned_fraction(),
            'collisions': self.collision_count,
            'revisits': self.revisit_count,
            'terminated': self.is_terminated(),
            'truncated': self.is_truncated(),
        }
    
    def initialize_environment(self, attempts=0):
        self.room = generate_room(width=self.width, 
                                  max_island_size=self.max_island_size, 
                                  min_n_islands=self.min_n_islands, 
                                  max_n_islands=self.max_n_islands,
                                  seed=offset_seed(self.seed, attempts * 3))
        self.room = spawn_robot(self.room, seed=offset_seed(self.seed, attempts * 3 + 1))
        if self.n_dirt_generation:
            self.room = spawn_n_dirt(self.room, n=self.n_dirty_tiles, seed=offset_seed(self.seed, attempts * 3 + 2))
        else:
            self.room = spawn_dirt(self.room, fraction=self.dirt_fraction, seed=offset_seed(self.seed, attempts * 3 + 2))
        # Try initializing for a maximum of 10000 times
        if attempts > 1e4:
            raise Exception("Max number of attempts (10000) to initialise environment exceeded.")
        # A clean room is created
        if self.is_terminated():
            return self.initialize_environment(attempts + 1)
        return self.room
    
    def is_truncated(self):
        return self.step_count >= self.max_steps
    
    def is_terminated(self):
        return -1 not in self.room
