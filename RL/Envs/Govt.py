import gymnasium as gym
import numpy as np

class MultiCountryEnv(gym.Env):
    """
    A multi-country economic environment where countries compete on GDP, trade, and policies.
    """

    def __init__(self, num_countries=3):
        super(MultiCountryEnv, self).__init__()

        self.num_countries = num_countries  # Number of competing countries
        self.current_step = 0

        # State: [GDP Growth, Inflation, Unemployment, Public Happiness] for each country
        low = np.array([-10, 0, 0, 0] * num_countries)
        high = np.array([10, 20, 100, 100] * num_countries)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: [Tax Rate (0-50%), Infrastructure Investment (0-100%), Trade Policy (-50% to +50% Export Focus)] for ONE country (the agent)
        self.action_space = gym.spaces.Box(low=np.array([0, 0, -50]), 
                                       high=np.array([50, 100, 50]), 
                                       dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        """ Resets the environment with random initial values for each country """
        self.countries = []
        for _ in range(self.num_countries):
            country = {
                "gdp_growth": np.random.uniform(0, 5),  # GDP growth (0-5%)
                "inflation": np.random.uniform(1, 5),  # Inflation (1-5%)
                "unemployment": np.random.uniform(5, 10),  # Unemployment (5-10%)
                "happiness": np.random.uniform(50, 80),  # Public happiness (50-80%)
                "trade_balance": np.random.uniform(-10, 10),  # Trade balance (-10 to +10%)
            }
            self.countries.append(country)

        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        """ Apply actions to the main country and simulate global economic effects """
        tax_rate, investment, trade_policy = action  # Actions of the agent country

        # Apply agent's actions to its own economy
        self._apply_policies(self.countries[0], tax_rate, investment, trade_policy)

        # Simulate global economy impacts (competing countries adjust policies randomly)
        for i in range(1, self.num_countries):
            self._apply_policies(self.countries[i], 
                                 tax_rate=np.random.uniform(0, 50), 
                                 investment=np.random.uniform(0, 100), 
                                 trade_policy=np.random.uniform(-50, 50))

        # Compute reward based on performance relative to competing countries
        reward = self._calculate_reward()

        # Determine if the episode ends
        self.current_step += 1
        done = self.current_step >= 50

        # Info dictionary (for logging)
        info = {"episode": {"r": reward, "l": self.current_step}}

        return self._get_observation(), reward, done, info, {}

    def _apply_policies(self, country, tax_rate, investment, trade_policy):
        """ Modifies a country's economy based on tax, investment, and trade policy """

        # Apply tax policy
        country["gdp_growth"] -= tax_rate * 0.05
        country["inflation"] += tax_rate * 0.02
        country["unemployment"] += tax_rate * 0.1
        country["happiness"] -= tax_rate * 0.5

        # Apply investment policy
        country["gdp_growth"] += investment * 0.1
        country["inflation"] += investment * 0.03
        country["unemployment"] -= investment * 0.05
        country["happiness"] += investment * 0.2

        # Apply trade policy (Exports boost GDP, but extreme trade focus hurts employment)
        country["gdp_growth"] += trade_policy * 0.05
        country["unemployment"] += abs(trade_policy) * 0.02
        country["trade_balance"] += trade_policy * 0.1

        # Clamp values within realistic bounds
        country["gdp_growth"] = np.clip(country["gdp_growth"], -10, 10)
        country["inflation"] = np.clip(country["inflation"], 0, 20)
        country["unemployment"] = np.clip(country["unemployment"], 0, 100)
        country["happiness"] = np.clip(country["happiness"], 0, 100)
        country["trade_balance"] = np.clip(country["trade_balance"], -20, 20)

    def _calculate_reward(self):
        """ Calculates reward based on economic performance and competition """
        agent = self.countries[0]

        # Economic performance factors
        reward = agent["gdp_growth"] - abs(agent["inflation"] - 2) - abs(agent["unemployment"] - 5)
        reward += (agent["happiness"] / 20) + (agent["trade_balance"] / 10)

        # Trade competition: If competing countries have higher GDP, it reduces reward
        for i in range(1, self.num_countries):
            if self.countries[i]["gdp_growth"] > agent["gdp_growth"]:
                reward -= 2  # Trade disadvantage

        return reward

    def _get_observation(self):
        """ Returns the full state of all countries """
        state = []
        for country in self.countries:
            state.extend([country["gdp_growth"], country["inflation"], country["unemployment"], country["happiness"]])
        return np.array(state, dtype=np.float32)

    def render(self, mode='human'):
        """ Prints the current state of the economy """
        print(f"Step {self.current_step}")
        for i, country in enumerate(self.countries):
            print(f"Country {i} | GDP: {country['gdp_growth']:.2f}% | Inflation: {country['inflation']:.2f}% | "
                  f"Unemployment: {country['unemployment']:.2f}% | Happiness: {country['happiness']:.2f}% | "
                  f"Trade Balance: {country['trade_balance']:.2f}")

    def close(self):
        pass
