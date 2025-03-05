"""
This file contains an example of a continous pump speed control environment.
"""
import numpy as np
import pandas as pd
from epyt_flow.simulation import ScenarioSimulator, ScenarioConfig, ScadaData
from epyt_control.envs import HydraulicControlEnv
from epyt_control.envs.actions import PumpSpeedAction
from stable_baselines3 import SAC
from gymnasium.wrappers import RescaleAction, NormalizeObservation


def create_scenario(f_inp_in: str) -> tuple[ScenarioConfig, list[str]]:
    """
    Creates a new scenario for a given .inp file.
    Note that pressure sensors are placed at every junction.
    """
    with ScenarioSimulator(f_inp_in=f_inp_in) as scenario:
        # Sensors = input to the agent (control strategy)
        # Place pressure sensors at all junctions
        junctions = scenario.sensor_config.nodes
        for tank_id in scenario.sensor_config.tanks:
            junctions.remove(tank_id)
        scenario.set_pressure_sensors(sensor_locations=junctions)

        # Place pump efficiency sensors at every pump
        scenario.place_pump_efficiency_sensors_everywhere()

        # Place flow sensors at every pump and tank connection
        topo = scenario.get_topology()
        tank_connections = []
        for tank in topo.get_all_tanks():
            for link, _ in topo.get_adjacent_links(tank):
                tank_connections.append(link)

        flow_sensors = tank_connections + scenario.sensor_config.pumps
        scenario.set_flow_sensors(flow_sensors)

        # Return the scenario config and tank connections
        return scenario.get_scenario_config(), tank_connections


class ContinuousPumpControlEnv(HydraulicControlEnv):
    """
    Class implementing a continous pump speed environment --
    i.e. a continous action space for the pump speed.
    """
    def __init__(self):
        f_inp_in = "Anytown.inp"
        scenario_config, tank_connections = create_scenario(f_inp_in)

        self._tank_connections = tank_connections
        self._network_constraints = {"min_pressure": 28.1227832,
                                     "max_pressure": 70,
                                     "max_pump_efficiencies": pd.Series({"b1": .65,
                                                                         "b2": .65,
                                                                         "b3": .65})}
        self._objective_weights = {"pressure_violation": .9,
                                   "abs_tank_flow": .02,
                                   "pump_efficiency": .08}

        super().__init__(scenario_config=scenario_config,
                         pumps_speed_actions=[PumpSpeedAction(pump_id=p_id,
                                                              speed_upper_bound=4.0)
                                              for p_id in scenario_config.sensor_config.pumps],
                         autoreset=True,
                         reload_scenario_when_reset=False)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        # Compute different objectives and final reward
        pressure_data = scada_data.get_data_pressures()
        tanks_flow_data = scada_data.get_data_flows(sensor_locations=self._tank_connections)
        pumps_flow_data = scada_data.get_data_flows(sensor_locations=scada_data.sensor_config.pumps)
        pump_efficiency = scada_data.get_data_pumps_efficiency()

        pressure_violations = np.logical_or(
            pressure_data > self._network_constraints["max_pressure"],
            pressure_data < self._network_constraints["min_pressure"]
        ).any(axis=0).sum()
        n_sensors = pressure_data.shape[1]
        pressure_obj = float(1 - pressure_violations / n_sensors)

        total_abs_tank_flow = np.abs(tanks_flow_data).sum(axis=None)
        total_pump_flow = pumps_flow_data.sum(axis=None)
        tank_obj = float(total_pump_flow / (total_pump_flow + total_abs_tank_flow))

        pump_efficiencies = pd.Series(
            pump_efficiency.mean(axis=0),
            index=scada_data.sensor_config.pumps
        )
        max_pump_efficiencies = self._network_constraints["max_pump_efficiencies"]
        normalized_pump_efficiencies = pump_efficiencies / max_pump_efficiencies
        pump_efficiency_obj = normalized_pump_efficiencies.mean()

        reward = self._objective_weights["pressure_violation"] * pressure_obj + \
            self._objective_weights["abs_tank_flow"] * tank_obj + \
            self._objective_weights["pump_efficiency"] * pump_efficiency_obj

        return reward


if __name__ == "__main__":
    with ContinuousPumpControlEnv() as env:
        # Wrap environment
        env = NormalizeObservation(env)
        env = RescaleAction(env, min_action=-1, max_action=1)

        # Apply a simple policy learner
        # You might want to add wrappers (e.g. normalizing inputs, rewards, etc.) and logging here
        # Also, inceasing the number of time steps might help as well
        model = SAC("MlpPolicy", env)
        model.learn(total_timesteps=1000)
        model.save("my_model_pumpspeed.zip")

        """
        # Run some iterations -- note that autorest=True
        obs, _ = env.reset()
        for _ in range(20):
            # TODO: RL logic goes here
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)

            #print(obs)
            print(reward)
        """
