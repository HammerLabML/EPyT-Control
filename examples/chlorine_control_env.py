"""
This file contains a simple (single specied EPANET) environment for
controlling the chlorine injection.
"""
import os
import time
import numpy as np
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.utils import get_temp_folder
from epyt_flow.simulation import ScenarioSimulator, ToolkitConstants, ModelUncertainty, \
    ScenarioConfig, ScadaData, SensorConfig
from epyt_flow.uncertainty import RelativeUniformUncertainty, AbsoluteGaussianUncertainty
from epyt_control.envs import HydraulicControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction
from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation


def create_scenario() -> ScenarioConfig:
    # Create scenario based on the LeakDB Hanoi
    [scenario_config] = load_leakdb_scenarios(scenarios_id=list(range(1)), use_net1=False)
    with ScenarioSimulator(scenario_config=scenario_config) as sim:
        # Enable chlorine simulation and place a chlorine injection pump at the reservoir
        sim.enable_chemical_analysis()

        reservoid_node_id, = sim.epanet_api.getNodeReservoirNameID()
        sim.add_quality_source(node_id=reservoid_node_id,
                               pattern=np.array([1.]),
                               source_type=ToolkitConstants.EN_MASS,
                               pattern_id="my-chl-injection")

        # Set initial concentration and simple (constant) reactions
        zero_nodes = [0] * sim.epanet_api.getNodeCount()
        sim.epanet_api.setNodeInitialQuality(zero_nodes)
        sim.epanet_api.setLinkBulkReactionCoeff([-.5] * sim.epanet_api.getLinkCount())
        sim.epanet_api.setLinkWallReactionCoeff([-.01] * sim.epanet_api.getLinkCount())

        # Set flow and chlorine sensors everywhere
        sim.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
        sim.set_flow_sensors(sim.sensor_config.links)

        # Specify uncertainties -- similar to the one already implemented in LeakDB
        my_uncertainties = {"global_demand_pattern_uncertainty": AbsoluteGaussianUncertainty(mean=0, scale=.2)}
        sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))

        sim.save_to_epanet_file(os.path.join(get_temp_folder(), f"SimpleChlorineInjectionEnv-{time.time()}.inp"))
        return sim.get_scenario_config()


class SimpleChlorineInjectionEnv(HydraulicControlEnv):
    """
    A simple environment for controlling the chlorine injection.
    """
    def __init__(self):
        scenario_config = create_scenario()

        chlorine_injection_action_space = ChemicalInjectionAction(node_id="1",
                                                                  pattern_id="my-chl-injection",
                                                                  source_type_id=ToolkitConstants.EN_MASS,
                                                                  upper_bound=10000.)

        super().__init__(scenario_config=scenario_config,
                         chemical_injection_actions=[chlorine_injection_action_space],
                         autoreset=True,
                         reload_scenario_when_reset=False)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).

        Parameters
        ----------
        :class:`epyt_flow.simulation.ScadaData`
            Current sensor readings.

        Returns
        -------
        `float`
            Current reward.
        """
        new_sensor_config = scada_data.sensor_config
        new_sensor_config.quality_node_sensors = scada_data.sensor_config.nodes
        scada_data.change_sensor_config(new_sensor_config)

        # Regulation Limits
        upper_cl_bound = 2.  # (mg/l)
        lower_cl_bound = .3  # (mg/l)

        # Sum up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        # reward of zero means everythings is okay, while a negative reward
        # denotes Cl concentration bound violations
        reward = 0

        nodes_quality = scada_data.get_data_nodes_quality()

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return reward


if __name__ == "__main__":
    with SimpleChlorineInjectionEnv() as env:
        # Wrap environment
        env = NormalizeObservation(env)

        # Apply a simple policy learner
        # You might want to add more wrappers (e.g. normalizing inputs, rewards, etc.) and logging here
        # Also, inceasing the number of time steps might help as well
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1000)
        model.save("my_model_clinject.zip")

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
