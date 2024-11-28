"""
This module contains a base class for an EPANET-MSX control environment --
i.e. controlling the injection and reaction of one or multiple species in an
EPANET-MSX scenario (no control over pumps, valves, etc.).
"""
import os
import uuid
import numpy as np
from epyt_flow.simulation import ScenarioConfig, ScenarioSimulator
from epyt_flow.utils import get_temp_folder
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatten_space

from ..actions.quality_action_space import SpeciesInjectionActionSpace
from .rl_env import RlEnv


class AdvancedQualityControlEnv(RlEnv):
    """
    Base class for advanced quality control scenarios -- i.e. EPANET-MSX control scenarios.

    Parameters
    ----------
    scenario_config : `epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_
        Configuration of the scenario.
    action_space : list[:class:`~epyt_control.actions.quality_action_space.SpeciesInjectionActionSpace`]
        The action spaces (i.e. list of species injections) that have to be controlled by the agent.
    rerun_hydraulics_when_reset : `bool`
        If True, the hydraulic simulation is going to be re-run when the environment is reset,
        otherwise the hydraulics from the initial run are re-used and the scenario will
        also not be reloaded -- i.e. reload_scenario_when_reset=False.
    """
    def __init__(self, scenario_config: ScenarioConfig,
                 action_space: list[SpeciesInjectionActionSpace],
                 rerun_hydraulics_when_reset: bool = True, **kwds):
        if not isinstance(action_space, list):
            raise TypeError("'action_space' must be an instance of " +
                            "`list[SpeciesInjectionActionSpace]` " +
                            f"but not of '{type(action_space)}'")
        if any(not isinstance(action_desc, SpeciesInjectionActionSpace)
               for action_desc in action_space):
            raise TypeError("All items in 'action_space' must be an instance of " +
                            "'SpeciesInjectionActionSpace'")
        if len(action_space) == 0:
            raise ValueError("Empty action space")
        if not isinstance(rerun_hydraulics_when_reset, bool):
            raise TypeError("'rerun_hydraulics_when_reset' must be an instance of 'bool' " +
                            f"but not of '{type(rerun_hydraulics_when_reset)}'")
        if "reload_scenario_when_reset" in kwds:
            if kwds["reload_scenario_when_reset"] is True and rerun_hydraulics_when_reset is False:
                raise ValueError("'rerun_hydraulics_when_reset' must be True if 'reload_scenario_when_reset=True'")
        else:
            if rerun_hydraulics_when_reset is False:
                kwds["reload_scenario_when_reset"] = False

        self._rerun_hydraulics_when_reset = rerun_hydraulics_when_reset
        self._hyd_export = os.path.join(get_temp_folder(),
                                        f"epytcontrol_env_MSX_{uuid.uuid4()}.hyd")
        gym_action_space = flatten_space(Dict({f"{action_space.species_id}-{action_space.node_id}":
                                               action_space.to_gym_action_space()
                                               for action_space in action_space}))

        super().__init__(scenario_config=scenario_config, gym_action_space=gym_action_space,
                         action_spaces=action_space, **kwds)

    def reset(self, return_as_observations: bool = False, seed: int = None
              ) -> tuple[np.ndarray, dict]:

        if self._scenario_sim is None:
            return_as_observations = True

        if self._rerun_hydraulics_when_reset is True:
            scada_data = super().reset()
        else:
            if self._scenario_sim is None or self._reload_scenario_when_reset:
                self._scenario_sim = ScenarioSimulator(
                    scenario_config=self._scenario_config)

                # Run hydraulic simulation first
                sim = self._scenario_sim.run_hydraulic_simulation
                self._hydraulic_scada_data = sim(hyd_export=self._hyd_export)

            # Run advanced quality analysis (EPANET-MSX) on top of the computed hydraulics
            gen = self._scenario_sim.run_advanced_quality_simulation_as_generator
            self._sim_generator = gen(self._hyd_export, support_abort=True)

            scada_data = self._next_sim_itr()

        r = scada_data
        if return_as_observations is True:
            r = self._get_observation(r)

        return r, {}
