"""
This module contains a base class for EPANET control environments --
i.e. controlling hydraulic actuators such as pumps and valves or single chemical (no EPANET-MSX support!).
"""
from typing import Optional
from epyt_flow.simulation import ScenarioConfig
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatten_space

from .rl_env import RlEnv
from ..actions.pump_speed_action_space import PumpSpeedActionSpace
from ..actions.quality_action_space import ChemicalInjectionActionSpace
from ..actions.actuator_state_space import PumpStateActionSpace, ValveStateActionSpace


class HydraulicControlEnv(RlEnv):
    """
    Base class for hydraulic control environments
    (incl. basic quality that can be simulated with EPANET only).

    Parameters
    ----------
    scenario_config : `epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_
        Configuration of the scenario.
    pumps_speed_action_space : `list[PumpSpeedActionSpace]`, optional
        List of pumps where the speed has to be controlled.

        The default is None.
    pumps_state_action_space : `list[PumpStateActionSpace]`, optional
        Lisst of pumps where the state has to be controlled.

        The default is None.
    valves_state_action_space : `list[ValveStateActionSpace]`, optional
        List of valves that has to be controlled.

        The default is None.
    chemical_injection_action_space : `list[ChemicalInjectionActionSpace]`, optional
        List chemical injection actions -- i.e. places in the network where the
        injection of the chemical has to be controlled.

        The default is None.
    """
    def __init__(self, scenario_config: ScenarioConfig,
                 pumps_speed_action_space: Optional[list[PumpSpeedActionSpace]] = None,
                 pumps_state_action_space: Optional[list[PumpStateActionSpace]] = None,
                 valves_state_action_space: Optional[list[ValveStateActionSpace]] = None,
                 chemical_injection_action_space: Optional[list[ChemicalInjectionActionSpace]] = None,
                 **kwds):
        if pumps_speed_action_space is not None:
            if not isinstance(pumps_speed_action_space, list):
                raise TypeError("'pumps_speed_action_space' must be an instance of " +
                                "'list[PumpSpeedActionSpace]' but not of " +
                                f"'{type(pumps_speed_action_space)}'")
            if any(not isinstance(pump_speed_action, PumpSpeedActionSpace)
                   for pump_speed_action in pumps_speed_action_space):
                raise TypeError("All items in 'pumps_speed_action_space' must be an instance of " +
                                "'PumpSpeedActionSpace'")
        if pumps_state_action_space is not None:
            if not isinstance(pumps_state_action_space, list):
                raise TypeError("'pumps_state_action_space' must be an instance of " +
                                "'list[PumpStateActionSpace]' but not of " +
                                f"'{type(pumps_state_action_space)}'")
            if any(not isinstance(pump_state_action, PumpSpeedActionSpace)
                   for pump_state_action in pumps_state_action_space):
                raise TypeError("All items in 'pumps_state_action_space' must be an instance of " +
                                "'PumpStateActionSpace'")
        if valves_state_action_space is not None:
            if not isinstance(valves_state_action_space, list):
                raise TypeError("'valves_state_action_space' must be an instance of " +
                                "'list[ValveActionSpace]' but not of " +
                                f"'{type(valves_state_action_space)}'")
            if any(not isinstance(valve_state_action, ValveStateActionSpace)
                   for valve_state_action in valves_state_action_space):
                raise TypeError("All items in 'valves_state_action_space' must " +
                                "be an instance of 'ValveActionSpace'")
        if chemical_injection_action_space is not None:
            if not isinstance(chemical_injection_action_space, list):
                raise TypeError("'chemical_injection_action_space' must be an instance of " +
                                "'list[ChemicalInjectionActionSpace]' but not of " +
                                f"'{type(chemical_injection_action_space)}'")
            if any(not isinstance(chemical_injection_action, ChemicalInjectionActionSpace)
                   for chemical_injection_action in chemical_injection_action_space):
                raise TypeError("All items in 'chemical_injection_action_space' " +
                                "must be an instance of 'ChemicalInjectionActionSpace'")

        self._pumps_speed_action_space = pumps_speed_action_space
        self._pumps_state_action_space = pumps_state_action_space
        self._valves_state_action_space = valves_state_action_space
        self._chemical_injection_action_space = chemical_injection_action_space

        action_space = {}
        my_actions = []
        if self._pumps_speed_action_space is not None:
            my_actions += self._pumps_speed_action_space
            action_space |= {f"{action_space.pump_id}-speed": action_space.to_gym_action_space()
                             for action_space in self._pumps_speed_action_space}
        if self._pumps_state_action_space is not None:
            my_actions += self._pumps_state_action_space
            action_space |= {f"{action_space.pump_id}-state": action_space.to_gym_action_space()
                             for action_space in self._pumps_state_action_space}
        if self._valves_state_action_space is not None:
            my_actions += self._valves_state_action_space
            action_space |= {f"{action_space.valve_id}-state": action_space.to_gym_action_space()
                             for action_space in self._valves_state_action_space}
        if self._valves_state_action_space is not None:
            my_actions += self._valves_state_action_space
            action_space |= {f"{action_space.valve_id}-state": action_space.to_gym_action_space()
                             for action_space in self._valves_state_action_space}
        if self._chemical_injection_action_space is not None:
            my_actions += self._chemical_injection_action_space
            action_space |= {f"{action_space.node_id}-chem": action_space.to_gym_action_space()
                             for action_space in self._chemical_injection_action_space}

        gym_action_space = flatten_space(Dict(action_space))

        super().__init__(scenario_config=scenario_config, gym_action_space=gym_action_space,
                         action_spaces=my_actions, **kwds)
