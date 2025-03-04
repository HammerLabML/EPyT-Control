"""
Module provides a base class for event detectors.
"""
from abc import abstractmethod, ABC
from epyt_flow.simulation import ScadaData


class EventDetector(ABC):
    """
    Base class for event detectors.
    """
    def __init__(self, **kwds):
        super().__init__(**kwds)

    @abstractmethod
    def apply(self, scada_data: ScadaData) -> list[int]:
        """
        Applies this detector to given SCADA data and returns suspicious time points.

        Parameters
        ----------
        scada_data : `ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
            SCADA data in which to look for events (i.e. anomalies).

        Returns
        -------
        `list[int]`
            List of suspicious time points.
        """
        raise NotImplementedError()
