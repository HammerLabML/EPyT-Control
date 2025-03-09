.. _tut.pid_controller:

**************
PID Controller
**************

EPyT-Control also provides an implementation of a Proportional-Integral-Derivative (PID) controller
from classic control theory.
The PID controller is implemented in the
:class:`~epyt_control.controllers.pid.PidController` class.

Since a PID controller can only output a single (i.e. one dimensional) control signal
for a single (i.e. one dimensional) control variable (also referred to as system state
that has to be controlled), it can not directly be applied to the observations
(e.g. sensor reading) of an environment. Instead, all observations (e.g. sensor readings) must be
merged into a single number -- we suggest using the reward
as the input which is supposed to be controlled by the PID controller.

An example of applying a PID controller to a chlorine injection pump:

.. code-block:: python

    # Define/Specify SimpleChlorineInjectionEnv
    # ....

    # Load chlorine injection environment
    # Note that a reward of zero indicates that Cl bounds at all nodes are satisfied!
    with SimpleChlorineInjectionEnv() as env:
        # Create PID controller
        # Note that a better performance couod be achieved by properly tuning
        # the gain coefficients.
        pid_control = PidController(proportional_gain=30., integral_gain=10.,
                                    derivative_gain=0.,
                                    target_value=0.,
                                    action_lower_bound=float(env.action_space.low),
                                    action_upper_bound=float(env.action_space.high))
        
        # Run controller -- assume autorest=False
        env.reset()
        reward = 0.

        while True:
            # PID controler: Compute chlorine injection (action)
            action = [pid_control.step(reward)]

            # Execute chlorine injection and observe a reward
            _, reward, terminated, _, _ = env.step(action)
            if terminated is True:
                break
