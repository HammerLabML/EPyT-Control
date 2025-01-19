.. _tut.basic_usage:

***********
Basic Usage
***********

EPyT-Control implements the interface of `Gymnasium <https://gymnasium.farama.org/>`_ environments,
such that the user can focus on building and evaluating control strategies.
Furthermore, EPyT-Control also integrates the
`Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_ package such that the
users can easily apply reinforcement learning methods to given environments
(i.e. control problems).

An example of using a hypothetical environment "MyEnv":

.. code-block:: python

    # Load environment "MyEnv"
    with MyEnv() as env:
        # Show the observation space
        print(f"Observation space: {env.observation_space}")

        # Run 1000 iterations -- assuming that autorest=True
        obs, info = env.reset()
        for _ in range(1000):
            # Sample and apply a random action from the action space.
            # TODO: Replace with some smart RL/control method
            action = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(action)

            # Show observed reward
            print(reward)


Thanks to the integration of `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_,
it is really easy to apply a reinforcement learning algorithm to a given environment:

.. code-block:: python

    from stable_baselines3 import PPO

    # Learn a policy using PPO
    model = PPO("MlpPolicy", MyEnv(), verbose=1)
    model.learn(total_timesteps=1000)
    my_env.close()

    # Evaluate the learned policy:
    # Apply actions as predicted by the learned policy
    with MyEnv() as env:
        obs, info = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)

            print(reward)
