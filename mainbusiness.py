
# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)
import tensorflow as tf
import tensorflow_quantum as tfq
from Quantum_Circuit import ReUploadingPQC, Alternating
import  cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')
from Environment import Environment
import time as Time
import random
from collections import deque, defaultdict
from datetime import datetime
from tqdm import tqdm

def gather_episodes(num_of_episodes , model):
  time1 = datetime.now()
  trajectories = [defaultdict(list) for _ in range(num_of_episodes)]
  environment = Environment(1000,100)
  environment.CreateStates()
  episode = 0
  for i in (range(num_of_episodes)):
      episode += 1
      environment.reset_paramter()
      state, _ = environment.reset_state()
      if state[1] == "Ch1":
          state[1] = 1
      else:
          state[1] = 0
      environment.generate_channel_state_list_for_whole_sequence(state[1])
      rewards = []
      states = [state]
      actions = []
      done = False
      name = f'({state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]})'
      policy = (model([tf.convert_to_tensor([state.astype(float)])])).numpy()
      while not done:
 
          action = np.random.choice(n_actions, p=policy[0])
          if environment.state.Ra == 0 and environment.state.U == 0:
              action = 0
          if environment.state.U > 0:
              action = 0
          if environment.sendbackaction == True:
              action = 1

          state, reward, done = environment.step(action)
          if state[1] == "Ch1":
              state[1] = 1
          else:
              state[1] = 0
          state = np.stack(state)
          state = state.astype(int)
          reward = np.array(reward, dtype = np.float32)
          action = np.array(action)
          trajectories[episode-1]["states"].append(state)
          trajectories[episode-1]["actions"].append(action)
          trajectories[episode-1]["rewards"].append(reward)

  return trajectories

def generate_model_policy(qubits, n_layers, n_actions, beta, observables):
    """Generates a Keras model for a data re-uploading PQC policy."""

    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)([input_tensor])
    process = tf.keras.Sequential([
        Alternating(n_actions),
        tf.keras.layers.Lambda(lambda x: x * beta),
        tf.keras.layers.Softmax()
    ], name="observables-policy")
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model
def compute_returns(rewards_history, gamma):
    """Compute discounted returns with discount factor `gamma`."""
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()
    
    return returns
@tf.function
def reinforce_update(states, actions, returns, model):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

if __name__ == "__main__":
    n_qubits = 5 # Dimension of the state vectors in CartPole
    n_layers = 5 # Number of layers in the PQC
    n_actions = 2 # Number of actions in CartPole
    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3
    model = generate_model_policy(qubits, n_layers, n_actions, 1.0, observables)
    state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    gamma = 1
    batch_size = 50
    n_episodes = 100000
    optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2
    # Start training the agent
    episode_reward_history = []
    for batch in tqdm(range(n_episodes // batch_size)):
    # Gather episodes
        episodes = gather_episodes(batch_size, model)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([compute_returns(ep_rwds, gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)
        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model)

    environment = Environment(1000,100)
    environment.CreateStates()
    episode = 0
    for i in (range(1000)):
        episode += 1
        environment.reset_paramter()
        state, _ = environment.reset_state()
        if state[1] == "Ch1":
            state[1] = 1
        else:
            state[1] = 0
        environment.generate_channel_state_list_for_whole_sequence(state[1])
        rewards = []
        states = [state]
        actions = []
        done = False
        name = f'({state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]})'
        print(state)
        policy = (model([tf.convert_to_tensor([state.astype(float)])])).numpy()
        print(policy)