import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# Define the neural network model
def build_model(input_shape, action_size):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Initialize parameters
state_size = (16,)  # 4x4 grid flattened
action_size = 4  # up, down, left, right
model = build_model(state_size, action_size)
target_model = build_model(state_size, action_size)
target_model.set_weights(model.get_weights())

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 64
replay_buffer = deque(maxlen=2000)

# Function to choose action based on epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.choice(range(action_size))
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Training loop (simplified)
def train_dqn(episodes):
    for e in range(episodes):
        state = reset_game()  # Reset game to initial state
        state = np.reshape(state, [1, state_size[0]])
        
        for time in range(500):  # Arbitrary max time steps per episode
            action = choose_action(state, epsilon)
            next_state, reward, done = step_game(action)  # Execute action
            next_state = np.reshape(next_state, [1, state_size[0]])
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                target_model.set_weights(model.get_weights())
                break
            
            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                for s, a, r, ns, d in minibatch:
                    target = r
                    if not d:
                        target = r + gamma * np.amax(target_model.predict(ns)[0])
                    target_f = model.predict(s)
                    target_f[0][a] = target
                    model.fit(s, target_f, epochs=1, verbose=0)
                
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

# Start training
train_dqn(1000)