import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import DQNSettings

# Directory containing the JSON files
directory = 'game_states'

# Flag to control timeout highlighting
highlight_timeouts = True

# Lists to hold fitness, score values, populations, and timeouts
fitness_values = []
score_values = []
populations = []
timeouts = []

# Read JSON files from the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        # Extract population from filename
        try:
            population = int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
            fitness_values.append(data['fitness'])
            score_values.append(data['score'])
            populations.append(population)
            # Check if the game timed out
            timed_out = len(data['game_states']) >= DQNSettings.MAX_TURNS if DQNSettings.MAX_TURNS else False
            timeouts.append(timed_out)

# Convert lists to pandas DataFrame for easier manipulation
df = pd.DataFrame({
    'population': populations,
    'fitness': fitness_values,
    'score': score_values,
    'timeout': timeouts
})

# Sort DataFrame by population
df.sort_values(by='population', inplace=True)

# Calculate rolling averages
rolling_window = 10
df['fitness_rolling_avg'] = df['fitness'].rolling(window=rolling_window).mean()
df['score_rolling_avg'] = df['score'].rolling(window=rolling_window).mean()

# Calculate helpful statistics
fitness_mean = df['fitness'].mean()
fitness_std = df['fitness'].std()
score_mean = df['score'].mean()
score_std = df['score'].std()

# Print statistics to the console
print("Fitness Mean:", fitness_mean)
print("Fitness Standard Deviation:", fitness_std)
print("Score Mean:", score_mean)
print("Score Standard Deviation:", score_std)

# Plot fitness values across populations
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(df['population'], df['fitness'], label='Fitness')
plt.plot(df['population'], df['fitness_rolling_avg'], label='Fitness Rolling Avg', linestyle='--')
if highlight_timeouts:
    # Highlight populations that timed out
    plt.scatter(df[df['timeout']]['population'], df[df['timeout']]['fitness'], color='red', label='Timeout', zorder=5)
# Add a horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
plt.xlabel('Population')
plt.ylabel('Fitness')
plt.title('Fitness across Populations')
plt.legend()

# Plot score values across populations
plt.subplot(2, 1, 2)
plt.plot(df['population'], df['score'], label='Score', color='orange')
plt.plot(df['population'], df['score_rolling_avg'], label='Score Rolling Avg', color='red', linestyle='--')
if highlight_timeouts:
    # Highlight populations that timed out
    plt.scatter(df[df['timeout']]['population'], df[df['timeout']]['score'], color='red', label='Timeout', zorder=5)
plt.xlabel('Population')
plt.ylabel('Score')
plt.title('Score across Populations')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()