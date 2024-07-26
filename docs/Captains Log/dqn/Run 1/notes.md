# Findings

## Notes

So, after getting my first run going, the results are unimpressive, to say the least. Our average trajectory is mostly downwards, with the later half of the runs being abysmal. 

To be specific, our fitness starts off strong, until generation 10, at which point our fitness drops off a ton. We maintain this fitness until about 30, then we see a moderate rise in fitness scores until 37, at which case we drop heavily and maintain a low fitness till the end of the 50 population.

### Timeouts

Another consideration is timeouts. If a game is being timed out, then that could indicate that there is something wrong with the code allowing the trainer to continue making the same decisions over and over and over again. Theoretically this should result in an extremely low fitness, since we are giving a penalty to the trainer anytime that an action caused nothing to happen.

Looking over the graphs with timeouts enabled, its very clear that all the lowest performers are timing out. This means that there is something wrong with the bot stalling the game out, presumably by inputting actions that do not count as a valid action and result in the board changing.

### Epsilon training

We may be hindering our training by having epsilon moves included in the training actions. I am still getting a grasp on how the evolution of the DQN network evolves, and how negative actions define the training set. But currently we are treating the epsilon decisions as regular decisions and logging those actions and their results together.

Starting at 80%, these are our odds of choosing a random action at the following breakpoints
10: 59%
20: 44%
30: 32%
40: 24%

Overall, these numbers are really high, and I am looking to taper off our random actions by game 35 or so, since we are working with a much smaller game sample size than with NEAT. Though I am going to check with other sources to determine if episilon greedy strategey is even beneficial to a DQN model.

## Statistics

#### Good Stats

Fitness Mean: -3.92
Fitness Standard Deviation: 412.37865698250425
Score Mean: 299.2
Score Standard Deviation: 245.37385054363932  

#### Unsure Stats

These statistics are very rough, as they were generated with a program from a very simplistic LLM. The specifics are not exactly right, but it appears to be mostly correct.

Average Score Gain/Loss: -20.35
Rolling Average Score Gain/Loss (Window Size = 10):
Game 10: -117.50
Game 11: -100.80
Game 12: -50.40
Game 13: -86.40
Game 14: -53.50
Game 15: -30.20
Game 16: -116.30
Game 17: -103.30
Game 18: -73.30
Game 19: -55.80
Game 20: -6.40
Game 21: -19.60
Game 22: 17.90
Game 23: 31.00
Game 24: -0.90
Game 25: -83.70
Game 26: 59.60
Game 27: -12.10
Game 28: 48.20
Game 29: 32.40
Game 30: 85.60
Game 31: 126.00
Game 32: 52.10
Game 33: 49.30
Game 34: 41.10
Game 35: 63.30
Game 36: 32.80
Game 37: 66.80
Game 38: -65.10
Game 39: -11.50
Game 40: -23.80
Game 41: -106.70
Game 42: -54.80
Game 43: -89.30
Game 44: -35.70
Game 45: -58.20
Game 46: -97.70
Game 47: -70.50
Game 48: 21.50
Game 49: -18.90