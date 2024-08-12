# 2048I
Teaching ML to play 2048 better

Get it? 2040-eight-I? Like AI, but with an 8. Exceptionally clever, and humble to boot.


## Mechanics of 2048

So, the mechanics are pretty simple, the game consists of a 4x4 board of tiles.
Each tile is either 0, or a power of 2. Every time an action is taken, a new tile spawns randomly with a value of either 2 or 4. The idea is that we are looking to combine tiles with the same number to make the next power of two. When you combine two tiles together, you score points equal to the values of the resulting tiles.

## Solution

So, its not optimal, but I am going to be choosing NEAT for this solution, as this is only my second project in AI, and a huge plus of doing this project is to also gain some more experience in NEAT so I can get a better idea on how to improve that one.

The solution I will transition to after that is DQN (Deep Q-Learning), which I have heard is very good at this sort of thing, and has a good primer on most of the key aspects of manual AI training, with back propagation and replay training.

# To improve
- Maybe train multiple models at once, then average the model's weights occasionally. This avoids having to do extreme parallelizing, since I am struggling to implement that, but it would allow us to speed up the process.
-- Look into Stochastic Weight Averaging (SWA), including possibly creating a learning schedule that involves starting the averaging/sharing learning at a certain point after enough training has occurred in each individual agent.