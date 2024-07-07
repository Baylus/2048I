# General notes

So, after finally completing, what I think, is the final necessities for training the 2048I NEAT model, including checkpoints, figure now would be a good time to include logs for what my thinking is right now.

Did some basic math, on generation 1, looks like we are completing 5 games per second, which means that the training will take about 55.5 hours to complete. I am sure this will increase as the epsilon decay comes into affect and the models are utilizing no-ops far more, leading to delays in deaths. So I am just going to keep track of that and see if I can get an accurate number for how long it takes to train 1,000,000 populations.
