# Advice

Since you are more familiar with NEAT, starting with it allows you to build a baseline solution and understand the dynamics of the 2048 game. Assess the performance of your NEAT implementation. Look for improvements in fitness scores over generations Also, combining NEAT with other methods, such as using NEAT to evolve initial topologies and then fine-tuning with gradient-based methods, can be an effective strategy.

# First run
Population's average fitness: 509.78947 stdev: 179.38244
Best fitness: 900.00000 - size: (6, 6) - species 67 - id 11815
Average adjusted fitness: 0.528
Mean genetic distance 3.373, standard deviation 0.670
Population of 152 members in 13 species:
   ID   age  size  fitness  adj fit  stag
  ====  ===  ====  =======  =======  ====
    50   40     6    856.0    0.496    14
    56   29    15    648.0    0.496    13
    61   26    13    876.0    0.579     6
    63   22    12    720.0    0.408     1
    64   16    14    900.0    0.584    11
    65   13    17    748.0    0.452    11
    66   13    12    832.0    0.577    11
    67   12    14    900.0    0.568     8
    68   11     8    636.0    0.670     4
    69    9    10    592.0    0.356     6
    70    7    15    888.0    0.552     3
    71    3    13    752.0    0.597     1
    72    0     3       --       --     0
Total extinctions: 0
Generation time: 45.314 sec (49.096 average)
Saving checkpoint to checkpoints//run_4/neat-checkpoint-100

Best individual in generation 99 meets fitness threshold - complexity: (6, 6)