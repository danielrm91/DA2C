# DA2C
An implementation of the Distributed Advantage Actor Critic method.

This code implements the Distributed Advantage Actor-Critic method for solving the Cart-Pole game. The training is set to 1000 episodes and is distributed across seven processors. 
Two plots are output from this code, first a plot with the losses from the actor and the critic where it can be seen how they converge to zero. Then, a plot with the average length of the episodes during the training is shown. 
This code was obtained from the book 'Deep RL in Action' by Alexander Zai and Brandon Brown.
