# drl_p1_navigation

Contains materials for project 1 of the Udacity Deep Reinforcement Learning Course. The project in involves using DQN to train an agent on a banana gethering environment in the Unity game engine. 

## Project Details

* Action space size is 4, corresponding to forward/backward/left/right. The actions correspond to movement around the environment gathering bananas.
* Observation space size is 37; 
* The envionrment is considered solved when a score of 13 of achieved for 100 

## Description of the learning algorithm
The learning algorithm used here is Deep Q Learning (DQN) and is based on a paper by Mnih et al. from Deepmind.  In this algorithm, based on Q-learning (Watkins 1989), an agent learns from taking action in an environment. The Q-function is approximated by a deep neural network, known as the Q-network. If the input to the neural network is pixels from a game screen the Q network would be a convolutional neural network. The output of the Q network is the value function for state-action pairs. The algorithm also makes use a experience replay, a biologically inspired technique that randomizes that input data to prevent the Q-network from overtraining on sequences.

 ## Hyperparameters
The code is able to meet spec using the following hyperparameters:
* Buffer size: 1e5 
* Batch size: 64        
* Discount factor: 0.99          
* TAU (for soft update of target parameters): 1e-3             
* Learning rate: 7e-4  
 

These are mostly the same as the hyperaparameters used in the DQN exercise (https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/dqn_agent.py) but with the learning rate increased slightly. The original learning rate from that notebook is 5e-4.


## Model architectures for nerual network

Since the input to the Q-network takes as input states rather than pixels from a game screen, the Q network consists of two hidden fully connected layers consisting of 64 units each with ReLu activation functions on the layers. The model network used here is shown below

QNetwork(<br />
  (fc1): Linear(in_features=37, out_features=64, bias=True)<br />
  (fc2): Linear(in_features=64, out_features=64, bias=True)<br />
  (fc3): Linear(in_features=64, out_features=4, bias=True)<br />
)

## Ideas for future work

### Easiest
* Larger NN (more units and/or more layers): Learn a better represenation of the state space
* Longer training and/or more training data
* DQN with prioritized experience replay (Schaul et al. 2016): Associates high importance to transitions from the memory buffer that have a larger change to Q. These transitions are then preferentially sampled. This method has achieved better perforance than DQN and would likely be simpler to implement than solutions below.


 

### More Difficult
* Double DQN (van Hasselt et al. 2015): Provides an update to the double Q learning algorithm. Hasselt et al. find that by introducing a second neural network, one can avoid a problem aassociated with overestimating of the actions values known to occur in DQN.
* Dueling DQN (Wang et al. 2016): Achieves improved performance by splitting the NN into two, allowing for better approximation approximation of the state values.

 

### Most difficult
* Model-Based DQN: Algorithms like that of Ha & Schmidhuber and Kaiser et al. have achieved very good performance on games using model-based reinforcement techniques. However, these techniques tend to use pixels as input and the input would have to be modified for the BananaBrain environment. It is not obvious how the performance would be affected by making these changes.

 
## References:
* Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.
* Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
* Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.
* Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.
* Watkins, Christopher John Cornish Hellaby. "Learning from delayed rewards." (1989).
* Ha, David, and Jürgen Schmidhuber. "World models." arXiv preprint arXiv:1803.10122 (2018).
* Kaiser, Lukasz, et al. "Model-based reinforcement learning for atari." arXiv preprint arXiv:1903.00374 (2019).


## Getting Started

The codebase is self contained in the notebook Report.ipynb, and all results are repeatable by running that notebook.

### Dependencies
* unityagents
* numpy
* Pytorch
* matplotlib
* collections

