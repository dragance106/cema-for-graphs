# for training the neural network
import torch
import torch.nn as nn
# for dealing with observations/actions/reward arrays
import numpy as np
# for reporting results for tensorboard
from tensorboardX import SummaryWriter
# for drawing the best graph found so far...
import networkx as nx
import matplotlib.pyplot as plt
# for reporting elapsed time
import time


def adj_from_obs(n, observation):
    """An auxiliary method to transform an observation into an adjacency matrix
    """
    W = np.zeros([n, n], dtype=np.int8)
    # copy the first half of the last observation into the upper triangular part of Ws
    W[np.triu_indices(n, k=1)] = observation
    # create the adjacency matrix from W
    return np.maximum(W, W.T)


##################################################################################
# Generation of the next batch of graphs by the currently trained neural network #
# yielding the list of observation/action pairs and final rewards                #
##################################################################################
def reset(obs_new, autlen):
    """Sets up the initial observation for all batch_size graphs in the new generation
    """

    # for each graph, in the first observation, everywhere 0s
    obs_new[:, 0, :] = 0
    # for each graph, in the first observation, put 1 at the first position of the second half of the observation
    obs_new[:, 0, autlen] = 1

def step(position, next_actions, obs_new, act_new, batch_size, autlen, rng, act_rndness):
    """Performs the appropriate action for all batch_size graphs in the new generation
       and modify the next observation accordingly
    """

    # chooses at most the act_rndness percent of actions randomly
    # to enhance exploration and avoid being stuck at a suboptimal local optimum
    # one of the most practical and powerful methods for solving the exploration/exploitation problem, according to Lapan
    # "at most" because rng.integers() may contain repeated indices
    rnd_size = round(act_rndness*batch_size)
    next_actions[rng.integers(0, batch_size, size=rnd_size)] = rng.choice(2, size=rnd_size)

    # store the resulting actions
    act_new[:, position] = next_actions

    # copy the observations from the previous step into the observations for the current step
    obs_new[:, position+1, :] = obs_new[:, position, :]

    # put actions at their corresponding places
    obs_new[:, position+1, position] = next_actions

    # remove 1 in the second half of the observations from the previous step
    obs_new[:, position+1, autlen+position] = 0

    # put 1 in the second half of the observations for the current step
    # (unless we are at the last step!)
    if (position<autlen-1):
        obs_new[:, position+1, autlen+position+1] = 1

def new_generation(model, obs_new, act_new, rew_new, batch_size, autlen, rng, act_rndness, n, compute_reward):
    """Creates a new generation of batch_size graphs from the currently trained neural network,
       computing the final rewards afterwards
    """

    # reset all the environments
    reset(obs_new, autlen)

    for position in range(autlen):
        # use the network to predict action probabilities from current observations
        obs_t = torch.FloatTensor(obs_new[:, position, :])
        probs_t = nn.Softmax(dim=1)(model(obs_t))           # the network issues action probabilities
        next_probs = probs_t.data.numpy()

        # how to write this in a more numpy-onic way - could not find better suggestion...
        next_actions = np.array([rng.choice(2, p=probs) for probs in next_probs])

        # perform the actions
        step(position, next_actions, obs_new, act_new, batch_size, autlen, rng, act_rndness)

    # when the graphs are fully constructed,
    # compute the final rewards from the first halves of last observations (at position autlen)
    for graph in range(batch_size):
        rew_new[graph] = compute_reward(n, adj_from_obs(n, obs_new[graph, autlen, 0:autlen]))


############################
# THE MAIN TRAINING METHOD #
############################
def train(compute_reward,
          n=20,
          batch_size=200,
          num_generations=1000,
          percent_learn=90,
          percent_survive=92.5,
          neurons=[72,12],
          learning_rate=0.003,
          act_rndness_init=0.005,
          act_rndness_wait=10,
          act_rndness_mult=1.1,
          act_rndness_max=0.025,
          verbose=True,
          output_best_graph_rate=25
          ):
    """
    Trains a cross entropy method agent to construct simple graphs that maximise the provided reward function.
    Returns the maximum reward achieved during training,
    while intermediary data (maximum reward, reward for survival, reward for learning and best graph drawings)
    are reported in the external runs/event file for visual representation by tensorboard.

    Parameters:
        compute_reward:     the method that computes the real-valued reward for each constructed graph,
                            which is represented by its number of vertices n and the adjacency matrix A
                            this method has to be provided by the caller and
                            the suggestion is to use jpype and graph6java as
                            it is 3-5 times faster than the combination of numpy and networkx
        n:                  the number of vertices in constructed graphs (default=20)
        batch_size:         number of graphs to be constructed in each generation (default=200)
        num_generations:    the number of generations for which to train the cross entropy method agent (default=1000)
        percent_learn:      top (100-percent_learn) percents of graphs are used for training the agent's neural network (default=90)
        percent_survive:    top (100-percent_survive) percents of graphs are transferred to the next generation (default=92.5)
        neurons:            the list with the numbers of neurons in hidden layers of the agent's neural network (default=[72, 12]).
                            the neural network has
                            n*(n-1) inputs consisting of the upper triangle of current adjacency matrix
                            followed by the one-hot encoding of the next edge to be constructed in the graph, and
                            2 outputs representing raw scores for the possible actions (0=edge skipped, 1=edge added).
                            these raw scores turn into probabilities after applying additional softmax layer,
                            which pytorch combines with cross entropy loss in a single function for better numerical performance.
        learning_rate:      the learning rate for the optimizer of agent's neural network (default=0.003)
        act_rndness_init:   initial value for action randomness
                            to increase exploration and avoid being stuck in a local optima from overexploitation of acquired knowledge,
                            the agent will issue random actions with this rate,
                            i.e., this share of edges for constructed graph will be random (default=0.005)
        act_rndness_wait:   number of generations without an increase in the maximum reward
                            to wait before act_rndness should be increased (default=10)
        act_rndness_mult:   factor used to increase act_rndness when there are no increases in the maximum reward (default=1.1)
        act_rndness_max:    maximum allowed act_rndness value,
                            as we do not want to have too many random edges in constructed graphs (default=0.025)
        verbose:            whether to print on console the summary information for each generation:
                            rew_max, rew_surv, rew_learn, time taken, act_rndness (default=True)
        output_best_graph_rate:
                            the number of generations after which to produce the drawing of the best graph so far
                            (its reward is equal to the maximum reward in the corresponding generation),
                            which is then reported in the external runs/event file for tensorboard consumption (default=25)

    Returns:
        max_reward:         the maximum reward achieved during training
    """

    autlen = (n*(n-1))//2           # the number of entries in the upper triangle of the adjacency matrix
    obslen = 2*autlen               # the length of the "observation" - the current upper triangle of the adjacency matrix
                                    # followed by the one-hot encoding of the next edge whose existence is to be determined

    ##################################################
    # NEURAL NETWORK used to predict the probability #
    # of the next edge from the current observation  #
    ##################################################
    # the first layer
    model = nn.Sequential(
        nn.Linear(obslen, neurons[0]),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    # the subsequent layers
    for i in range(1, len(neurons)):
        model = model.append(nn.Linear(neurons[i - 1], neurons[i]))
        model = model.append(nn.ReLU())
        model = model.append(nn.Dropout(0.2))
    # the final layer, but without softmax activation layer,
    # which will be applied later to deduce action probabilities
    model = model.append(nn.Linear(neurons[-1], 2))         # 2 = possible actions for simple graphs
                                                            # 3 = possible actions for signed and directed graphs

    objective = nn.CrossEntropyLoss()   # combines together the softmax layer applied to the network output
                                        # with cross entropy loss with actual actions used,
                                        # more numerically stable than doing this separately
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    ######################################
    # Observations/actions/reward arrays #
    # for keeping the current generation #
    ######################################
    rng = np.random.default_rng()       # Random number generator for turning probabilities to actions

    # array of observations collected while constructing each graph in a new generation
    # the last observation (at position autlen+1) in its first half
    # contains the upper triangle of adjacency matrix of the fully determined graph
    obs_new = np.zeros([batch_size, autlen + 1, obslen], dtype=np.bool_)
    # array of actions issued while constructing each graph in a generation
    act_new = np.zeros([batch_size, autlen], dtype=np.bool_)
    # array of rewards computed after each graph in a generation was constructed
    rew_new = np.zeros([batch_size], dtype=np.float64)

    # at the very beginning, we do not have any survivors
    obs_survive = None
    act_survive = None
    rew_survive = None

    ############################################################
    # Auxiliary variables for taking care of action randomness #
    ############################################################
    act_rndness = act_rndness_init      # probability with which random actions are issued in order to enhance exploration
    old_max_reward = None                # what was the maximum reward the last time when it improved?
    old_max_gen = None                  # when was the last generation when the maximum reward got improved?

    # for transferring results to tensorboard
    writer = SummaryWriter()

    #####################
    # THE TRAINING LOOP #
    #####################
    for gen in range(num_generations):
        # when did you start processing this generation?
        tic = time.perf_counter()

        # generate the next batch of graphs
        new_generation(model, obs_new, act_new, rew_new, batch_size, autlen, rng, act_rndness, n, compute_reward)

        # add earlier survivors to the new batch
        if gen==0:
            obs_full = obs_new
            act_full = act_new
            rew_full = rew_new
        else:
            obs_full = np.concatenate((obs_new, obs_survive), axis=0)
            act_full = np.concatenate((act_new, act_survive), axis=0)
            rew_full = np.concatenate((rew_new, rew_survive), axis=0)

        # select observation/action pairs for learning
        lrn_reward = np.percentile(rew_full, percent_learn)
        ind_learn = np.where(rew_full>=lrn_reward)[0]
        # do not learn from too many triplets - the old method
        # cut_off_size = round(batch_size*(100-percent_learn))
        # ind_learn = ind_learn[:cut_off_size]

        # observations and actions must be reshaped,
        # so that the first dimension is the number of pairs from which to learn,
        # and the second dimension corresponds to the number of model inputs/outputs = obslen/1
        num_learn_pairs = ind_learn.size * autlen
        obs_learn = obs_full[ind_learn, 0:autlen, :].reshape(num_learn_pairs, obslen)
        act_learn = act_full[ind_learn, :].reshape(num_learn_pairs)

        # trains the neural network on selected observation/action pairs
        optimizer.zero_grad()
        act_learn_scores = model(torch.FloatTensor(obs_learn))              # predicts raw score for each possible action
        loss = objective(act_learn_scores, torch.LongTensor(act_learn))     # combines softmax layer for the above raw scores
                                                                            # with cross entropy loss for the actual actions used
        loss.backward()
        optimizer.step()

        # select observation/action/reward triplets for surviving
        # but put an upper bound of batch_size on the number of survivors
        # so that the total size of new generation and survivors is also bounded - the new method
        cutoff_percent = percent_survive
        while cutoff_percent<99.9:
            srv_reward = np.percentile(rew_full, cutoff_percent)
            ind_survive = np.where(rew_full>=srv_reward)[0]
            if ind_survive.size <= batch_size:
                # we're done - there are not too many survivors!
                break
            else:
                # try with a higher cutoff percent
                cutoff_percent = (100+cutoff_percent)/2

        # just make sure that you stay within the limits if you came to the situation
        # that there are more than batch_size triplets with almost the same maximum reward
        ind_survive = ind_survive[:batch_size]
        num_survive = ind_survive.size

        # we need to copy() the subarrays below,
        # as observations, actions and rewards will be overwritten in the next generation
        obs_survive = obs_full[ind_survive, :, :].copy().reshape(num_survive, autlen+1, obslen)
        act_survive = act_full[ind_survive, :].copy().reshape(num_survive, autlen)
        rew_survive = rew_full[ind_survive].copy().reshape(num_survive)

        # a simple vns-like strategy to adjust action randomness
        # if the maximum reward has increased,
        #    reset act_rndness to act_rndness_init,
        # but if the maximum reward has not increased during the last act_rndness_wait steps,
        #    multiply act_rndness by act_rndness_mult,
        #    but also do not let it pass above act_rndness_max
        max_reward = np.max(rew_full)

        if gen==0:
            # pick up the starting max_reward and its generation
            old_max_reward = max_reward
            old_max_gen = 0
        elif max_reward > old_max_reward + 0.0001:
            # the maximum bound just increased, so reset action randomness
            act_rndness = act_rndness_init
            old_max_reward = max_reward
            old_max_gen = gen
        elif gen - old_max_gen >= act_rndness_wait:
            # every few generations without max bound increase, do increase action randomness
            act_rndness = min(act_rndness * act_rndness_mult, act_rndness_max)
            old_max_gen = gen

        # when did you finished processing this generation?
        toc = time.perf_counter()

        # if verbose is True, report some data to the console
        if verbose:
            print(f'gen={gen}, rew_max={max_reward:.8f}, rew_surv={srv_reward:.8f}, rew_learn={lrn_reward:.8f}, time={toc-tic:.4f}, act_rndness={act_rndness:.4f}')
        # this data always gets reported to runs/event file for tensorboard
        writer.add_scalar('rew_max', max_reward, gen)
        writer.add_scalar('rew_surv', srv_reward, gen)
        writer.add_scalar('rew_learn', lrn_reward, gen)
        writer.flush()

        # show the best graph found every output_best_graph_rate generations
        if gen % output_best_graph_rate == 0:
            if gen>0:
                plt.close('all')

            ind_maximum = np.argmax(rew_full)
            max_A = adj_from_obs(n, obs_full[ind_maximum, autlen, 0:autlen])
            gnx = nx.from_numpy_array(max_A)

            plt.figure(num=1, figsize=(4,4), dpi=300)
            nx.draw_kamada_kawai(gnx, node_size=80)

            # uncomment this to have the graph shown immediately (in pycharm, for example)
            # plt.ion()

            writer.add_figure('best graph', plt.figure(num=1), gen)
            # fig_name = f'fig-{gen}.png'
            # plt.savefig(fig_name, transparent=True)

    # freeing the resources...
    plt.close('all')
    writer.close()

    # what was the last maximum reward?
    ind_maximum = np.argmax(rew_full)
    max_A = adj_from_obs(n, obs_full[ind_maximum, autlen, 0:autlen])
    return max_reward, max_A
