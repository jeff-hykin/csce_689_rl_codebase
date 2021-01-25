from Solvers.DQN import DQN
from Solvers.Monte_Carlo import MonteCarlo, OffPolicyMC
from Solvers.Policy_Iteration import PolicyIteration
from Solvers.Q_Learning import ApproxQLearning
from Solvers.Q_Learning import QLearning
from Solvers.Random_Walk import RandomWalk
from Solvers.SARSA import Sarsa
from Solvers.Value_Iteration import ValueIteration, AsynchVI
from Solvers.REINFORCE import Reinforce
from Solvers.A2C import A2C

solvers = ['random', 'vi', 'pi', 'mc','avi', 'mcis', 'ql', 'sarsa', 'aql', 'dqn', 'reinforce', 'A2C']


def get_solver_class(name):
    if name == solvers[0]:
        return RandomWalk
    elif name == solvers[1]:
        return ValueIteration
    elif name == solvers[2]:
        return PolicyIteration
    elif name == solvers[3]:
        return MonteCarlo
    elif name == solvers[4]:
        return AsynchVI
    elif name == solvers[5]:
        return OffPolicyMC
    elif name == solvers[6]:
        return QLearning
    elif name == solvers[7]:
        return Sarsa
    elif name == solvers[8]:
        return ApproxQLearning
    elif name == solvers[9]:
        return DQN
    elif name == solvers[10]:
        return Reinforce
    elif name == solvers[11]:
        return A2C
    else:
        assert False, "unknown solver name {}. solver must be from {}".format(name, str(solvers))
