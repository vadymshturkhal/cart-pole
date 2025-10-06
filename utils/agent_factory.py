from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from agents.ase_ace_agent import ASEACEAgent


AGENTS = {
    "nstep_dqn": NStepDeepQLearningAgent, 
    "nstep_ddqn": NStepDoubleDeepQLearningAgent,
    "ase_ace": ASEACEAgent,
}


def build_agent(agent_name: str, state_dim: int, action_dim: int, hyperparams: dict):
    if agent_name == "nstep_dqn":
        return NStepDeepQLearningAgent(state_dim, action_dim, **hyperparams)

    if agent_name == "nstep_ddqn":
        return NStepDoubleDeepQLearningAgent(state_dim, action_dim, **hyperparams)

    if agent_name == "ase_ace":
        return ASEACEAgent(state_dim, action_dim, **hyperparams)

    raise ValueError(f"Unknown agent: {agent_name}")
