from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent

def build_agent(agent_name: str, state_dim: int, action_dim: int, hyperparams: dict):
    """
        Factory function to create an agent based on name.
    """

    if agent_name == "nstep_dqn":
        return NStepDeepQLearningAgent(state_dim, action_dim, **hyperparams)
    elif agent_name == "nstep_ddqn":
        return NStepDoubleDeepQLearningAgent(state_dim, action_dim, **hyperparams)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
