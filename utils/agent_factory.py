from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from agents.rainbow_agent import RainbowAgent
from agents.ase_ace_agent import ASEACEAgent


AGENTS = {
    "nstep_dqn": NStepDeepQLearningAgent, 
    "nstep_ddqn": NStepDoubleDeepQLearningAgent,
    # "rainbow": RainbowAgent,
    # "ase_ace": ASEACEAgent,
}

def build_agent(agent_name: str, state_dim: int, action_dim: int, hyperparams: dict):
    AgentClass = AGENTS.get(agent_name)
    
    if AgentClass is None:
        raise ValueError(f"❌ Unknown agent name: '{agent_name}'. Available: {list(AGENTS)}")
    
    return AgentClass(state_dim, action_dim, **hyperparams)
