from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from agents.ase_ace_agent import ASEACEAgent, ACConfig


AGENTS = {
    "nstep_dqn": NStepDeepQLearningAgent, 
    "nstep_ddqn": NStepDoubleDeepQLearningAgent,
    "aseace": ASEACEAgent,
}


def build_agent(agent_name: str, state_dim: int, action_dim: int, hyperparams: dict):
    if agent_name == "nstep_dqn":
        return NStepDeepQLearningAgent(state_dim, action_dim, **hyperparams)

    if agent_name == "nstep_ddqn":
        return NStepDoubleDeepQLearningAgent(
            state_dim, action_dim,
            gamma=hyperparams["gamma"], lr=hyperparams["lr"],
            buffer_size=int(hyperparams["buffer_size"]), batch_size=int(hyperparams["batch_size"]),
            n_step=int(hyperparams["n_step"]),
            eps_start=hyperparams["eps_start"], eps_end=hyperparams["eps_end"], eps_decay=int(hyperparams["eps_decay"])
        )

    if agent_name == "ase_ace":
        cfg = ACConfig(
            gamma=hyperparams["gamma"],
            lam_v=hyperparams["lam_v"],
            lam_pi=hyperparams["lam_pi"],
            alpha_v=hyperparams["alpha_v"],
            alpha_pi=hyperparams["alpha_pi"],
            max_steps=int(hyperparams["max_steps"]),
            sparse_reward=bool(hyperparams["sparse_reward"]),
        )
        return ASEACEAgent(cfg)

    raise ValueError(f"Unknown agent: {agent_name}")
