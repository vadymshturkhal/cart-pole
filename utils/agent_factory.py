# utils/agent_factory.py
from agents.nstep_dqn_agent import NStepDeepQLearningAgent
from agents.nstep_ddqn_agent import NStepDoubleDeepQLearningAgent
from agents.ase_ace_agent import ASEACEAgent, ACConfig


def build_agent(agent_name: str, state_dim: int, action_dim: int, hps: dict):
    if agent_name == "nstep_dqn":
        return NStepDeepQLearningAgent(
            state_dim, action_dim,
            gamma=hps["gamma"], lr=hps["lr"],
            buffer_size=int(hps["buffer_size"]), batch_size=int(hps["batch_size"]),
            n_step=int(hps["n_step"]),
            eps_start=hps["eps_start"], eps_end=hps["eps_end"], eps_decay=int(hps["eps_decay"])
        )

    if agent_name == "nstep_ddqn":
        return NStepDoubleDeepQLearningAgent(
            state_dim, action_dim,
            gamma=hps["gamma"], lr=hps["lr"],
            buffer_size=int(hps["buffer_size"]), batch_size=int(hps["batch_size"]),
            n_step=int(hps["n_step"]),
            eps_start=hps["eps_start"], eps_end=hps["eps_end"], eps_decay=int(hps["eps_decay"])
        )

    if agent_name == "ase_ace":
        cfg = ACConfig(
            gamma=hps["gamma"],
            lam_v=hps["lam_v"],
            lam_pi=hps["lam_pi"],
            alpha_v=hps["alpha_v"],
            alpha_pi=hps["alpha_pi"],
            max_steps=int(hps["max_steps"]),
            sparse_reward=bool(hps["sparse_reward"]),
        )
        return ASEACEAgent(cfg)

    raise ValueError(f"Unknown agent: {agent_name}")
