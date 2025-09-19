# CartPole N-step Q-learning 🏋️🤖

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-008080?logo=openai)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyTorch implementation of an **N-step Off-policy Q-learning agent** trained to solve **CartPole-v1** with [Gymnasium](https://gymnasium.farama.org/).  
The project features a **modular architecture**, configurable hyperparameters, reward logging, checkpointing, and rendering.

---

## Features
- 🧠 **N-step Off-policy Q-learning** with replay buffer  
- ⚙️ **Configurable architecture** (layers, activations, dropout) in `config.py`  
- 🎲 **Epsilon-greedy exploration** with flexible decay schedules  
- 📊 **Reward tracking** to CSV & TensorBoard  
- 💾 **Model checkpoint saving & loading**  
- 🎬 **Rendering** of trained agent  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/nstep-qlearning-cartpole.git
cd cart-pole
pip install -r requirements.txt
```

### GPU Acceleration (Optional)
By default, `requirements.txt` installs the **CPU version** of PyTorch for maximum portability.  
If you have a CUDA-enabled GPU, install a matching CUDA build of PyTorch from the [official wheels](https://pytorch.org/get-started/previous-versions/).

### CUDA 11.8 example:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1 example (latest builds):
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

---

## Training
Run training with:
```bash
python3 main.py
```

This will:
    Train the agent
    Save rewards to rewards.csv
    Save the model to trained_qnet.pth
    Plot the learning curve

---

## Testing & Rendering
```bash
python3 test_trained_agent.py
```
