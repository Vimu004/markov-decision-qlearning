# 🚢 Q-Learning Submarine Navigation

This project is a **Q-learning-based submarine navigation simulation** built with **Streamlit**.  
The agent learns to navigate a **dynamic grid world** while avoiding obstacles and reaching the goal.

## 📌 Features
- 🏗 **Custom Grid Size** – Define the grid dimensions dynamically.
- 🚧 **Custom Obstacles & Goal** – Place obstacles and set the goal before training.
- 🔄 **Live Q-table Updates** – Watch the agent learn in real-time.
- 🎚 **Adjustable Speed & Episodes** – Control training speed and the number of episodes.
- ▶⏹ **Start & Stop Training** – Run and pause Q-learning at any time.
- ✅ **Final Optimal Path Display** – Shows the shortest path after training.

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Vimu004/markov-decision-qlearning.git
cd q-learning-submarine
```
```bash
python3 -m venv venv
```

## Activate the Virtual Environment

### On Windows:
```bash
venv\Scripts\activate
```

### On macOS and Linux:
```bash
source venv/bin/activate
```

## Install Dependencies
Install the required packages using pip and the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Run the Application
Start the application using:

```bash
streamlit run app.py
```

## How to Use
- Set Grid Size, Obstacles, and Goal in the left sidebar.
- Adjust Speed & Number of Episodes for training.
- Click "▶ Start Training" to begin Q-learning.
- The agent moves step-by-step while the Q-table updates live.
- Click "⏹ Stop Training" anytime to pause.
- Once training is complete, the optimal learned path is displayed.

## Contributing
Pull requests are welcome!
If you find a bug or have a feature request, open an issue.

