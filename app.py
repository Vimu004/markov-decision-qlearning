import numpy as np
import random
import streamlit as st
import time
import ast  # To safely parse obstacle input

# Streamlit App
def main():
    st.title("🔍 Q-learning Submarine Navigation")

    # **NEW: User-defined Grid Size, Obstacles, and Goal**
    st.sidebar.header("Grid Settings")
    grid_size = st.sidebar.number_input("Grid Size (NxN)", min_value=3, max_value=10, value=4, step=1)
    
    obstacle_input = st.sidebar.text_area("Enter Obstacles (e.g., [(1,1), (2,2)])", "[(1,1), (2,2)]")
    goal_input = st.sidebar.text_input("Enter Goal Position (e.g., (3,3))", f"({grid_size-1},{grid_size-1})")

    # Convert user inputs safely
    try:
        obstacles = ast.literal_eval(obstacle_input)  # Convert string to list of tuples
        if not isinstance(obstacles, list) or not all(isinstance(x, tuple) and len(x) == 2 for x in obstacles):
            raise ValueError
        obstacles = [tuple(map(int, obs)) for obs in obstacles]  # Ensure integer coordinates
    except:
        st.sidebar.error("Invalid obstacle format! Use [(x,y), (x,y)] format.")

    try:
        GOAL = ast.literal_eval(goal_input)  # Convert goal input to tuple
        if not isinstance(GOAL, tuple) or len(GOAL) != 2:
            raise ValueError
        GOAL = tuple(map(int, GOAL))  # Ensure integer coordinates
    except:
        st.sidebar.error("Invalid goal format! Use (x,y) format.")

    # **Set Start Position**
    START = (0, 0)

    # **Q-learning Variables**
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
    Q = np.zeros((grid_size, grid_size, len(ACTIONS)))

    # **Movement Function**
    def move(state, action):
        moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        new_state = (state[0] + moves[action][0], state[1] + moves[action][1])

        if 0 <= new_state[0] < grid_size and 0 <= new_state[1] < grid_size and new_state not in obstacles:
            return new_state
        return state  # Stay in place if out of bounds or hitting obstacle

    # **NEW: Sidebar Controls**
    st.sidebar.header("Controls")
    speed = st.sidebar.slider("Adjust Speed", 0.1, 2.0, 0.5, 0.1)  
    num_episodes = st.sidebar.slider("Number of Learning Episodes", 10, 500, 100, 10)

    # **NEW: Start and Stop Buttons using Session State**
    if "training" not in st.session_state:
        st.session_state.training = False  # Default: Not running

    start_button = st.sidebar.button("▶ Start Training")
    stop_button = st.sidebar.button("⏹ Stop Training")

    if start_button:
        st.session_state.training = True  # Start Training
    if stop_button:
        st.session_state.training = False  # Stop Training

    # **NEW: Show training progress only when active**
    episode_placeholder = st.empty()  # Placeholder for episode display

    # **Rearrange layout for better visibility**
    col1, col2 = st.columns([2, 3])  # **NEW: Give more space to the Q-table (Right Side)**
    grid_placeholder = col1.empty()
    q_table_placeholder = col2.empty()
    final_grid_placeholder = st.empty()  # **NEW: Always display final learned path**

    # Initialize grid
    grid_display = np.full((grid_size, grid_size), "⬜", dtype="<U10")
    for obstacle in obstacles:
        if 0 <= obstacle[0] < grid_size and 0 <= obstacle[1] < grid_size:
            grid_display[obstacle] = "⛔"
    grid_display[GOAL] = "🏁"

    # **Only run training if Start button is pressed**
    if st.session_state.training:
        # Q-learning Hyperparameters
        alpha = 0.5  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.2  # Exploration rate

        # Run Q-learning
        for episode in range(1, num_episodes + 1):  # Start from 1 for better readability
            if not st.session_state.training:  # Stop training if the user presses Stop
                break

            episode_placeholder.write(f"## 🏁 Learning Progress: Episode {episode} / {num_episodes}")  # Show Episode

            state = START
            while state != GOAL:
                if not st.session_state.training:  # Stop training if needed
                    break

                # Choose action (ε-greedy)
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(ACTIONS)  # Explore
                else:
                    action = ACTIONS[np.argmax(Q[state])]  # Exploit

                new_state = move(state, action)
                reward = 10 if new_state == GOAL else (-5 if new_state in obstacles else -1)

                # Q-learning Update Rule
                action_index = ACTIONS.index(action)
                Q[state][action_index] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action_index])

                # Update grid to show submarine movement
                temp_grid = grid_display.copy()
                temp_grid[state] = "🚢"
                grid_placeholder.table(temp_grid)

                # **NEW: Make the Q-table wider**
                q_table_placeholder.dataframe(Q.reshape(grid_size * grid_size, len(ACTIONS)), width=700)

                time.sleep(speed)  # Adjust speed for visualization

                state = new_state  # Move to the new state

    # **NEW STEP: Find the Optimal Path After Training (Always Visible)**
    state, path = START, [START]
    while state != GOAL:
        action = ACTIONS[np.argmax(Q[state])]  # Choose best action from Q-table
        state = move(state, action)
        path.append(state)

    # **NEW STEP: Display the Optimal Learned Path**
    final_grid = np.full((grid_size, grid_size), "⬜", dtype="<U10")
    for obstacle in obstacles:
        final_grid[obstacle] = "⛔"
    final_grid[GOAL] = "🏁"

    # Mark the learned path
    for (x, y) in path:
        if (x, y) != GOAL:
            final_grid[x, y] = "🚢"

    final_grid_placeholder.write("### ✅ Optimal Path Learned (After Training)")
    final_grid_placeholder.table(final_grid)  # **NEW: Always visible even after stopping**

    st.success(f"🎉 Learning Complete! Final Path Learned.")

if __name__ == "__main__":
    main()
