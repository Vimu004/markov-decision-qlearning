import numpy as np
import random
import streamlit as st
import time
import ast  


def main():
    st.title("🔍 Q-learning Submarine Navigation")

    st.sidebar.header("Grid Settings")
    grid_size = st.sidebar.number_input("Grid Size (NxN)", min_value=3, max_value=10, value=4, step=1)
    
    obstacle_input = st.sidebar.text_area("Enter Obstacles (e.g., [(1,1), (2,2)])", "[(1,1), (2,2)]")
    goal_input = st.sidebar.text_input("Enter Goal Position (e.g., (3,3))", f"({grid_size-1},{grid_size-1})")

    try:
        obstacles = ast.literal_eval(obstacle_input)  
        if not isinstance(obstacles, list) or not all(isinstance(x, tuple) and len(x) == 2 for x in obstacles):
            raise ValueError
        obstacles = [tuple(map(int, obs)) for obs in obstacles]  
    except:
        st.sidebar.error("Invalid obstacle format! Use [(x,y), (x,y)] format.")

    try:
        GOAL = ast.literal_eval(goal_input)  
        if not isinstance(GOAL, tuple) or len(GOAL) != 2:
            raise ValueError
        GOAL = tuple(map(int, GOAL))  
    except:
        st.sidebar.error("Invalid goal format! Use (x,y) format.")

    START = (0, 0)

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
    Q = np.zeros((grid_size, grid_size, len(ACTIONS)))


    def move(state, action):
        moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        new_state = (state[0] + moves[action][0], state[1] + moves[action][1])

        if 0 <= new_state[0] < grid_size and 0 <= new_state[1] < grid_size and new_state not in obstacles:
            return new_state
        return state  

    st.sidebar.header("Controls")
    speed = st.sidebar.slider("Adjust Speed", 0.1, 2.0, 0.5, 0.1)  
    num_episodes = st.sidebar.slider("Number of Learning Episodes", 10, 500, 100, 10)

    if "training" not in st.session_state:
        st.session_state.training = False  

    start_button = st.sidebar.button("▶ Start Training")
    stop_button = st.sidebar.button("⏹ Stop Training")

    if start_button:
        st.session_state.training = True 
    if stop_button:
        st.session_state.training = False 

    episode_placeholder = st.empty()  

    col1, col2 = st.columns([2, 3])  
    grid_placeholder = col1.empty()
    q_table_placeholder = col2.empty()
    final_grid_placeholder = st.empty()  

    grid_display = np.full((grid_size, grid_size), "⬜", dtype="<U10")
    for obstacle in obstacles:
        if 0 <= obstacle[0] < grid_size and 0 <= obstacle[1] < grid_size:
            grid_display[obstacle] = "⛔"
    grid_display[GOAL] = "🏁"

    if st.session_state.training:
        alpha = 0.5 
        gamma = 0.9  
        epsilon = 0.2  

        for episode in range(1, num_episodes + 1):  
            if not st.session_state.training: 
                break

            episode_placeholder.write(f"## 🏁 Learning Progress: Episode {episode} / {num_episodes}")  

            state = START
            while state != GOAL:
                if not st.session_state.training:  
                    break

                if random.uniform(0, 1) < epsilon:
                    action = random.choice(ACTIONS)  
                else:
                    action = ACTIONS[np.argmax(Q[state])] 

                new_state = move(state, action)
                reward = 10 if new_state == GOAL else (-5 if new_state in obstacles else -1)


                action_index = ACTIONS.index(action)
                Q[state][action_index] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action_index])

                temp_grid = grid_display.copy()
                temp_grid[state] = "🚢"
                grid_placeholder.table(temp_grid)

                q_table_placeholder.dataframe(Q.reshape(grid_size * grid_size, len(ACTIONS)), width=700)

                time.sleep(speed)  

                state = new_state  

    state, path = START, [START]
    while state != GOAL:
        action = ACTIONS[np.argmax(Q[state])]  
        state = move(state, action)
        path.append(state)

    final_grid = np.full((grid_size, grid_size), "⬜", dtype="<U10")
    for obstacle in obstacles:
        final_grid[obstacle] = "⛔"
    final_grid[GOAL] = "🏁"

    for (x, y) in path:
        if (x, y) != GOAL:
            final_grid[x, y] = "🚢"

    final_grid_placeholder.write("### ✅ Optimal Path Learned (After Training)")
    final_grid_placeholder.table(final_grid)  

    st.success(f"🎉 Learning Complete! Final Path Learned.")

if __name__ == "__main__":
    main()
