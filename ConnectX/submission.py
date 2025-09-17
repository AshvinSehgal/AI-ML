import numpy as np
import matplotlib.pyplot as plt
import math
import random
from kaggle_environments import evaluate, make, utils
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Input, Dense
# import os
# import logging

# logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

env = make("connectx")

def countWindow(window, plr, opp, inarow):
    score = 0
    if window.count(plr) == inarow:
        score += 100000
    elif window.count(plr) == inarow - 1 and window.count(0) == 1:
        score += 1
    elif window.count(opp) == inarow - 1 and window.count(0) == 1:
        score -= 10
    elif window.count(opp) == inarow:
        score -= 10000
    return score

def get_score(state, plr, rows, cols, inarow):
    score = 0
    opp = 1 if plr == 2 else 2
    state = np.array(state).reshape(rows, cols)
    
    for r in range(rows):
        row = list(state[r, :])
        for c in range(inarow):
            window = row[c:c+inarow]
            score += countWindow(window, plr, opp, inarow)
            
    for c in range(cols):
        col = list(state[:, c])
        for r in range(cols - inarow):
            window = col[c:c+inarow]
            score += countWindow(window, plr, opp, inarow)
            
    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            window = [state[r+i][c+i] for i in range(inarow)]
            score += countWindow(window, plr, opp, inarow)
            
    for r in range(rows - inarow + 1, rows):
        for c in range(cols - inarow + 1):
            window = [state[r-i][c+i] for i in range(inarow)]
            score += countWindow(window, plr, opp, inarow)
            
    return score

def getValidMoves(state, cols):
    return [c for c in range(cols) if state[c] == 0]

def playMove(state, col, plr, rows):
    newS = state.copy()
    for r in range(rows - 1, -1, -1):
        if newS[r*7 + col] == 0:
            newS[r*7 + col] = plr
            break
    return newS

def winningMove(state, plr, rows, cols, inarow):
    state = np.array(state).reshape(rows, cols)

    for r in range(rows):
        for c in range(cols - inarow + 1):
            if all(state[r][c+i] == plr for i in range(inarow)):
                return True

    for c in range(cols):
        for r in range(rows - inarow + 1):
            if all(state[r+i][c] == plr for i in range(inarow)):
                return True

    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            if all(state[r+i][c+i] == plr for i in range(inarow)):
                return True

    for r in range(rows - inarow + 1, rows):
        for c in range(cols - inarow + 1):
            if all(state[r-i][c+i] == plr for i in range(inarow)):
                return True
    return False
    

def minimax(state, depth, alpha, beta, maxPlayer, plr, r, c, i):
    valid_moves = getValidMoves(state, c)
    opp = 1 if plr == 2 else 2
    if winningMove(state, plr, r, c, i):
        return None, 1e9
    elif winningMove(state, opp, r, c, i):
        return None, -1e9
    elif len(valid_moves) == 0 or depth == 0:
        return 0, get_score(state, plr, r, c, i)
    
    if maxPlayer:
        value = -math.inf
        best_col = random.choice(valid_moves)
        for col in valid_moves:
            childS = playMove(state, col, plr, r)
            _, newScore = minimax(childS, depth-1, alpha, beta, False, plr, r, c, i)
            if newScore > value:
                value = newScore
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        best_col = random.choice(valid_moves)
        for col in valid_moves:
            childS = playMove(state, col, opp, r)
            _, new_score = minimax(childS, depth-1, alpha, beta, True, plr, r, c, i)
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value
    
def agent(obs, config):
    action, _ = minimax(state=obs.board, depth=4, alpha=-math.inf, beta=math.inf, maxPlayer=True, plr=obs.mark, r=config.rows, c=config.columns, i=config.inarow)
    return action


# trainer = env.train([None, "random"])
# rewards = []
# r_mean = []
# for ep in range(1000):
#     state = trainer.reset()
#     done = False
#     reward = 0
#     i = 0
#     while i < 100000:
#         action = agent(state, env.configuration)
#         nextS, reward, done, _ = trainer.step(action)
#         reward = 0 if reward is None else reward
#         if done:
#             break
#         state = nextS
#         i += 1
#     rewards.append(reward)
#     r_mean.append(np.mean(rewards))
#     plt.plot(r_mean)
#     plt.savefig('r_mean')