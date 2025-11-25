"""
evaluate.py
Evaluates a trained policy for N episodes and prints average reward & average queue length.
Usage:
    python evaluate.py --model models/ppo_sumo_final.zip --routes ../routes --episodes 10
"""

import argparse
import glob
import random
import numpy as np
from stable_baselines3 import PPO
from data_collector_env import SumoTrafficEnv

def evaluate(model_path, route_files, episodes=5):
    model = PPO.load(model_path)
    rewards = []
    avg_queues = []
    for ep in range(episodes):
        route = random.choice(route_files)
        env = SumoTrafficEnv(route_file=route, logfile=f"eval_trans_{ep}.csv")
        obs, _ = env.reset(route)
        done = False
        total_reward = 0.0
        queue_sum = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            queue_sum += sum(obs[:4])
            steps += 1
        rewards.append(total_reward)
        avg_queues.append(queue_sum / max(1, steps))
        env.close()
    print(f"Avg reward over {episodes} episodes: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Avg queue per step: {np.mean(avg_queues):.2f}")
    return rewards, avg_queues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to saved model")
    parser.add_argument("--routes", default="../routes", help="path to route files")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    routes = glob.glob(args.routes + "/*.rou.xml")
    evaluate(args.model, routes, episodes=args.episodes)
