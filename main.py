import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import tensorflow as tf


def naive_policy(obs):
    # if isinstance(obs, tuple):
     #   obs = obs[0]
    angle = obs[0].all()
    return 0 if angle < 0 else 1


def random_policy(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    angle = obs[2]
    return 0 if np.random.uniform() < 0.5 else 1


def better_policy(obs):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    if isinstance(obs, tuple):
        obs = obs[0]
    angle = obs[2]
    input_ = np.array([[angle]])
    prediction = model.predict(input_, verbose=0)
    return int(np.random.rand() > prediction)


def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render()
    plt.imshow(img)
    plt.axis("off")
    return img


def naive_main(policy):
    debug = True
    env = gym.make("CartPole-v0", render_mode="rgb_array")
    obs = env.reset()
    env.render()
    step_count = 0

    # episodic reinforcement learning
    totals = []
    for episode in range(30):
        print("Episode number", episode)
        episode_rewards = 0
        obs = env.reset()
        for step in range(10000):
            action = policy(obs)
            obs, reward, done, _, _, = env.step(action)
            env.render()
            time.sleep(0.1)
            episode_rewards += reward
            if done:
                plot_environment(env)
                #plt.show()
                print("Game over. Number of steps = ", step)
                step_count += step
                env.render()
                time.sleep(3.14)
                break
        totals.append(episode_rewards)
    print(step_count/30)
    print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))


if __name__ == "__main__":
    naive_main(random_policy)
