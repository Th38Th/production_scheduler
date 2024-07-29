import os
import numpy as np
import random

import spektral as sp
from spektral.models import GeneralGNN

import gnn

import tensorflow as tf
from env import SmartFactoryEnv

from collections import deque


class SFDataset(sp.data.Dataset):
    def __init__(self, graph, **kwargs):
        self.graph = graph
        super().__init__(**kwargs)

    def __getitem__(self, item):
        return self.graphs[item]

    def read(self):
        output = []
        output.append(self.graph)
        return output


class DQGNN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        # Discount Factor
        self.gamma = 0.85

        # (Random) Exploration Probability
        self.epsilon = 0.5

        # Minimum Exploration Probability
        self.epsilon_min = 0.01

        # Exploration Probability Index
        self.epsilon_decay = 0.995

        # Learning Rate
        self.learning_rate = 0.005

        # Weight Update Rate in Target Model
        self.tau = 0.125

        # Working Model
        self.model = self.create_model()

        # Target Model
        self.target_model = self.create_model()

        self.__target_model_built__ = False

    @staticmethod
    def __get_loader__(graph):
        dataset = SFDataset(graph)
        loader = sp.data.loaders.SingleLoader(dataset)
        return loader

    def create_model(self):
        model = gnn.PMSPGNN(1, 1, 1, 1, node_hidden=32, edge_hidden=32, global_hidden=32)
        model.compile(loss="mean_squared_error",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def model_predict(self, input, **kwargs):
        actions = self.model.predict(input, **kwargs)
        return self.env.action_mask(actions)

    def target_predict(self, input, **kwargs):
        actions = self.target_model.predict(input, **kwargs)
        return self.env.action_mask(actions)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return min(state.n_edges-1, self.env.action_space.sample())
        __loader = DQGNN.__get_loader__(state)
        if not self.__target_model_built__:
            self.target_model.predict(__loader.load(), steps=__loader.steps_per_epoch)
            self.__target_model_built__ = True
        return np.argmax(self.model_predict(__loader.load(), steps=__loader.steps_per_epoch))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return False

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            __loader = DQGNN.__get_loader__(state)
            target = self.target_predict(__loader.load(), steps=__loader.steps_per_epoch)
            if done:
                target[action] = reward
            else:
                __loader = DQGNN.__get_loader__(new_state)
                q_future = max(self.target_predict(__loader.load(), steps=__loader.steps_per_epoch))
                target[action] = reward + q_future * self.gamma
            state.y = [target]
            __loader = DQGNN.__get_loader__(state)
            self.model.fit(__loader.load(), steps_per_epoch=__loader.steps_per_epoch, epochs=1, verbose=0)
        return True

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def generate_env():
    n_jobs_bounds = (40, 60)
    n_machines_bounds = (4, 20)
    n_job_classes = 5
    return SmartFactoryEnv(n_jobs_bounds, n_machines_bounds, n_job_classes)


def main():
    gamma = 0.9
    epsilon = .95

    env = generate_env()

    trials = 10

    dqn_agent = DQGNN(env=env)

    for trial in range(trials):
        cur_state = env.reset()
        done = False
        while not done:

            print("Jobs left: " + str(len(env.jobs)))

            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            if cur_state is not None:
                dqn_agent.remember(cur_state, action, reward, new_state, done)

            if dqn_agent.replay():
                dqn_agent.target_train()

            cur_state = new_state

        print(f"Completed trial {trial}.")
        env.render()

    dqn_agent.save_model("success.model")
    print(dqn_agent.model.summary())


if __name__ == "__main__":
    if os.path.isdir("success.model") \
            and input("Model exists. Do you want to load and test it? (Y/N)\n> ").lower() == "y":
        mdl = tf.keras.models.load_model("success.model")
        print(mdl.summary())

        env = generate_env()
        while True:
            problem = env.reset()
            done = False
            while not done:
                dataset = SFDataset(problem)
                loader = sp.data.loaders.SingleLoader(dataset)
                prediction = mdl.predict(loader.load(), steps=loader.steps_per_epoch)
                masked_actions = env.action_mask(prediction)
                action = np.argmax(masked_actions)
                problem, _, done, _ = env.step(action)
            env.render()
            if not input("Again? (Y/N)\n> ").lower() == 'y':
                break
    else:
        main()
