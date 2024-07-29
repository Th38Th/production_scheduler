import gym
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import spektral as sp
from scipy import sparse
from itertools import product
from gym import spaces

import utils as utl
import pprint

import model.model as mdl


def draw_gantt(schedule, figsize=(16, 6), verbose=False):
    fig, ax = plt.subplots(1, figsize=figsize)
    m_ids = list(map(lambda x: x[1][1].id, schedule.items()))
    j_ids = list(map(lambda x: x[0].id, schedule.items()))
    j_weights = list(map(lambda x: x[0].work_load, schedule.items()))
    j_classes = list(map(lambda x: x[0].job_class, schedule.items()))
    j_starts = list(map(lambda x: x[0].start_time + x[0].delay_time, schedule.items()))
    j_work_time = list(map(lambda x: x[0].work_time, schedule.items()))
    bars = ax.barh(m_ids, j_work_time, left=j_starts)
    cmap = plt.cm.get_cmap('hsv', len(m_ids))
    actual_bars = bars.get_children()
    for i in range(len(j_starts)):
        fontsize = 8
        bar_color = np.array(cmap(i)[:3]) * 255
        lum = np.sqrt((bar_color[0]**2) * 0.299 + (bar_color[1]**2) * 0.587 + (bar_color[2] ** 2) * 0.114)
        txt_color = np.ones(3) if lum <= 186 else np.zeros(3)
        label_data = (j_ids[i], j_weights[i], j_classes[i])
        label_text = 'Id: {}\nWeight: {}\nClass: {}'.format(*label_data) if verbose else "\n".join(map(str, label_data))
        plt.text(j_starts[i] + j_work_time[i] / 2, m_ids[i],
                 label_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color=txt_color)
    for i in range(len(actual_bars)):
        actual_bars[i].set_color(cmap(i))
    plt.title("Gantt Chart of current project")
    plt.show()


class GraphObservationSpace(gym.Space):

    def __init__(self, n_nodes_bounds, d_connectivity_bounds, node_feature_bounds, edge_weights_bounds,
                 edge_feature_bounds=None):
        super().__init__()
        self.n_nodes_bounds = n_nodes_bounds
        self.d_connectivity_bounds = d_connectivity_bounds
        self.edge_weights_bounds = edge_weights_bounds
        self.node_feature_bounds = node_feature_bounds
        self.edge_feature_bounds = edge_feature_bounds

    def sample(self):
        n_nodes = self.np_random.randint(self.n_nodes_bounds[0], self.n_nodes_bounds[1])
        x = np.array(
            [self.np_random.uniform(self.node_feature_bounds[0], self.node_feature_bounds[1]) for _ in range(n_nodes)])
        a = self.np_random.uniform(self.edge_weights_bounds[0], self.edge_weights_bounds[1],
                                   size=(n_nodes, n_nodes)) if self.edge_weights_bounds is not None else np.ones(
            (n_nodes, n_nodes))
        p = self.np_random.uniform(self.d_connectivity_bounds[0], self.d_connectivity_bounds[1])
        m = self.np_random.choice((0, 1), size=(n_nodes, n_nodes), p=(1 - p, p))
        a = np.multiply(m, a)
        n_edges = np.count_nonzero(a)
        a = sparse.coo_matrix(a)
        e = np.array(
            [self.np_random.uniform(self.edge_feature_bounds[0], self.edge_feature_bounds[1]) for _ in range(n_edges)]) \
            if self.edge_feature_bounds is not None else None
        return sp.data.Graph(x=x, a=a, e=e, u=0)

    def contains(self, graph):
        n_nodes_ok = self.n_nodes_bounds[0] < graph.n_nodes < self.n_nodes_bounds[1]
        d_connect_ok = self.d_connectivity_bounds[0] < graph.n_edges / (graph.n_nodes ** 2) < \
                       self.d_connectivity_bounds[1]
        if n_nodes_ok and d_connect_ok:
            nodes_within = self.node_feature_bounds[0] <= np.min(graph.x) and np.max(graph.x) <= \
                           self.node_feature_bounds[1]
            edges_within = self.edge_feature_bounds[0] <= np.min(graph.e) and np.max(graph.e) <= \
                           self.edge_feature_bounds[1]
            return nodes_within and edges_within
        return False


class SmartFactoryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    @property
    def n_edges_max(self):
        n_jobs_max = self.n_jobs_bounds[1]
        n_machines_max = self.n_machines_bounds[1]
        return n_jobs_max * n_machines_max

    def __init__(self, n_jobs_bounds, n_machines_bounds, n_job_classes, **kwargs):
        self.n_jobs_bounds = n_jobs_bounds
        self.n_machines_bounds = n_machines_bounds
        self.n_job_classes = n_job_classes

        self.job_worktime_bounds = kwargs.get("job_worktime_bounds") or (5, 10)
        self.job_workload_bounds = kwargs.get("job_workload_bounds") or (5, 20)
        self.job_arrival_bounds = kwargs.get("job_arrival_bounds") or (0, 5)
        self.class_delay_bounds = kwargs.get("class_delay_bounds") or (.25, 10)

        # An action represents the allocation of a job to a machine
        self.action_space = spaces.Discrete(self.n_edges_max + 1, start=-1)

        # An observation represents the state of the graph at a given point in time
        __n_nodes_bounds = np.array(self.n_jobs_bounds) + np.array(self.n_machines_bounds)
        __node_feature_bounds = np.array((self.job_worktime_bounds,
                                          self.job_workload_bounds,
                                          self.job_arrival_bounds,
                                          *[(0, 1) for _ in range(self.n_job_classes)],
                                          self.job_worktime_bounds,
                                          *[(0, 1) for _ in range(self.n_job_classes)],
                                          (0, 1))).transpose()

        __d_connect_bounds = np.array([self.n_machines_bounds, self.n_jobs_bounds]) / np.array(__n_nodes_bounds) ** 2

        self.observation_space = GraphObservationSpace(__n_nodes_bounds, __d_connect_bounds,
                                                       __node_feature_bounds, None, None)

        self.state = None
        self.t = None
        self.jobs = None
        self.machines = None
        self.schedule = None
        self.class_delays = None

    @property
    def allocation_space(self):
        return list(product(self.jobs, self.machines))

    @property
    def allocation_dict(self):
        allocation_space = self.allocation_space
        return {i: (allocation_space[i][0], allocation_space[i][1]) for i in range(len(allocation_space))}

    def objective(self, jobs, machines, schedule):
        return sum(map(lambda x: x.work_load * x.work_time, schedule.keys())) / \
               max(map(lambda x: x.finish_time, schedule.keys())) \
               if len(self.schedule) > 0 else - self.t

    def action_mask(self, actions):
        for i in range(len(actions)):
            actions[i] = actions[i] \
                if i < len(self.allocation_space) \
                   and self.allocation_space[i][1].is_free() \
                else 0
        return actions

    def step(self, action):

        f0 = self.objective(self.jobs, self.machines, self.schedule)

        for m in self.machines:
            if m.get_remaining_job_time(self.t) == 0:
                m.current_job = None

        if self.action_space.contains(action):
            if action != -1 and action < len(self.allocation_space):
                allocation = self.allocation_space[action]
                job = allocation[0]
                machine = allocation[1]
                if machine.is_free():
                    self.jobs.remove(job)
                    job.start_time = self.t
                    machine.current_job = job
                    if machine.last_run_job_class is not None:
                        delay = self.class_delays[job.job_class, machine.last_run_job_class]
                        job.delay_time = delay
                    self.schedule[job] = (self.t, machine)

        f1 = self.objective(self.jobs, self.machines, self.schedule)

        reward = f1 - f0

        done = (len(self.jobs) == 0)

        info = {'schedule': self.schedule}

        delta_t = 0
        filtered_machines = self.machines

        # If we've chosen to do nothing, ignore current free machine(s)
        # and skip to the closest time another machine becomes available
        if action == -1:
            filtered_machines = filter(lambda x: not x.is_free(), self.machines)
        if not any(map(lambda x: x.is_free(), self.machines)):
            # Skip ahead to the closest time when a machine becomes available
            delta_t = min(map(lambda x: x.get_remaining_job_time(self.t), filtered_machines))
        self.t += delta_t

        gph = utl.encode_pmsp(self.jobs, self.n_job_classes, self.machines, self.class_delays, self.t) if len(self.jobs) > 0 else None

        return gph, reward, done, info

    def render(self, mode="human"):
        draw_gantt(self.schedule)

    def generate_problem(self):
        n_jobs = self.np_random.randint(self.n_jobs_bounds[0], self.n_jobs_bounds[1])
        n_machines = self.np_random.randint(self.n_machines_bounds[0], self.n_machines_bounds[1])
        jobs = [mdl.Job('J' + str(i),
                        self.np_random.randint(0, self.n_job_classes),
                        self.np_random.randint(self.job_worktime_bounds[0], self.job_worktime_bounds[1]),
                        self.np_random.randint(self.job_workload_bounds[0], self.job_workload_bounds[1]),
                        0
                        ) for i in range(n_jobs)
                ]
        machines = [mdl.Machine('M' + str(i), 0, self.np_random.randint(10, 300)) for i in range(n_machines)]
        class_delays = np.random.uniform(self.class_delay_bounds[0], self.class_delay_bounds[1],
                                         size=(self.n_job_classes, self.n_job_classes))
        np.fill_diagonal(class_delays, 0)
        return jobs, machines, class_delays

    def reset(self):
        self.jobs, self.machines, self.class_delays = self.generate_problem()
        self.schedule = dict()
        self.t = 0
        return utl.encode_pmsp(self.jobs, self.n_job_classes, self.machines, self.class_delays, self.t)


if __name__ == '__main__':
    sfi = mdl.SmartFactory(name="MY FACTORY", job_csv="data/jobs.csv", machine_csv="data/machines.csv")
    n_jobs_bounds = (5, 20)
    n_machines_bounds = (2, 5)
    n_job_classes = 3
    sfi_env = SmartFactoryEnv(n_jobs_bounds, n_machines_bounds, n_job_classes)
    print(sfi_env.reset())
    done = False
    printr = pprint.PrettyPrinter(width=32)
    while not done:
        allocation_dict = sfi_env.allocation_dict
        print("Press the key corresponding to the allocation you want me to make.")
        allocation_descriptions = {i: "Allocate " + str(allocation_dict[i][0]) + "\nto " + str(allocation_dict[i][1])
                                   for i in allocation_dict.keys()}
        for itm in allocation_descriptions.items():
            print(str(itm[0]) + ": " + str(itm[1]))
        while True:
            action = int(input("> "))
            if action not in allocation_descriptions:
                print("That's not a valid action... Try again.")
            else:
                break
        print(f"You: \"{allocation_descriptions[action]}.\"")
        state, reward, done, info = sfi_env.step(action)
        print("Score: " + str(reward) + ".")
        if done:
            print("Finished! Your resulting schedule is: ")
            printr.pprint(info["schedule"])
