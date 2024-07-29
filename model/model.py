import pandas as pd


class Job:

    def __init__(self, id, job_class, work_time, work_load, arrival_time):
        self.id = id
        self.work_time = work_time
        self.work_load = work_load
        self.job_class = job_class
        self.arrival_time = arrival_time
        self.start_time = 0
        self.delay_time = 0

    def get_remaining_time(self, t=0):
        return max(0, self.finish_time - t)

    @property
    def finish_time(self):
        return self.start_time + self.delay_time + self.work_time

    def __repr__(self):
        return f"Job({self.id},{self.job_class},{self.work_time},{self.work_load},{self.arrival_time})"

    def __str__(self):
        return f"Job {self.id}, of class {self.job_class}, which takes {self.work_time}, is of priority {self.work_load} " \
               f"and arrived at {self.arrival_time}"


class Machine:

    def __init__(self, id, machine_class, capacity):
        self.id = id
        self.machine_class = machine_class
        self.machine_capacity = capacity
        self.last_run_job_class = None
        self.__current_job = None

    @property
    def current_job(self):
        return self.__current_job

    @current_job.setter
    def current_job(self, x):
        if self.current_job is not None:
            self.last_run_job_class = self.__current_job.job_class
        self.__current_job = x

    def get_remaining_job_time(self, t=0):
        c_job = self.current_job
        return c_job.get_remaining_time(t)\
            if c_job is not None else 0

    def is_free(self):
        return self.__current_job is None

    def __repr__(self):
        return f"Machine({self.id},{self.machine_class},{self.current_job},{self.last_run_job_class})"

    def __str__(self):
        return f"Machine {self.id}, of class {self.machine_class}" + (f", currently processing {self.current_job} "
                                                              if self.current_job is not None
                                                              else ", not currently processing any job")


class SmartFactory:

    def __new__(cls, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SmartFactory, cls).__new__(cls)
        return cls.instance

    def __init__(self, **kwargs):
        # Skip initialization if the Singleton Class instance has already been created
        if kwargs is None or len(kwargs) == 0:
            return
        self.__name = kwargs.get("name")
        job_csv_filename = kwargs.get("job_csv")
        if job_csv_filename is None:
            self.__job_classes = kwargs.get('job_classes') or dict()
            self.__jobs = []
        else:
            jobs_csv = pd.read_csv(job_csv_filename)
            self.job_classes = dict.fromkeys(jobs_csv["class"].unique())
            self.__jobs = [Job(*(row[1])) for row in jobs_csv.iterrows()]
        machine_csv_filename = kwargs.get("machine_csv")
        if machine_csv_filename is None:
            self.__machine_classes = kwargs.get('job_classes') or dict()
            self.__machines = []
        else:
            machine_csv = pd.read_csv(machine_csv_filename)
            self.machine_classes = dict.fromkeys(machine_csv["class"].unique())
            self.__machines = [Machine(*(row[1])) for row in machine_csv.iterrows()]
