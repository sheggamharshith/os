import argparse
import sys
from typing import Literal
from math import gcd
import heapq
from enum import Enum


class EventTypeEnum(Enum):
    arrival = "arrival"
    completed = "completed"
    preemption = "preemption"


def get_lcm(values: list):
    """helps to find lcm in the given processors"""
    lcm = 1
    for i in values:
        lcm = lcm * i // gcd(lcm, i)
    return lcm


class Process:
    """
    Process indicates the process what kind of task it is. don't get confuse between the process and processor.
    consider process also to be a task depending on situation. process can be directly attached to the processor task
    """

    def __init__(
        self, process_number, arrival_time, execution_time, relative_deadline, period
    ):
        self.process_number = process_number
        self.arrival_time = arrival_time
        self.relative_deadline = relative_deadline
        self.period = period
        self.execution_time = execution_time
        self.finish_times = []

    def get_process_finish_time(self):
        """_"""

        temp = ""
        for time in self.finish_times:
            temp += f"{time} units, "
        return temp

    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

    def __str__(self) -> str:
        return f"P{self.process_number}"

    def __repr__(self) -> str:
        return f"P{self.process_number}"

    def priority(self, algorithm: Literal["RM", "DM"]) -> float:
        """
        If Rm its inversely proportional to 1/period
        If Dm its inversely proportional to 1/relative_deadline

        TODO: need to do it for the leftover

        Args:
            algorithm (RM| DM): define type of algorithm

        Returns:
            float:priority
        """
        if algorithm == "RM":
            return 1 / self.period
        if algorithm == "DM":
            return 1 / self.relative_deadline


class Event:
    """
    Event
    """

    def __init__(self, arrival_time: int, process: Process, dead_line: int):
        self.arrival_time = arrival_time
        self.process = process
        self.dead_line = dead_line
        self.remaining_time = process.execution_time
        self.event_type: EventTypeEnum = EventTypeEnum.arrival
        self.finish_time: int | None = None

    def __lt__(self, other):
        return (
            self.arrival_time < other.arrival_time and self.dead_line < other.dead_line
        )

    def __eq__(self, other) -> bool:
        return self.process == other.process and self.arrival_time == other.arrival_time  # type: ignore

    def __repr__(self) -> str:
        return f"P{self.process.process_number} at AT {self.arrival_time} RT {self.remaining_time} FT {self.finish_time}"

    def __str__(self) -> str:
        return f"P{self.process.process_number} at AT {self.arrival_time} RT {self.remaining_time} FT {self.finish_time}"


class DeadLineNotMeetError(Exception):
    "raises when deadline is not meet"

    def __init__(self, event: Event) -> None:
        self.event = event


class ProcessesHelper:
    """_"""

    @staticmethod
    def get_average_cpu_utilization_time(processes: list[Process], total_time: int):
        """Helps to get average cpu utilization"""
        total_execution_time = sum(
            process.finish_times[-1]
            if process.finish_times
            else 0 - process.arrival_time
            for process in processes
        )
        return (total_execution_time / total_time) * 100

    @staticmethod
    def calculate_total_service_time(processes: list[Process]) -> int:
        """
        Calculate the total service time for a list of processes.

        Args:
            processes (list[Process]): List of processes.

        Returns:
            int: Total service time.
        """
        # Get the last finish time for each process and return the maximum
        # last_finish_times = [
        #    process.finish_times[-1] for process in processes if process.finish_times
        # ]
        # return max(last_finish_times, default=0)
        return (
            max(process.finish_times[-1] for process in processes) if processes else 0
        )

    @staticmethod
    def calculate_total_cpu_time(processes: list[Process]) -> int:
        """
        Calculate the total CPU time for a list of processes.

        Args:
            processes (List[Process]): List of processes.

        Returns:
            int: Total CPU time.
        """
        return sum(process.execution_time for process in processes)


def rate_monotonic_analysis(processes: list[Process]):
    """helps to check weather the process are feasible or not
    Need to verify the above
    """
    if not processes:
        print("Error: No processes provided for rate monotonic analysis.")
        return 0, 0  # Return dummy values

    total_utilization = sum(
        process.execution_time / process.period for process in processes
    )

    # Check if the length of processes is not zero before performing division
    feasibility_threshold = (
        (len(processes) * ((2 ** (1 / len(processes))) - 1))
        if len(processes) > 0
        else 0
    )

    return total_utilization, feasibility_threshold


def check_necessary_condition_for_edf(process: list[Process]):
    """
    helps to check the feasibility condition.

    Args:
        process (list[Process]): _description_
    """
    return sum(process.execution_time / process.period for process in process) < 1


def check_feasibility_condition_for_edf(process: list[Process]):
    """
    helps to check the necessary condition.

    Args:
        process (list[Process]): _description_
    """
    return (
        sum(process.execution_time / process.relative_deadline for process in process)
        < 1
    )


def schedule_rm(
    number_of_processor,
    processes: list[Process],
    process_switch: int,
    verbose: bool,
    detailed: bool,
):
    """TODO: write algo"""

    # first step is to check weather fesabile or not.
    total_utilization, feasibility_threshold = rate_monotonic_analysis(processes)
    print(total_utilization, feasibility_threshold)

    if total_utilization <= feasibility_threshold:
        print("There is no feasible schedule produced.")
        print(
            f"Total Utilization: {total_utilization} < Feasibility_threshold {feasibility_threshold}"
        )
        return
    print("processed")


def schedule_dm(
    number_of_processors: int,
    processes: list[Process],
    process_switch: int,
    verbose: bool,
    detailed: bool,
):
    """
    helps to schedule the dm.

    Args:
        number_of_processors (int): helps to number of processor.
        processes (list[Process]): list of process
        process_switch (int): process switch time
        verbose (bool): to display more log
        detailed (bool): detailed log flag
    """
    print("====================================================")
    print("Deadline Monotonic Scheduling Algorithm(DM):")
    print("====================================================")

    # Initialize the simulation clock, job queue, and event queue
    clock = 0
    job_queue = processes.copy()
    event_queue = []
    finished_queue = []

    # Use heap sort to build a min-heap based on relative deadlines
    heapq.heapify(job_queue)

    # Main simulation loop

    while job_queue:
        # Get the next job from the job queue
        current_job = heapq.heappop(job_queue)

        # Check for deadline miss
        if clock > current_job.relative_deadline:
            print("There is not a feasible schedule.")
            print(f"Schedule can be feasible from time 0 to {clock} units.")
            print(
                f"At time {clock} units, process {current_job.process_number} missed the deadline."
            )

            # Calculate and print statistics
            total_cpu_time_required = ProcessesHelper.calculate_total_service_time(
                processes
            )
            cpu_utilization = ProcessesHelper.get_average_cpu_utilization_time(
                processes, clock
            )
            print(
                f"From 0 to {clock}, Total CPU time required is {total_cpu_time_required} units"
            )
            print(f"CPU Utilization is {cpu_utilization:.1f}%")
            return

        # Schedule the process for execution
        current_job.finish_times.append(clock + current_job.execution_time)

        # Check if the process has completed its execution
        if current_job.execution_time == 0:
            finished_queue.append(current_job)
            continue

        # Update the remaining execution time for the process
        current_job.execution_time -= 1

        # Add the next event for the process to the event queue
        next_event_time = clock + 1
        next_event_deadline = current_job.relative_deadline + next_event_time
        heapq.heappush(
            event_queue, Event(next_event_time, current_job, next_event_deadline)
        )

        # Process events and update the simulation clock
        while event_queue and event_queue[0].arrival_time <= next_event_time:
            current_event = heapq.heappop(event_queue)
            clock = current_event.arrival_time

            # Update the remaining time for the process in the event
            current_event.remaining_time -= 1

            # Add the event back to the event queue if it is not completed
            if current_event.remaining_time > 0:
                heapq.heappush(event_queue, current_event)

        # Print preemption event if verbose mode is enabled
        if job_queue and job_queue[0].relative_deadline < next_event_deadline:
            preempted_process = job_queue[0]
            if verbose:
                print(
                    f"At time {next_event_time}: Process {current_job.process_number} is preempted by process {preempted_process}"
                )

    total_service_time = ProcessesHelper.calculate_total_service_time(processes)

    print("There is feasible schedule produced.")
    print(f"Total Time Required is {next_event_time-1} time units")
    print(
        f"CPU Utilization is {int((total_service_time / (next_event_time - 1)) * 100)} %"
    )
    print("====================================================")

    if detailed:
        print("\nFinal Detailed Information:")
        print("====================================================")
        for process in processes:
            print(f"Process {process}")
            print(f"Arrival time: {process.arrival_time} units")
            print(f"Service time: {process.execution_time} units")
            print(f"Relative Deadline: {process.relative_deadline} units")
            print(f"Period: {process.period} units")
            finished_times_str = [f"{time} units" for time in process.finish_times]
            print(f"Finish times: {', '.join(finished_times_str)}")
            print("====================================================")

    pass


class EdfScheduler:
    """_"""

    def __init__(
        self,
        num_processes: int,
        processes: list[Process],
        process_switch: int,
        verbose: bool,
        detailed: bool,
    ):
        self.current_time = 0
        self.current_event: None | Event = None
        self.finished_events: list[Event] = []
        self.waiting_queue: list[Event] = []
        self.fesable = True
        self.processes = processes
        self.num_process = num_processes
        self.process_switch = process_switch
        self.verbose = verbose
        self.detailed = detailed
        self.lcm = 100

    def initialize_process(self):
        """_"""
        self.print_starting()
        processes = sorted(
            self.processes,
            key=lambda event: (event.arrival_time, event.relative_deadline),
        )
        self.waiting_queue.extend(
            [
                Event(process.arrival_time, process, process.relative_deadline)
                for process in processes
            ]
        )
        self.find_lcm()

    def find_lcm(self):
        self.lcm = get_lcm([process.period for process in self.processes])

    def simulate(self):
        """_"""
        try:
            while self.waiting_queue:
                self.waiting_queue.sort(
                    key=lambda event: (event.arrival_time, event.dead_line)
                )
                current_event = self.waiting_queue.pop(0)
                self.current_time = max(current_event.arrival_time, self.current_time)

                if current_event.event_type == EventTypeEnum.arrival:
                    self.handle_arrival_event(current_event)
                elif current_event.event_type == EventTypeEnum.completed:
                    self.handle_completion(current_event)
                elif current_event.event_type == EventTypeEnum.preemption:
                    self.handle_preemption(current_event)
        except DeadLineNotMeetError as error:
            self.fesable = False
            if self.detailed:
                print(
                    f"There is not a feasible schedule. Schedule can be feasible from time 0 to {self.lcm} units.",
                    f"At time {self.current_time} units",
                    error.event,
                    f"process {current_event.process.process_number} missed the deadline \n",
                )
                # need to print cpu utilization

        self.print_summary()

    def handle_arrival_event(self, event: Event):
        """_"""
        if self.check_preemptive(event):
            event.event_type = EventTypeEnum.preemption
            self.waiting_queue.insert(0, event)
            return
        elif (
            self.current_time + event.remaining_time > event.dead_line
            or self.current_time + event.remaining_time > self.lcm
        ):
            raise DeadLineNotMeetError(event)
        else:
            self.current_time = self.current_time + event.remaining_time
            event.remaining_time = 0
            event.finish_time = self.current_time
            event.process.finish_times.append(self.current_time)
            event.event_type = EventTypeEnum.completed
            self.waiting_queue.insert(0, event)

    def check_preemptive(self, event: Event):
        """check preemptive"""
        preemptive = False
        temp_event = None

        # check if any event can be preempt by another event in waiting que.
        for next_event in self.waiting_queue:
            if (
                next_event.arrival_time <= event.arrival_time + event.remaining_time
                and next_event.dead_line < event.dead_line
            ):
                if temp_event is None:
                    temp_event = next_event
                elif temp_event > next_event:
                    temp_event = next_event
                preemptive = True
        return preemptive

    def get_preemptive_event(self, event: Event):
        # check if any event can be preempt by another event in waiting que.
        temp_event = None
        temp_index = None
        for index, next_event in enumerate(self.waiting_queue):
            if (
                next_event.arrival_time <= event.arrival_time + event.remaining_time
                and next_event.dead_line < event.dead_line
            ):
                if self.verbose:
                    print(f"current event: {event} cam ne preempt by {next_event}")

                if temp_event is None:
                    temp_event, temp_index = next_event, index
                elif temp_event > next_event:
                    temp_event, temp_index = next_event, index
                return next_event

        if temp_event and temp_index:
            self.waiting_queue.pop(temp_index)
            return temp_event
        return None

    def handle_preemption(self, event: Event):
        """_"""
        next_event = self.get_preemptive_event(event)
        event.remaining_time -= next_event.arrival_time - self.current_time
        self.current_time += (
            next_event.arrival_time - self.current_time + self.process_switch
        )
        event.arrival_time = self.current_time + next_event.remaining_time
        event.event_type = EventTypeEnum.arrival
        self.waiting_queue.insert(0, event)
        self.waiting_queue.insert(0, next_event)

    def handle_completion(self, event: Event):
        """_"""

        # to keep track of this event.
        self.finished_events.append(event)

        # generate a new event if its less than lcm.
        # need to check this again.
        if event.arrival_time + event.process.period < self.lcm:
            self.waiting_queue.append(
                Event(
                    event.arrival_time + event.process.period,
                    event.process,
                    event.dead_line + event.process.period,
                )
            )

    def print_starting(self):
        """_"""
        print("Earliest DeadLineFirst (EDF): \n")

    def print_summary(self):
        """_"""
        if self.fesable:
            print("There is feasible schedule produced ")
        processes = self.processes
        for process in processes:
            print(f"Process {process.process_number} \n")
            print(f"arrival time: {process.arrival_time}")
            print(f"service_time: {process.execution_time}")
            print(f"relative deadline: {process.relative_deadline}")
            print(f"period {process.period}")
            print(f"finish time:{process.get_process_finish_time()} \n")


class RateScheduler:
    """_"""

    def __init__(
        self,
        num_processes: int,
        processes: list[Process],
        process_switch: int,
        verbose: bool,
        detailed: bool,
    ):
        self.current_time = 0
        self.current_event: None | Event = None
        self.finished_events: list[Event] = []
        self.waiting_queue: list[Event] = []
        self.fesable = True
        self.processes = processes
        self.num_process = num_processes
        self.process_switch = process_switch
        self.verbose = verbose
        self.detailed = detailed
        self.lcm = 100

    def initialize_process(self):
        """_"""
        self.print_starting()
        processes = sorted(
            self.processes,
            key=lambda process: process.arrival_time,
        )
        self.waiting_queue.extend(
            [
                Event(process.arrival_time, process, process.relative_deadline)
                for process in processes
            ]
        )
        self.find_lcm()

    def find_lcm(self):
        self.lcm = get_lcm([process.period for process in self.processes])

    def simulate(self):
        """_"""
        try:
            while self.waiting_queue:
                self.waiting_queue.sort(
                    key=lambda event: (event.arrival_time, -event.process.priority("RM")),
                )
                current_event = self.waiting_queue.pop(0)
                self.current_time = max(current_event.arrival_time, self.current_time)

                if current_event.event_type == EventTypeEnum.arrival:
                    self.handle_arrival_event(current_event)
                elif current_event.event_type == EventTypeEnum.completed:
                    self.handle_completion(current_event)
                elif current_event.event_type == EventTypeEnum.preemption:
                    self.handle_preemption(current_event)
        except DeadLineNotMeetError as error:
            self.fesable = False
            if self.detailed:
                print(
                    f"There is not a feasible schedule. Schedule can be feasible from time 0 to {self.lcm} units.",
                    f"At time {self.current_time} units",
                    error.event,
                    f"process {current_event.process.process_number} missed the deadline \n",
                )
                # need to print cpu utilization

        self.print_summary()

    def handle_arrival_event(self, event: Event):
        """_"""
        if self.check_preemptive(event):
            event.event_type = EventTypeEnum.preemption
            self.waiting_queue.insert(0, event)
            return
        elif (
            self.current_time + event.remaining_time > event.dead_line
            or self.current_time + event.remaining_time > self.lcm
        ):
            raise DeadLineNotMeetError(event)
        else:
            self.current_time = self.current_time + event.remaining_time
            event.remaining_time = 0
            event.finish_time = self.current_time
            event.process.finish_times.append(self.current_time)
            event.event_type = EventTypeEnum.completed
            self.waiting_queue.insert(0, event)

    def check_preemptive(self, event: Event):
        """check preemptive"""
        preemptive = False
        temp_event = None

        # check if any event can be preempt by another event in waiting que.
        for next_event in self.waiting_queue:
            if (
                next_event.arrival_time <= event.arrival_time + event.remaining_time
                and next_event.process.priority("RM") > event.process.priority("RM")
            ):
                if temp_event is None:
                    temp_event = next_event
                elif temp_event > next_event:
                    temp_event = next_event
                preemptive = True
        return preemptive

    def get_preemptive_event(self, event: Event):
        """_"""
        # check if any event can be preempt by another event in waiting que.
        temp_event = None
        temp_index = None
        for index, next_event in enumerate(self.waiting_queue):
            if (
                next_event.arrival_time <= event.arrival_time + event.remaining_time
                and next_event.process.priority("RM") > event.process.priority("RM")
            ):
                if self.verbose:
                    print(f"current event: {event} cam ne preempt by {next_event}")

                if temp_event is None:
                    temp_event, temp_index = next_event, index
                elif temp_event > next_event:
                    temp_event, temp_index = next_event, index
                return next_event

        if temp_event and temp_index:
            self.waiting_queue.pop(temp_index)
            return temp_event
        return None

    def handle_preemption(self, event: Event):
        """_"""
        next_event = self.get_preemptive_event(event)
        event.remaining_time -= next_event.arrival_time - self.current_time
        self.current_time += (
            next_event.arrival_time - self.current_time + self.process_switch
        )
        event.arrival_time = self.current_time + next_event.remaining_time
        event.event_type = EventTypeEnum.arrival
        self.waiting_queue.insert(0, event)
        self.waiting_queue.insert(0, next_event)

    def handle_completion(self, event: Event):
        """_"""

        # to keep track of this event.
        self.finished_events.append(event)

        # generate a new event if its less than lcm.
        # need to check this again.
        if event.arrival_time + event.process.period < self.lcm:
            self.waiting_queue.append(
                Event(
                    event.arrival_time + event.process.period,
                    event.process,
                    event.dead_line + event.process.period,
                )
            )

    def print_starting(self):
        """_"""
        print("RateMonotonic (RM): \n")

    def print_summary(self):
        """_"""
        if self.fesable:
            print("There is feasible schedule produced ")
        processes = self.processes
        for process in processes:
            print(f"Process {process.process_number} \n")
            print(f"arrival time: {process.arrival_time}")
            print(f"service_time: {process.execution_time}")
            print(f"relative deadline: {process.relative_deadline}")
            print(f"period {process.period}")
            print(f"finish time:{process.get_process_finish_time()} \n")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Scheduling Simulator")

    parser.add_argument(
        "-d", action="store_true", help="Enable detailed information mode"
    )
    parser.add_argument("-v", action="store_true", help="Enable verbose mode")
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=["RM", "DM", "EDF"],
        help="Specify scheduling algorithm (RM, DM, EDF)",
    )

    parser.add_argument("input_file", nargs="?", help="Path to the input file")

    # compile the input args
    args = parser.parse_args()

    # get the detailed, verbose and algorithm.
    detailed = args.d
    verbose = args.v
    algorithm = args.algorithm
    input_file = args.input_file

    # read the lines
    # lines = sys.stdin.readlines()
    if args.input_file:
        with open(input_file, "r") as file:
            lines = file.readlines()
    else:
        lines = sys.stdin.readlines()

    # get the number of processor and processor switch
    num_processes, process_switch = map(int, lines[0].strip().split())

    # Read the process from next lines.
    processes = [Process(*map(int, line.strip().split())) for line in lines[1:]]

    # mapper will trigger the algorithm bassed on execution
    ALGO_MAPPER = {"RM": RateScheduler, "DM": schedule_dm, "EDF": EdfScheduler}

    # algorithm gives us set of algorithm that needs to be performed.
    if algorithm is None:
        algorithm = ALGO_MAPPER.values()
    else:
        algorithm = [ALGO_MAPPER[algorithm]]

    # apply multiple algo sequentially.
    for algo in algorithm:
        if algo in [EdfScheduler, RateScheduler]:
            test = algo(num_processes, processes, process_switch, verbose, detailed)
            test.initialize_process()
            test.simulate()
        algo(num_processes, processes, process_switch, verbose, detailed)


if __name__ == "__main__":
    main()
