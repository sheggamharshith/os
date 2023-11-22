import sys
import argparse
from typing import Literal
from math import gcd
import heapq


def find_lcm(values: list):
    """helps to find lcm in the given processors"""
    lcm = 1
    for i in values:
        lcm = lcm * i // gcd(lcm, i)
    return lcm


class Event:
    """
    Eevent
    """

    def __init__(self, time, event_type, process):
        self.time = time
        self.event_type = event_type
        self.process = process

    def __lt__(self, other):
        return self.time < other.time


class Processor:
    """
    Processor indicates the process number. also has meta data such as is busy, what's the current task is working on.
    """

    def __init__(self, number, current_task=None, busy=False) -> None:
        self.number = number
        self.is_busy = busy
        self.current_task = current_task


class Process:
    """
    Process indicates the process what kind of task it is. don't get confuse between the process and processor.
    consider process also to be a task depending on situation. process can be directly attached to the processor task
    """

    def __init__(self, process_number, arrival_time, relative_deadline, period):
        self.process_number = process_number
        self.arrival_time = arrival_time
        self.relative_deadline = relative_deadline
        self.period = period
        self.execution_time = None

    def __lt__(self, other):
        return self.period < other.period

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


def rate_monotonic_analysis(processes: list[Process]):
    """helps to check weather the process are feasible or not
    Need to verify the above
    """
    total_utilization = sum(
        process.execution_time / process.period for process in processes
    )
    feasibility_threshold = len(processes) * (2 ** (1 / len(processes)) - 1)

    return total_utilization, feasibility_threshold


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
    number_of_processor,
    processes,
    process_switch,
    verbose: bool,
    detailed: bool,
):
    """
    TODO: Write algo
    Args:
        processes (_type_): _description_
        process_switch (_type_): _description_
        verbose (bool): _description_
        detail (bool): _description_
    """
    pass


def schedule_edf(
    number_of_processor,
    processes,
    process_switch,
    verbose: bool,
    detailed: bool,
):
    """
    TODO: write algo

    Args:
        processes (_type_): _description_
        process_switch (_type_): _description_
        verbose (bool): _description_
        detail (bool): _description_
    """
    pass


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

    # compile the input args
    args = parser.parse_args()

    # get the detailed, verbose and algorithm.
    detailed = args.d
    verbose = args.v
    algorithm = args.algorithm

    # read the lines
    lines = sys.stdin.readlines()

    print(detailed, verbose, algorithm, lines)

    # get the number of processor and processor switch
    num_processes, process_switch = map(int, lines[0].strip().split())

    # Read the process from next lines.
    processes = [Process(*map(int, line.strip().split())) for line in lines[1:]]

    # mapper will trigger the algorithm bassed on execution
    ALGO_MAPPER = {"RM": schedule_rm, "DM": schedule_dm, "EDF": schedule_edf}

    # algorithm gives us set of algorithm that needs to be performed.
    if algorithm is None:
        algorithm = ALGO_MAPPER.values()
    else:
        algorithm = [ALGO_MAPPER[algorithm]]

    # apply multiple algo sequentially.
    for algo in algorithm:
        algo(num_processes, processes, process_switch, verbose, detailed)

    # Finished the analysis.
    print("analysis finished")


if __name__ == "__main__":
    main()
