import sys
import argparse
from typing import Literal
from math import gcd
import heapq
from collections import deque


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
        self.process: Process = process

    def __lt__(self, other):
        return self.time < other.time

    def __eq__(self, __value: object) -> bool:
        return self.process == __value.process


class Processor:
    """
    Processor indicates the process number. also has meta data such as is busy, what's the current task is working on.
    """

    def __init__(self, number, current_task=None, busy=False) -> None:
        self.number = number
        self.is_busy = busy
        self.current_task = current_task
        self.last_scheduled_time = 100


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
        self.execution_time = arrival_time + relative_deadline
        self.remaining_time = self.execution_time

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

    @property
    def deadline(self):
        return self.arrival_time + self.relative_deadline


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
        len(processes) * (2 ** (1 / len(processes)) - 1) if len(processes) > 0 else 0
    )

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
    number_of_processors,
    processes: list[Process],
    process_switch,
    verbose: bool,
    detailed: bool,
):
    current_time = 0
    process_queue = processes
    current_event = None
    processed_events: list[Event] = []
    finished_events: list[Event] = []
    waiting_queue: list[tuple[str, Event]] = []

    # Perform rate monotonic analysis to check feasibility
    total_utilization, feasibility_threshold = rate_monotonic_analysis(processes)

    if total_utilization > feasibility_threshold:
        print("There is not a feasible schedule.")
        print(
            f"Schedule can be feasible from time 0 to {find_lcm([process.period for process in processes])} units."
        )

        # Initialize missed_deadline_process variable
        missed_deadline_process = None

        # Loop through processes to find missed deadline
        for process in processes:
            if current_time >= process.deadline:
                missed_deadline_process = process.process_number
                break

        # Print missed deadline after the loop
        if missed_deadline_process is not None:
            print(
                f"At time {current_time}, process {missed_deadline_process} missed the deadline."
            )
        print(
            f"From 0 to {find_lcm([process.period for process in processes])}, Total CPU time required is {total_utilization * find_lcm([process.period for process in processes]):.2f} units."
        )
        print(f"CPU Utilization is {total_utilization * 100:.1f}%")
    else:
        print("There is a feasible schedule produced.")
        print(
            f"Total CPU time required is {total_utilization * find_lcm([process.period for process in processes]):.2f} units."
        )
        print(f"CPU Utilization is {total_utilization * 100:.1f}%")

    while len(waiting_queue) > 1 or (len(finished_events) < len(process_queue)):
        for process in process_queue:
            if process.arrival_time <= current_time:
                event = Event(process.arrival_time, "arrival", process)
                if event not in processed_events:
                    heapq.heappush(
                        waiting_queue,
                        (
                            process.deadline,  # Use absolute deadline for DM
                            event,
                        ),
                    )
                    processed_events.append(event)
                    if verbose:
                        print(
                            f"At time {current_time}: Process {process.process_number} arrived"
                        )

        if waiting_queue is None and process_queue is not None:
            current_time = current_time + 1
            if verbose:
                print("Process is idle at this time")
            continue

        if waiting_queue:
            absolute_deadline, event = heapq.heappop(waiting_queue)
            if current_event is None:
                current_event = event
            elif current_event.process.deadline > event.process.deadline:
                current_time = current_time + process_switch
                heapq.heappush(
                    waiting_queue,
                    (
                        absolute_deadline,
                        current_event,
                    ),
                )
                current_event = event
                if verbose:
                    print(
                        f"At time {current_time}: Process {current_event.process.process_number} is preempted by process {event.process.process_number}"
                    )
                continue
            else:
                heapq.heappush(waiting_queue, (absolute_deadline, event))

            task_start_time = max(current_time, current_event.time)

            if task_start_time < current_event.process.deadline:
                current_event.process.remaining_time -= 1
            else:
                print(
                    f"Missed a deadline for process {current_event.process.process_number} at time {current_time}"
                )
                break

            if current_event.process.remaining_time == 0:
                if verbose:
                    print(
                        f"Process {current_event.process.process_number} finished at time {current_time}"
                    )
                finished_events.append(current_event)
                current_event = None

        elif current_event:
            if task_start_time < current_event.process.deadline:
                current_event.process.remaining_time -= 1
            else:
                print(f"Missed a deadline for process {current_time}")
                break

            if current_event.process.remaining_time == 0:
                if verbose:
                    print(
                        f"Process {current_event.process.process_number} finished at time {current_time}"
                    )
                finished_events.append(current_event)
                current_event = None

        current_time += 1

    if detailed:
        print("\nFinal Detailed Information:")
        total_time_required = max(event.time for event in finished_events)
        cpu_utilization = total_utilization * 100
        print(f"\nTotal CPU time required is {total_time_required:.2f} units")
        print(f"CPU Utilization is {cpu_utilization:.1f}%")
        for event in finished_events:
            if event.process is not None:
                process = event.process
                print(f"Process {process.process_number}:")
                print(f"  arrival time: {process.arrival_time}")
                print(
                    f"  service time: {process.execution_time - process.remaining_time} units"
                )
                print(f"  relative deadline: {process.relative_deadline} units")
                print(f"  period: {process.period} units")
                print(f"  finish time: {event.time} units")

    pass


def schedule_edf(
    number_of_processors,
    processes: list[Process],
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
    current_time = 0
    # need to move it to the LCM
    max_time = 20
    process_queue = processes
    current_event = None
    processed_events: list[Event] = []
    finished_events: list[Event] = []
    waiting_queue: list[tuple[str, Event]] = []

    while len(waiting_queue) > 1 or (len(finished_events) < len(process_queue)):
        # check if there is any arraived process and push them into the wating heapmin que
        for process in process_queue:
            if process.arrival_time <= current_time:
                event = Event(process.arrival_time, "arrival", process)
                if event not in processed_events:
                    heapq.heappush(
                        waiting_queue,
                        (
                            process.relative_deadline,
                            event,
                        ),
                    )
                    processed_events.append(event)
                    if verbose:
                        print(
                            f"At time {current_time}: Process {process.process_number} arrived"
                        )

        if waiting_queue is None and process_queue is not None:
            current_time = current_time + 1
            if verbose:
                print("process is idle at this time")
            continue

        # execute the task in waiting que
        if waiting_queue:
            relative_deadline, event = heapq.heappop(waiting_queue)

            #  condition to check if new event is required or not.
            if current_event is None:
                current_event = event
            elif (
                current_event.process.relative_deadline
                > event.process.relative_deadline
            ):
                current_time = current_time + process_switch
                heapq.heappush(
                    waiting_queue,
                    (
                        relative_deadline,
                        current_event,
                    ),
                )
                current_event = event
                if verbose:
                    print(
                        f"At time {current_time}: Process {current_event.process.process_number} is preempted by process {event.process.process_number}"
                    )
                continue
            else:
                heapq.heappush(waiting_queue, (relative_deadline, event))

            task_start_time = max(current_time, current_event.time)

            # check if it can schedule or not
            if task_start_time < 100000:
                current_event.process.remaining_time -= 1

            else:
                print(f"missed a deadline for process {current_time}")
                break
            # check if remaining time is 0
            if current_event.process.remaining_time == 0:
                if verbose:
                    print(
                        f"process {current_event.process.process_number} finished at time {current_time}"
                    )
                finished_events.append(current_event)
                current_event = None

        elif current_event:
            # triggred when last event
            # check if it can schedule or not
            if task_start_time < 100000:
                current_event.process.remaining_time -= 1
            else:
                print(f"missed a deadline for process {current_time}")
                break

            # check if remaining time is 0
            if current_event.process.remaining_time == 0:
                if verbose:
                    print(
                        f"process {current_event.process.process_number} finished at time {current_time}"
                    )
                finished_events.append(current_event)
                current_event = None

        current_time += 1

    if detailed:
        print("\nFinal Detailed Information:")
        for event in finished_events:
            if event.process is not None:  # check
                process = event.process
                print(f"Process {process.process_number}:")
                print(f"  arrival time: {process.arrival_time}")
                print(
                    f"  service time: {process.execution_time - process.remaining_time} units"
                )
                print(f"  relative deadline: {process.relative_deadline} units")
                print(f"  period: {process.period} units")
                print(f"  finish time: {event.time} units")


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

    parser.add_argument("input_file", help="Path to the input file")

    # compile the input args
    args = parser.parse_args()

    # get the detailed, verbose and algorithm.
    detailed = args.d
    verbose = args.v
    algorithm = args.algorithm
    input_file = args.input_file

    # read the lines
    # lines = sys.stdin.readlines()

    with open(input_file, "r") as file:
        lines = file.readlines()

    print(detailed, verbose, algorithm, lines)

    # get the number of processor and processor switch
    num_processes, process_switch = map(int, lines[0].strip().split())

    # Read the process from next lines.
    processes = [Process(*map(int, line.strip().split())) for line in lines[1:]]

    print("Processes:", processes)  # Added this line for debugging

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
