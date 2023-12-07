import argparse
import sys
from typing import Literal
from math import gcd
import heapq
from collections import defaultdict


def find_lcm(values: list):
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

    def __lt__(self, other):
        return self.period < other.period

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
    Eevent
    """

    def __init__(self, arrival_time: int, process: Process, dead_line: int):
        self.arrival_time = arrival_time
        self.process = process
        self.dead_line = dead_line
        self.remaining_time = 1
        self.finish_time: int | None = None

    def __lt__(self, other):
        return self.dead_line < other.dead_line

    def __eq__(self, other) -> bool:
        return self.process == other.process and self.arrival_time == other.arrival_time  # type: ignore

    def __repr__(self) -> str:
        return f"P{self.process.process_number} at AT {self.arrival_time} RT {self.remaining_time} FT {self.finish_time}"

    def __str__(self) -> str:
        return f"P{self.process.process_number} at AT {self.arrival_time} RT {self.remaining_time} FT {self.finish_time}"


class ProcessesHelper:
    """_"""

    @staticmethod
    def get_total_scheduling_time(processes: list[Process]):
        """
        Helps to calculate the total scheduling time. This will helps ous to run the process
        in nu
        """
        arrival_times = [process.arrival_time for process in processes]
        process_periods = [process.period for process in processes]

        #  if the arrival time is same then follow the lcm method
        isSame = all(
            [arrival_times[0] == arrival_time for arrival_time in arrival_times]
        )

        if isSame:
            return arrival_times[0] + find_lcm(process_periods)
        else:
            return max(arrival_times) + 2 * find_lcm(process_periods)

    @staticmethod
    def get_process_availability_time(processes: list[Process]):
        """
        helps to get the availabilty of process
        """
        total_schedule_time = ProcessesHelper.get_total_scheduling_time(processes)
        result = defaultdict(list)  # type:ignore

        for process in processes:
            arrival_time = process.arrival_time
            period = process.period
            period_iteration = 0
            available_time = arrival_time + (period_iteration * period)
            while available_time <= total_schedule_time:
                result[process.process_number].append(available_time)
                period_iteration += 1
                available_time = arrival_time + (period_iteration * period)

        return result

    @staticmethod
    def get_process_deadline_time(processes: list[Process]):
        """
        Helps to get deadline
        """
        process_availability_timings = ProcessesHelper.get_process_availability_time(
            processes
        )
        result = defaultdict(list)  # type:ignore

        for process in processes:
            deadline_time = process.relative_deadline + process.arrival_time
            period = process.period

            while len(result[process.process_number]) != len(
                process_availability_timings[process.process_number]
            ):
                result[process.process_number].append(deadline_time)
                deadline_time += period
        return result

    @staticmethod
    def get_average_cpu_utilization_time(processes: list[Process]):
        """ "Helps to get average cpu utilization"""
        cpu_utilization = 0
        for process in processes:
            cpu_utilization += (
                process.finish_times[0]
                if process.finish_times
                else 0 - process.arrival_time
            ) / process.period
        return cpu_utilization * 100

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

    current_time = 1
    current_event = None
    finished_events: list[Event] = []
    waiting_queue: list[tuple[int, Event]] = []
    finished_events: list[Event] = []
    feasible = True
    # List to store missed deadlines
    missed_deadlines = []

    execution_time = ProcessesHelper.get_total_scheduling_time(processes)
    processes_availability_time = ProcessesHelper.get_process_availability_time(
        processes
    )
    processes_deadline_time = ProcessesHelper.get_process_deadline_time(processes)

    while current_time <= execution_time:
        if current_time <= execution_time:
            for process in processes:
                try:
                    if (
                        processes_availability_time[process.process_number][0]
                        <= current_time
                    ) and (current_time - 1) + process.execution_time <= execution_time:
                        temp_event = Event(
                            processes_availability_time[process.process_number][0],
                            process,
                            processes_deadline_time[process.process_number][0],
                        )
                        del processes_availability_time[process.process_number][0]
                        del processes_deadline_time[process.process_number][0]

                        heapq.heappush(
                            waiting_queue,
                            (
                                temp_event.dead_line,
                                temp_event,
                            ),
                        )
                        del temp_event
                except IndexError:
                    pass

        if not waiting_queue and current_event is None:
            current_time = current_time + 1
            continue

        if waiting_queue:
            absolute_deadline, future_event = heapq.heappop(waiting_queue)
            if current_event is None:
                current_event = future_event
            elif future_event < current_event:
                current_time = current_time + process_switch
                heapq.heappush(
                    waiting_queue,
                    (
                        absolute_deadline,
                        current_event,
                    ),
                )
                current_event = future_event
                if verbose:
                    print(
                        f"At time {current_time}: Process {current_event.process.process_number} is preempted by process with {future_event.process.process_number}"
                    )
                continue
            else:
                heapq.heappush(waiting_queue, (absolute_deadline, future_event))

            if current_event.dead_line >= current_time:
                current_event.remaining_time -= 1
            else:
                missed_deadlines.append((current_time, current_event.process))
                feasible = False
                break

            if current_event.remaining_time == 0:
                current_event.finish_time = current_time
                if verbose:
                    print(
                        f"Process {current_event.process.process_number} finished at time {current_time}"
                    )
                current_event.process.finish_times.append(current_time)
                finished_events.append(current_event)
                current_event = None

        elif current_event:
            if current_event.dead_line >= current_time:
                current_event.remaining_time -= 1
            else:
                missed_deadlines.append((current_time, current_event.process))
                feasible = False
                break

            if current_event.remaining_time == 0:
                current_event.finish_time = current_time
                if verbose:
                    print(
                        f"Process {current_event.process.process_number} finished at time {current_time}"
                    )
                current_event.process.finish_times.append(current_time)
                finished_events.append(current_event)
                current_event = None

        current_time += 1

    # Calculate total service time using helper method
    total_service_time = ProcessesHelper.calculate_total_service_time(processes)

    # Check feasibility using the total service time
    if total_service_time > execution_time:
        print("There is feasible schedule produced.")
        print(f"Total Time Required is {current_time-1} time units")
        print(
            f"CPU Utilization is {int(ProcessesHelper.get_average_cpu_utilization_time(processes))} %"
        )
        print("====================================================")
    else:
        print(
            f"There is not a feasible schedule. Schedule can be feasible from time 0 to {total_service_time}."
        )
        for missed_time, missed_process in missed_deadlines:
            print(
                f"At time {missed_time} units: process {missed_process} missed the deadline."
            )
        print(
            f"From time 0 to {total_service_time}, Total CPU time required is INPUT time units"
        )
        print(
            f"Average Cpu Utilization is {int(ProcessesHelper.get_average_cpu_utilization_time(processes))} %"
        )
        print("====================================================")

    if detailed:
        print("\nFinal Detailed Information:")
        print("====================================================")
        for process in processes:
            print(f"Process {process}")
            print(f"Arrival time {process.arrival_time} units")
            # print(
            #    f"Service time: {process.finish_times[-1] - process.arrival_time} units"
            # )
            print(f"Relative Deadline {process.relative_deadline} units")
            print(f"Period: {process.period} units")
            finished_times_str = [f"{time} units" for time in process.finish_times]
            print(f"Finish time: {finished_times_str}")
            print("====================================================")

    pass


def schedule_edf(
    number_of_processors: int,
    processes: list[Process],
    process_switch: int,
    verbose: bool,
    detailed: bool,
):
    """ """
    print("====================================================")
    print("Earliest Deadline First(EDF):")

    current_time = 1
    current_event = None
    finished_events: list[Event] = []
    waiting_queue: list[tuple[int, Event]] = []
    fesable = True

    execution_time = ProcessesHelper.get_total_scheduling_time(processes)
    processes_availability_time = ProcessesHelper.get_process_availability_time(
        processes
    )
    processes_deadline_time = ProcessesHelper.get_process_deadline_time(processes)

    # TODO: write fessabilty check.

    feasibility = check_feasibility_condition_for_edf(processes)
    necessary = check_necessary_condition_for_edf(processes)

    if not feasibility and not necessary:
        print(
            "Fesibility and necessary condition not stisfied, checking condition with lcm"
        )

    if not feasibility and necessary:
        # keep it provides why it can be failed
        print("If it satisfies necessary condition, but fails sufficient condition")

    while current_time <= execution_time:
        # check if there is any arrived process and push them into the waiting heap-min que
        if current_time <= execution_time:
            for process in processes:
                # since the timings are sorted we can directly check the least time with current time.
                try:
                    if (
                        processes_availability_time[process.process_number][0]
                        <= current_time
                    ) and (current_time - 1) + process.execution_time <= execution_time:
                        temp_event = Event(
                            processes_availability_time[process.process_number][0],
                            process,
                            processes_deadline_time[process.process_number][0],
                        )
                        # similar to poplet you can use the delete at 0 bothe deadline and arrival time
                        del processes_availability_time[process.process_number][0]
                        del processes_deadline_time[process.process_number][0]

                        # push the event to the waiting list array.
                        heapq.heappush(
                            waiting_queue,
                            (
                                # then the new realtive deadline will the  +
                                temp_event.dead_line,
                                temp_event,
                            ),
                        )
                        del temp_event
                # condition where there is no process available
                except IndexError:
                    pass
        # check for any ideal time in the processing.
        if not waiting_queue and current_event is None:
            current_time = current_time + 1
            continue

        # execute the task in waiting que
        if waiting_queue:
            relative_deadline, future_event = heapq.heappop(waiting_queue)
            #  condition to check if new event is required or not.
            if current_event is None:
                current_event = future_event
            elif future_event < current_event:
                if verbose:
                    print(
                        f"At time {current_time}: Process {current_event} with {future_event.process}"
                    )
                # This will skip time to future with adding process switch.
                # Since the program should be ideal at this process switch.
                current_time = current_time + process_switch
                heapq.heappush(
                    waiting_queue,
                    (
                        current_event.dead_line,
                        current_event,
                    ),
                )
                current_event = future_event
                continue
            else:
                heapq.heappush(waiting_queue, (relative_deadline, future_event))

            # check if it can schedule or not
            if current_event.dead_line >= current_time:
                current_event.remaining_time -= 1
            else:
                if detailed:
                    print(
                        f"There is not a feasible schedule.Schedule can be feasible from time 0 to {current_time}"
                    )
                    print(
                        f"At time {current_time} units, process {current_event} missed thedeadline."
                    )
                print("There is no feasible schedule produced.")
                print("====================================================")
                fesable = False
                break
            # check if remaining time is 0
            if current_event.remaining_time == 0:
                current_event.finish_time = current_time
                current_event.process.finish_times.append(current_time)
                finished_events.append(current_event)
                current_event = None

        elif current_event:
            # triggred when last event
            # check if it can schedule or not
            if current_event.dead_line > current_time - 1:
                current_event.remaining_time -= 1
            else:
                if detailed:
                    print(
                        f"There is not a feasible schedule.Schedule can be feasible from time 0 to {current_time}"
                    )
                    print(
                        f"At time {current_time} units, process {current_event} missed thedeadline."
                    )
                print("There is no feasible schedule produced.")
                print("====================================================")
                fesable = False
                break

            # check if remaining time is 0
            if current_event.remaining_time == 0:
                current_event.finish_time = current_time
                current_event.process.finish_times.append(current_time)
                finished_events.append(current_event)
                current_event = None

        current_time += 1

    if fesable:
        print("There is feasible schedule produced.")
        print(f"Total Time Required is {current_time-1} time units")
        print(
            f"Average Cpu Utilization is {int(ProcessesHelper.get_average_cpu_utilization_time(processes))} %"
        )
        print("====================================================")

    if detailed:
        for process in processes:
            print("====================================================")
            print(f"Process {process}")
            print(f"Arrival time {process.arrival_time} units")
            print(f"relative DeadLine {process.relative_deadline} units")
            print(f"period: {process.period} units")
            print(f"finish time: {process.finish_times}")
            print("====================================================")


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
    ALGO_MAPPER = {"RM": schedule_rm, "DM": schedule_dm, "EDF": schedule_edf}

    # algorithm gives us set of algorithm that needs to be performed.
    if algorithm is None:
        algorithm = ALGO_MAPPER.values()
    else:
        algorithm = [ALGO_MAPPER[algorithm]]

    # apply multiple algo sequentially.
    for algo in algorithm:
        algo(num_processes, processes, process_switch, verbose, detailed)


if __name__ == "__main__":
    main()
