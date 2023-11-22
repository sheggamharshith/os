import heapq
import sys
import argparse


class Event:
    def __init__(self, time, event_type, process_number):
        self.time = time
        self.event_type = event_type
        self.process_number = process_number

    def __lt__(self, other):
        return self.time < other.time


class Process:
    def __init__(self, process_number, arrival_time, relative_deadline, period):
        self.process_number = process_number
        self.arrival_time = arrival_time
        self.relative_deadline = relative_deadline
        self.period = period
        self.remaining_time = period
        #self.deadline = arrival_time + relative_deadline
        self.cpu_burst = 0 #this will be generated per the given distribution

    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

class Scheduler:
    def __init__(self, process, process_switch_time, algorithm):
        self.clock = 0
        self.event_queue = []
        self.processes = processes
        self.process_switch_time = process_switch_time
        self.algorithm = algorithm
        self.cpu_optimization = 0
        self.time_limit = float('inf')

    def schedule(self):
        while self.clock < self.time_limit and self.event_queue:
            current_event = heapq.heappop(self.event_queue)
            self.clock = current_event.time

            if current_event.event_type == 'arrival':
                self.handle_arrival(current_event.process_number)
            elif current_event.event_type == 'completion':
                self.handle_completion(current_event.process_number)
        
        self.calculate_statistics()

    def handle_arrival(self, process_number):
        # Implement arrival event handling here based on the scheduling algorithm
        pass

    def handle_completion(self, process_number):
        # Implement completion event handling here based on the scheduling algorithm
        pass

    def calculate_statistics(self):
        # Implement code to calculate and print statistics
        pass

def generate_processes(num_processes, process_switch_time):
    # Implement code to generate processes based on the given distribution
    pass

def parse_input_file(file_path):
    # Implement code to parse input from the file
    pass

def schedule_rm(number_of_processor, processes: list[Process], process_switch: int, verbose: bool, detailed: bool,):
    """TODO: write algo"""

    pass


def schedule_dm(number_of_processor, processes, process_switch, verbose: bool, detailed: bool,):
    """
    TODO: Write algo
    Args:
        processes (_type_): _description_
        process_switch (_type_): _description_
        verbose (bool): _description_
        detail (bool): _description_
    """
    pass


def schedule_edf(number_of_processor, processes, process_switch, verbose: bool, detailed: bool,):
    """
    TODO: write algo

    Args:
        processes (_type_): _description_
        process_switch (_type_): _description_
        verbose (bool): _description_
        detail (bool): _description_
    """
    pass


def simulate(processes, process_switch, scheduling_algorithm):
    """
    TODO: write simulation algo.
    """


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Scheduling Simulator")
    parser.add_argument("-d", action="store_true", help="Enable detailed information mode")
    parser.add_argument("-v", action="store_true", help="Enable verbose mode")
    parser.add_argument("-a", "--algorithm", choices=["RM", "DM", "EDF"], help="Specify scheduling algorithm (RM, DM, EDF)",)
    parser.add_argument("input_file", type=str, help="Path to the input file")

    processes, process_switch_time = parse_input_file(args.input_file)

    scheduler = Scheduler(processes, process_switch_time, args.a)

    # compile the input args
    args = parser.parse_args()

    if args.d:
        print("Detailed Information Mode Running")
    if args.v:
        print("Verbose Mode Running")


    # get the detailed, verbose and algorithm.
    detailed = args.d
    verbose = args.v
    algorithm = args.algorithm

    scheduler.schedule()
    
   '''
    # read the lines
    lines = sys.stdin.readlines()

    print(detailed, verbose, algorithm, lines)

    # get the number of processor and processor switch
    num_processes, process_switch = map(int, lines[0].strip().split())

    # Read the process from next lines.
    processes = [Process(*map(int, line.strip().split())) for line in lines[1:]]

    # mapper will trigger the algorithm bassed on execution
    ALGO_MAPPER = {"RM": schedule_rm, "DM": schedule_dm, "EDF": schedule_edf}
    ALGO_MAPPER[algorithm](
        num_processes,
        processes,
        process_switch,
        verbose,
        detailed,
    )

    # need to write the overall simulation method over here.
    '''


if __name__ == "__main__":
    main()
