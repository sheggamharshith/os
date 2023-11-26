import random


def generate_input_file(
    num_processes, process_switch_time, arrival_distribution="constant"
):
    with open("input_file.txt", "w") as file:
        # Write the number of processes and process switch overhead time
        file.write(f"{num_processes} {process_switch_time}\n")

        for process_number in range(1, num_processes + 1):
            # Generate arrival time based on the specified distribution
            if arrival_distribution == "constant":
                arrival_time = 0
            elif arrival_distribution == "exponential":
                arrival_time = int(random.expovariate(1 / 20))

            # Generate CPU burst, deadline, and period for each process
            cpu_burst = random.randint(0,10)
            deadline = cpu_burst + int(random.uniform(0, 10))
            period = deadline + int(random.uniform(0, 10))

            # Write process details to the file
            file.write(
                f"{process_number} {arrival_time:} {deadline} {period}\n"
            )


if __name__ == "__main__":
    num_processes = 5
    process_switch_time = 3  # or 5 for different switch overhead
    arrival_distribution = "constant"  # or "exponential"

    generate_input_file(num_processes, process_switch_time, arrival_distribution)
