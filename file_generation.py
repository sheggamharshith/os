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
            execution_time = int(random.uniform(1, 10))
            deadline = execution_time + int(random.uniform(1, 10))
            period = deadline + int(random.uniform(1, 10))

            # Write process details to the file
            file.write(
                f"{process_number} {arrival_time} {execution_time} {deadline} {period}\n"
            )


if __name__ == "__main__":
    num_processes = random.randint(5, 10)
    process_switch_time = random.randint(0, 5)  # or 5 for different switch overhead
    arrival_distribution = random.choice(["constant", "exponential"])  # picks random

    generate_input_file(num_processes, process_switch_time, arrival_distribution)
