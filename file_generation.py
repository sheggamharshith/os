import os
import random


# Define MaxProcess, change this value according your need.
MAX_PROCESSES = 10
# hardcoded the atleaset processor to min 5


class FileGeneration:
    """
    Helps to generate input file.
    """

    def __init__(self):
        self.file_name = None
        self.total_proc = 0
        self.arrival_time = None
        self.burst_time = None
        self.relative_deadline = None
        self.period = None

    def generate_input_file(self, file_name, is_distributed):
        self.total_proc = random.randint(5, MAX_PROCESSES - 6) + 5
        self.set_file_name(file_name)

        if self.file_exists(self.file_name):
            print("File already exists")
        else:
            print("Generating file ......")
            self.generate_file(is_distributed)
            print("file Generated")

    def generate_input_file_custom(
        self,
        file_name,
        total_proc,
        is_distributed,
    ):
        """
        Helps to generate file with custom total number of processor.

        Args:
            file_name (str): file name
            total_proc (int): total number of processor.
            is_distributed (bool): is distributed or not.
        """
        self.set_file_name(file_name)
        self.set_total_proc(total_proc)

        if total_proc >= 5:
            if self.file_exists(self.file_name):
                print("File already exists")
            else:
                print("Generating file ......")
                self.generate_file(is_distributed)
                print("File Generated")

    def generate_file(self, is_distributed):
        with open(self.file_name, "w", encoding="utf-8") as output_file:
            if is_distributed:
                proc_switch = 5
            else:
                proc_switch = 0

            output_file.write(f"{self.total_proc} {proc_switch}\n")
            print(f"{self.total_proc} {proc_switch}")

            for proc_num in range(1, self.total_proc + 1):
                if is_distributed:
                    self.randomize_dist()
                else:
                    self.randomize_not_dist()

                output_file.write(
                    f"{proc_num} {self.arrival_time} {self.burst_time} {self.relative_deadline} {self.period}\n"  # noqa:E501
                )
                print(
                    f"{proc_num} {self.arrival_time} {self.burst_time} {self.relative_deadline} {self.period}"  # noqa:E501
                )

    def set_file_name(self, file_name):
        self.file_name = f"input_files/{file_name}"

    def set_total_proc(self, total_proc):
        self.total_proc = total_proc

    def file_exists(self, file_name):
        return os.path.exists(file_name)

    def randomize_dist(self):
        arrival_time = random.expovariate(1) * 20
        self.arrival_time = round(arrival_time, 2)
        self.randomize_not_dist()

    def randomize_not_dist(self):
        self.burst_time = random.randint(30, 100)
        x = random.randint(1, 500)
        self.relative_deadline = self.burst_time + x
        y = random.randint(1, 1000)
        self.period = self.relative_deadline + y


if __name__ == "__main__":
    # Example usage
    MAX_PROCESSES = 10
    file_generation = FileGeneration()
    file_generation.generate_input_file("example.txt", is_distributed=True)
    file_generation.generate_input_file_custom(
        "example_custom.txt", total_proc=8, is_distributed=False
    )
