# Makefile

.PHONY: run

ARGS_RM = -a RM
ARGS_DM = -a DM
ARGS_EDF = -a EDF

INPUT_FILE_FEASIBLE = feasible.txt
INPUT_FILE_NOT_FEASIBLE = not_feasible.txt
INPUT_FILE_PREEMPTION = preemption.txt

TARGET ?= run_RM

# Targets for feasible input file
run_ALL:
	python3 main.py < $(INPUT_FILE_FEASIBLE)

run_ALLDetailed:
	python3 main.py -d < $(INPUT_FILE_FEASIBLE)

run_RMDetailed:
	python3 main.py -d $(ARGS_RM) < $(INPUT_FILE_FEASIBLE)

run_DMDetailed:
	python3 main.py -d $(ARGS_DM) < $(INPUT_FILE_FEASIBLE)

run_EDFDetailed:
	python3 main.py -d $(ARGS_EDF) < $(INPUT_FILE_FEASIBLE)

run_RMVerbose:
	python3 main.py -v $(ARGS_RM) < $(INPUT_FILE_FEASIBLE)

run_DMVerbose:
	python3 main.py -v $(ARGS_DM) < $(INPUT_FILE_FEASIBLE)

run_EDFVerbose:
	python3 main.py -v $(ARGS_EDF) < $(INPUT_FILE_FEASIBLE)

# Targets for not feasible input file
run_NOT_FEASIBLE_ALL:
	python3 main.py < $(INPUT_FILE_NOT_FEASIBLE)

run_NOT_FEASIBLE_ALLDetailed:
	python3 main.py -d < $(INPUT_FILE_NOT_FEASIBLE)

run_NOT_FEASIBLE_RMDetailed:
	python3 main.py -d $(ARGS_RM) < $(INPUT_FILE_NOT_FEASIBLE)

run_NOT_FEASIBLE_DMDetailed:
	python3 main.py -d $(ARGS_DM) < $(INPUT_FILE_NOT_FEASIBLE)

run_NOT_FEASIBLE_EDFDetailed:
	python3 main.py -d $(ARGS_EDF) < $(INPUT_FILE_NOT_FEASIBLE)

run_NOT_FEASIBLE_RMVerbose:
	python3 main.py -v $(ARGS_RM) < $(INPUT_FILE_NOT_FEASIBLE)

run_NOT_FEASIBLE_DMVerbose:
	python3 main.py -v $(ARGS_DM) < $(INPUT_FILE_NOT_FEASIBLE)

run_NOT_FEASIBLE_EDFVerbose:
	python3 main.py -v $(ARGS_EDF) < $(INPUT_FILE_NOT_FEASIBLE)

# Targets for preemption input file
run_PREEMPTION_ALL:
	python3 main.py -v < $(INPUT_FILE_PREEMPTION)

run_PREEMPTION_RMVerbose:
	python3 main.py -v $(ARGS_RM) < $(INPUT_FILE_PREEMPTION)

run_PREEMPTION_DMVerbose:
	python3 main.py -v $(ARGS_DM) < $(INPUT_FILE_PREEMPTION)

run_PREEMPTION_EDFVerbose:
	python3 main.py -v $(ARGS_EDF) < $(INPUT_FILE_PREEMPTION)


