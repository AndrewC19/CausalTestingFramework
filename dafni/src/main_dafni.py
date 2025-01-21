"""

Entrypoint script to run the causal testing framework on DAFNI

"""

from pathlib import Path
import argparse
import json
import pandas as pd
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_test_outcome import Positive, Negative, NoEffect, SomeEffect
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.json_front.json_class import JsonUtility
from causal_testing.specification.causal_dag import CausalDAG


class ValidationError(Exception):
    """
    Custom class to capture validation errors in this script
    """


def get_args(test_args=None) -> argparse.Namespace:
    """
    Function to parse arguments from the user using the CLI
    :param test_args: None
    :returns:
            - argparse.Namespace - A Namsespace consisting of the arguments to this script
    """
    parser = argparse.ArgumentParser(description="A script for running the CTF on DAFNI.")

    parser.add_argument("--data_path", required=True, help="Path to the input runtime data (.csv)", nargs="+")

    parser.add_argument(
        "--tests_path", required=True, help="Input configuration file path " "containing the causal tests (.json)"
    )

    parser.add_argument(
        "-i", "--ignore_cycles", action="store_true", help="Whether to ignore cycles in the DAG.", default=False
    )

    parser.add_argument(
        "--dag_path",
        required=True,
        help="Input configuration file path containing a valid DAG (.dot). "
        "Note: this must be supplied if the --tests argument isn't provided.",
    )

    parser.add_argument("--output_path", required=False, help="Path to the output directory.")

    parser.add_argument(
        "-f", default=False, help="(Optional) Failure flag to step the framework from running if a test has failed."
    )

    parser.add_argument(
        "-w",
        default=False,
        help="(Optional) Specify to overwrite any existing output files. "
        "This can lead to the loss of existing outputs if not "
        "careful",
    )

    args = parser.parse_args(test_args)

    # Convert these to Path objects for main()

    args.tests_path = Path(args.tests_path)

    if args.dag_path is not None:
        args.dag_path = Path(args.dag_path)

    if args.output_path is None:
        args.output_path = "data/outputs/causal_tests_results.json"

        Path(args.output_path).parent.mkdir(exist_ok=True)

    else:
        args.output_path = Path(args.output_path)

        args.output_path.parent.mkdir(exist_ok=True)

    return args


def parse_variables(causal_dag: CausalDAG) -> tuple:
    """
    Function to parse and validate the variables defined in the causal dag.
    :param causal_dag: an instancce of the CausalDAG class containing the relationships and variables (as attribtutes)
    :returns:
    - Tuple containing the inputs, outputs and constraints to pass into the modelling scenario
    """

    inputs = [
        Input(node, eval(eval(attributes["datatype"])))
        for node, attributes in causal_dag.graph.nodes(data=True)
        if eval(attributes["typestring"]) == "input"
    ]

    constraints = set()

    for (node, attributes), input_var in zip(causal_dag.graph.nodes(data=True), inputs):

        if "constraint" in attributes:

            constraints.add(input_var.z3 == attributes["constraint"])

    outputs = [
        Output(node, eval(eval(attributes["datatype"])))
        for node, attributes in causal_dag.graph.nodes(data=True)
        if eval(attributes["typestring"]) == "output"
    ]

    return inputs, outputs, constraints


def main():
    """
    Main entrypoint of the script:
    """
    args = get_args()

    try:
        # Step 0: Read in the runtime dataset(s)

        data_frame = pd.concat([pd.read_csv(d) for d in args.data_path])

        # Step 1: Read in the dag and parse the variables from the .dot file

        causal_dag = CausalDAG(args.dag_path)

        inputs, outputs, constraints = parse_variables(causal_dag)

        # Step 2: Set up the modeling scenario and estimator

        modelling_scenario = Scenario(variables=inputs + outputs, constraints=constraints)

        modelling_scenario.setup_treatment_variables()

        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        # Step 3: Define the expected variables

        expected_outcome_effects = {
            "Positive": Positive(),
            "Negative": Negative(),
            "NoEffect": NoEffect(),
            "SomeEffect": SomeEffect(),
        }

        # Step 4: Call the JSONUtility class to perform the causal tests

        json_utility = JsonUtility(args.output_path, output_overwrite=True)

        # Step 5: Set the path to the data.csv, dag.dot and causal_tests.json file
        json_utility.set_paths(args.tests_path, args.dag_path, args.data_path)

        # Step 6: Sets up all the necessary parts of the json_class needed to execute tests
        json_utility.setup(scenario=modelling_scenario, data=data_frame, ignore_cycles=args.ignore_cycles)

        # Step 7: Run the causal tests
        test_outcomes = json_utility.run_json_tests(
            effects=expected_outcome_effects, mutates={}, estimators=estimators, f_flag=args.f
        )

        # Step 8: Update, print and save the final outputs

        for test in test_outcomes:
            test.pop("estimator")

            test["result"] = test["result"].to_dict(json=True)

            test["result"].pop("treatment_value")

            test["result"].pop("control_value")

        with open(args.output_path, "w", encoding="utf-8") as f:
            print(json.dumps(test_outcomes, indent=2), file=f)

        print(json.dumps(test_outcomes, indent=2))

    except ValidationError as ve:
        print(f"Cannot validate the specified input configurations: {ve}")

    else:
        print(f"Execution successful. " f"Output file saved at {Path(args.output_path).parent.resolve()}")


if __name__ == "__main__":
    main()
