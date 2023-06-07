"""This module contains the JsonUtility class, details of using this class can be found here:
https://causal-testing-framework.readthedocs.io/en/latest/json_front_end.html"""

import argparse
import json
import logging

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import StatisticsError

import pandas as pd
import scipy
from fitter import Fitter, get_common_distributions

from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Meta, Output
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.base_test_case import BaseTestCase

logger = logging.getLogger(__name__)


class JsonUtility:
    """
    The JsonUtility Class provides the functionality to use structured JSON to setup and run causal tests on the
    CausalTestingFramework.

    :attr {Path} json_path: Path to the JSON input file.
    :attr {Path} dag_path: Path to the dag.dot file containing the Causal DAG.
    :attr {Path} data_path: Path to the csv data file.
    :attr {Input} inputs: Causal variables representing inputs.
    :attr {Output} outputs: Causal variables representing outputs.
    :attr {Meta} metas: Causal variables representing metavariables.
    :attr {pd.DataFrame}: Pandas DataFrame containing runtime data.
    :attr {dict} test_plan: Dictionary containing the key value pairs from the loaded json test plan.
    :attr {Scenario} scenario:
    :attr {CausalSpecification} causal_specification:
    """

    def __init__(self, output_path: str, output_overwrite: bool = False):
        self.input_paths = None
        self.variables = {"inputs": {}, "outputs": {}, "metas": {}}
        self.data = []
        self.test_plan = None
        self.scenario = None
        self.causal_specification = None
        self.output_path = Path(output_path)
        self.check_file_exists(self.output_path, output_overwrite)

    def set_paths(self, json_path: str, dag_path: str, data_paths: list[str] = None):
        """
        Takes a path of the directory containing all scenario specific files and creates individual paths for each file
        :param json_path: string path representation to .json file containing test specifications
        :param dag_path: string path representation to the .dot file containing the Causal DAG
        :param data_paths: string path representation to the data files
        """
        if data_paths is None:
            data_paths = []
        self.input_paths = JsonClassPaths(json_path=json_path, dag_path=dag_path, data_paths=data_paths)

    def setup(self, scenario: Scenario):
        """Function to populate all the necessary parts of the json_class needed to execute tests"""
        self.scenario = scenario
        self._get_scenario_variables()
        self.scenario.setup_treatment_variables()
        self.causal_specification = CausalSpecification(
            scenario=self.scenario, causal_dag=CausalDAG(self.input_paths.dag_path)
        )
        # Parse the JSON test plan
        with open(self.input_paths.json_path, encoding="utf-8") as f:
            self.test_plan = json.load(f)
        # Populate the data
        if self.input_paths.data_paths:
            self.data = pd.concat([pd.read_csv(data_file, header=0) for data_file in self.input_paths.data_paths])
        self._populate_metas()

    def _create_abstract_test_case(self, test, mutates, effects):
        assert len(test["mutations"]) == 1
        treatment_var = next(self.scenario.variables[v] for v in test["mutations"])

        if not treatment_var.distribution:
            fitter = Fitter(self.data[treatment_var.name], distributions=get_common_distributions())
            fitter.fit()
            (dist, params) = list(fitter.get_best(method="sumsquare_error").items())[0]
            treatment_var.distribution = getattr(scipy.stats, dist)(**params)
            self._append_to_file(treatment_var.name + f" {dist}({params})", logging.INFO)

        abstract_test = AbstractCausalTestCase(
            scenario=self.scenario,
            intervention_constraints=[mutates[v](k) for k, v in test["mutations"].items()],
            treatment_variable=treatment_var,
            expected_causal_effect={
                self.scenario.variables[variable]: effects[effect]
                for variable, effect in test["expected_effect"].items()
            },
            effect_modifiers={self.scenario.variables[v] for v in test["effect_modifiers"]}
            if "effect_modifiers" in test
            else {},
            estimate_type=test["estimate_type"],
            effect=test.get("effect", "total"),
        )
        return abstract_test

    def run_json_tests(self, effects: dict, estimators: dict, f_flag: bool = False, mutates: dict = None):
        """Runs and evaluates each test case specified in the JSON input

        :param effects: Dictionary mapping effect class instances to string representations.
        :param mutates: Dictionary mapping mutation functions to string representations.
        :param estimators: Dictionary mapping estimator classes to string representations.
        :param f_flag: Failure flag that if True the script will stop executing when a test fails.
        """
        failures = 0
        msg = ""
        for test in self.test_plan["tests"]:
            if "skip" in test and test["skip"]:
                continue
            test["estimator"] = estimators[test["estimator"]]
            if "mutations" in test:
                if "coefficient" in test["estimate_type"]:
                    base_test_case = BaseTestCase(
                        treatment_variable=next(self.scenario.variables[v] for v in test["mutations"]),
                        outcome_variable=next(self.scenario.variables[v] for v in test["expected_effect"]),
                        effect=test.get("effect", "direct"),
                    )
                    assert len(test["expected_effect"]) == 1, "Can only have one expected effect."
                    causal_test_case = CausalTestCase(
                        base_test_case=base_test_case,
                        expected_causal_effect=next(
                            effects[effect] for variable, effect in test["expected_effect"].items()
                        ),
                        estimate_type=test["estimate_type"],
                        effect_modifier_configuration={
                            self.scenario.variables[v] for v in test.get("effect_modifiers", [])
                        },
                    )
                    result = self._execute_test_case(causal_test_case=causal_test_case, test=test, f_flag=f_flag)
                    msg = (
                        f"Executing test: {test['name']} \n"
                        + f"  {causal_test_case} \n"
                        + "  "
                        + ("\n  ").join(str(result[1]).split("\n"))
                        + "==============\n"
                        + f"  Result: {'FAILED' if result[0] else 'Passed'}"
                    )
                    print(msg)
                else:
                    abstract_test = self._create_abstract_test_case(test, mutates, effects)
                    concrete_tests, _ = abstract_test.generate_concrete_tests(5, 0.05)
                    failures, _ = self._execute_tests(concrete_tests, test, f_flag)

                    msg = (
                        f"Executing test: {test['name']} \n"
                        + "  abstract_test \n"
                        + f"  {abstract_test} \n"
                        + f"  {abstract_test.treatment_variable.name},"
                        + f"  {abstract_test.treatment_variable.distribution} \n"
                        + f"  Number of concrete tests for test case: {str(len(concrete_tests))} \n"
                        + f"  {failures}/{len(concrete_tests)} failed for {test['name']}"
                    )
                self._append_to_file(msg, logging.INFO)
            else:
                outcome_variable = next(
                    iter(test["expected_effect"])
                )  # Take first key from dictionary of expected effect
                base_test_case = BaseTestCase(
                    treatment_variable=self.variables["inputs"][test["treatment_variable"]],
                    outcome_variable=self.variables["outputs"][outcome_variable],
                )

                causal_test_case = CausalTestCase(
                    base_test_case=base_test_case,
                    expected_causal_effect=effects[test["expected_effect"][outcome_variable]],
                    control_value=test["control_value"],
                    treatment_value=test["treatment_value"],
                    estimate_type=test["estimate_type"],
                )
                failed, _ = self._execute_test_case(causal_test_case=causal_test_case, test=test, f_flag=f_flag)

                msg = (
                    f"Executing concrete test: {test['name']} \n"
                    + f"treatment variable: {test['treatment_variable']} \n"
                    + f"outcome_variable = {outcome_variable} \n"
                    + f"control value = {test['control_value']}, treatment value = {test['treatment_value']} \n"
                    + f"Result: {'FAILED' if failed else 'Passed'}"
                )
                self._append_to_file(msg, logging.INFO)

    def _execute_tests(self, concrete_tests, test, f_flag):
        failures = 0
        details = []
        if "formula" in test:
            self._append_to_file(f"Estimator formula used for test: {test['formula']}")
        for concrete_test in concrete_tests:
            failed, result = self._execute_test_case(concrete_test, test, f_flag)
            details.append(result)
            if failed:
                failures += 1
        return failures, details

    def _populate_metas(self):
        """
        Populate data with meta-variable values and add distributions to Causal Testing Framework Variables
        """
        for meta in self.scenario.variables_of_type(Meta):
            meta.populate(self.data)

    def _execute_test_case(
        self, causal_test_case: CausalTestCase, test: Iterable[Mapping], f_flag: bool
    ) -> (bool, CausalTestResult):
        """Executes a singular test case, prints the results and returns the test case result
        :param causal_test_case: The concrete test case to be executed
        :param test: Single JSON test definition stored in a mapping (dict)
        :param f_flag: Failure flag that if True the script will stop executing when a test fails.
        :return: A boolean that if True indicates the causal test case passed and if false indicates the test case
         failed.
        :rtype: bool
        """
        failed = False

        causal_test_engine, estimation_model = self._setup_test(
            causal_test_case, test, test["conditions"] if "conditions" in test else None
        )
        causal_test_result = causal_test_engine.execute_test(
            estimation_model, causal_test_case, estimate_type=causal_test_case.estimate_type
        )

        test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)

        if causal_test_result.ci_low() is not None and causal_test_result.ci_high() is not None:
            result_string = (
                f"{causal_test_result.ci_low()} < {causal_test_result.test_value.value} <  "
                f"{causal_test_result.ci_high()}"
            )
        else:
            result_string = f"{causal_test_result.test_value.value} no confidence intervals"

        if not test_passes:
            if f_flag:
                raise StatisticsError(
                    f"{causal_test_case}\n    FAILED - expected {causal_test_case.expected_causal_effect}, "
                    f"got {result_string}"
                )
            failed = True
            logger.warning("   FAILED- expected %s, got %s", causal_test_case.expected_causal_effect, result_string)
        return failed, causal_test_result

    def _setup_test(
        self, causal_test_case: CausalTestCase, test: Mapping, conditions: list[str] = None
    ) -> tuple[CausalTestEngine, Estimator]:
        """Create the necessary inputs for a single test case
        :param causal_test_case: The concrete test case to be executed
        :param test: Single JSON test definition stored in a mapping (dict)
        :param conditions: A list of conditions which should be applied to the
        data. Conditions should be in the query format detailed at
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html
        :returns:
                - causal_test_engine - Test Engine instance for the test being run
                - estimation_model - Estimator instance for the test being run
        """

        data_collector = ObservationalDataCollector(
            self.scenario, self.data.query(" & ".join(conditions)) if conditions else self.data
        )
        causal_test_engine = CausalTestEngine(self.causal_specification, data_collector, index_col=0)

        minimal_adjustment_set = self.causal_specification.causal_dag.identification(causal_test_case.base_test_case)
        treatment_var = causal_test_case.treatment_variable
        minimal_adjustment_set = minimal_adjustment_set - {treatment_var}
        estimator_kwargs = {
            "treatment": treatment_var.name,
            "treatment_value": causal_test_case.treatment_value,
            "control_value": causal_test_case.control_value,
            "adjustment_set": minimal_adjustment_set,
            "outcome": causal_test_case.outcome_variable.name,
            "df": causal_test_engine.scenario_execution_data_df,
            "effect_modifiers": causal_test_case.effect_modifier_configuration,
        }
        if "formula" in test:
            estimator_kwargs["formula"] = test["formula"]
        estimation_model = test["estimator"](**estimator_kwargs)
        return causal_test_engine, estimation_model

    def _append_to_file(self, line: str, log_level: int = None):
        """Appends given line(s) to the current output file. If log_level is specified it also logs that message to the
        logging level.
        :param line: The line or lines of text to be appended to the file
        :param log_level: An integer representing the logging level as specified by pythons inbuilt logging module. It
        is possible to use the inbuilt logging level variables such as logging.INFO and logging.WARNING
        """
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if log_level:
            logger.log(level=log_level, msg=line)

    def _get_scenario_variables(self):
        for input_var in self.scenario.inputs():
            self.variables["inputs"][input_var.name] = input_var
        for output_var in self.scenario.outputs():
            self.variables["outputs"][output_var.name] = output_var
        for meta_var in self.scenario.metas():
            self.variables["metas"][meta_var.name] = meta_var

    @staticmethod
    def check_file_exists(output_path: Path, overwrite: bool):
        """Method that checks if the given path to an output file already exists. If overwrite is true the check is
        passed.
        :param output_path: File path for the output file of the JSON Frontend
        :param overwrite: bool that if true, the current file can be overwritten
        """
        if output_path.is_file():
            if overwrite:
                output_path.unlink()
            else:
                raise FileExistsError(f"Chosen file output ({output_path}) already exists")

    @staticmethod
    def get_args(test_args=None) -> argparse.Namespace:
        """Command-line arguments

        :return: parsed command line arguments
        """
        parser = argparse.ArgumentParser(
            description="A script for parsing json config files for the Causal Testing Framework"
        )
        parser.add_argument(
            "-f",
            help="if included, the script will stop if a test fails",
            action="store_true",
        )
        parser.add_argument(
            "-w",
            help="Specify to overwrite any existing output files. This can lead to the loss of existing outputs if not "
            "careful",
            action="store_true",
        )
        parser.add_argument(
            "--log_path",
            help="Specify a directory to change the location of the log file",
            default="./json_frontend.log",
        )
        parser.add_argument(
            "--data_path",
            help="Specify path to file containing runtime data",
            nargs="+",
        )
        parser.add_argument(
            "--dag_path",
            help="Specify path to file containing the DAG, normally a .dot file",
            required=True,
        )
        parser.add_argument(
            "--json_path",
            help="Specify path to file containing JSON tests, normally a .json file",
            required=True,
        )
        return parser.parse_args(test_args)


@dataclass
class JsonClassPaths:
    """
    A dataclass that converts strings of paths to Path objects for use in the JsonUtility class
    :param json_path: string path representation to .json file containing test specifications
    :param dag_path: string path representation to the .dot file containing the Causal DAG
    :param data_path: string path representation to the data file
    """

    json_path: Path
    dag_path: Path
    data_paths: list[Path]

    def __init__(self, json_path: str, dag_path: str, data_paths: str):
        self.json_path = Path(json_path)
        self.dag_path = Path(dag_path)
        self.data_paths = [Path(path) for path in data_paths]


@dataclass
class CausalVariables:
    """
    A dataclass that converts lists of dictionaries into lists of Causal Variables
    """

    def __init__(self, inputs: list[dict], outputs: list[dict], metas: list[dict]):
        self.inputs = [Input(**i) for i in inputs]
        self.outputs = [Output(**o) for o in outputs]
        self.metas = [Meta(**m) for m in metas] if metas else []

    def __iter__(self):
        for var in self.inputs + self.outputs + self.metas:
            yield var
