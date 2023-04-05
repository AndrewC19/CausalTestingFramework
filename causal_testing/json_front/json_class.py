"""This module contains the JsonUtility class, details of using this class can be found here:
https://causal-testing-framework.readthedocs.io/en/latest/json_front_end.html"""

import argparse
import json
import logging

from dataclasses import dataclass
from pathlib import Path

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
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import Estimator

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
    :attr {Scenario} modelling_scenario:
    :attr {CausalSpecification} causal_specification:
    """

    def __init__(self):
        self.paths = None
        self.variables = None
        self.data = []
        self.test_plan = None
        self.scenario = None
        self.causal_specification = None
        self.args = self.get_args()
        self.setup_logger(self.args.log_path)
        self._set_paths()

    def _set_paths(self):
        """
        Takes a path of the directory containing all scenario specific files and creates individual paths for each file
        """
        self.paths = JsonClassPaths(
            json_path=self.args.json_path, dag_path=self.args.dag_path, data_paths=self.args.data_path
        )

    def setup(self, scenario: Scenario):
        """Function to populate all the necessary parts of the json_class needed to execute tests"""
        self.scenario = scenario
        self.scenario.setup_treatment_variables()
        self.causal_specification = CausalSpecification(
            scenario=self.scenario, causal_dag=CausalDAG(self.paths.dag_path)
        )
        self._json_parse()
        self._populate_metas()

    def _create_abstract_test_case(self, test, mutates, effects):
        assert len(test["mutations"]) == 1
        abstract_test = AbstractCausalTestCase(
            scenario=self.scenario,
            intervention_constraints=[mutates[v](k) for k, v in test["mutations"].items()],
            treatment_variable=next(self.scenario.variables[v] for v in test["mutations"]),
            expected_causal_effect={
                self.scenario.variables[variable]: effects[effect]
                for variable, effect in test["expectedEffect"].items()
            },
            effect_modifiers={self.scenario.variables[v] for v in test["effect_modifiers"]}
            if "effect_modifiers" in test
            else {},
            estimate_type=test["estimate_type"],
            effect=test.get("effect", "total"),
        )
        return abstract_test

    def generate_tests(self, effects: dict, mutates: dict, estimators: dict):
        """Runs and evaluates each test case specified in the JSON input

        :param effects: Dictionary mapping effect class instances to string representations.
        :param mutates: Dictionary mapping mutation functions to string representations.
        :param estimators: Dictionary mapping estimator classes to string representations.
        """
        failures = 0
        for test in self.test_plan["tests"]:
            if "skip" in test and test["skip"]:
                continue
            abstract_test = self._create_abstract_test_case(test, mutates, effects)

            concrete_tests, dummy = abstract_test.generate_concrete_tests(5, 0.05)
            logger.info("Executing test: %s", test["name"])
            logger.info(abstract_test)
            logger.info([abstract_test.treatment_variable.name, abstract_test.treatment_variable.distribution])
            logger.info("Number of concrete tests for test case: %s", str(len(concrete_tests)))
            failures = self._execute_tests(concrete_tests, estimators, test, self.args.f_flag)
            logger.info("%s/%s failed for %s\n", failures, len(concrete_tests), test["name"])

    def _execute_tests(self, concrete_tests, estimators, test, f_flag):
        failures = 0
        for concrete_test in concrete_tests:
            failed = self._execute_test_case(concrete_test, estimators[test["estimator"]], f_flag)
            if failed:
                failures += 1
        return failures

    def _json_parse(self):
        """Parse a JSON input file into inputs, outputs, metas and a test plan"""
        with open(self.paths.json_path, encoding="utf-8") as f:
            self.test_plan = json.load(f)
        for data_file in self.paths.data_paths:
            df = pd.read_csv(data_file, header=0)
            self.data.append(df)
        self.data = pd.concat(self.data)

    def _populate_metas(self):
        """
        Populate data with meta-variable values and add distributions to Causal Testing Framework Variables
        """
        for meta in self.scenario.variables_of_type(Meta):
            meta.populate(self.data)
        for var in self.scenario.variables_of_type(Meta).union(self.scenario.variables_of_type(Output)):
            if not var.distribution:
                fitter = Fitter(self.data[var.name], distributions=get_common_distributions())
                fitter.fit()
                (dist, params) = list(fitter.get_best(method="sumsquare_error").items())[0]
                var.distribution = getattr(scipy.stats, dist)(**params)
                logger.info(var.name + f" {dist}({params})")

    def _execute_test_case(self, causal_test_case: CausalTestCase, estimator: Estimator, f_flag: bool) -> bool:
        """Executes a singular test case, prints the results and returns the test case result
        :param causal_test_case: The concrete test case to be executed
        :param f_flag: Failure flag that if True the script will stop executing when a test fails.
        :return: A boolean that if True indicates the causal test case passed and if false indicates the test case
         failed.
        :rtype: bool
        """
        failed = False

        causal_test_engine, estimation_model = self._setup_test(causal_test_case, estimator)
        causal_test_result = causal_test_engine.execute_test(
            estimation_model, causal_test_case, estimate_type=causal_test_case.estimate_type
        )

        test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)

        result_string = str()
        if causal_test_result.ci_low() and causal_test_result.ci_high():
            result_string = (
                f"{causal_test_result.ci_low()} < {causal_test_result.test_value.value} <  "
                f"{causal_test_result.ci_high()}"
            )
        else:
            result_string = f"{causal_test_result.test_value.value} no confidence intervals"
        if f_flag:
            assert test_passes, (
                f"{causal_test_case}\n    FAILED - expected {causal_test_case.expected_causal_effect}, "
                f"got {result_string}"
            )
        if not test_passes:
            failed = True
            logger.warning("   FAILED- expected %s, got %s", causal_test_case.expected_causal_effect, result_string)
        return failed

    def _setup_test(self, causal_test_case: CausalTestCase, estimator: Estimator) -> tuple[CausalTestEngine, Estimator]:
        """Create the necessary inputs for a single test case
        :param causal_test_case: The concrete test case to be executed
        :returns:
                - causal_test_engine - Test Engine instance for the test being run
                - estimation_model - Estimator instance for the test being run
        """

        data_collector = ObservationalDataCollector(self.scenario, self.data)
        causal_test_engine = CausalTestEngine(self.causal_specification, data_collector, index_col=0)

        minimal_adjustment_set = self.causal_specification.causal_dag.identification(causal_test_case.base_test_case)
        treatment_var = causal_test_case.treatment_variable
        minimal_adjustment_set = minimal_adjustment_set - {treatment_var}
        estimation_model = estimator(
            treatment=treatment_var.name,
            treatment_value=causal_test_case.treatment_value,
            control_value=causal_test_case.control_value,
            adjustment_set=minimal_adjustment_set,
            outcome=causal_test_case.outcome_variable.name,
            df=causal_test_engine.scenario_execution_data_df,
            effect_modifiers=causal_test_case.effect_modifier_configuration,
        )

        self.add_modelling_assumptions(estimation_model)

        return causal_test_engine, estimation_model

    def add_modelling_assumptions(self, estimation_model: Estimator):  # pylint: disable=unused-argument
        """Optional abstract method where user functionality can be written to determine what assumptions are required
        for specific test cases
        :param estimation_model: estimator model instance for the current running test.
        """
        return

    @staticmethod
    def setup_logger(log_path: str):
        """Setups up logging instance for the module and adds a FileHandler stream so all stdout prints are also
        sent to the logfile
        :param log_path: Path specifying location and name of the logging file to be used
        """
        setup_log = logging.getLogger(__name__)
        file_handler = logging.FileHandler(Path(log_path))
        setup_log.addHandler(file_handler)

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
            "--log_path",
            help="Specify a directory to change the location of the log file",
            default="./json_frontend.log",
        )
        parser.add_argument(
            "--data_path",
            help="Specify path to file containing runtime data",
            required=True,
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
