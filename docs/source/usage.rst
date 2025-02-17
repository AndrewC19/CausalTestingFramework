
Usage
-----

There are currently 3 methods of using the Causal Testing Framework; 1) :doc:`JSON Front End </frontends/json_front_end>`\, 2)
:doc:`Test Suites </frontends/test_suite>`\, or 3) directly as
described below.

The causal testing framework is made up of 3 main components: Specification, Testing, and Data Collection. The first
step is to specify the (part of the) system under test as a modelling ``Scenario``. Modelling scenarios specify the
observable variables and any constraints which exist between them. We currently support 3 types of variable:


* ``Input`` variables are input parameters to the system.
* ``Output`` variables are outputs from the system.
* ``Meta`` variables are not directly observable but are relevant to system testing, e.g. a model may take a ``location``
  parameter and expand this out into ``average_age`` and ``household_size`` variables "under the hood". These parameters can
  be made explicit by instantiating them as metavariables.

To instantiate a scenario, simply provide a set of variables and an optional set of constraints, e.g.

.. code-block:: python

   from causal_testing.specification.variable import Input, Output, Meta
   from causal_testing.specification.scenario import Scenario

   def some_populate_function():
       pass

   x = Input("x", int)  # Define an input with name "x" of type int
   y = Output("y", float)  # Define an output with name "y" of type float
   z = Meta("y", int, some_populate_function)  # Define a meta with name "z" of type int

   modelling_scenario = Scenario({x, y, z}, {x > z, z < 3})  # Define a scenario with the three variables and two constraints

Note that scenario constraints are primarily intended to help specify the region of the input space under test in a
manner consistent with the Category Partition Method. It is not intended to serve as a test oracle. Use constraints
sparingly and with caution to avoid introducing data selection bias. We use Z3 to handle constraints. For help with
this, check out `their documentation <https://ericpony.github.io/z3py-tutorial/guide-examples.htm>`_.

Having fully specified the modelling scenario, we are now ready to test. Causal tests are,
essentially `metamorphic tests <https://en.wikipedia.org/wiki/Metamorphic_testing>`_ which are executed using statistical
causal inference. A causal test expresses the change in a given output that we expect to see when we change a particular
input in a particular way. A causal test case is built from a base test case, which specifies the relationship between
the given output and input and the desired effect. This information is the minimum required to perform identification

.. code-block:: python

   from causal_testing.testing.base_test_case import BaseTestCase
   from causal_testing.testing.causal_test_case import CausalTestCase
   from causal_testing.testing.causal_test_outcome import Positive
   from causal_testing.testing.effect import Effect

   base_test_case = BaseTestCase(
      treatment_variable = x, # Set the treatment (input) variable to x
      outcome_variable = y, # set the outcome (output) variable to y
      effect = Effect.DIRECT.value) # effect type, current accepted types are direct and total

   causal_test_case = CausalTestCase(
      base_test_case = base_test_case,
      expected_causal_effect = Positive, # We expect to see a positive change as a result of this
      control_value = 0, # Set the unmodified (control) value for x to 0,
      treatment_value = 1, # Set the modified (treatment) value for x to ,1
      estimate_type = "ate")

Before we can run our test case, we first need data. There are two ways to acquire this: 1. run the model with the
specific input configurations we're interested in, 2. use data from previous model runs. For a small number of specific
tests where accuracy is critical, the first approach will yield the best results. To do this, you can use the
`ExperimentalEstimator` class. This will run the system directly and calculate the causal effect estimate from this.

Where there are many test cases, using pre-existing data is likely to be faster. If the program's behaviour can be
estimated statistically, the results should still be reliable as long as there is enough data for the estimator to work
as intended. This will vary depending on the program and the estimator. To use this method, simply instantiate
one of the other estimator classes with a Pandas dataframe containing the runtime data, e.g.

.. code-block:: python
   estimator = LinearRegressionEstimator(
         treatment_variable,
         treatment_value,
         control_value,
         minimal_adjustment_set,
         outcome_variable,
         df=pd.read_csv(observational_data_path),
     )


Whether using fresh or pre-existing data, a key aspect of causal inference is estimation. To actually execute a test, we
need an estimator. We currently support two estimators: linear regression and logistic regression. The estimators require the
minimal adjustment set from the causal_dag. This and the estimator can be instantiated as per
the `documentation <https://causal-testing-framework.readthedocs.io/en/latest/autoapi/causal_testing/testing/estimators/index.html>`_.

.. code-block:: python

   from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator

   minimal_adjustment_set = causal_dag.identification(base_test_case)
   estimation_model =  LinearRegressionEstimator(treatment=treatment, control=control, treatment_value=1, control_value=0, adjustment_set = minimal_adjustment_set, df = obs_df)



We can now execute the test using the estimation model. This returns a causal test result, from which we can extract
various information. Here, we simply assert that the observed result is (on average) what we expect to see.

.. code-block:: python

   causal_test_result = causal_test_case.execute_test(estimation_model)
   test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)
   assert test_passes, "Expected to see a positive change in y."

Multiple tests can be executed at once using the test engines :doc:`Test Suite </frontends/test_suite>` feature.
