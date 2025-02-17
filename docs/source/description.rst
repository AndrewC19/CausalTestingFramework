Background
=====================================

The Causal Testing Framework consists of 3 main components: 1) Causal Specification, 2) Causal Test Case and 3) Data Collection.

#.
   :doc:`Causal Specification <../modules/causal_specification>`\ : To apply graphical CI
   techniques for testing, we need a causal DAG which depicts causal relationships amongst inputs and outputs. To
   collect this information, users must create a *causal specification*. This comprises a set of scenarios which place
   constraints over input variables that capture the use-case of interest, a causal DAG corresponding to this scenario,
   and a series of high-level functional requirements that the user wishes to test. In causal testing, these
   requirements should describe how the model should respond to interventions (changes made to the input configuration).



#.
   :doc:`Causal Tests <../modules/causal_tests>`\ : With a causal specification in hand, we can now go about designing
   a series of test cases that interrogate the causal relationships of interest in the scenario-under-test. Informally,
   a causal test case is a triple ``(M, X, Delta, Y)``, where ``M`` is the modelling scenario, ``X`` is an input configuration,
   ``Delta`` is an intervention which should be applied to ``X``, and ``Y`` is the expected *causal effect* of that intervention on
   some output of interest. Therefore, a causal test case states the expected causal effect (``Y``) of a particular
   intervention (``Delta``) made to an input configuration (``X``). For each scenario, the user should create a suite of causal
   tests. Once a causal test case has been defined, it is executed as follows:

   a. Using the causal DAG, identify an estimand for the effect of the intervention on the output of interest. That is,
      a statistical procedure capable of estimating the causal effect of the intervention on the output.
   #. Collect the data to which the statistical procedure will be applied (see Data collection below).
   #. Apply a statistical model (e.g. linear regression or causal forest) to the data to obtain a point estimate for
      the causal effect. Depending on the estimator used, confidence intervals may also be obtained at a specified
      confidence level e.g. 0.05 corresponds to 95% confidence intervals (optional).
   #. Return the casual test result including a point estimate and 95% confidence intervals, usually quantifying the
      average treatment effect (ATE).
   #. Implement and apply a test oracle to the causal test result - that is, a procedure that determines whether the
      test should pass or fail based on the results. In the simplest case, this takes the form of an assertion which
      compares the point estimate to the expected causal effect specified in the causal test case.

For more information on each of these steps, follow the link to their respective documentation.
