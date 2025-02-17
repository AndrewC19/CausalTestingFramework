# Covasim Case Study: Prioritising Vaccination for the Elderly
In this case study, we demonstrate how to use causal testing to determine whether prioritising the elderly
has the expected effect on a series of vaccine-related outcomes. Specifically, cumulative infections, vaccines (number
of doses administered), vaccinated (number of agents vaccinated), and maximum doses per agent. As explained in Section
5.3 (Prioritising the elderly for vaccination), we expect that changing the Pfizer vaccine to additionally prioritise
the elderly should cause vaccines and vaccinated to decrease (more restrictive policy), infections to increase (less
are vaccinated), and no change to the maximum doses since this should always be 2.

This case study directly executes Covasim under two input configurations that differ only in their vaccine input: one
is the default Pfizer vaccine, the other is the same vaccine but additionally sub-targeting the elderly using a method
provided in the Covasim vaccine tutorial. We execute each of these input configurations 30 times and repeat this for
four test cases: one focusing on each of the four previously mentioned outputs.

Further details are provided in Section 5.3 (Prioritising the elderly for vaccination) of the paper.

>[!NOTE]
>This version of the CTF uses observational data to separate the software execution and testing.
Older versions of this framework simulate the data using a custom experimental data collector and the `covasim`
package (version 3.0.7) as outlined below.

## How to run
To run this case study:
1. Ensure all project dependencies are installed by running `pip install .` from the top
level of this directory (instructions are provided in the project README).
2. If necessary, install version 3.0.7 by running `pip install covasim==3.0.7`.
3. Change directory to `causal_testing/examples/covasim_/vaccinating_elderly`.
4. Run the command `python example_vaccine.py`.

This will run Covasim as described above and print out the causal test results for the effect of each input on each
output.
