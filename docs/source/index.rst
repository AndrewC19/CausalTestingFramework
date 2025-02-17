Welcome to the Causal Testing Framework
==========================================

|status| |ci-tests| |code-cov| |docs| |python| |pypi| |doi| |license|

Overview
**********

Causal testing is a :term:`causal inference`-driven framework for functional black-box testing. This framework utilises
graphical causal inference (CI) techniques for the specification and functional testing of software from a black-box
perspective. In this framework, we use causal directed acyclic graphs (DAGs) to express the anticipated cause-effect
relationships amongst the inputs and outputs of the system-under-test and the supporting mathematical framework to
design statistical procedures capable of making causal inferences. Each causal test case focuses on the causal effect
of an intervention made to the system-under test. That is, a prescribed change to the input configuration of the
system-under-test that is expected to cause a change to some output(s).

.. raw:: html

   <style>
   .zoom-overlay {
       position: fixed;
       top: 0;
       left: 0;
       width: 100%;
       height: 100%;
       background-color: rgba(0, 0, 0, 0.7);
       display: flex;
       align-items: center;
       justify-content: center;
       z-index: 9999;
   }

   .zoom-container {
       cursor: zoom-in;
       transition: transform 1s ease-in-out;
   }

   .zoom-container.zoomed {
       transform: scale(4);
       cursor: zoom-out;
   }

   .zoomable-image {
       max-width: 100%;
       max-height: 100%;
       margin: auto;
   }

   .zoom-container:hover {
       cursor: zoom-in;
   }
   </style>

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', function () {
       var image = document.querySelector('.zoomable-image');

       image.addEventListener('click', function () {
           var overlay = document.createElement('div');
           overlay.className = 'zoom-overlay';

           var container = document.createElement('div');
           container.className = 'zoom-container';
           container.style.cursor = 'zoom-out';

           var clonedImage = image.cloneNode();
           clonedImage.classList.add('zoomable-image');
           container.appendChild(clonedImage);
           overlay.appendChild(container);

           document.body.appendChild(overlay);

           overlay.addEventListener('click', function () {
               overlay.remove();
           });
       });
   });
   </script>



.. container:: zoom-container

   .. image:: /images/schematic.png
      :class: zoomable-image
      :alt: Zoomable Image


.. toctree::
   :hidden:
   :caption: Home

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Introduction

   description
   installation
   usage

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Module Descriptions

   /modules/causal_specification
   /modules/causal_tests

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:
   :titlesonly:

   /autoapi/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Front Ends

   frontends/json_front_end
   frontends/test_suite

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Glossary

   glossary

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   /dev/version_release
   /dev/documentation
   /dev/actions_and_webhooks

.. toctree::
   :caption: Useful Links
   :hidden:
   :maxdepth: 2

   Paper <https://dl.acm.org/doi/10.1145/3607184>
   Figshare <https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516>
   PyPI <https://pypi.org/project/causal-testing-framework/>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Credits

   credits

.. Define variables for our GH badges

.. |ci-tests| image:: https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml/badge.svg
   :target: https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml
   :alt: Continuous Integration Tests

.. |code-cov| image:: https://codecov.io/gh/CITCOM-project/CausalTestingFramework/branch/main/graph/badge.svg?token=04ijFVrb4a
   :target: https://codecov.io/gh/CITCOM-project/CausalTestingFramework
   :alt: Code coverage

.. |docs| image:: https://readthedocs.org/projects/causal-testing-framework/badge/?version=latest
   :target: https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation

.. |python| image:: https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FCITCOM-project%2FCausalTestingFramework%2Fmain%2Fpyproject.toml&query=%24.project%5B'requires-python'%5D&label=python
   :target: https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FCITCOM-project%2FCausalTestingFramework%2Fmain%2Fpyproject.toml&query=%24.project%5B'requires-python'%5D&label=python
   :alt: Python

.. |pypi| image:: https://img.shields.io/pypi/v/causal-testing-framework
   :target: https://pypi.org/project/causal-testing-framework/
   :alt: PyPI

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: Status

.. |doi| image:: https://t.ly/FCT1B
   :target: https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516
   :alt: DOI

.. |license| image:: https://img.shields.io/github/license/CITCOM-project/CausalTestingFramework
   :target: https://github.com/CITCOM-project/CausalTestingFramework
   :alt: License
