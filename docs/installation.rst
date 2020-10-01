.. highlight:: shell

============
Installation
============


Stable release
--------------
As module
^^^^^^^^^^


To install pyPOCQuantui, just run this command in your terminal:

.. code-block:: console

    $ pip install pyPOCQuant

Installing pyPOCQuantui this way ensures that you get always the latest release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

As stand alone executable
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install pyPOCQuantui on your system without installing Python yourself just download the
pre-compiled executable matching your operating system:

.. only:: html

    * `Download the latest in one portable for WINDOWS x64 systems <https://git.gitlab.com/csb.ethz/pypocquantui/releases/>`_
    
Install tesseract following these instructions depending your operating system:

.. only:: html

    * `Download the latest version of tesseract <https://tesseract-ocr.github.io/tessdoc/Home.html>`_
 
.. warning::
    Make sure tesseract is on PATH of your environment.


pypocquantui can then be used trough its graphical user interface (GUI) directly. 


From sources
------------

The latest sources for pypocquantui can be downloaded from the `Github repo`_.

1. Install system `Python3` or `miniconda3`.

.. only:: html

    * `Download miniconda  <https://docs.conda.io/en/latest/miniconda.html>`_


   If you have other Python installations it is good practice to install everything new into a separate environment. Also such an environment  can be later used to create a snapshot of your installation and shared  with other to build exactly the identical environment.

2. Create a new environment "pyPOCQuantEnv" with:

.. code-block:: console

   $ conda create -n pyPOCQuantEnv python=3.6
   $ activate pyPOCQuantEnv

.. note::
   Note: more information about conda environments can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_

3. You can clone the public repository:

.. code-block:: console

    $ git clone git://git.gitlab.com/csb.ethz/pypocquantui.git

Once you have a copy of the source, navigate into the directory and run:

.. code-block:: console

    $ fbs run

.. _Github repo: https://git.gitlab.com/csb.ethz/pypocquantui.git