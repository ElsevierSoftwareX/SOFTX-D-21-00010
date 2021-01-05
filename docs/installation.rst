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
    * `Download the latest in one portable for MAC x64 systems <https://git.gitlab.com/csb.ethz/pypocquantui/releases/>`_
    * `Download the latest in one portable for LINUX x64 systems <https://git.gitlab.com/csb.ethz/pypocquantui/releases/>`_

Install tesseract following these instructions depending your operating system:

.. only:: html

    * `Download the latest version of tesseract <https://tesseract-ocr.github.io/tessdoc/Home.html>`_

.. warning::
    Make sure tesseract is on PATH of your environment.


pypocquantui can then be used trough its graphical user interface (GUI) directly. 


From sources
------------

All platforms
^^^^^^^^^^^

The latest sources for pypocquantui can be downloaded from the `Github repo`_. 

*pyPOCQuant* requires python 3.6. It is recommended to use miniconda. When miniconda is installed, start the terminal and type:

1. Install system `Python3` or `miniconda3`.

.. only:: html

    * `Download miniconda  <https://docs.conda.io/en/latest/miniconda.html>`_


   If you have other Python installations it is good practice to install everything new into a separate environment. Also such an environment  can be later used to create a snapshot of your installation and shared  with other to build exactly the identical environment.

2. Create a new environment "pyPOCQuantEnv" with:

.. code-block:: console

   $ conda create -n pyPOCQuantEnv python=3.6
   $ activate pyPOCQuantEnv

.. note::
   More information about conda environments can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_

3. You can clone the public repository:

.. code-block:: console

    $ git clone git://git.gitlab.com/csb.ethz/pypocquantui.git

Once you have a copy of the source, navigate into the directory `pypocquantui` and install dependencies with:

.. code-block:: console
	$ cd ${pyPOCQuantUI_root_folder}
	$ pip install -r requirements/${platform}

.. note::
    `${platform}` is one of `win32.txt`, `linux.txt`, or `osx.txt`.

Then to start the UI run:

.. code-block:: console

    $ fbs run

If you use PyCharm make sure you open the project with its root folder and add 

 `/pypocquantui/src/main/python/main.py` 

to the run configuration.



Windows
^^^^^^^^

* `Install tesseract  <https://tesseract-ocr.github.io/tessdoc/Home.html>`_

Linux
^^^^^

Install the following dependences (instructions for Ubuntu Linux):

.. code-block:: console

	$ sudo apt install libzmq3-dev, tesseract-ocr, libzbar0


macOS
^^^^^^

To install the required dependencies we recommend to use the packaging manager `brew`. Install it from here if you have't allready  `Install brew  <https://brew.sh/>`_ .

.. code-block:: console

	$ brew install zbar
	$ brew install tesseract



Build from source
------------

To compile and create a pyPOCQuantUI installer, perform following steps. In the following `{ppcqui_root}` points to the root folder of the `pyPOCQuantUI` checked-out code.

Windows
^^^^^^^^

.. code-block:: console

	$ cd ${ppcqui_root}
	$ python ./make_build.py


You will find the installer in `${ppcqui_root}\target\pyPOCQuant`.

Linux
^^^^^

.. code-block:: console

	$ sudo apt install ruby ruby-dev rubygems build-essential
	$ sudo gem install --no-document fpm
	$ cd ${ppcqui_root}
	$ python ./make_build.py


This will create a `${ppcqui_root}/target/pyPOCQuant/pyPOCQuant.deb` package that can be installed and redistributed. 

.. code-block:: console

	$ sudo apt install ${ppcqui_root}/target/pyPOCQuant/pyPOCQuant.deb


Please notice that client machines will need to install also two dependences:

.. code-block:: console

	$ sudo apt install tesseract-ocr, libzbar0
	$ sudo apt install ${ppcqui_root}/target/pyPOCQuant/pyPOCQuant.deb


macOS
^^^^^^

.. code-block:: console

	$ cd ${ppcqui_root}
	$ python ./make_build.py

.. note::
	- Depending on your Python installation, you may need to use `pip3` instead of `pip`.

	- For both running it from source or with the compiled binaries `zbar` and `tesseract` needs to be installed and be on PATH. On Windows `zbar` libs are installed automatically.


.. _Github repo: https://git.gitlab.com/csb.ethz/pypocquantui.git