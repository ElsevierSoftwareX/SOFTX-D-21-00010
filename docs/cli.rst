==================
Command line usage
==================

**pyPOCQuant** can be used trough its command line interface (cli). It is convenient to process a large amount of different folder trough i.e a bash script.

To show the usage type:

.. code-block:: bash

   python -m pypocquant.pyPOCQuant --help


To run the pipline for a given folder and config type:

.. code-block:: bash

   python -m pypocquant.pyPOCQuant_FH -f path/to/images -s path/to/config.conf -w 10


To split and organize images of different kinds in one folder type:

.. code-block:: bash

   python -m pypocquant.split_images_by_strip_type_parallel -f path/to/images -o path/to/result_dir -w 10

