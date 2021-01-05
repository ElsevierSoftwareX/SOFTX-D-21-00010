==================

Scripting
==================

**pyPOCQuant** can be used directly from within python scripts and therefore being part of a larger workflow. It is convenient to process a large amount of different folder automatically. Or further automatically process results and generation of reports.

Minimal example with default settings. Add the following code to a file such as `example.py` wile replaceing the `input_folder_path` and `results_folder_path` to the example or your images : 

.. code-block:: bash

	from pypocquant.lib.pipeline import run_pipeline
	from pypocquant.lib.settings import default_settings

	# Get the default settings
	settings = default_settings()

	# Change settings manually as needed
	settings["sensor_band_names"] = ('igm', 'igg', 'ctl')

	# Alternatively, load existing settings file
	# from pypocquant.lib.settings import load_settings
	# settings = load_settings('full/path/to/settings/file.conf')

	# Set final argument
	input_folder_path = 'full/path/to/input/folder'
	results_folder_path = 'full/path/to/results/folder'
	max_workers = 8 

	# Run the pipeline
	run_pipeline(
    	input_folder_path,
    	results_folder_path,
    	**settings,
    	max_workers=max_workers
	)


and run it with:

.. code-block:: bash

   python -m example.py
