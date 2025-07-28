Installation
===========

Prerequisites
------------

- Python 3.8+
- MuJoCo physics engine
- Required Python packages (see requirements.txt)

Installation Steps
-----------------

1. Clone this repository:

   .. code-block:: bash

      git clone https://github.com/neumovelab/myoassist.git
      cd myoassist

2. Install the required packages:

   .. code-block:: bash

      pip install -r myoassist_reflex/config/requirements.txt

3. Verify installation:

   .. code-block:: python

      import myoassist_reflex
      print("MyoAssist installed successfully!")

Troubleshooting
--------------

Common installation issues:

1. **MuJoCo license**: Ensure you have a valid MuJoCo license
2. **Python version**: Make sure you're using Python 3.8 or higher
3. **Dependencies**: If you encounter import errors, try reinstalling dependencies

For more help, please open an issue on our `GitHub repository <https://github.com/neumovelab/myoassist/issues>`_. 