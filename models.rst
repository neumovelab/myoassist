Models and Tasks
================

MyoAssist provides a collection of musculoskeletal models for human locomotion simulation and exoskeleton control.

Available Models
---------------
+----------------+-------------+-------------+-------------+
| Device         | 22 (2D)     | 26 (3D)     | 80 (3D)     |
+================+=============+=============+=============+
| BASELINE       | myoLeg22_2D_BASELINE.xml | myoLeg26_BASELINE.xml | N/A |
+----------------+-------------+-------------+-------------+
| DEPHY          | myoLeg22_2D_DEPHY.xml | myoLeg26_DEPHY.xml | myoLeg80_DEPHY/ |
+----------------+-------------+-------------+-------------+
| HMEDI          | myoLeg22_2D_HMEDI.xml | myoLeg26_HMEDI.xml | myoLeg80_HMEDI/ |
+----------------+-------------+-------------+-------------+
| HUMOTECH       | myoLeg22_2D_HUMOTECH.xml | myoLeg26_HUMOTECH.xml | myoLeg80_HUMOTECH/ |
+----------------+-------------+-------------+-------------+
| OSL A          | myoLeg22_2D_OSL_A.xml | myoLeg26_OSL_A.xml | N/A |
+----------------+-------------+-------------+-------------+
| OSL KA         | N/A | N/A | myoLeg80_OSL_KA/ |
+----------------+-------------+-------------+-------------+
| TUTORIAL       | myoLeg22_2D_TUTORIAL.xml | myoLeg26_TUTORIAL.xml | N/A |
+----------------+-------------+-------------+-------------+


2D Models (22 Muscles)
~~~~~~~~~~~~~~~~~~~~~~

Located in `models/22muscle_2D/`, these models provide simplified 2D representations for faster simulation and easier analysis.

**Available configurations:**
- `myoLeg22_2D_BASELINE.xml` - Baseline model without exoskeleton
- `myoLeg22_2D_DEPHY.xml` - Model with Dephy exoskeleton
- `myoLeg22_2D_HMEDI.xml` - Model with HMEDI exoskeleton
- `myoLeg22_2D_HUMOTECH.xml` - Model with Humotech Caplex EXO-010 exoskeleton
- `myoLeg22_2D_OSL_A.xml` - Model with OSL A(ankle) prosthesis
- `myoLeg22_2D_TUTORIAL.xml` - Model with example exoskeleton


3D Models (26 Muscles)
~~~~~~~~~~~~~~~~~~~~~~

Located in `models/26muscle_3D/`, these models provide more realistic 3D representations.

**Available configurations:**
- `myoLeg26_BASELINE.xml` - Baseline model without exoskeleton
- `myoLeg26_DEPHY.xml` - Model with Dephy exoskeleton
- `myoLeg26_HMEDI.xml` - Model with HMEDI exoskeleton
- `myoLeg26_HUMOTECH.xml` - Model with Humotech Caplex EXO-010 exoskeleton
- `myoLeg26_OSL_A.xml` - Model with OSL A(ankle) prosthesis
- `myoLeg26_TUTORIAL.xml` - Model with example exoskeleton


High-Fidelity Models (80 Muscles)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Located in `models/80muscle/`, these models provide the most detailed representation with 80 muscles.

**Available configurations:**
- `myoLeg80_DEPHY/` - Model with Dephy exoskeleton
- `myoLeg80_HMEDI/` - Model with HMEDI exoskeleton
- `myoLeg80_HUMOTECH/` - Model with Humotech exoskeleton
- `myoLeg80_OSL_KA/` - Model with OSL KA exoskeleton


Exoskeleton Integration
----------------------

Each exoskeleton configuration includes:
- Hardware-specific geometry
- Actuator models
- Control interfaces
- Attachment points

For more information on exoskeleton controllers, see :doc:`controllers`. 