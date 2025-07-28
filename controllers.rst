Exoskeleton Controllers
======================

MyoAssist provides a modular framework for exoskeleton control with support for multiple controller architectures and hardware platforms.

Controller Architecture
---------------------

The exoskeleton control system consists of several key components:

- **Reference Generation**: Creates target trajectories for exoskeleton assistance
- **Controller Implementation**: Executes control algorithms (4-parameter, n-point spline)
- **Hardware Interface**: Manages communication with exoskeleton hardware
- **Safety Systems**: Ensures safe operation and emergency shutdown

4-Parameter Spline Controller
----------------------------

The 4-parameter spline controller provides a simple yet effective approach to exoskeleton assistance.

**Parameters:**
- **Onset**: When assistance begins in the gait cycle
- **Peak**: When maximum assistance is provided
- **Offset**: When assistance ends in the gait cycle
- **Magnitude**: Maximum assistance torque

**Advantages:**
- Simple parameterization
- Easy to optimize
- Intuitive control

**Use cases:**
- Initial controller design
- Educational applications
- Quick prototyping

N-Point Spline Controller
------------------------

The n-point spline controller provides more flexible assistance patterns.

**Features:**
- Multiple control points throughout the gait cycle
- Smooth interpolation between points
- Configurable number of control points
- Hardware-specific parameterization

**Advantages:**
- High flexibility
- Precise control
- Hardware-specific optimization

**Use cases:**
- Advanced research
- Clinical applications
- Hardware-specific optimization

Controller Implementation
-----------------------

Controllers are implemented in the `myoassist_reflex/exo/` directory:

.. code-block:: python

   from myoassist_reflex.exo import FourParamSplineCtrl, NPointSplineCtrl
   
   # 4-parameter controller
   ctrl_4param = FourParamSplineCtrl(
       onset=0.1,    # 10% of gait cycle
       peak=0.3,     # 30% of gait cycle
       offset=0.6,   # 60% of gait cycle
       magnitude=20   # 20 Nm assistance
   )
   
   # N-point controller
   ctrl_npoint = NPointSplineCtrl(
       control_points=[0.1, 0.3, 0.6, 0.8],
       torques=[0, 20, 15, 0]
   )

Hardware Support
---------------

MyoAssist supports multiple exoskeleton platforms:

**Dephy Exoskeleton**
- Ankle assistance
- Commercial hardware
- Real-time control interface

**HMEDI Exoskeleton**
- Hip-knee-ankle assistance
- Research platform
- Custom control interface

**Humotech Exoskeleton**
- Full leg assistance
- Commercial hardware
- Advanced control features

**OSL KA Exoskeleton**
- Knee-ankle assistance
- Research platform
- Modular design

Safety Features
--------------

All controllers include safety systems:

- **Torque limits**: Prevent excessive assistance
- **Velocity limits**: Ensure safe movement speeds
- **Emergency stop**: Immediate shutdown capability
- **Contact detection**: Automatic assistance adjustment

Optimization Integration
----------------------

Controllers are designed for optimization:

- **Parameter bounds**: Define feasible parameter ranges
- **Cost functions**: Evaluate controller performance
- **CMA-ES integration**: Efficient parameter optimization
- **Multi-objective optimization**: Balance multiple objectives

For more information on optimization, see :doc:`optimization`.

Controller Tuning
----------------

Guidelines for controller tuning:

1. **Start simple**: Use 4-parameter controller for initial design
2. **Define objectives**: Specify assistance goals and constraints
3. **Optimize parameters**: Use CMA-ES for parameter tuning
4. **Validate results**: Test on target hardware
5. **Iterate**: Refine based on performance feedback

For detailed optimization procedures, see :doc:`running_optimizations`. 