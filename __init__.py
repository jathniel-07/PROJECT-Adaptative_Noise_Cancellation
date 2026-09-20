"""
Hybrid CNN-LSTM + FxNLMS active noise control (ANC) prototype.

Comfort / intercom-clarity subsystem that cancels low-frequency engine,
road/track and wind noise reaching a vehicle crew's ears -- the same
class of technology used in noise-cancelling headphones and automotive
road-noise-cancellation systems. Not a weapons or targeting system.

Field-deployment runtime (`anc_prototype run`) reads ONLY the
reference microphone. No other vehicle sensor (tachometer, GPS speed,
IMU, CAN-bus telemetry, ...) is read anywhere in this package. See
README.md for the full calibrate / evaluate / run workflow.
"""

__version__ = "0.1.0"
