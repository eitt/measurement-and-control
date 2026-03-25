"""
code.py
=======
Legacy compatibility wrapper for the measurement-only baseline.

The older monolithic script in this repository mixed RUL estimation with a
control-oriented FOPID experiment. That control branch is no longer part of
the active project scope. Running this file now delegates to the measurement
baseline in ``main_universal.py``.
"""

from main_universal import main


if __name__ == "__main__":
    main()
