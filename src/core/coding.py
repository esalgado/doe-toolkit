"""
Factor value coding and decoding utilities.

This module provides functions to convert between coded values (-1, 0, +1)
and actual factor values (e.g., 150, 175, 200).

Coded values are used internally for:
- Numerical stability in model fitting
- Comparable effect estimates
- Standard DOE practices

Actual values are used for:
- User-facing displays
- Predictions with real settings
- Optimization constraints
"""

from typing import Dict, List, Union
import pandas as pd
import numpy as np

from src.core.factors import Factor


def encode_value(
    actual_value: float,
    min_val: float,
    max_val: float
) -> float:
    """
    Encode actual value to coded value (-1 to +1).
    
    Parameters
    ----------
    actual_value : float
        Actual factor value
    min_val : float
        Minimum of factor range
    max_val : float
        Maximum of factor range
    
    Returns
    -------
    float
        Coded value in range [-1, +1]
    
    Examples
    --------
    >>> encode_value(175, 150, 200)  # Temperature
    0.0
    >>> encode_value(150, 150, 200)
    -1.0
    >>> encode_value(200, 150, 200)
    1.0
    """
    center = (min_val + max_val) / 2
    half_range = (max_val - min_val) / 2
    return (actual_value - center) / half_range


def decode_value(
    coded_value: float,
    min_val: float,
    max_val: float
) -> float:
    """
    Decode coded value (-1 to +1) to actual value.
    
    Parameters
    ----------
    coded_value : float
        Coded value in range [-1, +1]
    min_val : float
        Minimum of factor range
    max_val : float
        Maximum of factor range
    
    Returns
    -------
    float
        Actual factor value
    
    Examples
    --------
    >>> decode_value(0.0, 150, 200)  # Temperature
    175.0
    >>> decode_value(-1.0, 150, 200)
    150.0
    >>> decode_value(1.0, 150, 200)
    200.0
    """
    center = (min_val + max_val) / 2
    half_range = (max_val - min_val) / 2
    return center + coded_value * half_range


def encode_design(
    design: pd.DataFrame,
    factors: List[Factor]
) -> pd.DataFrame:
    """
    Encode design matrix from actual to coded values.
    
    Only continuous factors are encoded. Discrete and categorical
    factors are left as-is.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with actual values
    factors : List[Factor]
        Factor definitions
    
    Returns
    -------
    pd.DataFrame
        Design matrix with coded values for continuous factors
    
    Examples
    --------
    >>> design = pd.DataFrame({
    ...     'Temperature': [150, 175, 200],
    ...     'Pressure': [50, 75, 100]
    ... })
    >>> factors = [
    ...     Factor('Temperature', FactorType.CONTINUOUS, levels=[150, 200]),
    ...     Factor('Pressure', FactorType.CONTINUOUS, levels=[50, 100])
    ... ]
    >>> encode_design(design, factors)
       Temperature  Pressure
    0        -1.0      -1.0
    1         0.0       0.0
    2         1.0       1.0
    """
    encoded = design.copy()
    
    for factor in factors:
        if factor.is_continuous() and factor.name in encoded.columns:
            min_val, max_val = factor.levels
            encoded[factor.name] = encoded[factor.name].apply(
                lambda x: encode_value(x, min_val, max_val)
            )
    
    return encoded


def decode_design(
    design: pd.DataFrame,
    factors: List[Factor]
) -> pd.DataFrame:
    """
    Decode design matrix from coded to actual values.
    
    Only continuous factors are decoded. Discrete and categorical
    factors are left as-is.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with coded values
    factors : List[Factor]
        Factor definitions
    
    Returns
    -------
    pd.DataFrame
        Design matrix with actual values for continuous factors
    
    Examples
    --------
    >>> design = pd.DataFrame({
    ...     'Temperature': [-1.0, 0.0, 1.0],
    ...     'Pressure': [-1.0, 0.0, 1.0]
    ... })
    >>> factors = [
    ...     Factor('Temperature', FactorType.CONTINUOUS, levels=[150, 200]),
    ...     Factor('Pressure', FactorType.CONTINUOUS, levels=[50, 100])
    ... ]
    >>> decode_design(design, factors)
       Temperature  Pressure
    0        150.0      50.0
    1        175.0      75.0
    2        200.0     100.0
    """
    decoded = design.copy()
    
    for factor in factors:
        if factor.is_continuous() and factor.name in decoded.columns:
            min_val, max_val = factor.levels
            decoded[factor.name] = decoded[factor.name].apply(
                lambda x: decode_value(x, min_val, max_val)
            )
    
    return decoded


def is_design_coded(
    design: pd.DataFrame,
    factors: List[Factor],
    tolerance: float = 0.1
) -> bool:
    """
    Detect if design matrix contains coded or actual values.
    
    Checks if continuous factor values are approximately in range [-1, +1]
    (coded) or in their actual ranges (not coded).
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix to check
    factors : List[Factor]
        Factor definitions
    tolerance : float, optional
        Tolerance for considering values as coded (default: 0.1)
        Values between -1-tol and +1+tol are considered coded
    
    Returns
    -------
    bool
        True if design appears to be coded, False if actual values
    
    Examples
    --------
    >>> # Coded design
    >>> design_coded = pd.DataFrame({'Temp': [-1, 0, 1]})
    >>> is_design_coded(design_coded, factors)
    True
    
    >>> # Actual values
    >>> design_actual = pd.DataFrame({'Temp': [150, 175, 200]})
    >>> is_design_coded(design_actual, factors)
    False
    """
    coded_votes = 0
    actual_votes = 0

    for factor in factors:
        if factor.is_continuous() and factor.name in design.columns:
            values = design[factor.name].values
            min_val, max_val = factor.levels

            coded_min = -1 - tolerance
            coded_max = 1 + tolerance

            if values.min() >= coded_min and values.max() <= coded_max:
                coded_votes += 1
            elif values.min() >= min_val - tolerance and values.max() <= max_val + tolerance:
                actual_votes += 1
            # else: ambiguous factor, abstain

    if coded_votes == 0 and actual_votes == 0:
        # No continuous factors found or all were ambiguous
        return False

    return coded_votes > actual_votes


def encode_settings_dict(
    settings: Dict[str, Union[float, str]],
    factors: List[Factor]
) -> Dict[str, Union[float, str]]:
    """
    Encode factor settings dictionary from actual to coded values.
    
    Parameters
    ----------
    settings : Dict[str, Union[float, str]]
        Dictionary of factor names to actual values
    factors : List[Factor]
        Factor definitions
    
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary with coded values for continuous factors
    
    Examples
    --------
    >>> settings = {'Temperature': 175.0, 'Pressure': 75.0}
    >>> encode_settings_dict(settings, factors)
    {'Temperature': 0.0, 'Pressure': 0.0}
    """
    encoded = settings.copy()
    
    for factor in factors:
        if factor.is_continuous() and factor.name in encoded:
            min_val, max_val = factor.levels
            encoded[factor.name] = encode_value(
                encoded[factor.name], min_val, max_val
            )
    
    return encoded


def decode_settings_dict(
    settings: Dict[str, Union[float, str]],
    factors: List[Factor]
) -> Dict[str, Union[float, str]]:
    """
    Decode factor settings dictionary from coded to actual values.
    
    Parameters
    ----------
    settings : Dict[str, Union[float, str]]
        Dictionary of factor names to coded values
    factors : List[Factor]
        Factor definitions
    
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary with actual values for continuous factors
    
    Examples
    --------
    >>> settings = {'Temperature': 0.0, 'Pressure': 0.0}
    >>> decode_settings_dict(settings, factors)
    {'Temperature': 175.0, 'Pressure': 75.0}
    """
    decoded = settings.copy()
    
    for factor in factors:
        if factor.is_continuous() and factor.name in decoded:
            min_val, max_val = factor.levels
            decoded[factor.name] = decode_value(
                decoded[factor.name], min_val, max_val
            )
    
    return decoded
