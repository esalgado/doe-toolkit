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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.core.factors import Factor, FactorType


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


# ---------------------------------------------------------------------------
# Typed design-space objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodingSpec:
    """
    Coding specification for a single continuous factor.

    Stores the linear mapping  ``x_coded = (x_natural - center) / scale``
    so that the low bound maps to -1 and the high bound maps to +1.

    Attributes
    ----------
    factor_name : str
        Name of the factor this spec belongs to.
    center : float
        Midpoint of the natural range, i.e. ``(low + high) / 2``.
    scale : float
        Half-range of the natural space, i.e. ``(high - low) / 2``.
        Must be strictly positive.

    Examples
    --------
    >>> spec = CodingSpec('Temperature', center=175.0, scale=25.0)
    >>> spec.encode(200.0)
    1.0
    >>> spec.decode(0.0)
    175.0
    """

    factor_name: str
    center: float
    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(
                f"CodingSpec for '{self.factor_name}': scale must be positive, "
                f"got {self.scale}"
            )

    def encode(self, natural_value: float) -> float:
        """
        Convert a natural-unit value to coded space.

        Parameters
        ----------
        natural_value : float
            Value in original engineering units.

        Returns
        -------
        float
            Coded value in [-1, +1].
        """
        return (natural_value - self.center) / self.scale

    def decode(self, coded_value: float) -> float:
        """
        Convert a coded value back to natural units.

        Parameters
        ----------
        coded_value : float
            Coded value, typically in [-1, +1].

        Returns
        -------
        float
            Value in original engineering units.
        """
        return self.center + coded_value * self.scale


@dataclass
class DesignSpace:
    """
    Unified description of a design's factor space and coding.

    Bundles the coding specification for every factor so that any
    DataFrame of design points can be unambiguously converted between
    natural and coded representations without touching factor definitions
    again.

    Attributes
    ----------
    factors : List[Factor]
        Ordered list of factor definitions (all types).
    coding_specs : Dict[str, CodingSpec]
        Mapping from factor name to its ``CodingSpec`` for every
        continuous factor.  Non-continuous factors have no entry.

    Examples
    --------
    >>> ds = DesignSpace.from_factors(factors)
    >>> coded_df = ds.encode_dataframe(natural_df)
    >>> natural_df2 = ds.decode_dataframe(coded_df)
    """

    factors: List[Factor]
    coding_specs: Dict[str, CodingSpec] = field(default_factory=dict)

    @classmethod
    def from_factors(cls, factors: List[Factor]) -> "DesignSpace":
        """
        Build a ``DesignSpace`` from a list of ``Factor`` objects.

        Continuous factors get a ``CodingSpec`` derived from their
        ``[low, high]`` levels.  Discrete and categorical factors are
        recorded in ``self.factors`` but have no coding spec.

        Parameters
        ----------
        factors : List[Factor]
            Factor definitions.

        Returns
        -------
        DesignSpace
            Fully initialised design space.

        Examples
        --------
        >>> ds = DesignSpace.from_factors(factors)
        """
        specs: Dict[str, CodingSpec] = {}
        for factor in factors:
            if factor.factor_type == FactorType.CONTINUOUS:
                low, high = factor.levels
                center = (low + high) / 2.0
                scale = (high - low) / 2.0
                specs[factor.name] = CodingSpec(
                    factor_name=factor.name,
                    center=center,
                    scale=scale,
                )
        return cls(factors=factors, coding_specs=specs)

    def encode_dataframe(self, natural_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode a natural-unit DataFrame to coded space.

        Only columns that correspond to continuous factors are transformed.
        Metadata columns (``StdOrder``, ``RunOrder``, ``Block``, etc.) and
        non-continuous factor columns are copied through unchanged.

        Parameters
        ----------
        natural_df : pd.DataFrame
            Design matrix in natural (engineering) units.

        Returns
        -------
        pd.DataFrame
            Design matrix with continuous factor columns in coded space.

        Notes
        -----
        The returned DataFrame shares no memory with the input.

        Examples
        --------
        >>> coded_df = ds.encode_dataframe(natural_df)
        """
        result = natural_df.copy()
        for name, spec in self.coding_specs.items():
            if name in result.columns:
                result[name] = result[name].apply(spec.encode)
        return result

    def decode_dataframe(self, coded_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode a coded DataFrame back to natural units.

        Only columns that correspond to continuous factors are transformed.
        Metadata columns and non-continuous factor columns are copied
        through unchanged.

        Parameters
        ----------
        coded_df : pd.DataFrame
            Design matrix with continuous factor columns in coded space.

        Returns
        -------
        pd.DataFrame
            Design matrix in natural (engineering) units.

        Examples
        --------
        >>> natural_df = ds.decode_dataframe(coded_df)
        """
        result = coded_df.copy()
        for name, spec in self.coding_specs.items():
            if name in result.columns:
                result[name] = result[name].apply(spec.decode)
        return result

    def encode_settings(self, settings: Dict[str, Union[float, str]]) -> Dict[str, Union[float, str]]:
        """
        Encode a factor-settings dict from natural to coded values.

        Parameters
        ----------
        settings : Dict[str, Union[float, str]]
            Factor name → natural value mapping.

        Returns
        -------
        Dict[str, Union[float, str]]
            Same mapping with continuous factors encoded to [-1, +1].

        Examples
        --------
        >>> ds.encode_settings({'Temperature': 175.0, 'Pressure': 75.0})
        {'Temperature': 0.0, 'Pressure': 0.0}
        """
        result = settings.copy()
        for name, spec in self.coding_specs.items():
            if name in result and isinstance(result[name], (int, float)):
                result[name] = spec.encode(float(result[name]))
        return result

    def decode_settings(self, settings: Dict[str, Union[float, str]]) -> Dict[str, Union[float, str]]:
        """
        Decode a factor-settings dict from coded to natural values.

        Parameters
        ----------
        settings : Dict[str, Union[float, str]]
            Factor name → coded value mapping.

        Returns
        -------
        Dict[str, Union[float, str]]
            Same mapping with continuous factors decoded to natural units.

        Examples
        --------
        >>> ds.decode_settings({'Temperature': 0.0, 'Pressure': 0.0})
        {'Temperature': 175.0, 'Pressure': 75.0}
        """
        result = settings.copy()
        for name, spec in self.coding_specs.items():
            if name in result and isinstance(result[name], (int, float)):
                result[name] = spec.decode(float(result[name]))
        return result

    @property
    def factor_names(self) -> List[str]:
        """
        Ordered list of all factor names.

        Returns
        -------
        List[str]
        """
        return [f.name for f in self.factors]

    @property
    def continuous_factor_names(self) -> List[str]:
        """
        Names of continuous factors only.

        Returns
        -------
        List[str]
        """
        return list(self.coding_specs.keys())


@dataclass
class DesignNatural:
    """
    A design matrix in natural (engineering) units with its design space.

    This is the canonical user-facing representation.  All display,
    export, and CSV round-trips use this object.  The ``design_space``
    attribute carries the coding specs needed to produce the coded form
    on demand.

    Attributes
    ----------
    df : pd.DataFrame
        Design matrix in natural units.  Includes factor columns plus
        metadata columns (``StdOrder``, ``RunOrder``, ``Block``, etc.).
    design_space : DesignSpace
        Unified factor and coding specification.

    Examples
    --------
    >>> dn = DesignNatural(df=natural_df, design_space=ds)
    >>> dc = dn.to_coded()
    """

    df: pd.DataFrame
    design_space: DesignSpace

    def to_coded(self) -> "DesignCoded":
        """
        Convert to coded representation.

        Returns
        -------
        DesignCoded
            The same design with continuous factor columns in coded space.

        Examples
        --------
        >>> dc = dn.to_coded()
        """
        return DesignCoded(
            df=self.design_space.encode_dataframe(self.df),
            design_space=self.design_space,
        )

    @property
    def factor_df(self) -> pd.DataFrame:
        """
        Slice containing only the factor columns (no metadata).

        Returns
        -------
        pd.DataFrame
        """
        cols = [c for c in self.design_space.factor_names if c in self.df.columns]
        return self.df[cols]


@dataclass
class DesignCoded:
    """
    A design matrix in coded space with its design space.

    This is the canonical mathematical representation used for model
    fitting, D-efficiency calculations, and all linear-algebra operations.
    Never displayed directly to the user.

    Attributes
    ----------
    df : pd.DataFrame
        Design matrix with continuous factor columns in [-1, +1] coded
        space.  Metadata columns (``StdOrder``, etc.) are preserved as-is.
    design_space : DesignSpace
        Unified factor and coding specification.

    Examples
    --------
    >>> dc = DesignCoded(df=coded_df, design_space=ds)
    >>> dn = dc.to_natural()
    """

    df: pd.DataFrame
    design_space: DesignSpace

    def to_natural(self) -> DesignNatural:
        """
        Convert to natural-unit representation.

        Returns
        -------
        DesignNatural
            The same design with continuous factor columns in natural units.

        Examples
        --------
        >>> dn = dc.to_natural()
        """
        return DesignNatural(
            df=self.design_space.decode_dataframe(self.df),
            design_space=self.design_space,
        )

    @property
    def factor_df(self) -> pd.DataFrame:
        """
        Slice containing only the factor columns (no metadata).

        Returns
        -------
        pd.DataFrame
        """
        cols = [c for c in self.design_space.factor_names if c in self.df.columns]
        return self.df[cols]
