"""
Factor Name Sanitization and Validation.

Factor names must be valid Python identifiers that are safe to embed in
Patsy / statsmodels formula strings.  This module provides utilities for
cleaning, validating, and suggesting alternatives for factor names that
would otherwise cause formula-parsing errors.

References
----------
.. [1] Patsy formula syntax: https://patsy.readthedocs.io/en/latest/formulas.html
"""

import re
import keyword
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Reserved name sets
# ---------------------------------------------------------------------------

# Tokens that have special meaning inside a Patsy formula string.
PATSY_RESERVED: List[str] = [
    'C', 'Q', 'I', 'T', 'cr', 'cc', 'bs', 'te', 'np',
    'Intercept', 'intercept',
]

# Common single-letter or short statistical names that would shadow
# built-ins or create ambiguous models.
COMMON_RESERVED: List[str] = [
    'y', 'x', 'n', 'k', 'p', 'df', 'se', 'sd',
    'mean', 'std', 'var', 'sum', 'min', 'max',
    'True', 'False', 'None',
]

# Combined set: Patsy tokens + statistical shorthands + Python keywords.
ALL_RESERVED: List[str] = PATSY_RESERVED + COMMON_RESERVED + list(keyword.kwlist)

# Characters illegal in Patsy factor names (formula operators and punctuation).
_ILLEGAL_CHARS_RE = re.compile(r'[^\w]')  # anything not alphanumeric / underscore
_CONSECUTIVE_UNDERSCORES_RE = re.compile(r'_+')


# ---------------------------------------------------------------------------
# Core sanitization
# ---------------------------------------------------------------------------


def sanitize_factor_name(name: str) -> Tuple[str, bool]:
    """
    Sanitize a factor name so it is safe inside Patsy formula strings.

    Transformations applied in order:

    1. Strip leading/trailing whitespace.
    2. Replace spaces and hyphens with underscores.
    3. Remove all characters that are not alphanumeric or underscores.
    4. Collapse consecutive underscores to a single underscore.
    5. Strip leading/trailing underscores.
    6. Prepend ``f_`` if the result starts with a digit.
    7. Prepend ``f_`` if the result is a Python keyword or reserved name.
    8. Fall back to ``factor`` if the result is still empty.

    Parameters
    ----------
    name : str
        Raw factor name (e.g. from user input or a CSV header).

    Returns
    -------
    clean_name : str
        Sanitized name guaranteed to be a valid Python identifier that is
        safe to use in a Patsy formula.
    was_modified : bool
        ``True`` if the returned name differs from the input.

    Examples
    --------
    >>> sanitize_factor_name("Temperature (°C)")
    ('Temperature_C', True)
    >>> sanitize_factor_name("2nd_run")
    ('f_2nd_run', True)
    >>> sanitize_factor_name("Pressure")
    ('Pressure', False)
    >>> sanitize_factor_name("for")
    ('f_for', True)

    Notes
    -----
    Callers should display a warning to the user whenever ``was_modified``
    is ``True`` so they are aware their column has been renamed.
    """
    original = name
    clean = name

    # 1. Strip whitespace.
    clean = clean.strip()

    # 2. Replace spaces and hyphens with underscores.
    clean = clean.replace(' ', '_').replace('-', '_')

    # 3. Remove illegal characters (keep alphanumerics and underscores).
    clean = _ILLEGAL_CHARS_RE.sub('', clean)

    # 4. Collapse consecutive underscores.
    clean = _CONSECUTIVE_UNDERSCORES_RE.sub('_', clean)

    # 5. Strip leading/trailing underscores.
    clean = clean.strip('_')

    # 6. Prepend f_ if starts with a digit.
    if clean and clean[0].isdigit():
        clean = 'f_' + clean

    # 7. Prepend f_ if reserved keyword.
    if clean in ALL_RESERVED:
        clean = 'f_' + clean

    # 8. Fallback.
    if not clean:
        clean = 'factor'

    was_modified = clean != original
    return clean, was_modified


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_factor_name(name: str) -> List[str]:
    """
    Validate a factor name and return a list of problems found.

    Does *not* modify the name — use :func:`sanitize_factor_name` for that.

    Parameters
    ----------
    name : str
        Factor name to validate.

    Returns
    -------
    List[str]
        Human-readable problem descriptions.  Empty list means the name is
        valid.

    Examples
    --------
    >>> validate_factor_name("Temp (°C)")
    ["Contains illegal characters: '(', '°', ')'"]
    >>> validate_factor_name("for")
    ["'for' is a Python keyword and cannot be used as a factor name"]
    >>> validate_factor_name("Temperature")
    []
    """
    problems: List[str] = []

    if not name or not name.strip():
        problems.append("Factor name cannot be empty.")
        return problems

    # Illegal characters.
    illegal = sorted({c for c in name if not (c.isalnum() or c == '_' or c == ' ')})
    if illegal:
        problems.append(
            f"Contains illegal characters: {', '.join(repr(c) for c in illegal)}"
        )

    # Starts with digit.
    stripped = name.strip()
    if stripped and stripped[0].isdigit():
        problems.append(
            f"Factor name '{name}' starts with a digit, which is not allowed."
        )

    # Python keyword.
    if stripped in keyword.kwlist:
        problems.append(
            f"'{name}' is a Python keyword and cannot be used as a factor name."
        )

    # Patsy reserved.
    if stripped in PATSY_RESERVED:
        problems.append(
            f"'{name}' is reserved by the Patsy formula parser and cannot be used."
        )

    return problems


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def suggest_alternative_names(name: str) -> List[str]:
    """
    Suggest valid alternative names for a factor that fails validation.

    Parameters
    ----------
    name : str
        The problematic factor name.

    Returns
    -------
    List[str]
        Up to three alternative name suggestions.  Always non-empty.

    Examples
    --------
    >>> suggest_alternative_names("for")
    ['f_for', 'for_factor', 'factor_1']
    >>> suggest_alternative_names("Temp (°C)")
    ['Temp_C', 'Temperature', 'factor_1']
    """
    clean, _ = sanitize_factor_name(name)
    suggestions: List[str] = []

    # Primary: sanitized version.
    if clean not in suggestions:
        suggestions.append(clean)

    # Secondary: append _factor.
    alt = clean + '_factor'
    if alt not in suggestions:
        suggestions.append(alt)

    # Tertiary: generic fallback.
    generic = 'factor_1'
    if generic not in suggestions:
        suggestions.append(generic)

    return suggestions[:3]


# ---------------------------------------------------------------------------
# Sanitization report
# ---------------------------------------------------------------------------


def get_sanitization_report(name: str) -> Dict:
    """
    Return a detailed report of the sanitization steps applied to *name*.

    Parameters
    ----------
    name : str
        Raw factor name.

    Returns
    -------
    dict
        Keys:

        ``original`` : str
            The input name.
        ``sanitized`` : str
            The cleaned name.
        ``was_modified`` : bool
            Whether any change was made.
        ``changes`` : List[str]
            Human-readable description of each transformation applied.
        ``problems`` : List[str]
            Validation problems found in the original name.

    Examples
    --------
    >>> report = get_sanitization_report("Temp (°C)")
    >>> report['changes']
    ["Removed illegal characters: '(', '°', ')'", 'Stripped trailing underscores']
    """
    clean, was_modified = sanitize_factor_name(name)
    problems = validate_factor_name(name)
    changes: List[str] = []

    if not was_modified:
        return {
            'original': name,
            'sanitized': clean,
            'was_modified': False,
            'changes': [],
            'problems': problems,
        }

    # Describe what changed.
    stripped = name.strip()
    if stripped != name:
        changes.append("Stripped leading/trailing whitespace.")

    spaces_or_hyphens = [c for c in stripped if c in (' ', '-')]
    if spaces_or_hyphens:
        changes.append("Replaced spaces/hyphens with underscores.")

    illegal = sorted({c for c in stripped if not (c.isalnum() or c in ('_', ' ', '-'))})
    if illegal:
        changes.append(
            f"Removed illegal characters: {', '.join(repr(c) for c in illegal)}"
        )

    if re.search(r'__', name):
        changes.append("Collapsed consecutive underscores.")

    if stripped.strip('_') != stripped:
        changes.append("Stripped leading/trailing underscores.")

    if clean.startswith('f_') and stripped and stripped[0].isdigit():
        changes.append("Prepended 'f_' because name started with a digit.")

    if clean.startswith('f_') and stripped in ALL_RESERVED:
        changes.append(f"Prepended 'f_' because '{stripped}' is a reserved name.")

    if not changes:
        changes.append("Name was modified (see sanitized value).")

    return {
        'original': name,
        'sanitized': clean,
        'was_modified': was_modified,
        'changes': changes,
        'problems': problems,
    }
