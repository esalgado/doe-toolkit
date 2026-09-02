"""
ANOVA Analysis Module for Design of Experiments.

Model Term Notation:
- Main effects: 'A', 'B', 'Temperature'
- Interactions: 'A*B', 'Temperature*Pressure'
- Quadratic: 'I(A**2)', 'I(Temperature**2)'
  (uses patsy I() identity operator for Python exponentiation)

Shared primitives (ANOVAResults, parse_model_term, enforce_hierarchy,
quadratic) live in ``analysis_base`` to avoid circular imports with
``split_plot_analysis``.
"""

import warnings
import re
from typing import List, Dict, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.coding import DesignSpace
from src.core.analysis_base import (
    ANOVAResults,
    enforce_hierarchy,
    parse_model_term,
    quadratic,
)
from src.core.split_plot_analysis import fit_split_plot_anova


def generate_model_terms(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    include_intercept: bool = True
) -> List[str]:
    """
    Generate standard model terms using Python/patsy notation.
    
    Raises
    ------
    ValueError
        If model_type='quadratic' but no continuous factors present
    
    Examples
    --------
    >>> generate_model_terms(factors, 'quadratic')
    ['1', 'A', 'B', 'A*B', 'I(A**2)', 'I(B**2)']
    """
    # Validate quadratic requirement
    if model_type == 'quadratic':
        if not any(f.is_continuous() for f in factors):
            raise ValueError(
                "Quadratic model requires at least one continuous factor. "
                "Use 'interaction' model for categorical factors only."
            )
        
    terms = []
    
    if include_intercept:
        terms.append('1')
    
    # Main effects
    factor_names = [f.name for f in factors]
    terms.extend(factor_names)
    
    # Two-way interactions
    if model_type in ('interaction', 'quadratic'):
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                terms.append(f"{factor_names[i]}*{factor_names[j]}")
    
    # Quadratic terms - use patsy I() notation directly
    if model_type == 'quadratic':
        for factor in factors:
            if factor.is_continuous():
                terms.append(f"I({factor.name}**2)")
    
    return terms


# parse_model_term is imported from analysis_base


# enforce_hierarchy is imported from analysis_base


# quadratic is imported from analysis_base


def detect_split_plot_structure(design: pd.DataFrame, factors: List[Factor]) -> Dict:
    """Detect split-plot structure from factor changeability."""
    very_hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
    hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.HARD]
    easy = [f.name for f in factors if f.changeability == ChangeabilityLevel.EASY]
    
    return {
        'is_split_plot': len(hard) > 0 or len(very_hard) > 0,
        'whole_plot_factors': very_hard + hard,
        'sub_plot_factors': easy,
        'has_blocking': 'Block' in design.columns,
        'whole_plot_column': 'WholePlot' if 'WholePlot' in design.columns else None
    }


def prepare_analysis_data(
    design: pd.DataFrame,
    response: Union[np.ndarray, pd.Series],
    factors: List[Factor],
    response_name: str = "Response"
) -> pd.DataFrame:
    """Prepare data for analysis."""
    if len(response) != len(design):
        raise ValueError(f"Response length mismatch: {len(response)} != {len(design)}")

    factor_names = [f.name for f in factors]
    analysis_df = design[factor_names].copy()

    analysis_df[response_name] = response
    
    for col in ['Block', 'WholePlot', 'VeryHardPlot', 'Replicate', 'RunOrder', 'StdOrder']:
        if col in design.columns:
            analysis_df[col] = design[col]
    
    return analysis_df



def validate_model_terms(terms: List[str], factors: List[Factor], design: pd.DataFrame) -> None:
    """Validate model terms compatibility."""
    factor_dict = {f.name: f for f in factors}

    # Operators that involve a transform — the primary factor must be continuous
    _transform_operators = {
        'transform', 'transform_power', 'transform_cross', 'transform_power_cross'
    }

    for term in terms:
        if term == '1':
            continue

        factor_list, operator = parse_model_term(term)

        for fname in factor_list:
            if fname not in factor_dict:
                raise ValueError(f"Factor '{fname}' in '{term}' not found")

        if operator == '**':
            # Plain quadratic: I(A**2)
            factor = factor_dict[factor_list[0]]
            if not factor.is_continuous():
                raise ValueError(f"Quadratic '{term}' requires continuous factor")
            unique_vals = design[factor.name].nunique()
            if unique_vals <= 2:
                warnings.warn(f"Quadratic '{term}': only {unique_vals} levels")

        elif operator in _transform_operators:
            # Transform terms: the first factor in factor_list is always the
            # raw factor being transformed — it must be continuous.
            primary_factor = factor_dict[factor_list[0]]
            if not primary_factor.is_continuous():
                raise ValueError(
                    f"Transform term '{term}' requires a continuous factor; "
                    f"'{factor_list[0]}' is not continuous."
                )


# ANOVAResults is imported from analysis_base


class ANOVAAnalysis:
    """ANOVA analysis for experimental designs."""

    def __init__(
        self,
        design: pd.DataFrame,
        response: Union[np.ndarray, pd.Series],
        factors: List[Factor],
        response_name: str = "Response",
        is_split_plot: Optional[bool] = None,
        block_as_random: bool = False
    ):
        self.factors = factors
        self.response = np.array(response)
        self.response_name = response_name
        self.block_as_random = block_as_random

        # Build the design space and encode to coded [-1, +1] space for the
        # primary analysis data.  The incoming ``design`` may be in natural
        # units (the canonical session-state format since Phase 2) or already
        # coded (legacy callers).  Using DesignSpace guarantees the coded
        # matrix is always available for polynomial models.
        #
        # Transform terms (np.log, np.sqrt, I(1/x), np.exp) must be evaluated
        # in natural units — applying log to a coded value of -1 produces NaN.
        # self.natural_data retains the original factor values so that Patsy
        # can correctly evaluate transform terms when they appear in a model.
        self._design_space = DesignSpace.from_factors(factors)
        self.design = self._design_space.encode_dataframe(design)

        self.data = prepare_analysis_data(
            self.design, response, factors, response_name
        )
        # Natural-unit data frame: decoded back from coded so it is always
        # consistent with self.design regardless of what the caller passed in.
        self.natural_data = prepare_analysis_data(
            self._design_space.decode_dataframe(self.design), response, factors, response_name
        )
        self.rename_map = {}
        self.design_structure = detect_split_plot_structure(self.design, factors)

        if is_split_plot is not None:
            self.design_structure['is_split_plot'] = is_split_plot

        self.current_model = None
        self.current_results = None
    
    def fit(self, model_terms: List[str], enforce_hierarchy_flag: bool = True) -> ANOVAResults:
        """Fit ANOVA model. Use I(A**2) notation for quadratic terms."""
        validate_model_terms(model_terms, self.factors, self.design)
        
        if enforce_hierarchy_flag:
            factor_names = [f.name for f in self.factors]
            complete_terms, added = enforce_hierarchy(model_terms, factor_names)
            
            if added:
                warnings.warn(f"Added for hierarchy: {added}")
            
            model_terms = complete_terms
        
        self.current_model = model_terms
        self._validate_degrees_of_freedom(model_terms)
        if self.design_structure['is_split_plot']:
            self._validate_split_plot_degrees_of_freedom()
            results = self._fit_mixed_effects_model(model_terms)
        else:
            results = self._fit_fixed_effects_model(model_terms)
        
        self.current_results = results
        return results
    
    @staticmethod
    def _has_transform_terms(model_terms: List[str]) -> bool:
        """
        Return True if any term in *model_terms* is a transform term.

        Transform terms require evaluation in natural (engineering) units.
        Patsy cannot evaluate ``np.log(A)`` meaningfully when ``A`` is coded
        to ``[-1, +1]`` because negative coded values produce NaN/complex
        results for log and sqrt.

        Parameters
        ----------
        model_terms : List[str]
            Model terms in patsy notation.

        Returns
        -------
        bool
        """
        from src.core.analysis_base import _TRANSFORM_PREFIXES
        return any(
            any(term.startswith(pfx) for pfx in _TRANSFORM_PREFIXES)
            or (term.startswith('I(') and any(
                pfx in term for pfx in _TRANSFORM_PREFIXES
            ))
            for term in model_terms
        )

    def _select_fit_data(self, model_terms: List[str]) -> pd.DataFrame:
        """
        Choose the data frame to pass to Patsy based on whether transform
        terms are present.

        - Polynomial-only models use ``self.data`` (coded space): coefficients
          are comparable across factors and numerically stable.
        - Models containing any transform term use ``self.natural_data``
          (natural units): transforms are applied to meaningful physical
          values, not to arbitrary [-1, +1] coded values.

        Parameters
        ----------
        model_terms : List[str]
            Model terms in patsy notation.

        Returns
        -------
        pd.DataFrame
            The appropriate data frame for OLS fitting.
        """
        if self._has_transform_terms(model_terms):
            return self.natural_data
        return self.data

    def _fit_fixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """Fit fixed effects ANOVA."""
        formula = self._build_formula(model_terms)
        fit_data = self._select_fit_data(model_terms)

        if self.design_structure['has_blocking'] and not self.block_as_random:
            # Cast Block to str so patsy treats it as categorical without
            # requiring C() notation (which conflicts with factors named "C").
            fit_data = fit_data.copy()
            fit_data['Block'] = fit_data['Block'].astype(str)
            formula += " + Block"
            fitted_model = ols(formula, data=fit_data).fit()
        elif self.design_structure['has_blocking'] and self.block_as_random:
            model = mixedlm(formula, data=fit_data, groups=fit_data['Block'], re_formula='1')
            fitted_model = model.fit(method='lbfgs')
        else:
            fitted_model = ols(formula, data=fit_data).fit()

        return self._build_results_object(fitted_model, model_terms, False)
    
    def _fit_mixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """
        Fit a two-strata split-plot ANOVA model.

        Delegates to split_plot_analysis.fit_split_plot_anova which uses the
        Yates / expected-mean-squares approach:
        - Whole-plot terms tested against whole-plot error (MS_WP)
        - Subplot terms tested against subplot error (MS_SP)

        This avoids the LinAlgError that occurs when whole-plot factors are
        passed to mixedlm alongside a random WholePlot intercept, which makes
        the design matrix singular because hard factors do not vary within any
        whole-plot.
        """
        if self.design_structure['whole_plot_column'] is None:
            raise ValueError(
                "Split-plot analysis requires a 'WholePlot' column in the design. "
                "Ensure the design was generated with hard-to-change factors so that "
                "whole-plot groupings are recorded."
            )

        return fit_split_plot_anova(
            data=self._select_fit_data(model_terms),
            factors=self.factors,
            model_terms=model_terms,
            response_name=self.response_name,
            whole_plot_col=self.design_structure['whole_plot_column'],
            design_structure=self.design_structure,
        )
    
    def _build_formula(self, model_terms: List[str]) -> str:
        """Build formula - terms already in patsy notation."""
        terms = [t for t in self._wrap_categorical_terms(model_terms) if t != '1']
        if not terms:
            # Intercept-only model
            formula_rhs = '1'
        else:
            formula_rhs = ' + '.join(terms)
        return f"{self.response_name} ~ {formula_rhs}"

    def _wrap_categorical_terms(self, model_terms: List[str]) -> List[str]:
        """
        Wrap categorical factor names in patsy ``C(...)`` so they are always
        treated as categorical, regardless of the data column dtype.

        Without this, a categorical factor whose levels are numeric-looking
        labels (e.g. lot/batch IDs) can be silently fitted as a continuous
        predictor: patsy decides categorical-vs-continuous purely from the
        column dtype, and ``float64`` columns get a single linear coefficient
        (DF=1) instead of k-1 dummy columns.
        """
        categorical = {f.name for f in self.factors if f.is_categorical()}
        if not categorical:
            return list(model_terms)

        wrapped_terms = []
        for term in model_terms:
            if term == '1':
                wrapped_terms.append(term)
                continue
            new_term = term
            for name in sorted(categorical, key=len, reverse=True):
                if re.search(r'C\(' + re.escape(name) + r'\)', new_term):
                    continue
                pattern = r'(?<![\w)])' + re.escape(name) + r'(?![\w])'
                new_term = re.sub(pattern, f'C({name})', new_term)
            wrapped_terms.append(new_term)
        return wrapped_terms
    
    @staticmethod
    def _strip_c_wrappers(label: str) -> str:
        """
        Remove patsy ``C(...)`` wrappers from a term label so results are
        reported using plain factor names, e.g.:
          ``C(Egg_lot)`` -> ``Egg_lot``
          ``C(Egg_lot)[T.41007666]`` -> ``Egg_lot[T.41007666]``
          ``C(Egg_lot):Egg_percent`` -> ``Egg_lot:Egg_percent``
        """
        return re.sub(r'C\(([^()]*)\)', r'\1', label)

    def _build_results_object(self, fitted_model, model_terms: List[str], is_split_plot: bool) -> ANOVAResults:
        """Build results from fitted model."""
        try:
            if hasattr(fitted_model, 'anova_table'):
                anova_table = fitted_model.anova_table()
            else:
                anova_table = sm.stats.anova_lm(fitted_model, typ=2)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Could not compute ANOVA: {e}")
            anova_table = pd.DataFrame()
        
        if not anova_table.empty:
            anova_table = anova_table.copy()
            anova_table.index = [
                self._strip_c_wrappers(str(i)) for i in anova_table.index
            ]
        
        effect_estimates = pd.DataFrame({
            'Coefficient': fitted_model.params,
            'Std_Error': fitted_model.bse,
            't_value': fitted_model.tvalues,
            'p_value': fitted_model.pvalues
        })
        effect_estimates.index = [
            self._strip_c_wrappers(str(i)) for i in effect_estimates.index
        ]
        residuals = fitted_model.resid
        fitted_values = fitted_model.fittedvalues
        
        if hasattr(fitted_model, 'rsquared'):
            r_squared = fitted_model.rsquared
            adj_r_squared = fitted_model.rsquared_adj
        else:
            ss_total = np.sum((self.data[self.response_name] - self.data[self.response_name].mean())**2)
            ss_resid = np.sum(residuals**2)
            r_squared = 1 - ss_resid / ss_total
            n, p = len(self.data), len(fitted_model.params)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        rmse = np.sqrt(np.mean(residuals**2))
        diagnostics = self._compute_diagnostics(residuals, fitted_values)
        
        # LogWorth
        logworth_df = effect_estimates[effect_estimates.index != 'Intercept'].copy()
        logworth_values = []
        for p in logworth_df['p_value']:
            if pd.isna(p) or p <= 0:
                logworth_values.append(np.nan)
            elif p < 1e-16:
                logworth_values.append(16.0)
            else:
                logworth_values.append(-np.log10(p))
        logworth_df['LogWorth'] = logworth_values
        
        return ANOVAResults(
            anova_table=anova_table,
            effect_estimates=effect_estimates,
            logworth=logworth_df,
            residuals=residuals,
            fitted_values=fitted_values,
            fitted_model=fitted_model,
            diagnostics=diagnostics,
            model_terms=model_terms,
            is_split_plot=is_split_plot,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            rmse=rmse
        )
    
    def _compute_diagnostics(self, residuals: np.ndarray, fitted_values: np.ndarray) -> Dict:
        """Compute diagnostics."""
        diagnostics = {}
        if len(residuals) <= 5000:
            stat, pval = stats.shapiro(residuals)
            diagnostics['shapiro_wilk'] = {'statistic': stat, 'p_value': pval}
        return diagnostics
    
    def _validate_degrees_of_freedom(self, model_terms: List[str]) -> None:
        """Validate DF and warn about saturation."""
        n_runs = len(self.data)
        n_params = 1 + self._estimate_n_parameters(model_terms)
        df_error = n_runs - n_params

        if df_error < 0:
            raise ValueError(
                f"Model is oversaturated: {n_runs} runs for {n_params} parameters "
                f"(df_error = {df_error}). Reduce model complexity or add more runs."
            )
        elif df_error == 0:
            warnings.warn(
                f"Model is saturated (df_error = 0): No degrees of freedom for error estimation. "
                f"Statistical tests will not be valid. Consider adding runs or simplifying model."
            )
        elif df_error < 3:
            warnings.warn(f"Low df_error = {df_error}: Inference may be unreliable")

    def _estimate_n_parameters(self, model_terms: List[str]) -> int:
        """
        Estimate the number of model parameters consumed by each term,
        counting categorical dummy expansion.

        A categorical main effect with k levels is fit as ``k-1`` dummy
        columns; an interaction ``A*B`` consumes ``df_A * df_B`` columns, where
        ``df`` of a categorical factor is ``k-1`` and ``1`` otherwise.
        """
        factor_df: Dict[str, int] = {}
        for factor in self.factors:
            if factor.is_categorical():
                n_levels = len(factor.levels)
                if n_levels < 2:
                    n_levels = self.data[factor.name].nunique()
                factor_df[factor.name] = max(n_levels - 1, 0)
            else:
                factor_df[factor.name] = 1

        total = 0
        for term in model_terms:
            if term == '1':
                continue
            factor_list, op = parse_model_term(term)
            if op == '*':
                dfs = [factor_df.get(name, 1) for name in factor_list]
                if not dfs or any(df <= 0 for df in dfs):
                    continue
                total += int(np.prod(dfs))
            else:
                name = factor_list[0] if factor_list else ''
                total += factor_df.get(name, 1)
        return total

    def _validate_split_plot_degrees_of_freedom(self) -> None:
        """Warn when the whole-plot stratum has too few groups for reliable inference."""
        whole_plot_col = self.design_structure.get('whole_plot_column')
        if whole_plot_col is None:
            return
        n_wp = self.data[whole_plot_col].nunique()
        if n_wp < 3:
            warnings.warn(f"Only {n_wp} whole-plots, need ≥3 for reliable whole-plot inference.")
    
    def update_model(
        self,
        terms_to_add: Optional[List[str]] = None,
        terms_to_remove: Optional[List[str]] = None,
        enforce_hierarchy_flag: bool = True
    ) -> ANOVAResults:
        """Update model by adding/removing terms."""
        if self.current_model is None:
            raise ValueError("No model fitted yet")
        
        new_terms = self.current_model.copy()
        
        if terms_to_add:
            new_terms.extend(terms_to_add)
        
        if terms_to_remove:
            new_terms = [t for t in new_terms if t not in terms_to_remove]
        
        seen = set()
        new_terms = [t for t in new_terms if not (t in seen or seen.add(t))]
        
        return self.fit(new_terms, enforce_hierarchy_flag)

    def coded_single_df_coefficients(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Design-Expert style coded regression coefficients for single-df terms.

        Design Expert reports half-normal effect magnitudes using the *coded*
        factor scale (numeric factors re-encoded to [-1, +1]).  The main fit
        keeps numeric factors in natural units, so `AnovaAnalysis` re-fits the
        active model here with numeric columns re-encoded ``(x - mid) / half``;
        each single-df slope is then directly comparable to Design Expert's
        coded coefficient (for this design, ``% egg yolk`` -> ~5.33).

        Interaction terms whose full-model p-value exceeds ``alpha`` are
        dropped from the refit, mirroring Design Expert's automatic model
        reduction when reporting coefficients (the ANOVA table still reflects
        the full model).  Blocking is retained exactly as in the main fit.

        Parameters
        ----------
        alpha : float
            Significance threshold for retaining interaction terms in the
            refit.

        Returns
        -------
        Dict[str, float]
            Mapping ``{base_term: coded_coefficient}`` for every single-df
            term in the current model.  Multi-df terms, the intercept and
            nuisance (Block) terms are omitted.  An empty dict is returned
            whenever the coded refit is not well defined (split-plot /
            mixed-block / transform models, or no fitted model).
        """
        if self.current_model is None or self.current_results is None:
            return {}
        if self.current_results.is_split_plot or self.block_as_random:
            return {}
        if self._has_transform_terms(self.current_model):
            return {}

        refit = self._code_refit(alpha)
        if refit is None:
            return {}
        fitted, reduced = refit
        params = {
            self._strip_c_wrappers(str(k)): float(v)
            for k, v in fitted.params.items()
        }

        result: Dict[str, float] = {}
        for term in reduced:
            base = term.replace('*', ':')
            matches = {
                key: value
                for key, value in params.items()
                if re.sub(r'\[[^\]]*\]', '', key) == base
            }
            if len(matches) != 1:
                continue
            key, value = next(iter(matches.items()))
            if '[' in key:
                result[base] = value / 2.0
            else:
                result[base] = value
        return result

    def _code_refit(self, alpha: float = 0.05):
        """
        Re-fit the active model with numeric factors re-encoded to the coded
        [-1, +1] scale used by Design Expert.  Interaction terms whose
        full-model p-value exceeds ``alpha`` are dropped (mirroring Design
        Expert's automatic model reduction when reporting coefficients; the
        ANOVA table still reflects the full model).  Blocking is retained
        exactly as in the main fit.

        Returns
        -------
        tuple or None
            ``(fitted, reduced)`` where *fitted* is the statsmodels OLS
            result on the coded data and *reduced* is the list of retained
            model terms.  ``None`` if the coded refit is not well defined.
        """
        model_terms = [t for t in self.current_model if t != '1']
        anova = self.current_results.anova_table
        if model_terms and anova is not None and not anova.empty:
            p_col = next(
                (c for c in ('PR(>F)', 'F_p_value', 'p_value')
                 if c in anova.columns),
                None,
            )
        else:
            p_col = None

        mains = [t for t in model_terms if '*' not in t]
        interactions = [t for t in model_terms if '*' in t]
        reduced = list(mains)
        for term in interactions:
            base = term.replace('*', ':')
            p = None
            if p_col is not None and base in anova.index:
                p = anova.loc[base, p_col]
            if p is None or pd.isna(p) or float(p) <= alpha:
                reduced.append(term)
        if not reduced:
            return None

        coded = self.data.copy()
        for factor in self.factors:
            if not (factor.is_continuous() or factor.is_discrete_numeric()):
                continue
            lo, hi = factor.min_value, factor.max_value
            if lo is None or hi is None or not (hi > lo):
                continue
            if factor.name not in coded.columns:
                continue
            try:
                values = coded[factor.name].astype(float)
            except (TypeError, ValueError):
                continue
            half = (hi - lo) / 2.0
            coded[factor.name] = (values - (lo + hi) / 2.0) / half

        blocking = bool(self.design_structure.get('has_blocking'))
        formula_terms = [
            t for t in self._wrap_categorical_terms(reduced) if t != '1'
        ]
        rhs = ' + '.join(formula_terms) if formula_terms else '1'
        formula = f"{self.response_name} ~ {rhs}"
        warnings.simplefilter('ignore')
        if blocking:
            coded['Block'] = coded['Block'].astype(str)
            formula += " + Block"
        try:
            fitted = ols(formula, data=coded).fit()
        except Exception:
            fitted = None
        finally:
            warnings.resetwarnings()
        if fitted is None:
            return None
        return fitted, reduced

    def coded_effect_se(self, alpha: float = 0.05) -> Optional[float]:
        """
        Standard error of a coded ([-1, +1]) single-df effect, from the coded
        reference refit used for the half-normal panel: ``sqrt(2 * MSE / N)``.

        This is the fully data-driven slope for the Design-Expert style
        reference diagonal (there are no fitted per-experiment constants),
        so it transfers to any design and keeps a significant term standing
        clearly off the line.

        Returns
        -------
        Optional[float]
            The coded-effect standard error, or ``None`` when the coded refit
            is not well defined.
        """
        if self.current_model is None or self.current_results is None:
            return None
        if self.current_results.is_split_plot or self.block_as_random:
            return None
        if self._has_transform_terms(self.current_model):
            return None
        refit = self._code_refit(alpha)
        if refit is None:
            return None
        fitted, _ = refit
        n = fitted.nobs
        scale = float(getattr(fitted, 'scale', np.nan))
        if not bool(n) or not np.isfinite(scale) or scale <= 0:
            return None
        return float(np.sqrt(2.0 * scale / float(n)))