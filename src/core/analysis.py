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
    compute_actual_coefficients,
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
    
    for term in terms:
        if term == '1':
            continue
        
        factor_list, operator = parse_model_term(term)
        
        for fname in factor_list:
            if fname not in factor_dict:
                raise ValueError(f"Factor '{fname}' in '{term}' not found")
        
        if operator == '**':
            factor = factor_dict[factor_list[0]]
            if not factor.is_continuous():
                raise ValueError(f"Quadratic '{term}' requires continuous factor")
            
            unique_vals = design[factor.name].nunique()
            if unique_vals <= 2:
                warnings.warn(f"Quadratic '{term}': only {unique_vals} levels")


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

        # Build the design space and always encode to coded [-1, +1] space
        # before analysis.  The incoming ``design`` may be in natural units
        # (the canonical session-state format since Phase 2) or already coded
        # (legacy callers).  Using DesignSpace guarantees the model is always
        # fit on a coded matrix, making coefficients directly comparable and
        # compute_actual_coefficients() unconditionally correct.
        self._design_space = DesignSpace.from_factors(factors)
        self.design = self._design_space.encode_dataframe(design)

        self.data = prepare_analysis_data(
            self.design, response, factors, response_name
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
    
    def _fit_fixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """Fit fixed effects ANOVA."""
        formula = self._build_formula(model_terms)

        if self.design_structure['has_blocking'] and not self.block_as_random:
            # Cast Block to str so patsy treats it as categorical without
            # requiring C() notation (which conflicts with factors named "C").
            fit_data = self.data.copy()
            fit_data['Block'] = fit_data['Block'].astype(str)
            formula += " + Block"
            fitted_model = ols(formula, data=fit_data).fit()
        elif self.design_structure['has_blocking'] and self.block_as_random:
            model = mixedlm(formula, data=self.data, groups=self.data['Block'], re_formula='1')
            fitted_model = model.fit(method='lbfgs')
        else:
            fitted_model = ols(formula, data=self.data).fit()
        
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
            data=self.data,
            factors=self.factors,
            model_terms=model_terms,
            response_name=self.response_name,
            whole_plot_col=self.design_structure['whole_plot_column'],
            design_structure=self.design_structure,
        )
    
    def _build_formula(self, model_terms: List[str]) -> str:
        """Build formula - terms already in patsy notation."""
        terms = [t for t in model_terms if t != '1']
        if not terms:
            # Intercept-only model
            formula_rhs = '1'
        else:
            formula_rhs = ' + '.join(terms)
        return f"{self.response_name} ~ {formula_rhs}"
    
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
        
        effect_estimates = pd.DataFrame({
            'Coefficient': fitted_model.params,
            'Std_Error': fitted_model.bse,
            't_value': fitted_model.tvalues,
            'p_value': fitted_model.pvalues
        })
        effect_estimates = compute_actual_coefficients(effect_estimates, self.factors)
        
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
        n_params = len([t for t in model_terms if t != '1']) + 1
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