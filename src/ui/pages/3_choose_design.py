"""
Step 2: Choose Design Type

Select and configure experimental design type.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from src.ui.utils.state_management import (
    initialize_session_state,
    can_access_step,
    invalidate_downstream_state
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.fractional_factorial import FractionalFactorial
from src.core.design_validation import validate_design, warnings_for
from src.core.selection import trim_non_estimable_terms
from src.ui.components.constraint_builder import (
    show_constraint_builder,
    show_constraint_help
)

_RESOLUTION_PRESENTATION = {
    3: (
        "III", "Screening Only", "🟥", "#FDECEC", "#B42318", "#7A271A",
        "Main effects may be confounded with two-factor interactions. "
        "Recommended only when run count is highly constrained."
    ),
    4: (
        "IV", "Good", "🟨", "#FFF4CC", "#B54708", "#7A2E0E",
        "Main effects are clear. Some two-factor interactions may be aliased "
        "with other interactions. Suitable for screening studies."
    ),
    5: (
        "V", "Excellent", "🟩", "#EAF7EE", "#2E7D32", "#1B5E20",
        "Main effects and two-factor interactions are estimable, assuming "
        "higher-order interactions are negligible. Suitable for "
        "characterization studies."
    ),
    6: (
        "VI", "Excellent", "🟩", "#EAF7EE", "#2E7D32", "#1B5E20",
        "Very high resolution design with minimal aliasing among low-order effects."
    ),
    7: (
        "VII", "Excellent", "🟩", "#EAF7EE", "#2E7D32", "#1B5E20",
        "Very high resolution design with minimal aliasing among low-order effects."
    )
}


def _render_resolution_card(resolution, total_runs):
    roman, quality, icon, background, border, text, interpretation = (
        _RESOLUTION_PRESENTATION[resolution]
    )
    with st.container(border=True):
        badge_col, details_col = st.columns([3, 1])
        with badge_col:
            st.markdown(
                f'<span style="display:inline-block;background:{background};'
                f'color:{text};border:1px solid {border};border-radius:999px;'
                f'padding:4px 10px;font-weight:700;font-size:0.9rem">'
                f'{icon}&nbsp; Resolution {roman} ({quality})</span>',
                unsafe_allow_html=True
            )
            st.caption(interpretation)
        with details_col:
            st.markdown(f"**Quality:** {quality}")
            st.markdown(f"**Estimated runs:** {total_runs}")


# Initialize state
initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar
build_standard_sidebar()

# Check access
if not can_access_step(3):
    st.warning("⚠️ Please complete Steps 1-2 first")
    st.stop()

st.title("Step 3: Choose Design Type")

factors = st.session_state['factors']

# Check if model is selected
model_selected = 'model_terms' in st.session_state and st.session_state['model_terms']

if not model_selected:
    st.warning(
        "⚠️ **No analysis model selected yet.** "
        "Some design types (especially D-Optimal) require knowing the model upfront. "
        "We recommend completing Step 2 first."
    )
    if st.button("← Return to Step 2: Select Model", type="primary"):
        st.switch_page("pages/2_select_model.py")
else:
    # Show selected model
    from src.ui.components.model_builder import format_full_equation
    model_terms = st.session_state['model_terms']
    equation = format_full_equation(model_terms, "Y")
    
    with st.expander("📋 Selected Model (from Step 2)", expanded=False):
        st.markdown(f"**{equation}**")
        st.caption(f"{len(model_terms)} terms to estimate")
        
        # Analyze model complexity for recommendations
        has_quadratic = any(t.startswith('I(') and '**2' in t for t in model_terms)
        has_interactions = any('*' in t and not t.startswith('I(') for t in model_terms)
        
        if has_quadratic:
            st.info("💡 Your model includes quadratic terms - Response Surface designs (CCD/Box-Behnken) are recommended.")
        elif has_interactions:
            st.info("💡 Your model includes interactions - Full Factorial or D-Optimal designs work well.")
        else:
            st.info("💡 Your model is linear - Most design types will work.")

def _predict_2_level_counts(design_type, config, factors):
    """Per-factor level counts implied by the chosen design, when determinable.

    Only designs that can *only* observe factors at two levels are handled
    here.  Designs whose level structure is not yet fixed (CCD, Box-Behnken,
    D-optimal, Latin hypercube, split-plot, imported/custom data) return an
    empty dict and defer to Step 4, where the generated design's real
    per-factor level counts are the authoritative estimability rule.
    """
    if design_type == "Fractional Factorial":
        return {f.name: 2 for f in factors if f.is_continuous()}
    if design_type == "Full Factorial":
        n_levels = int(config.get('n_levels', 2) or 2)
        n_center = int(config.get('n_center_points', 0) or 0)
        if n_levels >= 3 or n_center > 0:
            return {}
        return {f.name: 2 for f in factors if f.is_continuous()}
    return {}


def _apply_step2_quadratic_suppression(factors):
    """Early Step-3 correction: suppress (not merely warn about) quadratics a
    2-level design cannot estimate.

    With only two factor levels the squared column is constant (identical to
    the intercept), so the quadratic coefficient is not identifiable.  The
    removal persists in ``st.session_state['model_terms']``; the same
    estimability rule is re-applied authoritatively at Step 4 against the
    generated design's observed level counts.
    """
    design_type = st.session_state.get('design_type')
    config = st.session_state.get('design_config') or {}
    predicted = _predict_2_level_counts(design_type, config, factors)
    if not predicted:
        return []
    model_terms = st.session_state.get('model_terms') or []
    kept, removed = trim_non_estimable_terms(model_terms, factors, predicted)
    if removed:
        st.session_state['model_terms'] = kept
        st.session_state['suppressed_quadratics'] = list(removed)
        return list(removed)
    return [
        t for t in (st.session_state.get('suppressed_quadratics') or [])
        if t not in model_terms
    ]


def _show_quadratic_suppression_notice(suppressed):
    if not suppressed:
        return
    names = ", ".join(f"`{t}`" for t in suppressed)
    st.warning(
        f"**Non-estimable terms removed:** {names}\n\n"
        "This design observes continuous factors at only two levels, so "
        "quadratic terms are constant (aliased with the intercept) and "
        "cannot be estimated. Use a response-surface design (CCD or "
        "Box-Behnken) if you need curvature, or edit the model in Step 2."
    )


st.markdown(f"""
You have defined **{len(factors)} factors**. Now choose the design type that best suits your objectives.
""")

# Show factor summary
with st.expander("📋 View Defined Factors"):
    factor_summary = []
    for f in factors:
        if f.is_continuous():
            levels_str = f"[{f.levels[0]}, {f.levels[1]}]"
        else:
            levels_str = f"{len(f.levels)} levels"
        
        factor_summary.append({
            'Name': f.name,
            'Type': f.factor_type.value.replace('_', ' ').title(),
            'Levels': levels_str,
            'Changeability': f.changeability.value.title()
        })
    
    st.dataframe(pd.DataFrame(factor_summary), width='stretch', hide_index=True)

st.divider()

# Design type selection
st.subheader("Select Design Type")

# Determine available designs based on factors
#
# Availability is not decided here.  The shared validation service mirrors the
# constraints enforced by the generators in src/core/, so the UI cannot offer a
# design the backend will refuse, nor hide one it can build.  Adding a design
# type means updating src/core/design_validation.py, not this page.

# Design type descriptions
design_options = {
    "Full Factorial": {
        "description": "All possible combinations of factor levels. Best for small experiments.",
        "when_to_use": "2-4 factors, want to estimate all interactions",
        "runs": "2^k for 2-level factors (grows exponentially)"
    },
    "Fractional Factorial": {
        "description": "Subset of full factorial. Efficient screening for many factors.",
        "when_to_use": "3+ two-level factors, willing to sacrifice some interactions",
        "runs": "2^(k-p) where p is the fraction"
    },
    "Response Surface (CCD)": {
        "description": "Central Composite Design for quadratic models and optimization.",
        "when_to_use": "Continuous factors, need to model curvature",
        "runs": "2^k + 2k + center points"
    },
    "Response Surface (Box-Behnken)": {
        "description": "Efficient response surface design (no corner points).",
        "when_to_use": "3+ continuous factors, avoid extreme combinations",
        "runs": "Fewer than CCD, no axial points at ±α"
    },
    "D-Optimal": {
        "description": "Computer-generated optimal design with constraints.",
        "when_to_use": "Constrained design space, irregular regions, mixed factors",
        "runs": "User-specified (typically p+1 to 2p where p=parameters)"
    },
    "Latin Hypercube": {
        "description": "Space-filling design for exploration and screening.",
        "when_to_use": "Initial exploration, many factors, computer experiments",
        "runs": "User-specified (flexible)"
    },
    "Split-Plot": {
        "description": "Hierarchical design for hard-to-change factors.",
        "when_to_use": "Some factors are expensive or slow to change",
        "runs": "Based on whole-plot structure"
    }
}

design_validity = {
    name: validate_design(factors, name) for name in design_options
}

# Display design options
design_choice = None
current_design = st.session_state.get('design_type')

for design_name, design_info in design_options.items():
    if design_validity[design_name].is_valid:
        # Check if this is the selected design
        is_selected = (design_name == current_design)
        
        # Customize expander label based on selection
        if is_selected:
            expander_label = f"✅ {design_name} (Currently Selected)"
        else:
            expander_label = f"✓ {design_name}"
        
        with st.expander(expander_label, expanded=is_selected):
            st.markdown(f"**{design_info['description']}**")
            st.markdown(f"**When to use:** {design_info['when_to_use']}")
            st.markdown(f"**Typical runs:** {design_info['runs']}")
            
            if is_selected:
                st.info("🎯 This design is currently selected. Modify configuration below or choose a different design.")

            # Caveats that do not block selection, e.g. a stratified Latin
            # Hypercube over a categorical factor.
            for warning in design_validity[design_name].warnings:
                st.warning(warning.message)
            
            if st.button(f"Select {design_name}", key=f"select_{design_name}", disabled=is_selected, type="primary" if not is_selected else "secondary"):
                design_choice = design_name
    else:
        with st.expander(f"🔒 {design_name} (Not Available)", expanded=False):
            st.markdown(f"**{design_info['description']}**")
            
            # Explain why not available, using the same rules that locked it.
            for error in design_validity[design_name].errors:
                st.warning(error.message)

            # A project can be loaded with a design its own factors no longer
            # support (e.g. a factor gained a level after the project was
            # saved).  The selection and its configuration are preserved
            # untouched -- switching is the user's call -- but the mismatch is
            # called out here so it is not silently carried into generation.
            if design_name == current_design:
                st.error(
                    f"⚠️ **{design_name} is the design saved in this project, but "
                    f"the current factors do not support it.** The design type and "
                    f"its configuration have been kept as saved. Select a "
                    f"different design above, or adjust the factors back, to "
                    f"generate a design."
                )

# If user selected a design, show configuration
if design_choice:
    st.session_state['design_type'] = design_choice
    invalidate_downstream_state(from_step=2)
    st.rerun()

# If design already selected, show configuration
if st.session_state.get('design_type'):
    st.divider()
    st.subheader(f"⚙️ Configure {st.session_state['design_type']}")
    
    design_type = st.session_state['design_type']
    
    # Whether the selected design can actually be built from these factors.
    # The configuration widgets below stay rendered and editable either way, so
    # a loaded project's design is preserved as saved -- but the generator
    # previews further down raise ValueError for exactly this reason, and their
    # handlers would report it as a missing generator set.
    selected_design_valid = design_validity[design_type].is_valid

    # Configuration forms for each design type
    if design_type == "Full Factorial":
        st.markdown("**Full Factorial Configuration**")
        
        # Number of levels per factor (for continuous/discrete)
        n_levels = st.radio(
            "Number of Levels",
            [2, 3],
            help="2-level: factorial corners only. 3-level: includes midpoints.",
            horizontal=True
        )
        
        n_center_points = st.number_input(
            "Number of Center Points",
            min_value=0,
            max_value=10,
            value=3,
            help="Center points estimate pure error and check curvature"
        )
        
        n_replicates = st.number_input(
            "Number of Replicates",
            min_value=1,
            max_value=10,
            value=1,
            help="Full repetitions of entire design"
        )
        
        n_blocks = st.number_input(
            "Number of Blocks",
            min_value=1,
            max_value=10,
            value=1,
            help="Divide runs into blocks to account for nuisance variation. Use 1 for no blocking."
        )
        
        randomize = st.checkbox("Randomize Run Order", value=True)
        
        # Store config
        st.session_state['design_config'] = {
            'n_levels': n_levels,
            'n_center_points': n_center_points,
            'n_replicates': n_replicates,
            'n_blocks': n_blocks,
            'randomize': randomize
        }
        
        # 2-level full factorials without center points observe every factor
        # at exactly two levels, so quadratic terms are not estimable —
        # suppress them rather than only warning about them.
        if model_selected:
            _show_quadratic_suppression_notice(
                _apply_step2_quadratic_suppression(factors)
            )
        
        # Estimate runs
        base_runs = n_levels ** len(factors)
        total_runs = (base_runs + n_center_points) * n_replicates
        st.info(f"**Estimated runs:** {total_runs}")
    
    elif design_type == "Fractional Factorial":
        st.markdown("**Fractional Factorial Configuration**")
        
        # Check model compatibility: fractional factorials are 2-level, so
        # quadratic terms are not estimable — suppress them rather than only
        # warning about them.
        if model_selected:
            _show_quadratic_suppression_notice(
                _apply_step2_quadratic_suppression(factors)
            )
        
        k = len(factors)
        
        fractions = ["1/2", "1/4", "1/8", "1/16"]
        valid_fractions = []

        for frac in fractions:
            fraction_p = int(frac.split('/')[1]).bit_length() - 1
            if k - fraction_p >= 3 or (k == 3 and frac == "1/2"):
                valid_fractions.append(frac)

        # Open the widget on the fraction this project is actually configured
        # for.  design_config is restored verbatim on project load and is never
        # cleared by invalidate_downstream_state, but a selectbox with no
        # `index` always opens at 0 ('1/2'). That made the widget and the
        # stored configuration disagree -- the widget showed one fraction while
        # Step 4 generated from another. Fall back to 0 when the stored
        # fraction is no longer valid for this k.
        _stored_fraction = st.session_state.get('design_config', {}).get('fraction')
        _start_index = (
            valid_fractions.index(_stored_fraction)
            if _stored_fraction in valid_fractions
            else 0
        )
        fraction = st.selectbox(
            "Fraction Size",
            valid_fractions,
            index=_start_index,
            help="Smaller fractions = fewer runs but more aliasing"
        )

        p = int(fraction.split('/')[1]).bit_length() - 1
        total_runs = 2 ** (k - p)
        achieved_resolution = None

        generator_mode = st.radio(
            "Generator Specification",
            ["Standard (Recommended)", "Custom"],
            help="Standard generators from Montgomery/Box-Hunter-Hunter"
        )

        if generator_mode == "Custom":
            st.warning("Custom generators require knowledge of alias structure")

            generators_input = st.text_area(
                f"Generators (one per line, need {p})",
                placeholder="E=ABCD\nF=ABC",
                help="Format: NewFactor=Expression (e.g., E=ABCD)"
            )

            custom_generators = [g.strip() for g in generators_input.split('\n') if g.strip()]

            min_options = ["No minimum (as generated)"] + [f"Resolution {res}" for res in range(3, 8)]
            min_choice = st.selectbox(
                "Minimum Resolution (optional)",
                min_options,
                index=0,
                help="Your custom generators must achieve at least this resolution, "
                     "or the design will be rejected."
            )
            resolution = None if min_choice.startswith("No minimum") else int(min_choice.split()[-1])

            if custom_generators:
                try:
                    trial = FractionalFactorial(
                        factors=factors,
                        fraction=fraction,
                        resolution=resolution,
                        generators=custom_generators
                    )
                    achieved_resolution = trial.resolution
                    if resolution is not None and achieved_resolution < resolution:
                        st.error(
                            f"Your generators achieve Resolution {achieved_resolution}, "
                            f"below the selected minimum Resolution {resolution}."
                        )
                    else:
                        message = f"Generators valid — achieve Resolution {achieved_resolution}."
                        if resolution is not None:
                            message += " Meets the selected minimum."
                        st.success(message)
                except Exception as e:
                    st.error(f"Invalid generators: {e}")

        else:
            try:
                standard_preview = FractionalFactorial(
                    factors=factors,
                    fraction=fraction
                )
                achieved_resolution = standard_preview.resolution
            except ValueError:
                if not selected_design_valid:
                    # The ValueError is the blocking factor issue already
                    # reported above, not a missing generator set. Reporting it
                    # as the latter would send the user hunting for a fraction
                    # that can never work for these factors.
                    st.warning(
                        "Design preview paused: this design cannot be generated "
                        "from the current factors (see the reason above). "
                        "Resolve that first and the resolution badge will update."
                    )
                else:
                    alternatives = []
                    for alt_frac in valid_fractions:
                        alt_p = int(alt_frac.split('/')[1]).bit_length() - 1
                        if alt_p == p:
                            continue
                        try:
                            alt_resolution = FractionalFactorial(
                                factors=factors,
                                fraction=alt_frac
                            ).resolution
                        except ValueError:
                            continue
                        alternatives.append(
                            f"{alt_frac} → {2 ** (k - alt_p)} runs (Res {alt_resolution})"
                        )

                    guidance = (
                        f"No standard generator set exists for {k} factors at "
                        f"a {fraction} fraction ({total_runs} runs)."
                    )
                    if alternatives:
                        guidance += (
                            " Standard sets are available at: "
                            + ", ".join(alternatives)
                            + ". Choose one of these fractions, or switch to Custom generators."
                        )
                    else:
                        guidance += " Switch to Custom generators to build the design."
                    st.warning(guidance)
            resolution = None
            custom_generators = None

        if achieved_resolution is not None:
            _render_resolution_card(achieved_resolution, total_runs)
            if k == 3:
                saved_runs = 2 ** k - total_runs
                st.warning(
                    f"Only {saved_runs} runs are saved compared with the full "
                    "factorial. Consider a full factorial unless experimental "
                    "cost is extremely constrained."
                )

        n_blocks = st.number_input(
            "Number of Blocks",
            min_value=1,
            max_value=10,
            value=1,
            help="Divide runs into blocks to account for nuisance variation. Use 1 for no blocking."
        )

        randomize = st.checkbox("Randomize Run Order", value=True)

        st.session_state['design_config'] = {
            'fraction': fraction,
            'resolution': resolution,
            'generator_mode': generator_mode,
            'custom_generators': custom_generators,
            'n_blocks': n_blocks,
            'randomize': randomize
        }

        if achieved_resolution is None:
            st.info(f"**Estimated runs:** {total_runs}")
    
    elif design_type in ["Response Surface (CCD)", "Response Surface (Box-Behnken)"]:
        st.markdown(f"**{design_type} Configuration**")
        
        if "CCD" in design_type:
            alpha = st.selectbox(
                "Axial Distance (α)",
                ["Rotatable", "Orthogonal", "Face-centered (α=1)"],
                help="α determines axial point distance from center"
            )
            
# Map the display label to the semantic value consumed by the
            # core generator (mirrors rsm_config.alpha_for_label). The old code
            # computed a wrong numeric float here (k**0.5) that Step 4 ignored.
            if "Face-centered" in alpha:
                alpha_value = 'face'
            elif "Orthogonal" in alpha:
                alpha_value = 'orthogonal'
            else:  # Rotatable
                alpha_value = 'rotatable'
            
            st.session_state['design_config'] = {
                'alpha': alpha_value,
                'alpha_type': alpha
            }
            # Step 4 reads ccd_alpha when building the CCD.
            st.session_state['ccd_alpha'] = alpha_value
        else:
            st.session_state['design_config'] = {}
        
        n_center_points = st.number_input(
            "Number of Center Points",
            min_value=1,
            max_value=10,
            value=5,
            help="RSM designs need multiple center points"
        )
        
        st.session_state['design_config']['n_center_points'] = n_center_points
        
        # Live readout of the axial distance the core generator will use
        if "CCD" in design_type:
            from src.core.response_surface import CentralCompositeDesign
            
            try:
                _preview_ccd = CentralCompositeDesign(
                    factors=factors,
                    alpha=alpha_value,
                    center_points=n_center_points
                )
                st.caption(
                    f"Axial distance (α) with {n_center_points} center point(s): "
                    f"**{_preview_ccd.alpha:.4f}**"
                )
            except Exception as e:
                if not selected_design_valid:
                    st.caption(
                        "α unavailable: this design cannot be generated from "
                        "the current factors (see the reason above)."
                    )
                else:
                    st.caption(f"α unavailable: {e}")
        
        randomize = st.checkbox("Randomize Run Order", value=True)
        st.session_state['design_config']['randomize'] = randomize
        
        # Estimate runs
        k = len(factors)
        if "CCD" in design_type:
            factorial_runs = 2 ** k
            axial_runs = 2 * k
            total_runs = factorial_runs + axial_runs + n_center_points
        else:  # Box-Behnken
            # Approximate formula
            total_runs = 2 * k * (k - 1) + n_center_points
        
        st.info(f"**Estimated runs:** {total_runs}")
    
    elif design_type == "D-Optimal":
        st.markdown("**D-Optimal Design Configuration**")
        
        # Get model terms from Step 2
        if 'model_terms' not in st.session_state or not st.session_state['model_terms']:
            st.error("⚠️ No model defined. Please return to Step 2 to select analysis model.")
            if st.button("← Go to Step 2: Select Model", type="primary"):
                st.switch_page("pages/2_select_model.py")
            st.stop()
        
        model_terms = st.session_state['model_terms']
        
        # Display selected model
        from src.ui.components.model_builder import format_full_equation
        equation = format_full_equation(model_terms, "Y")
        
        st.info(
            f"🎯 **Selected Model (from Step 2):**\n\n"
            f"{equation}\n\n"
            f"This model has **{len(model_terms)} terms** to estimate. "
            f"D-optimal design will be optimized for these exact terms."
        )
        
        if st.button("✏️ Edit Model", key="edit_model_button"):
            st.switch_page("pages/2_select_model.py")
        
        st.divider()
        
        # Number of runs
        min_runs = len(model_terms)
        
        n_runs = st.number_input(
            f"Number of Runs (minimum: {min_runs})",
            min_value=min_runs,
            max_value=min_runs * 5,
            value=min_runs * 2,
            help="More runs = better precision"
        )
        
        st.divider()
        
        # Constraints (full implementation)
        show_constraint_builder(factors)
        show_constraint_help()
        
        # Store config with constraints
        constraints = st.session_state.get('constraints', [])
        
        st.session_state['design_config'] = {
            'n_runs': n_runs,
            'n_constraints': len(constraints),
            'model_terms': model_terms  # Store for preview/generation
        }
        
        if constraints:
            st.info(f"**Number of runs:** {n_runs} | **Constraints:** {len(constraints)}")
        else:
            st.info(f"**Number of runs:** {n_runs}")
    
    elif design_type == "Latin Hypercube":
        st.markdown("**Latin Hypercube Sampling Configuration**")
        
        n_runs = st.number_input(
            "Number of Runs",
            min_value=len(factors) + 1,
            max_value=1000,
            value=len(factors) * 10,
            help="Typically 5-10 times number of factors"
        )
        
        criterion = st.selectbox(
            "Optimization Criterion",
            ["None", "Maximin", "Correlation"],
            help="Criterion for optimizing space-filling"
        )
        
        st.session_state['design_config'] = {
            'n_runs': n_runs,
            'criterion': criterion
        }
        
        st.info(f"**Number of runs:** {n_runs}")
    
    elif design_type == "Split-Plot":
        st.markdown("**Split-Plot Design Configuration**")
        
        # Show which factors are hard/easy
        hard_factors = [f.name for f in factors if f.changeability == ChangeabilityLevel.HARD]
        very_hard_factors = [f.name for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
        easy_factors = [f.name for f in factors if f.changeability == ChangeabilityLevel.EASY]
        
        st.info(
            f"**Hard-to-change factors:** {', '.join(hard_factors + very_hard_factors)}\n\n"
            f"**Easy-to-change factors:** {', '.join(easy_factors)}"
        )
        
        n_whole_plots = st.number_input(
            "Number of Whole-Plots",
            min_value=2,
            max_value=32,
            value=4,
            help="Number of settings for hard-to-change factors"
        )
        
        n_subplot_runs = st.number_input(
            "Runs per Whole-Plot",
            min_value=2,
            max_value=16,
            value=4,
            help="Combinations of easy factors at each whole-plot setting"
        )
        
        randomize_whole_plots = st.checkbox("Randomize Whole-Plot Order", value=True)
        randomize_subplots = st.checkbox("Randomize Sub-Plot Order", value=True)
        
        st.session_state['design_config'] = {
            'n_whole_plots': n_whole_plots,
            'n_subplot_runs': n_subplot_runs,
            'randomize_whole_plots': randomize_whole_plots,
            'randomize_subplots': randomize_subplots
        }
        
        total_runs = n_whole_plots * n_subplot_runs
        st.info(f"**Total runs:** {total_runs}")
    
    # Generate button
    st.divider()
    
    if st.button("Generate Design →", type="primary", width='stretch'):
        st.session_state['current_step'] = 4
        st.switch_page("pages/4_preview_design.py")

# Navigation
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Model", width='stretch'):
        st.session_state['current_step'] = 2
        st.switch_page("pages/2_select_model.py")

with col2:
    if st.session_state.get('design_type') and st.session_state.get('design_config'):
        if st.button("Preview Design →", width='stretch'):
            st.session_state['current_step'] = 4
            st.switch_page("pages/4_preview_design.py")