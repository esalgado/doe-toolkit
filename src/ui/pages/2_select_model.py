"""
Step 2: Select Analysis Model

Define the statistical model BEFORE choosing the design type.
This ensures D-optimal and other model-dependent designs know the correct structure.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.ui.utils.state_management import (
    initialize_session_state,
    can_access_step,
    invalidate_downstream_state
)
from src.ui.components.model_builder import display_model_builder
from src.core.design_validation import DESIGN_TYPES, validate_design

# ==================== PAGE CONFIG ====================

# The two response surface designs are presented as one row on the
# compatibility panel.  When the whole group is blocked, the blocked list must
# name both designs -- otherwise Box-Behnken disappears from the panel without
# ever being reported as unusable.
RSM_GROUP_LABEL = "Response Surface (CCD, Box-Behnken)"

st.set_page_config(
    page_title="Select Model - DOE Toolkit",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar
build_standard_sidebar()

# ==================== ACCESS CONTROL ====================

if not can_access_step(2):
    st.error("⚠️ Complete Step 1 (Define Factors) first")
    st.stop()

# ==================== MAIN PAGE ====================

st.title("🔧 Step 2: Select Analysis Model")

st.markdown("""
### Why Define the Model First?

Different design types support different models:
- **D-optimal designs** require knowing the model upfront (they optimize for those specific terms)
- **Fractional factorial designs** may alias certain interactions
- **Full factorial designs** can fit any model up to full interactions

Define your anticipated model now to:
1. Enable design generation that supports your analysis goals
2. Get warnings if a design type can't fit your model
3. Ensure efficient experimental designs
""")

st.info("""
💡 **Don't worry if you're unsure!** You can always simplify the model during analysis or use 
augmentation to upgrade your design later.
""")

# ==================== MODEL DEFINITION ====================

# Get factors from state
factors = st.session_state.get('factors', [])

if not factors:
    st.warning("⚠️ No factors defined. Return to Step 1.")
    st.stop()

# Display current factors
with st.expander("📋 Defined Factors", expanded=False):
    factor_data = []
    for f in factors:
        factor_type = (
            "Continuous" if f.is_continuous() else
            "Discrete" if f.factor_type.name == "DISCRETE_NUMERIC" else
            "Categorical"
        )
        if f.is_continuous():
            levels_str = f"[{f.levels[0]}, {f.levels[1]}]"
        else:
            levels_str = f"{len(f.levels)} levels"
        
        factor_data.append({
            "Factor": f.name,
            "Type": factor_type,
            "Levels": levels_str,
            "Changeability": f.changeability.name.replace('_', ' ').title()
        })
    
    import pandas as pd
    st.dataframe(
        pd.DataFrame(factor_data),
        width='stretch',
        hide_index=True
    )

st.markdown("---")

# ==================== MODEL BUILDER ====================

# Initialize model terms if not present
if 'model_terms' not in st.session_state or st.session_state['model_terms'] is None:
    # Default: Linear model with intercept
    st.session_state['model_terms'] = ['1'] + [f.name for f in factors]

# Get current terms from session state
current_terms = st.session_state.get('model_terms', ['1'])

# Display model builder - this shows the UI and returns potentially updated terms
updated_terms = display_model_builder(
    factors=factors,
    current_terms=current_terms.copy(),  # Pass a copy to avoid mutation issues
    response_name="Y",
    key_prefix="step2"
)

# Check if terms actually changed
# Use list comparison (order matters for display)
terms_changed = (updated_terms != current_terms)

if terms_changed:
    # Terms changed - update session state immediately
    st.session_state['model_terms'] = updated_terms
    # Invalidate downstream steps since model changed  
    invalidate_downstream_state(from_step=2)
    # Rerun to refresh the entire page with new model
    st.rerun()

# ==================== MODEL COMPATIBILITY GUIDANCE ====================

st.markdown("---")

st.subheader("📊 Design Type Recommendations")

# Analyze model to suggest best designs
terms = st.session_state['model_terms']
has_intercept = '1' in terms
has_main_effects = any(t != '1' and '*' not in t and not t.startswith('I(') for t in terms)
has_interactions = any('*' in t and not t.startswith('I(') for t in terms)
has_quadratic = any(t.startswith('I(') and '**2' in t for t in terms)
has_cubic_or_higher = any(t.startswith('I(') and '**' in t and '**2' not in t for t in terms)

# Determine model complexity
if has_cubic_or_higher:
    model_complexity = "Very High (Cubic+ terms)"
elif has_quadratic:
    model_complexity = "High (Quadratic)"
elif has_interactions:
    model_complexity = "Medium (Interactions)"
else:
    model_complexity = "Low (Main effects only)"

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Model Complexity", model_complexity)
    st.metric("Terms to Estimate", len(terms))
    st.metric("Minimum Runs Needed", len(terms))

with col2:
    st.markdown("**✅ Compatible Design Types:**")
    
    compatible = []
    warnings = []
    blocked = []
    
    # Factor feasibility comes from the shared validation service, so this
    # panel agrees with what Step 3 will actually let you build.  The model
    # shape (curvature, interactions) is a separate, softer axis layered on
    # top below.
    feasible = {
        name: validate_design(factors, name, model_terms=terms)
        for name in DESIGN_TYPES
    }
    
    def _rsm_label(members, feasible):
        """Name the response surface designs that are actually available.

        Only meaningful on the available path: when every member is usable the
        merged group can be called "CCD, Box-Behnken", and when only CCD
        survives the block below never runs for this group.
        """
        if all(feasible[m].is_valid for m in members):
            return RSM_GROUP_LABEL
        return "Response Surface (CCD)"
    
    def _note(design_name):
        """Model-shape advice for a design whose factors are feasible."""
        if design_name == "Full Factorial":
            return "Can fit any model"
        if design_name == "Fractional Factorial":
            return "Check resolution for interactions"
        if design_name.startswith("Response Surface"):
            return "Designed for quadratic models"
        if design_name == "D-Optimal":
            return "Can fit this exact model efficiently"
        return "Good for main effects screening"
    
    # The two response surface designs differ only in their factor-count
    # minimum, so they are presented as one row rather than two near-identical
    # bullets.  Order matches Step 3.
    groups = [
        (["Full Factorial"], "Full Factorial"),
        (["Fractional Factorial"], "Fractional Factorial"),
        (["Response Surface (CCD)", "Response Surface (Box-Behnken)"], None),
        (["D-Optimal"], "D-Optimal"),
        (["Latin Hypercube"], "Latin Hypercube"),
        (["Split-Plot"], "Split-Plot"),
    ]
    
    for members, label in groups:
        usable = [m for m in members if feasible[m].is_valid]
        if not usable:
            # Hard blocker: no generator in this group can be built.  The
            # reason is resolved now, while the result object is in hand.
            blocked.append((
                label or RSM_GROUP_LABEL,
                feasible[members[0]].reason(),
            ))
            continue

        if label is None:
            label = _rsm_label(members, feasible)
        lead = usable[0]

        if has_quadratic and lead in ("Fractional Factorial", "Latin Hypercube"):
            warnings.append(f"⚠️ **{label}** - Cannot fit quadratic terms")
            continue
        if lead.startswith("Response Surface") and not has_quadratic:
            warnings.append(
                f"⚠️ **{label}** - Overqualified (no quadratic terms in model)"
            )
            continue
        if lead == "Latin Hypercube" and has_interactions:
            warnings.append(
                f"⚠️ **{label}** - Better for screening, not interaction models"
            )
            continue
        compatible.append(f"✅ **{label}** - {_note(lead)}")
        # Non-blocking caveats from the service, e.g. a stratified Latin
        # Hypercube over a categorical factor.
        for issue in feasible[lead].warnings:
            warnings.append(f"⚠️ **{label}** - {issue.message}")
    
    for item in compatible:
        st.markdown(item)
    
    if warnings:
        st.markdown("**Compatibility Notes:**")
        for item in warnings:
            st.markdown(item)

    if blocked:
        st.markdown("**Unavailable for the current factors:**")
        for blocked_label, reason in blocked:
            st.markdown(f"❌ **{blocked_label}** - {reason}")

# ==================== MODEL SUMMARY ====================

st.markdown("---")

st.subheader("📝 Model Summary")

from src.ui.components.model_builder import format_full_equation

equation = format_full_equation(terms, "Y")

# Display equation in a clean info box (matches analyze page style)
st.info(f"**Model Equation:**\n\n{equation}", icon="📐")

st.caption("""
This model will be used to:
- Guide design type selection (next step)
- Pre-populate analysis model (Step 6)
- Enable D-optimal design generation with correct structure
""")

# ==================== MODEL TERM BREAKDOWN ====================

with st.expander("🔍 Term Details", expanded=False):
    from src.ui.components.model_builder import format_term_for_display

    _TRANSFORM_STARTS_S2 = ("np.log(", "np.sqrt(", "np.exp(", "I(1/")

    def _is_nonlinear_s2(t: str) -> bool:
        return t.startswith("I(") or any(t.startswith(p) for p in _TRANSFORM_STARTS_S2)

    intercept_terms = [t for t in terms if t == '1']
    main_effects = [
        t for t in terms if t != '1' and not _is_nonlinear_s2(t) and '*' not in t
    ]
    interactions = [
        t for t in terms if not _is_nonlinear_s2(t) and '*' in t
    ]
    powers = [
        t for t in terms if _is_nonlinear_s2(t)
    ]

    if intercept_terms:
        st.markdown("**Intercept:**")
        for t in intercept_terms:
            st.write(f"  • {format_term_for_display(t)}")

    if main_effects:
        st.markdown("**Main Effects:**")
        for t in main_effects:
            st.write(f"  • {format_term_for_display(t)}")

    if interactions:
        st.markdown("**Interactions:**")
        for t in interactions:
            st.write(f"  • {format_term_for_display(t)}")

    if powers:
        st.markdown("**Power / Transform Terms:**")
        for t in powers:
            st.write(f"  • {format_term_for_display(t)}")

# ==================== NAVIGATION ====================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("← Back to Factors", width='stretch'):
        st.switch_page("pages/1_define_factors.py")

with col3:
    # Model selection complete if we have at least one term
    if len(terms) > 0:
        st.session_state['step_2_complete'] = True
        if st.button("Next: Choose Design →", width='stretch', type="primary"):
            st.switch_page("pages/3_choose_design.py")
    else:
        st.warning("Select at least one model term to continue")
