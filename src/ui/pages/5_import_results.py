"""
Step 5: Import Experimental Results (DOE-Toolkit CSV Only)

Simplified import workflow - only accepts DOE-Toolkit formatted CSVs with metadata.
Users must use the template format for all imports.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from typing import Dict, List, Optional

from src.ui.utils.state_management import (
    initialize_session_state,
    invalidate_downstream_state
)
from src.core.factors import Factor, FactorType
from src.ui.utils.csv_parser import (
    parse_doe_csv,
    validate_csv_structure,
    ParseResult
)


def extract_responses_from_design(
    design_data: pd.DataFrame,
    factors: List[Factor],
    response_definitions: Optional[List[Dict]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract response columns from design DataFrame.
    
    Parameters
    ----------
    design_data : pd.DataFrame
        Design with factor and response columns
    factors : List[Factor]
        Factor definitions to exclude from responses
    response_definitions : Optional[List[Dict]]
        Optional list of response definitions to filter
    
    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of response names to data arrays
    """
    factor_names = {f.name for f in factors}
    meta_cols = {'StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase'}
    
    # Get expected response names if provided
    if response_definitions:
        response_names = {r['name'] for r in response_definitions}
    else:
        response_names = None
    
    responses = {}
    for col in design_data.columns:
        # Skip factor and metadata columns
        if col in factor_names or col in meta_cols:
            continue
        
        # If response names specified, only include those
        if response_names is not None and col not in response_names:
            continue
        
        # Extract non-empty data
        col_data = design_data[col]
        if col_data.notna().any():
            responses[col] = col_data.values
    
    return responses


# Initialize state
initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar
build_standard_sidebar()

st.title("Step 5: Import Experimental Results")

st.info(
    "📋 **DOE-Toolkit CSV Format Required**\n\n"
    "This page only accepts CSVs exported from DOE-Toolkit or created using our template format. "
    "The CSV must include metadata headers defining factors and responses.\n\n"
    "To create a compatible CSV:\n"
    "1. Export your design from Step 4 (Preview Design)\n"
    "2. Add your experimental results to the response columns\n"
    "3. Upload the completed file here"
)

# Show current data status if exists
if st.session_state.get('design') is not None and st.session_state.get('responses'):
    with st.expander("📊 Currently Loaded Data", expanded=False):
        design = st.session_state['design']
        responses = st.session_state['responses']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Runs", len(design))
        with col2:
            st.metric("Factors", len(st.session_state.get('factors', [])))
        with col3:
            st.metric("Responses", len(responses))
        
        st.markdown("**Preview (first 10 rows):**")
        preview_df = design.copy()
        for name, data in responses.items():
            preview_df[name] = data
        st.dataframe(preview_df.head(10), use_container_width=True)

st.divider()

# File upload
st.subheader("📤 Upload DOE-Toolkit CSV")

uploaded_file = st.file_uploader(
    "Choose a DOE-Toolkit formatted CSV file",
    type=['csv'],
    key='results_upload',
    help="Must be a CSV exported from DOE-Toolkit with metadata headers"
)

if uploaded_file:
    # Read file content
    file_content = uploaded_file.getvalue().decode('utf-8')
    
    # Parse as DOE-Toolkit format
    parse_result = parse_doe_csv(file_content)
    
    if parse_result.is_valid:
        st.success("✓ Valid DOE-Toolkit CSV detected!")
        
        # Determine import path
        has_session_design = st.session_state.get('design') is not None
        has_session_factors = len(st.session_state.get('factors', [])) > 0
        
        # Show summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Factors in CSV", len(parse_result.factors))
        with col2:
            response_count = len(parse_result.response_definitions) if parse_result.response_definitions else 0
            st.metric("Responses in CSV", response_count)
        
        st.divider()
        
        # Display factor summary table
        st.markdown("### 📋 Factors")
        factor_table_data = []
        for factor in parse_result.factors:
            if factor.factor_type == FactorType.CONTINUOUS:
                levels_display = f"[{factor.min_value}, {factor.max_value}]"
            elif factor.factor_type == FactorType.DISCRETE_NUMERIC:
                levels_display = ", ".join(str(v) for v in factor.levels)
            else:  # CATEGORICAL
                levels_display = ", ".join(str(v) for v in factor.levels)
            
            factor_table_data.append({
                'Name': factor.name,
                'Type': factor.factor_type.value,
                'Levels/Range': levels_display,
                'Units': factor.units or '',
                'Changeability': factor.changeability.value
            })
        
        st.dataframe(
            pd.DataFrame(factor_table_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Display response summary table
        if parse_result.response_definitions:
            st.markdown("### 📊 Responses")
            response_table_data = []
            for resp in parse_result.response_definitions:
                response_table_data.append({
                    'Name': resp['name'],
                    'Units': resp.get('units', '') or ''
                })
            
            st.dataframe(
                pd.DataFrame(response_table_data),
                use_container_width=True,
                hide_index=True
            )
        
        # Show design data preview
        with st.expander("🔍 Design Data Preview (first 10 rows)", expanded=False):
            st.dataframe(parse_result.design_data.head(10), use_container_width=True)
        
        st.divider()
        
        # PATH 1: Fresh session (no existing design)
        if not has_session_design:
            st.markdown("### ✅ Ready to Import")
            st.caption("This will load the factors, design, and responses into the current session.")
            
            if st.button("📥 Import All Data", type="primary", use_container_width=True):
                # Load into session
                st.session_state['factors'] = parse_result.factors
                st.session_state['design'] = parse_result.design_data
                st.session_state['response_definitions'] = parse_result.response_definitions
                st.session_state['design_metadata'] = parse_result.metadata
                
                # Extract responses
                responses = extract_responses_from_design(
                    parse_result.design_data,
                    parse_result.factors,
                    parse_result.response_definitions
                )
                
                if responses:
                    st.session_state['responses'] = responses
                    st.session_state['response_names'] = list(responses.keys())
                    st.success(f"✓ Imported {len(parse_result.factors)} factors, {len(responses)} responses")
                    st.info("👉 Data loaded! Click 'Analyze Results →' below to continue.")
                else:
                    # Still set empty responses to maintain consistency
                    st.session_state['responses'] = {}
                    st.session_state['response_names'] = []
                    st.warning("⚠️ No response data found in CSV (columns are empty)")
                    st.info("👉 Design imported! You can view the design or add response data later. Click 'Analyze Design →' below.")
                
                st.rerun()
        
        # PATH 2: Active session with existing factors
        elif has_session_factors:
            st.markdown("### 🔄 Factor Comparison")
            
            # Validate factor compatibility
            is_valid, errors = validate_csv_structure(parse_result, st.session_state.get('factors'))
            
            if is_valid:
                st.success("✓ CSV factors match session factors!")
                
                # Check for response mismatch
                session_responses = set(st.session_state.get('responses', {}).keys())
                csv_responses = {r['name'] for r in parse_result.response_definitions}
                
                if csv_responses and session_responses and csv_responses != session_responses:
                    st.warning("⚠️ Response names differ:")
                    st.caption(f"Session: {', '.join(sorted(session_responses))}")
                    st.caption(f"CSV: {', '.join(sorted(csv_responses))}")
                
                if st.button("📥 Import Results (Keep Session Factors)", type="primary", use_container_width=True):
                    # Extract responses
                    responses = extract_responses_from_design(
                        parse_result.design_data,
                        st.session_state['factors'],
                        parse_result.response_definitions
                    )
                    
                    if responses:
                        st.session_state['responses'] = responses
                        st.session_state['response_names'] = list(responses.keys())
                        st.session_state['response_definitions'] = parse_result.response_definitions
                        st.success(f"✓ Imported {len(responses)} response(s)")
                        st.info("👉 Results loaded! Click 'Analyze Results →' below to continue.")
                    else:
                        # Set empty responses to maintain consistency
                        st.session_state['responses'] = {}
                        st.session_state['response_names'] = []
                        st.warning("⚠️ No response data in CSV")
                        st.info("👉 Click 'Analyze Design →' to view design structure.")
                    
                    st.rerun()
            
            else:
                # Factor mismatch - show comparison
                st.error("❌ Factor mismatch detected!")
                
                st.markdown("**Differences:**")
                for error in errors:
                    st.error(f"• {error}")
                
                # Show comparison table
                comparison_data = []
                for i in range(max(len(parse_result.factors), len(st.session_state['factors']))):
                    csv_name = parse_result.factors[i].name if i < len(parse_result.factors) else '—'
                    session_name = st.session_state['factors'][i].name if i < len(st.session_state['factors']) else '—'
                    
                    comparison_data.append({
                        'CSV Factor': csv_name,
                        'Session Factor': session_name,
                        'Match': '✓' if csv_name == session_name and csv_name != '—' else '✗'
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                
                st.divider()
                
                # Resolution options
                st.markdown("### How to proceed:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Replace Session with CSV Factors", use_container_width=True):
                        st.session_state['factors'] = parse_result.factors
                        st.session_state['design'] = parse_result.design_data
                        st.session_state['response_definitions'] = parse_result.response_definitions
                        
                        # Extract responses
                        responses = extract_responses_from_design(
                            parse_result.design_data,
                            parse_result.factors,
                            parse_result.response_definitions
                        )
                        
                        if responses:
                            st.session_state['responses'] = responses
                            st.session_state['response_names'] = list(responses.keys())
                            st.success(f"✓ Replaced with {len(parse_result.factors)} factors, {len(responses)} responses")
                        
                        st.rerun()
                
                with col2:
                    if st.button("📤 Re-upload Corrected CSV", use_container_width=True):
                        st.info("Please upload a CSV that matches your session factors, or start a new session.")
    
    else:
        # Invalid CSV format
        st.error("❌ Invalid CSV Format")
        st.markdown(
            "This file is **not** a valid DOE-Toolkit formatted CSV.\n\n"
            "**Required format:**\n"
            "- Must include metadata headers (# DOE-TOOLKIT DESIGN)\n"
            "- Must define factors in metadata block\n"
            "- Must define responses in metadata block\n"
            "- Must include design data section"
        )
        
        if parse_result.error:
            with st.expander("🔍 Parse Error Details"):
                st.code(parse_result.error)
        
        st.divider()
        
        st.markdown("### 📝 How to create a valid CSV:")
        st.markdown(
            "1. **Option A:** Export your design from Step 4 (Preview Design)\n"
            "2. **Option B:** Download the CSV template (coming soon)\n"
            "3. Add your experimental results to the response columns\n"
            "4. Upload the completed file here"
        )

# Navigation
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("← Back to Design", use_container_width=True):
        st.switch_page("pages/4_preview_design.py")

with col3:
    # Enable navigation if we have either design or responses loaded
    has_design = st.session_state.get('design') is not None
    has_responses = st.session_state.get('responses') is not None and len(st.session_state.get('responses', {})) > 0
    can_proceed = has_design or has_responses
    
    # Determine button text and help based on state
    if not can_proceed:
        button_text = "Import Data First →"
        help_text = "Upload and import a DOE-Toolkit CSV to continue"
    elif has_design and not has_responses:
        button_text = "Analyze Design →"
        help_text = "View design without response data"
    else:
        button_text = "Analyze Results →"
        help_text = "Analyze experimental results"
    
    if st.button(
        button_text,
        type="primary", 
        use_container_width=True,
        disabled=not can_proceed,
        help=help_text
    ):
        st.session_state['current_step'] = 6
        st.switch_page("pages/6_analyze.py")
