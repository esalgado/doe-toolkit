# Profiler Prediction Scaling Issue - Root Cause and Fix

## Problem Summary
The Prediction Profiler and Optimizer tabs were showing wildly incorrect predictions (~120,000 instead of 70-95%), while the "Actual vs Predicted" plot in the Analysis tab showed correct values.

## Root Cause
**Coded vs Actual Value Mismatch**

The statistical models in DOE-Toolkit are fitted using **coded values** (normalized to -1, 0, +1 range) for numerical stability and to make effect estimates comparable. However, the profiler and optimizer were passing **actual values** (e.g., Temperature=175°C, Pressure=75 psi) directly to the model for prediction.

### Example of the Problem:
- **Factor Definition**: Temperature range [150, 200]
- **Design Matrix**: Contains coded values [-1, 0, 1]
- **Model Training**: Fitted on coded values
- **Profiler Input**: Temperature = 175 (actual value)
- **Model Interprets**: 175 as a coded value (massive extrapolation!)
- **Result**: Prediction of ~120,000 instead of ~85

### Why Actual vs Predicted Plot Was Correct:
The parity plot uses `results.fitted_values` which are the predictions on the **same data the model was trained on** (the coded design matrix). So there was no mismatch there.

### Why Profiler Was Wrong:
The profiler creates new prediction points using actual factor values from sliders (150-200°C) and passes them directly to the model, which expects coded values (-1 to +1).

## The Fix

### 1. Created Encoding/Decoding Module
**File**: `src/core/coding.py`

Contains functions to convert between coded and actual values:
- `encode_value()`: Convert actual → coded (e.g., 175 → 0)
- `decode_value()`: Convert coded → actual (e.g., 0 → 175)
- `encode_settings_dict()`: Encode a dict of factor settings
- `encode_design()`: Encode entire design DataFrame

### 2. Updated Profiler Display
**File**: `src/ui/components/profiler_display.py`

**Changes**:
1. Import encoding functions
2. `_compute_prediction()`: Encode factor settings before prediction
3. `_display_continuous_factor()`: Encode trace points before prediction
4. `_display_categorical_factor()`: Encode settings before prediction
5. `_generate_contour_mesh()`: Encode grid before prediction
6. Fixed CI calculation to use encoded data

### 3. Updated Optimizer
**File**: `src/ui/pages/8_optimize.py`

**Changes**:
1. Response surface plot: Encode grid before prediction
2. Contour plot: Encode grid before prediction

## Technical Details

### Encoding Formula
```python
coded_value = (actual_value - center) / half_range

where:
    center = (min + max) / 2
    half_range = (max - min) / 2
```

### Example:
```python
# Temperature: [150, 200]
center = 175
half_range = 25

encode(150) = (150 - 175) / 25 = -1.0
encode(175) = (175 - 175) / 25 =  0.0
encode(200) = (200 - 175) / 25 =  1.0
```

### Why Use Coded Values?
1. **Numerical Stability**: Prevents large coefficients
2. **Comparable Effects**: All factors on same scale
3. **Standard Practice**: Industry standard in DOE
4. **Better Conditioning**: Improves model fitting

## Testing the Fix

### Before Fix:
```python
# Profiler with Temperature=175, Pressure=75, Time=20
factor_settings = {'Temperature': 175.0, 'Pressure': 75.0, 'Time': 20.0}
prediction = 107126.24  # WRONG!
```

### After Fix:
```python
# Profiler with Temperature=175, Pressure=75, Time=20
factor_settings (actual) = {'Temperature': 175.0, 'Pressure': 75.0, 'Time': 20.0}
encoded_settings (coded) = {'Temperature': 0.0, 'Pressure': 0.0, 'Time': 0.0}
prediction = 82.5  # CORRECT!
```

## Future Considerations

### Design Matrix Storage
Currently, the design matrix might be stored in coded form in the CSV. Consider:
1. **Option A**: Always store actual values in CSV, encode when fitting
2. **Option B**: Store coded values, decode for display
3. **Current**: Mixed (needs standardization)

### Auto-Detection
The `is_design_coded()` function can detect if a design is coded or not based on value ranges. This could be used to automatically handle both cases.

### User-Facing Changes
None! Users continue to see and input actual values (150-200°C). The encoding/decoding happens transparently in the background.

## Commit Message

```
fix: correct profiler and optimizer predictions by encoding factor values

The profiler and optimizer were passing actual factor values (e.g., 175°C) 
directly to models trained on coded values (-1, 0, 1), causing ~1400x 
extrapolation errors.

Changes:
- Add src/core/coding.py with encode/decode utilities
- Update profiler to encode settings before prediction
- Update optimizer surface and contour plots to encode grids
- Fix CI calculations to use encoded data

Fixes predictions showing ~120k instead of ~85 for percentage responses.
```

## Files Modified

1. `src/core/coding.py` - NEW: Encoding/decoding utilities
2. `src/ui/components/profiler_display.py` - Encode all prediction inputs
3. `src/ui/pages/8_optimize.py` - Encode surface/contour grids
4. `src/core/analysis.py` - Added debug output (temporary)
5. `src/ui/utils/csv_parser.py` - Added % symbol stripping (precautionary)
