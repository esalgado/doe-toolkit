# Plain CSV Import Removal - Summary

## Changes Made

### 1. Simplified 5_import_results.py
**What changed:**
- Removed all plain CSV import functionality
- Removed auto-classification workflow
- Removed encoding/decoding logic
- Removed editable verification table
- Kept only DOE-Toolkit metadata CSV import

**New behavior:**
- Only accepts DOE-Toolkit formatted CSVs with metadata headers
- Shows read-only summary tables of factors and responses
- Clear error message if CSV format is invalid
- Simplified import workflow (no user editing)

### 2. Removed/Stubbed Helper Files

**Files made into stubs (no longer functional):**
- `src/ui/components/column_verification_table.py` - Editable factor/response table
- `src/ui/utils/column_classifier.py` - Auto-classification of CSV columns
- `src/ui/utils/data_encoder.py` - Encoding/decoding utilities

**Why stubbed instead of deleted:**
- Prevents import errors if other code references them
- Clear documentation of what was removed
- Easy to restore if needed

### 3. Cleaned Up UI

**Import page now shows:**
- Clear requirements message at top
- Factor summary table (read-only)
- Response summary table (read-only)
- Design data preview (expandable)
- Import button (or factor comparison if session active)

**Removed:**
- Plain CSV upload option
- Auto-classification UI
- Editable verification table
- Encoding preview
- Manual column reclassification

## Why This Change?

### Problems with Plain CSV Import:
1. **Complexity:** Too many edge cases and error handling
2. **User confusion:** Multiple import paths confusing
3. **Indentation errors:** Complex nested logic hard to maintain
4. **Encoding issues:** Continuous factors needed special handling
5. **Validation complexity:** Too many things could go wrong

### Benefits of DOE-Toolkit Only:
1. ✅ **Simpler workflow:** One path, one format
2. ✅ **Less error-prone:** Metadata defines everything
3. ✅ **Easier maintenance:** Less code, clearer logic
4. ✅ **Better UX:** Clear expectations, no guessing
5. ✅ **Proper data:** Metadata ensures correctness

## User Workflow Now

### To import results:

1. **Create design** in Steps 1-3
2. **Export CSV** from Step 4 (Preview Design)
3. **Run experiments** and fill in response columns
4. **Upload completed CSV** in Step 5
5. **Analyze** in Step 6

### If user has plain CSV:
- They must create a DOE-Toolkit formatted CSV
- Can do this by:
  - Defining factors in Step 1
  - Generating design in Steps 2-4
  - Exporting template
  - Copying their data into template

## Files Modified

```
src/ui/pages/5_import_results.py              [COMPLETELY REWRITTEN]
src/ui/components/column_verification_table.py [STUBBED - removed functionality]
src/ui/utils/column_classifier.py             [STUBBED - removed functionality]
src/ui/utils/data_encoder.py                  [STUBBED - removed functionality]
```

## Code Reduction

**Before:**
- 5_import_results.py: ~460 lines
- column_verification_table.py: ~380 lines
- column_classifier.py: ~200 lines (estimated)
- data_encoder.py: ~280 lines
- **Total: ~1,320 lines**

**After:**
- 5_import_results.py: ~290 lines
- Other files: ~10 lines (stubs)
- **Total: ~300 lines**

**Reduction: ~1,020 lines removed (77% reduction)**

## Testing Checklist

- [ ] DOE-Toolkit CSV imports successfully
- [ ] Factor table displays correctly
- [ ] Response table displays correctly  
- [ ] Design preview shows data
- [ ] Fresh session import works
- [ ] Existing session comparison works
- [ ] Factor mismatch shows error
- [ ] Invalid CSV shows clear error message
- [ ] Navigation buttons work
- [ ] No import errors from stubbed files

## Migration Notes

**For users who relied on plain CSV:**
- No direct migration path
- Must create DOE-Toolkit formatted CSV
- Template export feature recommended for next version

**For developers:**
- Stubbed files can be fully deleted later
- Consider adding CSV template download feature
- Consider adding CSV format converter tool

## Git Commit Message

```
refactor: remove plain CSV import, keep only DOE-Toolkit format

BREAKING CHANGE: Plain CSV import no longer supported

- Simplify import workflow to only accept DOE-Toolkit formatted CSVs
- Remove auto-classification and verification table
- Remove encoding/decoding utilities
- Replace editable tables with read-only summaries
- Reduce codebase by ~1,020 lines (77% reduction)
- Improve maintainability and reduce complexity

Benefits:
- Single, clear import path
- Less error-prone
- Easier to maintain
- Better user experience with clear requirements

Stubbed files (for clean imports):
- column_verification_table.py
- column_classifier.py  
- data_encoder.py

Users must now export from Step 4 or use DOE-Toolkit template format.
```

## Next Steps

1. Test import workflow thoroughly
2. Update user documentation
3. Add CSV template download feature (recommended)
4. Consider adding format converter tool (optional)
5. Remove stub files in future PR (after confirming no dependencies)
