"""Quick test to verify SAR analysis process is intact."""

print("=== SAR Analysis Integration Test ===\n")

# Test imports
from analyz import SARAnalyzer, FileHandler, Plotter, InsightsGenerator
import numpy as np
print("✓ All imports successful\n")

# Test SARAnalyzer initialization
data = np.random.rand(2, 50, 50) * 0.1
analyzer = SARAnalyzer(data)
print("✓ SARAnalyzer initialized\n")

# List public methods
print("Public methods (application analyses):")
methods = [m for m in dir(analyzer) if not m.startswith('_') and callable(getattr(analyzer, m))]
print(f"  Total: {len(methods)} methods")
for m in sorted(methods):
    print(f"  - {m}")

# Test a simple analysis
print("\n--- Testing Oil Spill Detection ---")
result, stats = analyzer.oil_spill_detection()
print(f"✓ Analysis completed")
print(f"  Result shape: {result.shape}")
print(f"  Stats keys: {len(stats)} keys")

# Test insights generation
print("\n--- Testing Insights Generation ---")
insights = InsightsGenerator.generate_sar_insights('Oil Spill Detection', result, stats)
print(f"✓ Insights generated")
print(f"  Summary: {insights['summary'][:60]}...")

# Verify old methods are gone
print("\n--- Verifying Old Methods Removed ---")
old_methods = ['lee_filter', 'frost_filter', 'median_filter', 'backscatter_analysis', 'coherence_estimation']
all_removed = True
for method in old_methods:
    exists = hasattr(analyzer, method)
    status = "❌ STILL EXISTS" if exists else "✓ Removed"
    print(f"  {method}: {status}")
    if exists:
        all_removed = False

print(f"\n{'✅ ALL TESTS PASSED' if all_removed else '❌ SOME TESTS FAILED'}")
print("\n=== Process Verification COMPLETE ===")
