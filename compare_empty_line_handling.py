import torch
import math
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

def compare_empty_line_handling():
    """Compare original vs corrected empty line handling in PPL chunking"""
    
    code_to_be_analyzed = """def evaluate_blind(self, code, **kwargs):

    suffix = kwargs.get('suffix', self.get('suffix', ''))
    blind = kwargs.get('blind', False)

    action = self.actions.get('evaluate_blind', {})
    payload_action = action.get('evaluate_blind')
    call_name = action.get('call', 'inject')

    # Skip if something is missing or call function is not set
    if not action or not payload_action or not call_name or not hasattr(self, call_name):
        return

    expected_delay = self._get_expected_delay()

    if '%(code_b64)s' in payload_action:
        log.debug('[b64 encoding] %s' % code)
        execution_code = payload_action % ({
            'code_b64' : base64.urlsafe_b64encode(code),
            'delay' : expected_delay
        })
    else:
        execution_code = payload_action % ({
            'code' : code,
            'delay' : expected_delay
        })

    return getattr(self, call_name)(
        code = execution_code,
        prefix = prefix,
        suffix = suffix,
        blind=True
    )"""

    print("="*80)
    print("COMPARISON: Empty Line Handling in PPL Chunking")
    print("="*80)
    
    lines = code_to_be_analyzed.split('\n')
    
    # Simulate original approach (includes empty lines in smoothing)
    def original_smoothing(values, window_size=3):
        """Original smoothing that includes empty lines"""
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            
            window_values = []
            for j in range(start_idx, end_idx):
                if not math.isinf(values[j]) and not math.isnan(values[j]):
                    window_values.append(values[j])
            
            if window_values:
                smoothed.append(sum(window_values) / len(window_values))
            else:
                smoothed.append(values[i])
        
        return smoothed
    
    # Simulate corrected approach (excludes empty lines from smoothing)
    def corrected_smoothing(values, lines, window_size=3):
        """Corrected smoothing that excludes empty lines"""
        smoothed = []
        
        # Identify non-empty line indices
        non_empty_indices = [i for i, line in enumerate(lines) if line.strip() != '']
        
        for i in range(len(values)):
            if lines[i].strip() == '':  # Empty line
                smoothed.append(values[i])  # Keep original value
            else:
                # Find position in non-empty indices
                try:
                    pos_in_non_empty = non_empty_indices.index(i)
                except ValueError:
                    smoothed.append(values[i])
                    continue
                
                # Get window around this position in non-empty lines
                start_pos = max(0, pos_in_non_empty - window_size // 2)
                end_pos = min(len(non_empty_indices), pos_in_non_empty + window_size // 2 + 1)
                
                # Get values from non-empty lines in the window
                window_values = []
                for j in range(start_pos, end_pos):
                    idx = non_empty_indices[j]
                    val = values[idx]
                    if not math.isinf(val) and not math.isnan(val) and val > 0:
                        window_values.append(val)
                
                if window_values:
                    smoothed.append(sum(window_values) / len(window_values))
                else:
                    smoothed.append(values[i])
        
        return smoothed
    
    # Create sample PPL values (simulated)
    sample_ppls = []
    for i, line in enumerate(lines):
        if line.strip() == '':
            sample_ppls.append(1.0)  # Empty line PPL
        else:
            # Simulate varying PPL values
            if 'def ' in line:
                sample_ppls.append(101.65)
            elif 'return' in line and len(line.strip()) < 20:
                sample_ppls.append(1.50)
            elif line.strip().startswith('#'):
                sample_ppls.append(17.72)
            elif 'kwargs.get' in line:
                sample_ppls.append(8.39)
            elif 'action' in line:
                sample_ppls.append(8.17)
            elif 'if ' in line:
                sample_ppls.append(12.41)
            elif 'else:' in line:
                sample_ppls.append(1.36)
            elif line.strip().startswith("'"):
                sample_ppls.append(2.52)
            else:
                sample_ppls.append(5.0 + (i % 10))  # Varying values
    
    # Apply both smoothing approaches
    original_smoothed = original_smoothing(sample_ppls, window_size=3)
    corrected_smoothed = corrected_smoothing(sample_ppls, lines, window_size=3)
    
    print(f"Total lines: {len(lines)}")
    print(f"Empty lines: {len([l for l in lines if l.strip() == ''])}")
    print(f"Non-empty lines: {len([l for l in lines if l.strip() != ''])}")
    print()
    
    print("Line-by-line comparison:")
    print(f"{'Line':>4} {'Empty':>5} {'Original':>10} {'Corrected':>10} {'Difference':>10} {'Content'}")
    print("-" * 80)
    
    for i, (line, orig_ppl, orig_smooth, corr_smooth) in enumerate(zip(lines, sample_ppls, original_smoothed, corrected_smoothed)):
        is_empty = line.strip() == ''
        diff = abs(orig_smooth - corr_smooth)
        content = repr(line[:40] + "..." if len(line) > 40 else line)
        
        print(f"{i:4d} {'Yes' if is_empty else 'No':>5} {orig_smooth:10.4f} {corr_smooth:10.4f} {diff:10.4f} {content}")
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES:")
    print("="*80)
    
    # Find lines where smoothing differs significantly
    significant_diffs = []
    for i, (orig_smooth, corr_smooth) in enumerate(zip(original_smoothed, corrected_smoothed)):
        diff = abs(orig_smooth - corr_smooth)
        if diff > 0.1 and lines[i].strip() != '':  # Non-empty lines with significant difference
            significant_diffs.append((i, diff, orig_smooth, corr_smooth))
    
    print(f"\nLines with significant smoothing differences (> 0.1):")
    for line_idx, diff, orig, corr in significant_diffs:
        print(f"Line {line_idx}: Original={orig:.4f}, Corrected={corr:.4f}, Diff={diff:.4f}")
        print(f"  Content: {repr(lines[line_idx])}")
    
    # Show impact on empty lines
    empty_line_indices = [i for i, line in enumerate(lines) if line.strip() == '']
    print(f"\nEmpty line smoothing values:")
    for idx in empty_line_indices:
        print(f"Line {idx}: Original={original_smoothed[idx]:.4f}, Corrected={corrected_smoothed[idx]:.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("Original approach:")
    print("- Includes empty lines (PPL=1.0) in smoothing windows")
    print("- Can artificially lower smoothed PPL values near empty lines")
    print("- May create false local minimums")
    
    print("\nCorrected approach:")
    print("- Excludes empty lines from smoothing calculations")
    print("- Only considers non-empty lines for smoothing windows")
    print("- Preserves original line indices for visualization")
    print("- More accurate representation of code complexity patterns")

if __name__ == "__main__":
    compare_empty_line_handling() 