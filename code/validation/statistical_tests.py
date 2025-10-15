"""
Statistical Tests for Model Comparison

Implements rigorous statistical tests to determine if differences between
models are statistically significant, addressing the critical gap where
models are selected based on point estimates without significance testing.

Key Tests:
1. Diebold-Mariano (DM) Test - Compare forecast accuracy of two models
2. Model Confidence Set (MCS) - Identify set of statistically equivalent models
3. Harvey-Leybourne-Newbold correction - Small sample adjustment for DM test

Scientific Justification:
- Cannot claim model A is "better" than B without statistical test
- Small differences in MAPE may be due to random variation
- Need p-value < 0.05 to claim significant difference
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import logging


class ModelComparisonTests:
    """
    Statistical tests for comparing forecast models.
    
    Addresses the scientific gap where model selection is based on
    point estimates (e.g., MAPE = 3.72% vs 3.85%) without testing
    if the difference is statistically significant.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize test framework.
        
        Args:
            alpha: Significance level (default: 0.05 for 95% confidence)
        """
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
    
    def diebold_mariano_test(self, errors_a: np.ndarray, errors_b: np.ndarray, 
                            h: int = 1, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Diebold-Mariano test for comparing forecast accuracy.
        
        Tests the null hypothesis that two models have equal forecast accuracy.
        
        H0: Models have equal forecast accuracy
        Ha: Model accuracy differs (two-sided) or one is better (one-sided)
        
        Args:
            errors_a: Forecast errors from model A (can be absolute or squared)
            errors_b: Forecast errors from model B (same type as errors_a)
            h: Forecast horizon (for autocorrelation adjustment)
            alternative: 'two-sided', 'less' (A < B), or 'greater' (A > B)
        
        Returns:
            dict: {
                'statistic': float,        # DM test statistic
                'p_value': float,          # p-value
                'conclusion': str,         # Plain English conclusion
                'significant': bool,       # True if difference is significant
                'mean_loss_diff': float,   # Average difference in loss
                'interpretation': str      # Detailed interpretation
            }
        
        References:
            Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
            Journal of Business & Economic Statistics, 20(1), 134-144.
        """
        if len(errors_a) != len(errors_b):
            raise ValueError("Error arrays must have same length")
        
        if len(errors_a) < 3:
            raise ValueError("Need at least 3 observations for DM test")
        
        # Convert to squared errors if not already
        loss_a = errors_a ** 2
        loss_b = errors_b ** 2
        
        # Loss differential
        d = loss_a - loss_b
        d_mean = np.mean(d)
        
        # Variance with Newey-West HAC correction for autocorrelation
        if h > 1:
            # Include autocorrelation up to h-1 lags
            gamma_0 = np.var(d, ddof=1)
            gamma_sum = 0
            for lag in range(1, h):
                if lag < len(d):
                    gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
                    gamma_sum += gamma_lag
            d_var = gamma_0 + 2 * gamma_sum
        else:
            d_var = np.var(d, ddof=1)
        
        # DM statistic
        T = len(d)
        dm_stat = d_mean / np.sqrt(d_var / T)
        
        # Harvey-Leybourne-Newbold small sample correction
        # More accurate for small samples (T < 50)
        if T < 50:
            dm_stat = dm_stat * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        
        # Calculate p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        elif alternative == 'less':  # Test if model A is better (lower error)
            p_value = stats.norm.cdf(dm_stat)
        elif alternative == 'greater':  # Test if model B is better
            p_value = 1 - stats.norm.cdf(dm_stat)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
        
        # Determine significance
        significant = p_value < self.alpha
        
        # Generate conclusion
        if alternative == 'two-sided':
            if significant:
                if d_mean < 0:
                    conclusion = "Model A is significantly more accurate than Model B"
                else:
                    conclusion = "Model B is significantly more accurate than Model A"
            else:
                conclusion = "No significant difference between models"
        else:
            if significant:
                conclusion = f"Significant difference (p = {p_value:.4f})"
            else:
                conclusion = "No significant difference"
        
        # Interpretation
        if significant:
            interpretation = (
                f"✅ Significant at α={self.alpha} level (p = {p_value:.4f}). "
                f"Can claim one model is better than the other."
            )
        else:
            interpretation = (
                f"⚠️ Not significant at α={self.alpha} level (p = {p_value:.4f}). "
                f"Difference may be due to random variation. Consider using ensemble."
            )
        
        return {
            'statistic': float(dm_stat),
            'p_value': float(p_value),
            'conclusion': conclusion,
            'significant': significant,
            'mean_loss_diff': float(d_mean),
            'interpretation': interpretation,
            'sample_size': T,
            'horizon': h
        }
    
    def compare_multiple_models(self, errors_dict: Dict[str, np.ndarray],
                                reference_model: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple models using pairwise DM tests.
        
        Args:
            errors_dict: Dictionary of {model_name: error_array}
            reference_model: Optional reference model to compare against.
                           If None, compares all pairs.
        
        Returns:
            DataFrame with pairwise comparison results
        """
        results = []
        models = list(errors_dict.keys())
        
        if reference_model is None:
            # All pairwise comparisons
            for i, model_a in enumerate(models):
                for model_b in models[i+1:]:
                    dm_result = self.diebold_mariano_test(
                        errors_dict[model_a],
                        errors_dict[model_b],
                        alternative='two-sided'
                    )
                    
                    results.append({
                        'model_a': model_a,
                        'model_b': model_b,
                        'dm_statistic': dm_result['statistic'],
                        'p_value': dm_result['p_value'],
                        'significant': dm_result['significant'],
                        'better_model': model_a if dm_result['mean_loss_diff'] < 0 else model_b
                    })
        else:
            # Compare all models against reference
            if reference_model not in errors_dict:
                raise ValueError(f"Reference model '{reference_model}' not found")
            
            for model in models:
                if model == reference_model:
                    continue
                
                dm_result = self.diebold_mariano_test(
                    errors_dict[reference_model],
                    errors_dict[model],
                    alternative='two-sided'
                )
                
                results.append({
                    'model': model,
                    'vs_reference': reference_model,
                    'dm_statistic': dm_result['statistic'],
                    'p_value': dm_result['p_value'],
                    'significant': dm_result['significant'],
                    'better_model': reference_model if dm_result['mean_loss_diff'] < 0 else model
                })
        
        return pd.DataFrame(results)
    
    def model_confidence_set(self, errors_dict: Dict[str, np.ndarray], 
                            alpha: float = 0.10) -> Dict[str, Any]:
        """
        Model Confidence Set (MCS) procedure to identify statistically
        equivalent models.
        
        Returns a set of models that cannot be distinguished statistically.
        This is better than picking a single "best" model when differences
        are not significant.
        
        Args:
            errors_dict: Dictionary of {model_name: error_array}
            alpha: Significance level (default: 0.10, less conservative)
        
        Returns:
            dict: {
                'included_models': list,      # Models in MCS
                'excluded_models': list,      # Models rejected from MCS
                'final_p_value': float,       # Final test p-value
                'recommendation': str         # What to do
            }
        
        References:
            Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
            Econometrica, 79(2), 453-497.
        """
        self.logger.info(f"Computing Model Confidence Set (α = {alpha})")
        
        # Start with all models
        models = list(errors_dict.keys())
        remaining = set(models)
        
        # Convert errors to losses (squared errors)
        losses = {name: errors**2 for name, errors in errors_dict.items()}
        
        # Iteratively eliminate worst performing models
        eliminated_order = []
        
        while len(remaining) > 1:
            # Calculate average loss for remaining models
            avg_losses = {
                model: np.mean(losses[model]) 
                for model in remaining
            }
            
            # Find worst model
            worst_model = max(avg_losses, key=avg_losses.get)
            
            # Find best model
            best_model = min(avg_losses, key=avg_losses.get)
            
            # Test if worst is significantly worse than best
            dm_result = self.diebold_mariano_test(
                errors_dict[worst_model],
                errors_dict[best_model],
                alternative='two-sided'
            )
            
            # If significantly worse, eliminate
            if dm_result['p_value'] < alpha and dm_result['mean_loss_diff'] > 0:
                self.logger.info(
                    f"Eliminating {worst_model} (p = {dm_result['p_value']:.4f})"
                )
                remaining.remove(worst_model)
                eliminated_order.append(worst_model)
            else:
                # Cannot eliminate any more models
                self.logger.info(
                    f"Cannot eliminate more models (p = {dm_result['p_value']:.4f} >= {alpha})"
                )
                break
        
        included = list(remaining)
        excluded = eliminated_order
        
        # Recommendation
        if len(included) == 1:
            recommendation = (
                f"Use {included[0]} - clearly superior to other models"
            )
        elif len(included) <= 3:
            recommendation = (
                f"Use ensemble of {len(included)} models: {', '.join(included)}. "
                f"These models are statistically indistinguishable."
            )
        else:
            recommendation = (
                f"Use ensemble of top {len(included)} models. "
                f"No single model is clearly best."
            )
        
        return {
            'included_models': included,
            'excluded_models': excluded,
            'n_included': len(included),
            'n_excluded': len(excluded),
            'final_p_value': dm_result['p_value'] if 'dm_result' in locals() else None,
            'recommendation': recommendation,
            'alpha': alpha
        }
    
    def rank_models_with_significance(self, errors_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Rank models and test if differences are significant.
        
        Args:
            errors_dict: Dictionary of {model_name: error_array}
        
        Returns:
            DataFrame with model rankings and significance tests
        """
        # Calculate average loss for each model
        avg_losses = {
            name: np.mean(errors**2) 
            for name, errors in errors_dict.items()
        }
        
        # Sort by average loss
        sorted_models = sorted(avg_losses.items(), key=lambda x: x[1])
        
        results = []
        
        for rank, (model, loss) in enumerate(sorted_models, 1):
            # Compare to best model (rank 1)
            if rank == 1:
                results.append({
                    'rank': rank,
                    'model': model,
                    'avg_squared_error': loss,
                    'vs_best_p_value': np.nan,
                    'significantly_worse': False,
                    'interpretation': 'Best model (reference)'
                })
            else:
                best_model = sorted_models[0][0]
                dm_result = self.diebold_mariano_test(
                    errors_dict[model],
                    errors_dict[best_model],
                    alternative='two-sided'
                )
                
                significantly_worse = (
                    dm_result['significant'] and 
                    dm_result['mean_loss_diff'] > 0
                )
                
                if significantly_worse:
                    interp = f"⚠️ Significantly worse than {best_model}"
                else:
                    interp = f"✅ Not significantly different from {best_model}"
                
                results.append({
                    'rank': rank,
                    'model': model,
                    'avg_squared_error': loss,
                    'vs_best_p_value': dm_result['p_value'],
                    'significantly_worse': significantly_worse,
                    'interpretation': interp
                })
        
        return pd.DataFrame(results)


def format_comparison_results(dm_result: Dict[str, Any], 
                              model_a: str, model_b: str) -> str:
    """
    Format DM test results for display.
    
    Args:
        dm_result: Result from diebold_mariano_test()
        model_a: Name of first model
        model_b: Name of second model
    
    Returns:
        Formatted string
    """
    output = f"""
╔═══════════════════════════════════════════════════════════════╗
║  Diebold-Mariano Test: {model_a} vs {model_b}
╚═══════════════════════════════════════════════════════════════╝

H0: Models have equal forecast accuracy
Ha: Models have different forecast accuracy

Test Statistic: {dm_result['statistic']:.4f}
P-Value:        {dm_result['p_value']:.4f}
Sample Size:    {dm_result['sample_size']}

Conclusion: {dm_result['conclusion']}

{dm_result['interpretation']}
"""
    
    return output.strip()
