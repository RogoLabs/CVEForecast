"""
Residual Diagnostics for Time Series Models

Implements comprehensive diagnostic tests to validate model assumptions
and identify potential issues with model adequacy.

Key Tests:
1. Ljung-Box Test - Residual autocorrelation (white noise test)
2. Jarque-Bera Test - Normality of residuals
3. ARCH Test - Heteroskedasticity (variance instability)
4. Q-Q Plot - Visual normality assessment
5. ACF/PACF Plots - Autocorrelation structure

Scientific Justification:
- Time series models assume residuals are white noise
- Violations indicate model inadequacy or misspecification
- Diagnostic checks are standard practice in forecasting
"""

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


class ResidualDiagnostics:
    """
    Comprehensive residual diagnostic suite for time series models.

    Validates key assumptions:
    1. Residuals are uncorrelated (white noise)
    2. Residuals are normally distributed
    3. Residuals have constant variance (homoskedastic)
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize diagnostic suite.

        Args:
            alpha: Significance level for tests (default: 0.05)
        """
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def run_full_diagnostics(
        self, residuals: np.ndarray, model_name: str = 'Model', lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete diagnostic suite on residuals.

        Args:
            residuals: Model residuals (actuals - forecasts)
            model_name: Name for reporting
            lags: Number of lags for Ljung-Box (default: min(10, len/5))

        Returns:
            dict: Complete diagnostic results
        """
        self.logger.info(f'Running diagnostics for {model_name}')

        # Remove any NaN values
        residuals = residuals[~np.isnan(residuals)]

        if len(residuals) < 10:
            self.logger.warning(f'Only {len(residuals)} residuals, skipping diagnostics')
            return {'sufficient_data': False}

        # Default lags
        if lags is None:
            lags = min(10, len(residuals) // 5)

        results = {'model_name': model_name, 'n_residuals': len(residuals), 'sufficient_data': True}

        # 1. Ljung-Box Test (autocorrelation)
        results['ljung_box'] = self.ljung_box_test(residuals, lags=lags)

        # 2. Normality Tests
        results['normality'] = self.normality_tests(residuals)

        # 3. ARCH Test (heteroskedasticity)
        results['arch'] = self.arch_test(residuals, lags=min(5, lags))

        # 4. Descriptive Statistics
        results['statistics'] = self.residual_statistics(residuals)

        # 5. Overall Assessment
        results['assessment'] = self.overall_assessment(results)

        return results

    def ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """
        Ljung-Box test for residual autocorrelation.

        H0: Residuals are independently distributed (white noise)
        Ha: Residuals exhibit autocorrelation

        If p-value < 0.05, reject H0 → residuals are autocorrelated
        This indicates model inadequacy (not capturing all patterns)

        Args:
            residuals: Model residuals
            lags: Number of lags to test

        Returns:
            dict: Test results
        """
        try:
            lb_result = acorr_ljungbox(residuals, lags=lags, return_df=False)

            # Get test statistics and p-values
            lb_stat = lb_result[0][-1]  # Last lag statistic
            p_value = lb_result[1][-1]  # Last lag p-value

            # Check if any lag is significant
            any_significant = np.any(lb_result[1] < self.alpha)

            passed = not any_significant

            if passed:
                interpretation = (
                    f'✅ Residuals appear to be white noise (p = {p_value:.4f}). '
                    f'No significant autocorrelation detected.'
                )
                status = 'PASS'
            else:
                significant_lags = np.where(lb_result[1] < self.alpha)[0] + 1
                interpretation = (
                    f'❌ Residuals exhibit autocorrelation at lags {significant_lags.tolist()} '
                    f'(p = {p_value:.4f}). Model may not be capturing all patterns.'
                )
                status = 'FAIL'

            return {
                'test': 'Ljung-Box',
                'statistic': float(lb_stat),
                'p_value': float(p_value),
                'lags_tested': lags,
                'passed': passed,
                'status': status,
                'interpretation': interpretation,
                'significant_lags': significant_lags.tolist() if not passed else [],
            }

        except Exception as e:
            self.logger.error(f'Ljung-Box test failed: {e}')
            return {'test': 'Ljung-Box', 'passed': None, 'status': 'ERROR', 'error': str(e)}

    def normality_tests(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Test residual normality using multiple tests.

        H0: Residuals are normally distributed
        Ha: Residuals are not normal

        Many models assume normal innovations. Violations suggest:
        - Need for transformations
        - Different error distribution
        - Presence of outliers

        Args:
            residuals: Model residuals

        Returns:
            dict: Normality test results
        """
        results = {}

        # 1. Jarque-Bera Test
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)

            jb_passed = jb_pvalue >= self.alpha

            if jb_passed:
                jb_interp = f'✅ Residuals appear approximately normal (JB p = {jb_pvalue:.4f})'
            else:
                jb_interp = (
                    f'❌ Residuals deviate from normality (JB p = {jb_pvalue:.4f}). '
                    f'May need transformation or robust methods.'
                )

            results['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_pvalue),
                'passed': jb_passed,
                'interpretation': jb_interp,
            }
        except Exception as e:
            self.logger.error(f'Jarque-Bera test failed: {e}')
            results['jarque_bera'] = {'error': str(e)}

        # 2. Shapiro-Wilk Test (better for small samples)
        if len(residuals) < 5000:  # SW test is computationally intensive
            try:
                sw_stat, sw_pvalue = stats.shapiro(residuals)

                sw_passed = sw_pvalue >= self.alpha

                results['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_pvalue),
                    'passed': sw_passed,
                }
            except Exception as e:
                self.logger.warning(f'Shapiro-Wilk test failed: {e}')

        # 3. Descriptive statistics
        results['skewness'] = float(stats.skew(residuals))
        results['kurtosis'] = float(stats.kurtosis(residuals))

        # Interpretation of skewness and kurtosis
        if abs(results['skewness']) > 1:
            results['skewness_note'] = '⚠️ High skewness (|skew| > 1)'
        else:
            results['skewness_note'] = '✅ Skewness acceptable'

        if abs(results['kurtosis']) > 3:
            results['kurtosis_note'] = '⚠️ High kurtosis (heavy tails)'
        else:
            results['kurtosis_note'] = '✅ Kurtosis acceptable'

        # Overall normality assessment
        jb_passed = results.get('jarque_bera', {}).get('passed', False)
        results['overall_passed'] = jb_passed

        if jb_passed:
            results['status'] = 'PASS'
        else:
            results['status'] = 'FAIL'

        return results

    def arch_test(self, residuals: np.ndarray, lags: int = 5) -> Dict[str, Any]:
        """
        ARCH test for conditional heteroskedasticity.

        H0: No ARCH effects (constant variance)
        Ha: ARCH effects present (variance changes over time)

        If p-value < 0.05, reject H0 → heteroskedasticity present
        This suggests:
        - Variance is not constant
        - May need GARCH modeling
        - Prediction intervals may be miscalibrated

        Args:
            residuals: Model residuals
            lags: Number of lags for ARCH test

        Returns:
            dict: ARCH test results
        """
        try:
            # Need at least 2*lags + 2 observations
            min_obs = 2 * lags + 2
            if len(residuals) < min_obs:
                return {
                    'test': 'ARCH',
                    'status': 'SKIPPED',
                    'reason': f'Insufficient data (need {min_obs}, have {len(residuals)})',
                }

            lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=lags)

            passed = lm_pvalue >= self.alpha

            if passed:
                interpretation = (
                    f'✅ No significant ARCH effects (p = {lm_pvalue:.4f}). Residual variance appears constant.'
                )
                status = 'PASS'
            else:
                interpretation = (
                    f'❌ ARCH effects detected (p = {lm_pvalue:.4f}). '
                    f'Residual variance is not constant. Consider GARCH model '
                    f'or robust standard errors.'
                )
                status = 'FAIL'

            return {
                'test': 'ARCH',
                'lm_statistic': float(lm_stat),
                'lm_p_value': float(lm_pvalue),
                'f_statistic': float(f_stat),
                'f_p_value': float(f_pvalue),
                'lags_tested': lags,
                'passed': passed,
                'status': status,
                'interpretation': interpretation,
            }

        except Exception as e:
            self.logger.error(f'ARCH test failed: {e}')
            return {'test': 'ARCH', 'status': 'ERROR', 'error': str(e)}

    def residual_statistics(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Calculate descriptive statistics for residuals.

        Args:
            residuals: Model residuals

        Returns:
            dict: Descriptive statistics
        """
        return {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'median': float(np.median(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'q75': float(np.percentile(residuals, 75)),
            'iqr': float(np.percentile(residuals, 75) - np.percentile(residuals, 25)),
        }

    def overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall diagnostic assessment.

        Args:
            results: Complete diagnostic results

        Returns:
            dict: Overall assessment
        """
        # Check each test
        ljung_box_passed = results.get('ljung_box', {}).get('passed', None)
        normality_passed = results.get('normality', {}).get('overall_passed', None)
        arch_passed = results.get('arch', {}).get('passed', None)

        # Count passes/fails
        tests = [ljung_box_passed, normality_passed, arch_passed]
        tests = [t for t in tests if t is not None]

        if not tests:
            return {'overall_status': 'UNKNOWN', 'message': 'No diagnostic tests completed'}

        n_passed = sum(tests)
        n_total = len(tests)
        pass_rate = n_passed / n_total

        # Determine overall status
        if pass_rate == 1.0:
            status = 'EXCELLENT'
            grade = 'A'
            message = '✅ All diagnostic tests passed. Model assumptions appear valid.'
        elif pass_rate >= 0.67:
            status = 'ACCEPTABLE'
            grade = 'B'
            message = '⚠️ Most diagnostic tests passed. Minor assumption violations.'
        elif pass_rate >= 0.33:
            status = 'CONCERNING'
            grade = 'C'
            message = '⚠️ Several diagnostic tests failed. Model adequacy questionable.'
        else:
            status = 'PROBLEMATIC'
            grade = 'D'
            message = '❌ Most diagnostic tests failed. Model likely inadequate.'

        # Specific issues
        issues = []
        if ljung_box_passed is False:
            issues.append('Residual autocorrelation detected')
        if normality_passed is False:
            issues.append('Residuals not normally distributed')
        if arch_passed is False:
            issues.append('Heteroskedasticity present')

        recommendations = []
        if 'autocorrelation' in ' '.join(issues).lower():
            recommendations.append('Consider adding more lags or different model form')
        if 'normal' in ' '.join(issues).lower():
            recommendations.append('Consider Box-Cox transformation or robust methods')
        if 'heteroskedasticity' in ' '.join(issues).lower():
            recommendations.append('Consider GARCH model or robust standard errors')

        return {
            'overall_status': status,
            'grade': grade,
            'pass_rate': pass_rate,
            'tests_passed': n_passed,
            'tests_total': n_total,
            'message': message,
            'issues': issues,
            'recommendations': recommendations,
        }

    def plot_diagnostics(self, residuals: np.ndarray, model_name: str = 'Model', save_path: Optional[str] = None):
        """
        Generate comprehensive diagnostic plots.

        Creates 4-panel diagnostic plot:
        1. Residuals over time
        2. Q-Q plot (normality)
        3. ACF plot (autocorrelation)
        4. Histogram with normal overlay

        Args:
            residuals: Model residuals
            model_name: Model name for title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Residual Diagnostics: {model_name}', fontsize=16, fontweight='bold')

        # 1. Residuals over time
        axes[0, 0].plot(residuals, 'o-', markersize=4, alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=np.std(residuals), color='orange', linestyle=':', alpha=0.5, label='±1 std')
        axes[0, 0].axhline(y=-np.std(residuals), color='orange', linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Q-Q Plot
        stats.probplot(residuals, dist='norm', plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Check)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. ACF Plot
        plot_acf(residuals, lags=min(20, len(residuals) // 4), ax=axes[1, 0], alpha=0.05)
        axes[1, 0].set_title('Autocorrelation Function (ACF)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Histogram with normal overlay
        axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')

        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')

        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Residual Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f'Diagnostic plot saved to {save_path}')
        else:
            plt.show()

        plt.close()


def format_diagnostic_report(results: Dict[str, Any]) -> str:
    """
    Format diagnostic results as readable report.

    Args:
        results: Results from run_full_diagnostics()

    Returns:
        Formatted string report
    """
    if not results.get('sufficient_data', False):
        return '⚠️ Insufficient data for diagnostics'

    model_name = results.get('model_name', 'Model')
    n_residuals = results.get('n_residuals', 0)

    lines = ['', '=' * 70]
    lines.append(f'RESIDUAL DIAGNOSTICS: {model_name}')
    lines.append('=' * 70)
    lines.append(f'Sample Size: {n_residuals} residuals')
    lines.append('')

    # Ljung-Box Test
    lb = results.get('ljung_box', {})
    if lb.get('status') != 'ERROR':
        lines.append('1. Ljung-Box Test (Autocorrelation)')
        lines.append(f'   Status: {lb.get("status", "UNKNOWN")}')
        lines.append(f'   Statistic: {lb.get("statistic", 0):.4f}')
        lines.append(f'   P-value: {lb.get("p_value", 0):.4f}')
        lines.append(f'   {lb.get("interpretation", "")}')
        lines.append('')

    # Normality Tests
    norm = results.get('normality', {})
    if norm:
        lines.append('2. Normality Tests')
        lines.append(f'   Status: {norm.get("status", "UNKNOWN")}')

        jb = norm.get('jarque_bera', {})
        if jb:
            lines.append(f'   Jarque-Bera p-value: {jb.get("p_value", 0):.4f}')

        lines.append(f'   Skewness: {norm.get("skewness", 0):.4f} - {norm.get("skewness_note", "")}')
        lines.append(f'   Kurtosis: {norm.get("kurtosis", 0):.4f} - {norm.get("kurtosis_note", "")}')
        lines.append('')

    # ARCH Test
    arch = results.get('arch', {})
    if arch.get('status') not in ['ERROR', 'SKIPPED']:
        lines.append('3. ARCH Test (Heteroskedasticity)')
        lines.append(f'   Status: {arch.get("status", "UNKNOWN")}')
        lines.append(f'   LM Statistic: {arch.get("lm_statistic", 0):.4f}')
        lines.append(f'   P-value: {arch.get("lm_p_value", 0):.4f}')
        lines.append(f'   {arch.get("interpretation", "")}')
        lines.append('')

    # Overall Assessment
    assessment = results.get('assessment', {})
    if assessment:
        lines.append('=' * 70)
        lines.append('OVERALL ASSESSMENT')
        lines.append('=' * 70)
        lines.append(f'Status: {assessment.get("overall_status", "UNKNOWN")}')
        lines.append(f'Grade: {assessment.get("grade", "N/A")}')
        lines.append(f'Tests Passed: {assessment.get("tests_passed", 0)}/{assessment.get("tests_total", 0)}')
        lines.append(f'\n{assessment.get("message", "")}')

        issues = assessment.get('issues', [])
        if issues:
            lines.append('\nIssues Identified:')
            for issue in issues:
                lines.append(f'  - {issue}')

        recommendations = assessment.get('recommendations', [])
        if recommendations:
            lines.append('\nRecommendations:')
            for rec in recommendations:
                lines.append(f'  - {rec}')

        lines.append('=' * 70)

    return '\n'.join(lines)
