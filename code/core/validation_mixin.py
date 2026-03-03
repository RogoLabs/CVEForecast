"""
Validation Mixin - Shared validation methods for all forecasters.

Provides common validation, diagnostic, and testing functionality
that can be used by any forecaster implementation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from diagnostics.horizon_analysis import HorizonAnalysis, format_horizon_report
from diagnostics.residual_analysis import ResidualDiagnostics, format_diagnostic_report
from validation.statistical_tests import ModelComparisonTests
from validation.time_series_cv import RobustTimeSeriesValidator


class ValidationMixin:
    """
    Mixin providing shared validation and diagnostic methods.

    Can be used by any forecaster to add standardized validation
    capabilities without code duplication.
    """

    def perform_cross_validation(self, n_splits: int = 5, forecast_horizon: int = 12) -> Dict[str, Any]:
        """
        Perform time series cross-validation on all models.

        Args:
            n_splits: Number of CV folds
            forecast_horizon: Forecast horizon in periods

        Returns:
            CV results for each model
        """
        if not hasattr(self, 'series') or self.series is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        if not hasattr(self, 'model_results') or not self.model_results:
            raise ValueError('No models trained. Call train_all_models() first.')

        logger = logging.getLogger(self.__class__.__name__)
        logger.info('=' * 70)
        logger.info('PERFORMING TIME SERIES CROSS-VALIDATION')
        logger.info('=' * 70)
        logger.info(f'Configuration: {n_splits}-fold expanding window CV')

        validator = RobustTimeSeriesValidator(n_splits=n_splits, min_train_size=24)
        cv_results = {}

        for model_name, model_data in self.model_results.items():
            if not model_data.get('trained', False):
                continue

            logger.info(f'\nCross-validating: {model_name}')

            try:
                # Create fresh model instance
                model = self.create_model(model_name, model_data['hyperparameters'])

                # Perform CV
                cv_result = validator.validate_model(
                    data=self.series, model=model, forecast_horizon=forecast_horizon, metric_name='MAPE'
                )

                cv_results[model_name] = cv_result

                if cv_result['is_valid']:
                    logger.info(
                        f'✅ {model_name}: CV MAPE = {cv_result["mean_error"]:.2f}% ± '
                        f'{cv_result["std_error"]:.2f}% ({cv_result["n_successful_folds"]} folds)'
                    )

                    # Update model_results
                    model_data['cv_metrics'] = {
                        'cv_mape_mean': cv_result['mean_error'],
                        'cv_mape_std': cv_result['std_error'],
                        'cv_validated': True,
                    }

            except Exception as e:
                logger.error(f'❌ CV failed for {model_name}: {e}')
                cv_results[model_name] = {'is_valid': False, 'error': str(e)}

        logger.info('=' * 70)
        return cv_results

    def perform_statistical_tests(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical significance tests between models.

        Args:
            cv_results: Results from perform_cross_validation()

        Returns:
            Statistical test results
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.info('=' * 70)
        logger.info('PERFORMING STATISTICAL SIGNIFICANCE TESTS')
        logger.info('=' * 70)

        tester = ModelComparisonTests(alpha=0.05)

        # Extract errors
        errors_dict = {}
        for model_name, cv_result in cv_results.items():
            if cv_result.get('is_valid', False):
                errors_dict[model_name] = np.array(cv_result['errors_by_fold'])

        if len(errors_dict) < 2:
            logger.warning('Need at least 2 models for comparison')
            return {}

        # Model rankings
        rankings = tester.rank_models_with_significance(errors_dict)
        logger.info('\nModel Rankings:')
        for _, row in rankings.iterrows():
            logger.info(
                f'  {row["rank"]}. {row["model"]:20s} Avg Loss: {row["avg_squared_error"]:.2f}  {row["interpretation"]}'
            )

        # Model Confidence Set
        mcs_result = tester.model_confidence_set(errors_dict, alpha=0.10)
        logger.info(f'\nModel Confidence Set: {", ".join(mcs_result["included_models"])}')
        logger.info(f'Recommendation: {mcs_result["recommendation"]}')

        logger.info('=' * 70)

        return {'rankings': rankings, 'mcs_result': mcs_result, 'errors_dict': errors_dict}

    def run_residual_diagnostics(self, models_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run residual diagnostics on models.

        Args:
            models_to_test: List of model names to test (None = all)

        Returns:
            Diagnostic results
        """
        if not hasattr(self, 'series') or self.series is None:
            raise ValueError('Data not loaded')

        if not hasattr(self, 'model_results') or not self.model_results:
            raise ValueError('No models trained')

        logger = logging.getLogger(self.__class__.__name__)
        logger.info('=' * 70)
        logger.info('RUNNING RESIDUAL DIAGNOSTICS')
        logger.info('=' * 70)

        diagnostics = ResidualDiagnostics(alpha=0.05)
        diagnostic_results = {}

        # Determine which models to test
        if models_to_test is None:
            models_to_test = [name for name, data in self.model_results.items() if data.get('trained', False)]

        for model_name in models_to_test:
            model_data = self.model_results.get(model_name)
            if not model_data or not model_data.get('trained', False):
                continue

            logger.info(f'\nDiagnosing: {model_name}')

            try:
                # Split data
                split_ratio = 0.8
                split_point = int(split_ratio * len(self.series))
                train_data = self.series[:split_point]
                val_data = self.series[split_point:]

                # Train and predict
                model = self.create_model(model_name, model_data['hyperparameters'])
                model.fit(train_data)
                predictions = model.predict(len(val_data))

                # Calculate residuals
                residuals = val_data.values().flatten() - predictions.values().flatten()

                # Run diagnostics
                diag_results = diagnostics.run_full_diagnostics(residuals, model_name)
                diagnostic_results[model_name] = diag_results

                logger.info(format_diagnostic_report(diag_results))

                # Update model_results
                assessment = diag_results.get('assessment', {})
                model_data['diagnostics'] = {
                    'grade': assessment.get('grade', 'N/A'),
                    'status': assessment.get('overall_status', 'UNKNOWN'),
                    'pass_rate': assessment.get('pass_rate', 0),
                }

            except Exception as e:
                logger.error(f'❌ Diagnostics failed for {model_name}: {e}')
                diagnostic_results[model_name] = {'error': str(e)}

        logger.info('=' * 70)
        return diagnostic_results

    def run_horizon_analysis(self, horizons: List[int] = [1, 3, 6, 12], top_n: int = 5) -> Dict[str, Any]:
        """
        Analyze performance across forecast horizons.

        Args:
            horizons: Horizons to evaluate
            top_n: Number of top models to analyze

        Returns:
            Horizon analysis results
        """
        if not hasattr(self, 'series') or self.series is None:
            raise ValueError('Data not loaded')

        if not hasattr(self, 'model_results') or not self.model_results:
            raise ValueError('No models trained')

        logger = logging.getLogger(self.__class__.__name__)
        logger.info('=' * 70)
        logger.info('RUNNING HORIZON ANALYSIS')
        logger.info('=' * 70)

        analyzer = HorizonAnalysis()
        horizon_results = {}

        # Get top N models by MAPE
        top_models = sorted(
            [(name, data) for name, data in self.model_results.items() if data.get('trained', False)],
            key=lambda x: x[1]['metrics'].get('mape', float('inf')),
        )[:top_n]

        for model_name, model_data in top_models:
            logger.info(f'\nAnalyzing horizons for {model_name}')

            try:
                model = self.create_model(model_name, model_data['hyperparameters'])
                results = analyzer.evaluate_by_horizon(model, self.series, horizons=horizons, n_splits=3)

                horizon_results[model_name] = results
                logger.info(format_horizon_report(results, model_name))

                # Update model_results
                summary = results.get('summary', {})
                model_data['horizon_analysis'] = {
                    'best_horizon': summary.get('best_horizon'),
                    'degradation_rate': summary.get('degradation_rate'),
                }

            except Exception as e:
                logger.error(f'❌ Horizon analysis failed for {model_name}: {e}')
                horizon_results[model_name] = {'error': str(e)}

        logger.info('=' * 70)
        return horizon_results

    def save_validation_results(self, output_dir: str = 'validation_results'):
        """
        Save all validation results to files.

        Args:
            output_dir: Directory to save results
        """
        import json

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger = logging.getLogger(self.__class__.__name__)

        # Save CV results
        if hasattr(self, 'validation_results') and 'cv' in self.validation_results:
            cv_file = output_path / 'cv_results.json'
            with open(cv_file, 'w') as f:
                json.dump(self.validation_results['cv'], f, indent=2, default=str)
            logger.info(f'✓ CV results saved to {cv_file}')

        # Save diagnostic results
        if hasattr(self, 'diagnostic_results'):
            diag_file = output_path / 'diagnostics.json'
            with open(diag_file, 'w') as f:
                json.dump(self.diagnostic_results, f, indent=2, default=str)
            logger.info(f'✓ Diagnostic results saved to {diag_file}')

        logger.info(f'All validation results saved to {output_path}')
