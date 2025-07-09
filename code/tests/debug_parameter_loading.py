#!/usr/bin/env python3
"""
Debug why optimized parameters aren't being used despite successful deployment.
"""

import json

def debug_parameter_loading():
    print('🔍 DEBUGGING WHY OPTIMIZED PARAMETERS AREN\'T BEING USED')
    print('=' * 60)

    # Check if hyperparameters.json was actually updated
    with open('hyperparameters.json', 'r') as f:
        config = json.load(f)

    models_config = config['models']
    test_models = ['DLinear', 'TiDE', 'NHiTS']

    print('📋 VERIFYING HYPERPARAMETERS.JSON WAS UPDATED:')
    for model_name in test_models:
        if model_name in models_config:
            params = models_config[model_name]['default_params']
            print(f'✅ {model_name}: {len(params)} parameters')
            # Check for key optimized parameter that should indicate it's updated
            if 'n_epochs' in params:
                n_epochs = params['n_epochs']
                if n_epochs >= 50:  # Optimized values should be 50+ instead of default ~10-20
                    print(f'   ✅ OPTIMIZED: n_epochs = {n_epochs} (indicates optimized parameters)')
                else:
                    print(f'   ❌ DEFAULT: n_epochs = {n_epochs} (indicates default parameters)')
            else:
                print(f'   ❓ No n_epochs parameter found')
        else:
            print(f'❌ {model_name}: Not found in config')

    print('\n🔍 TESTING PARAMETER LOADING IN PRODUCTION SYSTEM:')
    print('=' * 50)

    try:
        from analysis import CVEForecastAnalyzer
        analyzer = CVEForecastAnalyzer()
        
        for model_name in test_models:
            params = analyzer._get_model_hyperparameters(model_name)
            if params and 'n_epochs' in params:
                n_epochs = params['n_epochs']
                print(f'📊 {model_name} production params: n_epochs = {n_epochs}')
                if n_epochs >= 50:
                    print(f'   ✅ Production system IS using optimized parameters')
                else:
                    print(f'   ❌ Production system NOT using optimized parameters')
            else:
                print(f'❌ {model_name}: Failed to load parameters in production')
                
    except Exception as e:
        print(f'❌ Error testing production parameter loading: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_parameter_loading()
