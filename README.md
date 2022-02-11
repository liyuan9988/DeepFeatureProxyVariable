# DeepFeatureProxyVariable
Code for "Learning Deep Features in Instrumental Variable Regression" (https://arxiv.org/abs/2010.07154))

## How to Run codes?

1. Install all dependencies
   ```
   pip install -r requirements.txt
   ```
2. Create empty directories for logging
   ```
   mkdir logs
   mkdir dumps
   ```
3. Run codes
   ```
   python main.py <path-to-configs> <problem_setting>
   ```
   `<problem_setting>` can be selected from `ate` and `ope`, which corresponds to ate experiments and ope experiments in the paper. Make sure to input the corresponding config file to each setting. The result can be found in `dumps` folder. You can run in parallel by specifing  `-t` option.
