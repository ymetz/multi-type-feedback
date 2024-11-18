import yaml
import os
import argparse
import numpy as np
import traceback
import pandas as pd
from typing import List

yaml.add_multi_constructor('!!', lambda loader, suffix, node: None)
yaml.add_multi_constructor('tag:yaml.org,2002:python/name', lambda loader, suffix, node: None, Loader=yaml.SafeLoader)

def process_algorithm_results(model_base_path: str, algorithm: str) -> pd.DataFrame:
    """Process results for a single algorithm."""
    df = pd.DataFrame()
    algorithm_path = os.path.join(model_base_path, algorithm)
    
    if not os.path.exists(algorithm_path):
        print(f"Path not found for algorithm {algorithm}: {algorithm_path}")
        return df
        
    for run_dir in os.listdir(algorithm_path):
        env = run_dir.split("_")[0]
        full_run_dir = os.path.join(algorithm_path, run_dir)
        
        try:
            # Read arguments and config
            args_path = os.path.join(full_run_dir, env, "args.yml")
            config_path = os.path.join(full_run_dir, env, "config.yml")
            
            if not os.path.isfile(args_path):
                print(f"Args file not found: {args_path}")
                continue
                
            # Read YAML files
            with open(args_path, "r") as theyaml:
                next(theyaml)  # Skip first line
                run_arguments = yaml.safe_load(theyaml)
                
            with open(config_path, "r") as theyaml:
                next(theyaml)  # Skip first line
                config = yaml.safe_load(theyaml)
            
            # Combine all parameters
            the_dict = {}
            for elem in run_arguments[0]:
                the_dict[elem[0]] = elem[1]
            for elem in config[0]:
                the_dict[elem[0]] = elem[1]
            
            # Load evaluation results
            eval_path = os.path.join(full_run_dir, "evaluations.npz")
            evals = np.mean(np.load(eval_path)["results"][-1])
            
            # Create result dictionary
            result_dict = {
                "algorithm": algorithm,
                "run": run_dir,
                "env": the_dict["env"],
                "seed": the_dict["seed"],
                "eval_score": evals
            }
            
            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {full_run_dir}:")
            print(traceback.format_exc())
            continue
            
    return df

def main():
    parser = argparse.ArgumentParser(
        prog='CollectResults',
        description='Collecting results from multiple algorithm runs'
    )
    parser.add_argument('--model-base-path', default="gt_agents")
    parser.add_argument('--algorithms', nargs='+', default=["ppo", "sac"],
                        help='List of algorithms to process (default: ppo sac)')
    args = parser.parse_args()
    
    # Process each algorithm
    all_results = pd.DataFrame()
    for algorithm in args.algorithms:
        print(f"Processing algorithm: {algorithm}")
        algorithm_df = process_algorithm_results(args.model_base_path, algorithm)
        all_results = pd.concat([all_results, algorithm_df], ignore_index=True)
    
    # Sort and save results
    if not all_results.empty:
        all_results = all_results.sort_values(['env', 'algorithm'])
        output_path = os.path.join(args.model_base_path, "collected_results.csv")
        all_results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        print("No results were collected!")

if __name__ == "__main__":
    main()