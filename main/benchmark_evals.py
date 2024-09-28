import yaml
import os
import argparse
import numpy as np
import traceback
import pickle
import pandas as pd

yaml.add_multi_constructor('!!', lambda loader, suffix, node: None)
yaml.add_multi_constructor('tag:yaml.org,2002:python/name', lambda loader, suffix, node: None, Loader=yaml.SafeLoader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='CollectResults',
                    description='Collecting results from runs')
    parser.add_argument('--model-base-path', default="gt_agents")
    parser.add_argument('--algorithm', default="ppo")

    args = parser.parse_args()

    df = pd.DataFrame()

    for run_dir in os.listdir(os.path.join(args.model_base_path, args.algorithm)):

        env = run_dir.split("_")[0]

        # save scores for each run
        run_dir = os.path.join(args.model_base_path, args.algorithm, run_dir)

        if not os.path.isfile(os.path.join(run_dir, env, "args.yml")):
            print(os.path.join(run_dir, env), " not found")
            exit(0)
        with open(os.path.join(run_dir, env, "args.yml"), "r") as theyaml:
            for i in range(1):
                _ = theyaml.readline()    
            run_arguments = yaml.safe_load(theyaml)
        with open(os.path.join(run_dir, env, "config.yml"), "r") as theyaml:
            for i in range(1):
                _ = theyaml.readline()    
            config = yaml.safe_load(theyaml)
    
        
        the_dict = {}
        for elem in run_arguments[0]:
            the_dict[elem[0]] = elem[1]
        for elem in config[0]:
            the_dict[elem[0]] = elem[1]
    
        try: 
            evals = np.mean(np.load(os.path.join(run_dir, "evaluations.npz"))["results"][-1])
            the_dict = {"run": run_dir.split("/")[-1], "env": the_dict["env"], "seed": the_dict["seed"], "eval_score": evals}

            df = pd.concat([df, pd.DataFrame([the_dict])], ignore_index=True)

            df = df.sort_values('env')
            df.to_csv(f"collected_results_{args.algorithm}.csv")
        except Exception:
            print(traceback.format_exc())
            exit(0)