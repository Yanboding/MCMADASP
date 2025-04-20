import json
from environment import AdvanceSchedulingEnv
from experiments import Config

if __name__ == '__main__':
    config = Config.from_real_scale()
    env_params = config.env_params
    init_state = config.init_state
    t = 1
    env = AdvanceSchedulingEnv(**env_params)
    replication = 1000
    num_sample_path = 1000
    with open('table.dat', 'w') as f:
        for _ in range(replication):
            sample_path = env.reset_arrivals(t=t)
            for sample_path_number in [1]+[i for i in range(100, num_sample_path+1, 100)]:
                parameter = {
                    "sample_path": sample_path.tolist(),
                    "sample_path_number": sample_path_number
                }
                line = "python run.py --params '" + json.dumps(parameter) + "'\n"
                f.write(line)