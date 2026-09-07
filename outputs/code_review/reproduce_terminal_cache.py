import os,sys,tempfile
sys.path.insert(0,'/Users/yanboding/Desktop/PhdTopic/MCMADASP')
import numpy as np
import gurobipy as gp
import run
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction
config=get_config_by_type('toy'); e=config.env
state=(np.r_[1.,np.zeros(e.planning_horizon-1)],np.zeros(e.planning_horizon),np.zeros(e.num_types))
gf=AbsorptionLinearPenaltyFunction(e); theta=np.zeros(gf.number_of_coefficients); theta[0]=1; gf.set_coefficients(theta)
old=os.getcwd()
with tempfile.TemporaryDirectory(prefix='penalty-terminal-cache-') as tmp, gp.Env(empty=True) as grb:
    os.chdir(tmp)
    grb.setParam('OutputFlag',0); grb.start()
    common=dict(uid='same-record',experiment_name='same_experiment',policy_id='myopic',agent_name='myopic',agent_args={},env_args=config.args,init_state=state,sample_path=np.zeros((0,e.num_types)),warm_up_periods=0,generating_function=gf,grb_env=grb,period_weights=[1.])
    absorbed=run.calculate_policy_costs_with_penalty(**common,terminal='absorbed')
    reused=run.calculate_policy_costs_with_penalty(**common,terminal='truncated')
    common['experiment_name']='fresh_experiment'
    fresh=run.calculate_policy_costs_with_penalty(**common,terminal='truncated')
    print('TERMINAL_CACHE', {'absorbed_penalty':absorbed['total_penalty'],'truncated_cached_penalty':reused['total_penalty'],'truncated_fresh_penalty':fresh['total_penalty']})
    os.chdir(old)
