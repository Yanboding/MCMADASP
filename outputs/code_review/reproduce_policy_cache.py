import sys
sys.path.insert(0, '/Users/yanboding/Desktop/PhdTopic/MCMADASP')
import numpy as np
import gurobipy as gp
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction
from importance_sampling import FixedLengthProposal
from decision_maker import ApproxQAgent

config=get_config_by_type('toy')
e=config.env
state=(np.zeros(e.planning_horizon), np.zeros(e.planning_horizon), np.array([1., 2.]))
zero_action=(np.zeros((e.booking_window_size,e.num_types)), np.zeros(e.planning_horizon))

def new_agent(gf, grb):
    return ApproxQAgent(e, discount_factor=e.discount_factor, sample_path_number=1,
        sample_path_proposal=FixedLengthProposal(1), generating_function=gf,
        grb_env=grb, subproblem_grb_envs=[grb])

with gp.Env(empty=True) as grb:
    grb.setParam('OutputFlag',0)
    grb.start()
    gf=AbsorptionLinearPenaltyFunction(e)
    n=gf.number_of_coefficients
    gf.set_coefficients(np.zeros(n))
    a=new_agent(gf,grb)
    fixed=a.approx_Q_solve(state,1,action=zero_action)
    after=a.approx_Q_solve(state,1)
    fresh=new_agent(gf,grb).approx_Q_solve(state,1)
    print('FIXED_ACTION_LEAK', {'fixed_obj': fixed[0], 'later_free_obj':after[0], 'fresh_free_obj':fresh[0], 'later_scheduled':float(after[1][0].sum()), 'fresh_scheduled':float(fresh[1][0].sum())})
    a=new_agent(gf,grb)
    base=a.approx_Q_solve(state,1)
    theta=np.zeros(n)
    theta[2*e.planning_horizon:2*e.planning_horizon+e.num_types] = -1000
    gf.set_coefficients(theta)
    stale=a.approx_Q_solve(state,1)
    fresh=new_agent(gf,grb).approx_Q_solve(state,1)
    print('COEFFICIENT_CACHE', {'before_obj':base[0], 'after_coeff_update_obj':stale[0], 'fresh_obj':fresh[0], 'stale_scheduled':float(stale[1][0].sum()), 'fresh_scheduled':float(fresh[1][0].sum())})
    fx=np.zeros((e.booking_window_size,e.num_types)); fx[0,0]=1
    fixed_noncanonical=(fx, np.minimum(e.convert_action_to_booking_slots(fx), e.overtime_capacity))
    gf.set_coefficients(np.zeros(n))
    a=new_agent(gf,grb)
    obj,returned,_=a.solve(state,1,action=fixed_noncanonical)
    print('FIXED_ACTION_REPAIR',{'input_overtime':fixed_noncanonical[1].tolist(),'returned_overtime':returned[1].tolist(),'reported_obj':obj,'returned_actual_cost':float(e.cost_fn(state,returned))})
print('EXTENSIVE_FORM_METHOD',hasattr(ApproxQAgent,'extensive_form_train'))
