"""
NetrAI Module 5: Telemedicine Simulation
Estimates PHC workload, referral rates, and cost savings.
"""

import numpy as np


def simulate_phc_deployment(screening_results, phc_config=None):
    """
    Simulate deployment at a Primary Health Center.
    
    Args:
        screening_results: list of dicts with keys 'grade', 'is_referable'
        phc_config: dict with PHC parameters (optional)
    
    Returns:
        dict with simulation metrics
    """
    if phc_config is None:
        phc_config = {
            'patients_per_day': 50,
            'manual_screening_time_min': 15,
            'ai_screening_time_min': 0.5,
            'ophthalmologist_consult_cost_inr': 500,
            'ai_screening_cost_inr': 20,
            'referral_travel_cost_inr': 300,
        }
    
    n = len(screening_results)
    if n == 0:
        return {'error': 'No screening results provided'}
    
    grades = [r['grade'] for r in screening_results]
    referrals = sum(1 for r in screening_results if r.get('is_referable', False))
    
    grade_dist = {i: grades.count(i) for i in range(5)}
    referral_rate = referrals / n
    
    # Time savings
    manual_time_hr = n * phc_config['manual_screening_time_min'] / 60
    ai_time_hr = n * phc_config['ai_screening_time_min'] / 60
    time_saved_hr = manual_time_hr - ai_time_hr
    
    # Cost savings
    manual_cost = n * phc_config['ophthalmologist_consult_cost_inr']
    ai_cost = (n * phc_config['ai_screening_cost_inr'] + 
               referrals * phc_config['ophthalmologist_consult_cost_inr'] +
               referrals * phc_config['referral_travel_cost_inr'])
    cost_saved = manual_cost - ai_cost
    
    # Capacity
    daily_capacity_manual = int(8 * 60 / phc_config['manual_screening_time_min'])
    daily_capacity_ai = phc_config['patients_per_day']
    capacity_multiplier = daily_capacity_ai / max(daily_capacity_manual, 1)
    
    return {
        'total_screened': n,
        'grade_distribution': grade_dist,
        'referrals': referrals,
        'referral_rate': round(referral_rate, 3),
        'non_referable': n - referrals,
        'time_manual_hr': round(manual_time_hr, 1),
        'time_ai_hr': round(ai_time_hr, 1),
        'time_saved_hr': round(time_saved_hr, 1),
        'cost_manual_inr': manual_cost,
        'cost_ai_inr': ai_cost,
        'cost_saved_inr': cost_saved,
        'daily_capacity_manual': daily_capacity_manual,
        'daily_capacity_ai': daily_capacity_ai,
        'capacity_multiplier': round(capacity_multiplier, 1),
    }


def format_simulation_report(sim_results):
    """Format simulation results as a readable string."""
    r = sim_results
    lines = [
        f"PHC Deployment Simulation ({r['total_screened']} patients)",
        f"{'='*50}",
        f"Referral Rate: {r['referral_rate']*100:.1f}% ({r['referrals']}/{r['total_screened']})",
        f"Grade Distribution: {r['grade_distribution']}",
        f"",
        f"Time: {r['time_manual_hr']}h manual vs {r['time_ai_hr']}h AI ({r['time_saved_hr']}h saved)",
        f"Cost: INR {r['cost_manual_inr']:,} manual vs INR {r['cost_ai_inr']:,} AI (INR {r['cost_saved_inr']:,} saved)",
        f"Capacity: {r['daily_capacity_manual']} → {r['daily_capacity_ai']} patients/day ({r['capacity_multiplier']}x)",
    ]
    return '\n'.join(lines)
