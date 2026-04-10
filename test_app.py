import sys, os
sys.path.insert(0, os.path.abspath('d:/2026/apex-gpt'))
from app import predictor
from app.city_rates import get_cost_estimate
from app.main import _build_cost

predictor.load_model()
predictor.load_meta()

res = predictor.predict(area=1000, unit='sqft', floors=1, building_type=0, quality=1, city='bangalore')
cost = _build_cost(res['materials'], 'bangalore', res['total_area_sqft'], 1)

print('API Test -> OK')
print(f'City: {cost.city}')
print(f'R2:   {res["model_r2_scores"]["cement_bags"]:.4f}')
print(f'Materials: {res["materials"]}')
print(f'Total Estimate: Rs {cost.total_cost_inr:,}')
print(f'Rate per sqft: Rs {cost.cost_per_sqft}')
