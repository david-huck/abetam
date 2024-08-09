from batch import BatchResult
import matplotlib
matplotlib.use('Agg')


def test_batch_run():

    batch_params = {
        "N": [50],
        "province": ["Ontario"],
        "random_seed": range(0,4),
        "n_segregation_steps": [41],
        "price_weight_mode": [0.6],
        "ts_step_length": ["W"],
        "start_year": 2020,
        "refurbishment_rate": 0.03,
        "hp_subsidy": 0.3,
        "fossil_ban_year": 2030
    }

    res = BatchResult.from_parameters(batch_params, max_steps=80)
    res.save()