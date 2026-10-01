from types import SimpleNamespace

sslp_5_25 = SimpleNamespace(
    n_locations=5,
    n_clients=25,
    time_limit=60,
    mip_gap=0.01,
    verbose=0,
    seed=7,
    siplib_instance_names=["sslp_5_25_50", "sslp_5_25_100"],
    data_path='./env/sslp/data'
)

sslp_10_50 = SimpleNamespace(
    n_locations=10,
    n_clients=50,
    time_limit=60,
    verbose=0,
    seed=7,
    siplib_instance_names=["sslp_10_50_50"],
    data_path='./env/sslp/data'
)

sslp_15_45 = SimpleNamespace(
    n_locations=15,
    n_clients=45,
    time_limit=60,
    verbose=0,
    seed=7,
    siplib_instance_names=["sslp_15_45_15"],
    data_path='./env/sslp/data'
)