from configs.config_utils import (
    build_configs,
    register_composition,
    INTERESTING_COMPOSITIONS,
)

# RAW BAYES RISKS
for n_1, n_2 in [("OUTER", "outer"), ("INNER", "inner"), ("CENTRAL", "central")]:
    register_composition(
        f"COMPOSITE BAYES ALL {n_1}",
        build_configs(
            ["risk_bayes_{logscore,brierscore,spherical,zero-one}_" + f"{n_2}_T=1.0"]
        ),
    )

# for n_1, n_2 in [("OUTER", "outer"), ("INNER", "inner"), ("CENTRAL", "central")]:
#     register_composition(
#         f"COMPOSITE BAYES (LBS) {n_1}",
#         build_configs([
#         "risk_bayes_{logscore,brierscore,spherical}_" + f"{n_2}_T=1.0"
#     ]),
#     )


# # BAYES AND DENSITY BASED RISKS

for n_1, n_2 in [("OUTER", "outer"), ("INNER", "inner"), ("CENTRAL", "central")]:
    register_composition(
        f"COMPOSITE BAYES ALL {n_1} + M",
        build_configs(
            ["risk_bayes_{logscore,brierscore,spherical,zero-one}_" + f"{n_2}_T=1.0"]
        ),
        ['mahalanobis_label="Mahalanobis score"'],
    )

# for n_1, n_2 in [("OUTER", "outer"), ("INNER", "inner"), ("CENTRAL", "central")]:
#     register_composition(
#         f"COMPOSITE BAYES (LBS) {n_1} + M",
#         build_configs([
#         "risk_bayes_{logscore,brierscore,spherical}_" + f"{n_2}_T=1.0"
#     ]),
#     ['mahalanobis_label="Mahalanobis score"']
#     )

# for n_1, n_2 in [("OUTER", "outer"), ("INNER", "inner"), ("CENTRAL", "central")]:
#     register_composition(
#         f"COMPOSITE BAYES ALL {n_1} + GMM",
#         build_configs([
#         "risk_bayes_{logscore,brierscore,spherical,zero-one}_" + f"{n_2}_T=1.0"
#     ]),
#     ['gmm_label="GMM score"']
#     )

# for n_1, n_2 in [("OUTER", "outer"), ("INNER", "inner"), ("CENTRAL", "central")]:
#     register_composition(
#         f"COMPOSITE BAYES (LBS) {n_1} + GMM",
#         build_configs([
#         "risk_bayes_{logscore,brierscore,spherical}_" + f"{n_2}_T=1.0"
#     ]),
#     ['gmm_label="GMM score"']
#     )


# PURE EXCESS RISKS
for n_1, n_2, n_3, n_4 in [
    ("OUTER", "OUTER", "outer", "outer"),
    # ("OUTER", "INNER", "outer", "inner"),
    # ("OUTER", "CENTRAL", "outer", "central"),
    # ("INNER", "OUTER", "inner", "outer"),
    # ("INNER", "INNER", "inner", "inner"),
    # ("INNER", "CENTRAL", "inner", "central"),
    # ("CENTRAL", "OUTER", "central", "outer"),
    # ("CENTRAL", "INNER", "central", "inner"),
    # ("CENTRAL", "CENTRAL", "central", "central")
]:
    register_composition(
        f"COMPOSITE EXCESS ALL {n_1} {n_2}",
        build_configs(
            [
                "risk_excess_{logscore,brierscore,spherical,zero-one}_"
                + f"{n_3}_{n_4}_T=1.0"
            ]
        ),
    )

# for n_1, n_2, n_3, n_4 in [
#     ("OUTER", "OUTER", "outer", "outer"),
#     ("OUTER", "INNER", "outer", "inner"),
#     ("OUTER", "CENTRAL", "outer", "central"),
#     ("INNER", "OUTER", "inner", "outer"),
#     ("INNER", "INNER", "inner", "inner"),
#     ("INNER", "CENTRAL", "inner", "central"),
#     ("CENTRAL", "OUTER", "central", "outer"),
#     ("CENTRAL", "INNER", "central", "inner"),
#     ("CENTRAL", "CENTRAL", "central", "central")]:
#     register_composition(
#         f"COMPOSITE EXCESS LBS {n_1} {n_2}",
#         build_configs([
#         "risk_excess_{logscore,brierscore,spherical}_" + f"{n_3}_{n_4}_T=1.0"
#     ]),
# )

# EXCESS RISKS + MAHALANOBIS
for n_1, n_2, n_3, n_4 in [
    ("OUTER", "OUTER", "outer", "outer"),
    # ("OUTER", "INNER", "outer", "inner"),
    # ("OUTER", "CENTRAL", "outer", "central"),
    # ("INNER", "OUTER", "inner", "outer"),
    # ("INNER", "INNER", "inner", "inner"),
    # ("INNER", "CENTRAL", "inner", "central"),
    # ("CENTRAL", "OUTER", "central", "outer"),
    # ("CENTRAL", "INNER", "central", "inner"),
    # ("CENTRAL", "CENTRAL", "central", "central")
]:
    register_composition(
        f"COMPOSITE EXCESS ALL {n_1} {n_2} + M",
        build_configs(
            [
                "risk_excess_{logscore,brierscore,spherical,zero-one}_"
                + f"{n_3}_{n_4}_T=1.0"
            ]
        ),
        ['mahalanobis_label="Mahalanobis score"'],
    )


# EXCESS ALEATORIC TOTAL MAHALANOBIS
for n_1, n_2, n_3, n_4 in [
    ("OUTER", "OUTER", "outer", "outer"),
]:
    register_composition(
        f"COMPOSITE EAT LOGSCORE {n_1} {n_2} + M",
        "risk_excess_logscore_" + f"{n_3}_{n_4}_T=1.0",
        "risk_total_logscore_" + f"{n_3}_{n_4}_T=1.0",
        "risk_bayes_logscore_" + f"{n_3}_T=1.0",
        ['mahalanobis_label="Mahalanobis score"'],
    )
