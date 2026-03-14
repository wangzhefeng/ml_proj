from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# API
gnb = GaussianNB(
    priors = None,
    # var_smoothing = 1e-09
)

# Attributes
# gnb.class_prior_
# gnb.class_count_
# gnb.theta_
# gnb.sigma_
# gnb.esplion_

# Method
# .fit()
# .get_params()
# .partial_fit()
# .predict()
# .predict_log_proba()
# .predict_proba()
# .score()
# .set_params()