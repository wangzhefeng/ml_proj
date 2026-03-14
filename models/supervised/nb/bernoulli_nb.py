from sklearn.naive_bayes import BernoulliNB


# API
bnb = BernoulliNB(
    alpha = 1.0,
    binarize = 0.0,
    fit_prior = True, 
    class_prior = None,
)

# Attributes
# bnb.class_log_prior_
# bnb.feature_log_prob_
# bnb.class_count_
# bnb.feature_count_

# Method
# .fit()
# .get_params()
# .partial_fit()
# .predict()
# .predict_log_proba()
# .predict_proba()
# .score()
# .set_params()
