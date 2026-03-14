from sklearn.naive_bayes import MultinomialNB


# API
mnb = MultinomialNB(
    alpha = 1.0,
    fit_prior = True, 
    class_prior = None,
    norm = False
)

# Attributes
# cnb.class_log_prior_
# cnb.feature_log_prob_
# cnb.coef_
# cnb.class_count_
# cnb.feature_count_
# cnb.feature_all_

# Method
# .fit()
# .get_params()
# .partial_fit()
# .predict()
# .predict_log_proba()
# .predict_proba()
# .score()
# .set_params()