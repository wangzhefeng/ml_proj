from sklearn.naive_bayes import MultinomialNB


# API
mnb = MultinomialNB(
    alpha = 1.0,
    fit_prior = True, 
    class_prior = None,
)

# Attributes
# mnb.class_log_prior_
# mnb.intercept_
# mnb.feature_log_prob_
# mnb.coef_
# mnb.class_count_
# mnb.feature_count_

# Method
# .fit()
# .get_params()
# .partial_fit()
# .predict()
# .predict_log_proba()
# .predict_proba()
# .score()
# .set_params()