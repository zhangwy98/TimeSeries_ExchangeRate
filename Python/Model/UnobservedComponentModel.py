import statsmodels.api as sm

class UnobservedComponentModel:
    def __init__(self):
        self.model_name = "UnobservedComponentModel"
        return

    def fit(self, ts):
        unrestricted_model = {
            'level': 'local linear trend', 'cycle': False , 'seasonal': 12
        }

        model = sm.tsa.UnobservedComponents(endog=ts, **unrestricted_model)
        self.trained_model = model.fit()
        return self

    def predict(self, next_n_prediction):
        prediction = self.trained_model.forecast(steps=next_n_prediction)
        return prediction