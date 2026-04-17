import forecasting
from sklearn.model_selection import cross_validate

X, y = forecasting.get_Xy()

regressor = forecasting.make_pipeline().fit(X, y)
test_X = forecasting.get_test_X()
print(regressor.predict(test_X))

print("gradient boosting")
regressor = forecasting.make_pipeline()
cv_results = cross_validate(
    regressor, X, y, cv=forecasting.Splitter(), scoring="neg_mean_absolute_error"
)
print(cv_results["test_score"].mean())

print("previous month")
regressor = forecasting.make_pipeline("prev_month")
cv_results = cross_validate(
    regressor, X, y, cv=forecasting.Splitter(), scoring="neg_mean_absolute_error"
)
print(cv_results["test_score"].mean())

print("dummy")
regressor = forecasting.make_pipeline("dummy")
cv_results = cross_validate(
    regressor, X, y, cv=forecasting.Splitter(), scoring="neg_mean_absolute_error"
)
print(cv_results["test_score"].mean())
