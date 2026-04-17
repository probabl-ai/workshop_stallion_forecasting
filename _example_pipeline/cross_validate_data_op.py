import forecasting

data_op = forecasting.make_data_op()
env = forecasting.get_env()
data_op.skb.full_report(env, title="fit_predict")

learner = data_op.skb.make_learner().fit(env)
learner.report(
    environment=forecasting.get_test_env(), mode="predict", title="predict test"
)

print("gradient boosting")
cv_results = data_op.skb.cross_validate(env)
print(cv_results["test_score"].mean())

print("previous month")
data_op = forecasting.make_data_op("prev_month")
cv_results = data_op.skb.cross_validate(env)
print(cv_results["test_score"].mean())

print("dummy")
data_op = forecasting.make_data_op("dummy")
cv_results = data_op.skb.cross_validate(env)
print(cv_results["test_score"].mean())
