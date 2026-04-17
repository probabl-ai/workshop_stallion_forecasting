import skore

import forecasting

skore.login()

data_op = forecasting.make_data_op()
env = forecasting.get_env()
data_op.skb.full_report(env, title="fit_predict")

learner = data_op.skb.make_learner().fit(env)
learner.report(
    environment=forecasting.get_test_env(), mode="predict", title="predict test"
)

print("gradient boosting")
report = skore.evaluate(data_op, data=env, splitter=forecasting.Splitter())

project = skore.Project("jerome-workspace-1/stallion-forecasting", mode="hub")
project.put("gradient_boosting", report)
