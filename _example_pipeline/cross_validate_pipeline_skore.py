import skore

import forecasting

skore.login()

X, y = forecasting.get_Xy()
regressor = forecasting.make_pipeline()

print("gradient boosting")
report = skore.evaluate(regressor, X=X, y=y, splitter=forecasting.Splitter())

project = skore.Project("jerome-workspace-1/stallion-forecasting", mode="hub")
project.put("gradient_boosting_pipeline", report)
