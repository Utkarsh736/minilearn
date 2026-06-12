import numpy as np

from minilearn import LinearRegression


def test_linear_regression_learns_simple_line():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegression(lr=0.01, epochs=1000)
    model.fit(X, y)

    prediction = model.predict([[5.0]])
    np.testing.assert_allclose(prediction, np.array([11.0]), rtol=1e-1, atol=1e-1)
