"""
MOGPTK - a Python toolkit for multi output Gaussian processes

Example:
    >>> import mogptk
    ...
    >>> x, y = get_your_data()
    ...
    >>> data = mogptk.DataSet()
    >>> data.append(mogptk.LoadFunction(lambda x: np.sin(5*x[:,0]), n=200, start=0.0, end=4.0, name='A'))
    >>> data.append(mogptk.LoadFunction(lambda x: np.sin(6*x[:,0])+2, n=200, start=0.0, end=4.0, var=0.03, name='B'))
    >>> data.append(mogptk.LoadFunction(lambda x: np.sin(6*x[:,0])+2 - np.sin(4*x[:,0]), n=20, start=0.0, end=4.0, var=0.03, name='C'))
    ...
    >>> data.set_pred_range(0.0, 5.0, n=200)
    ...
    >>> mosm = mogptk.MOSM(data, Q=3)
    >>> mosm.estimate_params()
    >>> mosm.train()
    >>> mosm.predict()
    ...
    >>> data.plot()
"""
