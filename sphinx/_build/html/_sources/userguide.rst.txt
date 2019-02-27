---------------------------
User Guide
---------------------------
.. module:: AftFore

Set up
_____________________

Requirements
========================

    :mod:`AftFore` depends on the following external packages:

    - Python 2.7
    - Numpy
    - Scipy
    - Matplotlib
    - Pandas

    Please see `Installing the SciPy Stack <https://www.scipy.org/install.html>`_ for installing Numpy, Scipy, and Matplotlib.

Download
========================

    A souce code is available at `https://github.com/omitakahiro/AftFore/ <https://github.com/omitakahiro/AftFore/>`_.

    Please see the below figure to download the zipped source file. After unzipping the downloaded file, please put the AftFore folder (in the unzipped folder) to your working directory.

    .. image:: github.jpg
        :width: 800px



Data Preparation
_____________________
    A data file is an ascii file that contains the list of the relative time after the main shock [day] (the first column) and magnitude (the second column) of the main shock and its aftershocks. The first event is identified as the main shock, and the time of the main shock must be set to 0. A sample aftershock data of the 1995 Kobe earthquake of M7.3 is included in the source file (./AftFore/Kobe.txt).

    .. note::
        AftFore uses all the aftershock data including earthquake with magnitude smaller than the completeness magnitude for the estimation, so please inclue all the available data in your data file.


    A data file is like:

    .. code-block:: none

        0.000000 7.3
        0.001652 4.4
        0.001890 4.2
        0.002048 4.5
        0.002455 5.2
        0.002933 3.8
        0.003422 4.2
        0.003651 4.0
        0.004033 3.3
        0.004396 5.0
        0.005003 3.6
        0.005305 3.8
        0.012113 3.2
        0.012714 3.1
        0.012822 3.8
        ...

Program Workflow
_____________________

    .. image:: scheme.jpg
        :width: 600px
        :align: center

    The forecast procedure consists of two steps.

    1. | The model parameters are estimated from the aftershock data in the learning period, done by a function :py:func:`Est`. We here use the prior probability distribution for the Bayesian estimation, which is set to be the default values used in *Omi et al.,* (2016) unless otherwise set.

    2. | The forecast for the testing period is produced based on the estimated parameter sets, done by a function :py:func:`Fore`.

    A function :py:func:`EstFore` performs the both two tasks and provides a simple way to produce the forecasts. If the user just wants to generate a forecast, :py:func:`EstFore` is enough. The user may use :py:func:`Est` or :py:func:`Fore` for the advanced purpose.


Quick Start
_____________________
    .. note::

        Before running the code, the user needs to (1) put the downloaded AftFore folder in your working directory and (2) prepare the data file.

    The below code shows the simplest way to generate a forecast by using :py:func:`EstFore`. The parameters that the user has to set are :var:`t_learn` (the range of the learning period), :var:`t_test` (the range of the testing period), and :var:`Data` (the path of the data file). For example, this code uses the data file './AftFore/Kobe.txt' and  generates a forecast for the testing period [1.0, 2.0] (day) based on the data in the learning period [0.0, 1.0] (day). The estimation is carried out with the default prior probability distribution. See :var:`prior` for a method to customize the prior.

    .. code-block:: Python

        import AftFore as aft

        t_learn = [0.0, 1.0]            # The range of the learning period [day].
        t_test  = [1.0, 2.0]            # The range of the testing period [day].
        Data    = './AftFore/Kobe.txt'  # The path of the date file

        aft.EstFore(Data, t_learn, t_test)

    This process takes about a few minutes. The forecast result (the number of earthquakes with :math:`M>M_t` in the testing period, and the probability to have at least one earthquake with :math:`M>M_t` in the testing period) is saved in :var:`fore.txt`.

    .. code-block:: none

        # (M_t) (expected_number) (lower bound of 95% interval) (upper bound of 95% interval) (probability)
        0.95         682.218         564         832      1.0000
        1.05         576.029         477         699      1.0000
        1.15         486.368         403         589      1.0000
        1.25         410.664         340         496      1.0000
        1.35         346.743         287         419      1.0000
        1.45         292.772         241         354      1.0000
        1.55         247.201         202         300      1.0000
        1.65         208.723         170         255      1.0000
        1.75         176.235         142         217      1.0000
        1.85         148.804         119         185      1.0000
        1.95         125.642          99         158      1.0000
        2.05         106.085          82         135      1.0000
        ...

    :py:func:`EstFore` also generates files, :var:`param.pkl`, :var:`param.pdf`, and :var:`fore.pdf`. See :ref:`genfiles` for the details.

    .. note::

       :py:func:`EstFore` may raise some warnings, but the user does not have to care about this for the most cases.

Code Examples
_____________________

    The user needs to import AftFore at first before running the following examples.

    .. code-block:: Python

        import AftFore as aft

.. _paraest:

Parameter Estimation
========================

    The parameter estimation is carried out by :py:func:`Est`. Here the estimation is carried out under the defalut prior probability distribution. See :var:`prior` for a method to customize the prior.

    .. code-block:: Python

        t_learn = [0.0, 1.0]            # The range of the learning period [day].
        Data    = './AftFore/Kobe.txt'  # The path of the date file

        param = aft.Est(Data, t_learn)

    :py:func:`Est` returns a parameter object :var:`param` that contains the estimated parameter values and other information, which will be used for producing forecasts. :py:func:`Est` also generates: :var:`param.pkl` (a pickle file that saves :var:`param`) and :var:`param.pdf` (a figure summarizing the estimated parameters).

    The :var:`param` can be loaded from :var:`param.pkl`.

    .. code-block:: Python
        :name: pickle

        import pickle

        param = pickle.load(open('param.pkl','rb'))



Forecast Generation
========================

    The forecast is produced using the estimated parameter sets :var:`param` by :py:func:`Fore`.

    .. code-block:: Python

        t_test  = [1.0, 2.0]            # The range of the testing period [day].

        aft.Fore(param, t_test)

    :py:func:`Fore` generates :var:`fore.txt` and :var:`fore.pdf` that summarize the forecast result.

    The user may want to compare the forecast with the observation. By setting a keyword *Data_test* to the path of the data file, the observation in the testing period is plotted in :var:`fore.pdf`. See :var:`fore.pdf` for the detail.

    .. code-block:: Python

        t_test  = [1.0, 2.0]             # The range of the testing period [day].
        Data_test = './AftFore/Kobe.txt' # The path of the data file including the data in the testing period

        aft.Fore(param, t_test, Data_test=Data_test)

Tips
_____________________

.. _tips-param1:

How to extract the estimated parameter values
================================================
    The estimated parameter values are stored in the object :var:`param["para"]` (please see also :var:`param` for the detail).

    The MAP parameters:

    .. code-block:: Python

        [k,p,c,beta] = param["para"][["k", "p", "c", "beta"]]
        print( "k: %f" % k )
        print( "p: %f" % p )
        print( "c: %f" % c )
        print( "beta: %f" % beta )

        """ OUTPUT
        k: 0.021769
        p: 1.037202
        c: 0.015635
        beta: 1.691913
        """

    Please also see the next tips for forecasting the aftershock activity from these parameters.

.. _tips-param2:

How to calculate the estimation uncertainty
================================================
    The sets of the parameters sampled from the posterior distribution is stored in the object :var:`param["para_mcmc"]` (please see also :var:`param` for the detail). The estimation uncertainty can be obtained by

    .. code-block:: Python

        print( param["para_mcmc"][["k", "p", "c", "beta"]].std() )

        """ OUTPUT
        k        0.008331
        p        0.068125
        c        0.006976
        beta     0.071577
        """

How the parameter values are related to the earthquake occurrence rate or earthquake number
================================================================================================

    In our model, the occurrence rate of an aftershock of magnithde :math:`M` at time :math:`t` is given by

    .. math:: \lambda(t,M) = \frac{k}{(t+c)^p}\beta e^{-\beta(M-M_0)},

    where :math:`M_0` represents the mainshock magnitude.

    For a given threshold magnitude :math:`M_{th}`, the occurrence rate of an aftershock of :math:`M>M_{th}` at time :math:`t` is obtained by

    .. math:: \lambda(t) = \frac{k}{(t+c)^p} e^{\beta(M_0-M_{th})}.

    The expected number of aftershocks of :math:`M>M_{th}` in the time interval :math:`[t_1,t_2]` is also obtained by

    .. math:: n = \int_{t_1}^{t_2}dt \lambda(t) = \frac{k}{-p+1} [(t_2+c)^{-p+1}-(t_1+c)^{-p+1}] e^{\beta(M_0-M_{th})}.

How to show the N-T plot (the cumulative number of aftershocks versus time), compared with the forecast (Experimental)
=========================================================================================================================

    A function :py:func:`NT_plot` shows the time evolution of the cumulative number of aftershocks of *M > mag_th* in the time interval [0, *t_test_end*], compared with the forecast. The confidence interval is evaluated in a Bayesian way, using the parameter sets sampled from the posterior distribution (para_mcmc in param).

    .. code-block:: Python

        # parameter estimation
        t_learn = [0.0, 1.0]            # The range of the learning period [day].
        Data    = './AftFore/Kobe.txt'  # The path of the date file

        param = aft.Est(Data, t_learn)

        # showing the NT plot with the forecast
        mag_th   = 1.95               # The magnitude threshold M_th fore forecasting
        t_test_end  = 5.0             # The end of the forecasting period [day]. The time interval [0, t_test_end]

        aft.NT_plot(Data, t_test_end, mag_th, param)

    The output figure is saved as "NT.pdf".

    .. image:: NT.png
        :width: 300px

AftFore Reference
_____________________

Function
========================
    .. py:function:: Est(Data,t_learn,prior=None)

        :Keywords: If *prior* is *None*, the default prior probability distribution is used for the estimation. See :var:`prior` for a method to customize the prior.
        :Returns: :var:`param`
        :Generates: :var:`param.pkl` and :var:`param.pdf`

    .. py:function:: Fore(param,t_test,Data_test=None)

        :Keywords: If a keywrod *Data_test* is set to the path of the data file, the observation in the test period is added in the :var:`fore.pdf`.
        :Generates: :var:`fore.txt` and :var:`fore.pdf`

    .. py:function:: EstFore(Data,t_learn,t_test,prior=None)

        :Keywords: If *prior* is *None*, the default prior probability distribution is used for the estimation. See :var:`prior` for a method to customize the prior.
        :Generates: :var:`param.pkl`, :var:`param.pdf`, :var:`fore.txt`, and :var:`fore.pdf`

Object
========================

    .. var:: t_learn

        [float, float]

        The lower and upper range of the leraning period. The unit is 'day'.

    .. var:: t_test

        [float, float]

        The lower and upper range of the testing period. The unit is 'day'.

    .. var:: Data

        string

        The path of the data file

    .. var:: param

        A parameter object. Also see :ref:`tips-param1` and :ref:`tips-param2`.

            param["para"]: the MAP estimated parameters (pandas.Series)

            param["para_mcmc"]: the parameter set sampled from the posterior distribution (pandas.DataFrame)

            param["mag_ref"]: the main shock magnitude

            param["t"]: the estimation period

    .. var:: prior

        A prior object (optional: the default prior is used unless otherwise set)

        The user can introduce the prior for the parameters {:math:`K,p,c,\beta,\sigma,\mu1`} (Please see `here <./method.pdf>`_ for the detail of each parameter). A prior object is a list where each element defines the prior probability distribution of each parameter. A prior probability distribution of each parameter is specified by a list *[para_name,prior_type,mu,sigma]*.

            **para_name**: string

                A name of the parameter (string) from {'k','p','c','beta','sigma','mu1'}

            **para_type**: string

                A probability distribution function from the below list

                'n': the normal distribution :math:`p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp[-\frac{(x-\mu)^2}{2\sigma^2}]`

                'ln': the log-normal distribution :math:`p(x)=\frac{1}{x\sqrt{2\pi\sigma^2}}\exp[-\frac{(\ln(x)-\mu)^2}{2\sigma^2}]`

                'f': the Dirac's delta :math:`p(x)=\delta(x-\mu)`  (This prior fixes a given parameter to be :math:`\mu`)

            **mu**: float
                the :math:`\mu`-value of the prior probability distribution function

            **sigma**: float
                the :math:`\sigma`-value of the prior probability distribution function. Set *sigma* to 0, if *'f'* is chosen for *para_type*.

        For example, if the user want to use the normal distribution with :math:`\mu=2.0` and :math:`\sigma=0.5` as the prior of the parameter :math:`\beta`, the prior object is given as:

        .. code-block:: Python

            prior = [['beta','n',2.0,0.5]]

        The user can introduce the prior for multiple parameters. The below code is the setting of the default prior.

        .. code-block:: Python

            import numpy as np
            prior =[]
            prior.append(['beta','n',0.85*np.log(10.0),0.15*np.log(10.0)])
            prior.append(['p','n',1.05,0.13])
            prior.append(['c','ln',-4.02,1.42])
            prior.append(['sigma','ln',np.log(0.2),1.0])

        Finally the prior object is passed to the keyword argument *prior* in :py:func:`Est` or :py:func:`EstFore`.

        .. code-block:: Python

            aft.EstFore(Data,t_lean,t_test,prior=prior)

        .. note::

            When the user uses the customized prior, it is strongly recommended to introduce a prior for the :math:`\sigma`-parameter for the computational stability. For example, the prior ['sigma','ln',np.log(0.2),1.0] is used for the :math:`\sigma`-parameter in the default prior.

.. _genfiles:

Generated Files
========================

    .. var:: param.pkl

        A pickle file that stores a parameter object :var:`param`. See :ref:`paraest` for the method to retrive :var:`param` from the pickle file.

    .. var:: param.pdf

        A figure showing (1) the MT plot superinposed with the time-varying magnitude of 50% detection rate and (2) the scatter plot matrix of the estimated parameter sets (red dots: MAP parameter values, gray dots: sampled parameter sets).

        .. image:: mu.jpg
            :height: 380px

        .. image:: sm.jpg
            :width: 400px

    .. var:: fore.txt

        A list of the target magnitude :math:`M_t`, the expected number of earthquakes in the testing period with :math:`M>M_t`, its 95% confidencial interval, the probability to have at least one earthquake with :math:`M>M_t` in the testing period.

    .. code-block:: none

        # (M_t) (expected_number) (lower bound of 95% interval) (upper bound of 95% interval) (probability)
        0.95         682.218         564         832      1.0000
        1.05         576.029         477         699      1.0000
        1.15         486.368         403         589      1.0000
        1.25         410.664         340         496      1.0000
        1.35         346.743         287         419      1.0000
        1.45         292.772         241         354      1.0000
        1.55         247.201         202         300      1.0000
        1.65         208.723         170         255      1.0000
        1.75         176.235         142         217      1.0000
        1.85         148.804         119         185      1.0000
        1.95         125.642          99         158      1.0000
        2.05         106.085          82         135      1.0000
        ...

    .. var:: fore.pdf

        A figure showing the forecast result.

        .. image:: fore1.jpg
            :width: 450px

        If a keyword *Data_test* is set in :py:func:`Fore`, the observation in the testing period is added.

        .. image:: fore2.jpg
            :width: 450px
