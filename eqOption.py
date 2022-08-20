"""
Author: Adrian Haerle
Date: 17.06.2021
"""

import numpy as np
from scipy.stats import norm
from datetime import date


class EqOption(object):
    def __init__(self, option_type, strike, expiry, t0, spot, r, q, sigma):
        """
        Basic vanilla equity option class with risk metrics.

        :param option_type: (1=Call & -1=Put)
        :param strike: absolute strike
        :param expiry: expiry date as datetime.date
        :param t0: current date as datetime.date
        :param spot: spot underlying
        :param r: annualized interest rates for respective time to expiry
        :param q: annualized dividend yield for respective time to expiry (discrete model)
        :param sigma: annualized implied volatility
        """
        self.option_type = option_type
        self.strike = float(strike)
        self.expiry = expiry
        self.t0 = t0
        self.spot = spot
        self.r = r
        self.q = q
        self.sigma = sigma
        self.time_to_expiry = (self.t0, self.expiry)
        self.forward = self.spot * np.exp((self.r - self.q) * self.time_to_expiry)

    @property
    def option_type(self):
        return self._option_type

    @option_type.setter
    def option_type(self, _type):
        if isinstance(_type, str):
            if _type == "c":
                self._option_type = 1
            elif _type == "p":
                self._option_type = -1
            else:
                raise ValueError("Value inserted ")
        else:
            self._option_type = int(_type)

    @property
    def time_to_expiry(self):
        return self._time_to_expiry

    @time_to_expiry.setter
    def time_to_expiry(self, current_dates):
        try:
            t0, expiry = current_dates
        except ValueError:
            raise ValueError("Pass an iterable with two items in the following format: (t0, expiry)")
        else:
            time_to_expiry = (expiry - t0).days / 365.2425
            if time_to_expiry > 0.0:
                self._time_to_expiry = time_to_expiry
            else:
                raise ValueError("Time to expiry smaller than 0.0 years is not possible")

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, vol):
        if vol > 0.0:
            self._sigma = float(vol)
        else:
            raise ValueError("Implied volatility smaller than 0.0 is not possible")

    @property
    def d1(self):
        return (np.log(self.forward / self.strike) + (self.sigma**2.0) * self.time_to_expiry * 0.5) / (self.sigma * self.time_to_expiry ** 0.5)

    @property
    def d2(self):
        return self.d1 - (self.sigma * self.time_to_expiry ** 0.5)

    def _price(self):
        """
        :return: Option price according to BSM
        """
        p = np.exp(-self.r * self.time_to_expiry) * (self.option_type * (self.forward * norm.cdf(self.option_type * self.d1)) - (self.strike * norm.cdf(self.option_type * self.d2)) * self.option_type)
        return p

    @property
    def price(self):
        """
        :return: Option price according to BSM
        """
        return self._price()

    @property
    def delta(self):
        return self._delta()

    @property
    def delta_volume(self):
        return self._delta_volume()

    @property
    def vega(self):
        return self._vega()

    @property
    def theta(self):
        return self._theta()

    @property
    def rho(self):
        return self._rho()

    @property
    def gamma(self):
        return self._gamma()

    @property
    def gamma_volume(self):
        return self._gamma_volume()

    @property
    def vanna(self):
        return self._vanna()

    @property
    def vanna_volume(self):
        return self._vanna_volume()

    @property
    def charm(self):
        return self._charm()

    @property
    def charm_volume(self):
        return self._charm_volume()

    @property
    def vomma(self):
        return self._vomma()

    @property
    def veta(self):
        return self._veta()

    def _delta(self):
        """
        Measures the rate of change of the option value with respect to changes in the underlying asset's price.
        :return: d(Value of Option) / d(Price of underlying)
        """
        if self.option_type == 1:
            return np.exp(-self.q * self.time_to_expiry) * norm.cdf(self.d1)
        else:
            return np.exp(-self.q * self.time_to_expiry) * (norm.cdf(self.d1) - 1)

    def _delta_volume(self):
        """
        Measures nominal volume in absolute terms of the underlying that is equivalent to option sensitivity.
        :return: -
        """
        return self.spot * self._delta()

    def _vega(self):
        """
        Measures sensitivity to changes of implied volatility (1%-point).
        :return: d(Value of Option) / d(Volatility)
        """
        return 0.01 * self.spot * np.exp(-self.q * self.time_to_expiry) * norm.pdf(self.d1) * self.time_to_expiry ** 0.5

    def _theta(self):
        """
        Measures the sensitivity of the option value with respect to the passage of time for 1 calendar day - aka "time decay".
        :return: d(Value of Option) / d(Time)
        """
        exp_r = np.exp(-self.r * self.time_to_expiry)
        exp_q = np.exp(-self.q * self.time_to_expiry)
        first_part = -self.spot * norm.pdf(self.d1) * self.sigma * exp_q / (2.0 * self.time_to_expiry ** 0.5)
        second_part = self.option_type * self.q * self.spot * norm.cdf(self.d1 * self.option_type) * exp_q
        third_part = -self.option_type * self.r * self.strike * exp_r * norm.cdf(self.d2 * self.option_type)
        return 1.0 / 365.2425 * (first_part + second_part + third_part)

    def _rho(self):
        """
        Measures the sensitivity of the option value with respect to the interest rate (1% increase).
        :return: d(Value of Option) / d(Interest Rate)
        """
        return 0.01 * self.option_type * self.strike * self.time_to_expiry * np.exp(-self.r * self.time_to_expiry) * norm.cdf(self.d2 * self.option_type)

    def _gamma(self):
        """
        Measures the sensitivity of delta with respect to changes in the underlying price (for a 1% change of the underlying).
        :return: d(Delta) / d(Price of underlying) --> d2(Value of Option) / d2(Price of underlying)
        """
        gamma_simple = (norm.pdf(self.d1) * np.exp(-self.q * self.time_to_expiry)) / (self.spot * self.sigma * self.time_to_expiry ** 0.5)
        return self.spot * gamma_simple * 0.01

    def _gamma_volume(self):
        """
        Measures nominal volume in absolute terms of the underlying that is equivalent to option's gamma sensitivity.
        :return: -
        """
        return self.spot * self._gamma()

    def _vanna(self):
        """
        Measures the sensitivity of the option delta with respect to changes in the implied volatility (1%-point change).
        :return: d(Delta) / d(Volatility) --> d2(Value of Option) / d(Price of underlying)d(Volatility)
        """
        return 0.01 * -np.exp(-self.q * self.time_to_expiry) * norm.cdf(self.d1) * self.d2 / self.sigma

    def _vanna_volume(self):
        """
        Measures nominal volume in absolute terms of the underlying that is equivalent to option's vanna sensitivity.
        :return: -
        """
        return self.spot * self._vanna()

    def _charm(self):
        """
        Measures the sensitivity of delta with respect to the passage of time (1 calendar day).
        :return: d(Delta) / d(Time) --> d2(Value of Option) / d(Price of underlying)d(Time)
        """
        charm_simple = self.q * np.exp(-self.q * self.time_to_expiry) * norm.cdf(self.d1) - np.exp(-self.q * self.time_to_expiry) * norm.pdf(self.d1) * (2 * (self.r - self.q) * self.time_to_expiry - self.d2 * self.sigma * self.time_to_expiry ** 0.5) / (2 * self.time_to_expiry * self.sigma * self.time_to_expiry ** 0.5)
        return self.option_type * charm_simple * 1.0 / 365.2425

    def _charm_volume(self):
        """
        Measures nominal volume in absolute terms of the underlying that is equivalent to option's charm sensitivity.
        :return: -
        """
        return self.spot * self._charm()

    def _vomma(self):
        """
        Measures sensitivity of vega with respect to volatility - aka "vega convexity".
        :return: d(Vega) / d(Volatility) --> d2(Value of Option) / d2(Volatility)
        """
        return 0.0001 * self.spot * np.exp(-self.q * self.time_to_expiry) * norm.pdf(self.d1) * self.time_to_expiry ** 0.5 * self.d1 * self.d2 / self.sigma

    def _veta(self):
        """
        Measures the sensitivity of vega with respect to the passage of time (1 calendar day).
        :return: d(Vega) / d(Time) --> d2(Value of Option) / d(Volatility)d(Time)
        """
        first_half = self.spot * np.exp(-self.q * self.time_to_expiry) * norm.pdf(self.d1) * self.time_to_expiry ** 0.5
        second_half = self.q + (self.r - self.q) * self.d1 / (self.sigma * self.time_to_expiry ** 0.5) - (1 + self.d1 * self.d2) / (2 * self.time_to_expiry)
        return first_half * second_half * 1.0 / 365.2425 * 0.01
